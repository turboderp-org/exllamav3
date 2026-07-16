from __future__ import annotations
import torch
from ..model.model import Model
from ..cache.cache import Cache
from ..cache.recurrent import RecurrentCache
from ..tokenizer.tokenizer import Tokenizer
from ..constants import PAGE_SIZE
from ..util import cuda_sync_active
from .pagetable import PageTable
from .job import Job
from .filter import Filter
from concurrent.futures import ThreadPoolExecutor
from .sampler import Sampler
from .visualizer import CacheVisualizer
import time
import threading
from ..tokenizer import MMEmbedding
from ..util import profile_opt

class Generator:

    def __init__(
        self,
        model: Model,
        cache: Cache,
        tokenizer: Tokenizer,
        max_batch_size: int = 256,
        max_chunk_size: int = 2048,
        max_q_size: int = 8,
        draft_model: Model | None = None,
        draft_cache: Cache | None = None,
        num_draft_tokens: int | None = None,
        show_visualizer: bool = False,
        enable_defrag: bool = True,
        recurrent_cache_size: int = 4 * 1024**3,
        recurrent_checkpoint_interval: int = None,
        recurrent_checkpoint_interval_pp: int = 32768,
        ngram_match_min: int = 0,
        **kwargs
    ):
        """
        Initialize generator

        :param model:
            The model (loaded)

        :param cache:
            Paged cache

        :param tokenizer:
            Tokenizer

        :param max_batch_size:
            The maximum number of sequences to process in parallel. The generator will also limit this
            dynamically considering the available cache space.

        :param max_chunk_size:
            Maximum number of tokens to process in parallel during prefill (prompt ingestion). Should not
            exceed the model's max_input_len but can be lowered to trade off prompt speed for a shorter
            interruption to ongoing jobs when a new job is started.

        :param max_q_size:
            Maximum number of tokens to evaluate per sequence during generation. Leave this at the default
            (8) unless there's a good reason to increase it.

        :param draft_model:
            Draft model. Enables speculative decoding with draft, and must be specified along with
            draft_cache. Note that speculative decoding with many parallel jobs is likely not advantageous.

        :param draft_cache:
            Cache allocated for draft model. Must be same size as main cache.

        :param num_draft_tokens:
            Number of future tokens to draft. Default is 4 unless the draft model has a preference (e.g.
            from DFlash block size)

        :param ngram_match_min:
            Minimum number of tokens to match for n-gram draft (0 = disabled).

        :param show_visualizer:
            Open window to render visualization of cache (for debug/demonstration purposes)

        :param enable_defrag:
            Defragment cache periodically

        :param recurrent_cache_size:
            Size of recurrent cache, in bytes. Recurrent cache resides in system RAM. Default is 4 GB.
            Ignored if model doesn't use recurrent states

        :param recurrent_checkpoint_interval:
            Minimum number of tokens between recurrent checkpoints in model output and the tail end of
            the prompt. Must be a multiple of the page size. Model architecture determines default

        :param recurrent_checkpoint_interval_pp]:
            Minimum number of tokens between recurrent checkpoints during prompt ingestion. Must be a
            multiple of the page size. Default is 32768 tokens

        :param kwargs:
        """

        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        cfg = self.model.config
        self.padded_vocab_size = ((cfg.vocab_size + 31) // 32) * 32

        # Paging
        self.pagetable = PageTable(self, cache)
        self.max_total_tokens = PAGE_SIZE * self.pagetable.max_pages

        # Draft model
        self.draft_model = draft_model
        self.draft_cache = draft_cache
        if draft_model:
            assert not ngram_match_min, \
                "Cannot use both draft model and n-gram draft."
            assert draft_cache is not None, \
                "Must supply cache for draft model"
            assert draft_cache.max_num_tokens == cache.max_num_tokens, \
                "Cache and draft cache must be same size"
            assert not draft_model.caps.get("recurrent_states"), \
                "Speculative decoding with recurrent draft model not supported."
            if num_draft_tokens:
                self.num_draft_tokens = num_draft_tokens
            else:
                self.num_draft_tokens = draft_model.caps.get("default_draft_size", 4)
        elif ngram_match_min:
            self.num_draft_tokens = num_draft_tokens if num_draft_tokens is not None else 4
        else:
            self.num_draft_tokens = 0

        self.ngram_match_min = ngram_match_min
        max_q_size = max(self.num_draft_tokens + 1, max_q_size)

        # Chunking/partitioning
        self.max_batch_size = max_batch_size
        self.max_chunk_size = max_chunk_size

        # Job queues
        self.job_serial = 0
        self.pending_jobs = []
        self.active_jobs = []

        # Filter threads
        self.filter_pool = ThreadPoolExecutor(max_workers = 16)
        self.filter_queue = []

        # Pinned staging buffer for batched token readback in iterate_gen
        self.sample_pinned = None
        self.staging_buffers = {}

        # Buffers
        if draft_model or ngram_match_min:
            self.draft_input_ids_pinned = torch.empty(
                (max_batch_size, 1),
                dtype = torch.long,
                pin_memory = False
            )
            self.draft_ids_pinned = torch.empty(
                (max_batch_size, self.num_draft_tokens),
                dtype = torch.long,
                pin_memory = False
            )

        # Visualizer
        if show_visualizer:
            self.visualizer = CacheVisualizer(self.pagetable.max_pages)
        else:
            self.visualizer = None

        # Defrag
        self.enable_defrag = enable_defrag

        # Recurrent cache
        self.recurrent_cache_size = recurrent_cache_size
        if self.model.caps.get("recurrent_states"):
            self.recurrent_cache = RecurrentCache(self.model, recurrent_cache_size)
            # Limit batch size if cache has recurrent states
            self.max_batch_size = min(self.max_batch_size, cache.num_slots)
        else:
            self.recurrent_cache = None
        if recurrent_checkpoint_interval is None:
            recurrent_checkpoint_interval = model.caps.get("default_recurrent_checkpoint_interval", 2048)

        assert recurrent_checkpoint_interval % PAGE_SIZE == 0 and recurrent_checkpoint_interval % PAGE_SIZE == 0, \
            "checkpoint interval must be a multiple of the page size (256)"
        def ceil_span(a, b):
            return (a + b - 1) // b * b
        recurrent_checkpoint_interval = recurrent_checkpoint_interval
        self.recurrent_checkpoint_interval = recurrent_checkpoint_interval
        self.recurrent_checkpoint_interval_pp = ceil_span(recurrent_checkpoint_interval_pp, self.max_chunk_size)

        # Drafting mode
        if draft_model is not None and draft_model.caps.get("attach_target"):
            draft_model.attach_to(model)
        self.dflash_draft = self.draft_model is not None and self.draft_model.caps.get("dflash_draft", False)
        self.mtp_draft = self.draft_model is not None and self.draft_model.caps.get("mtp_draft", False)


    def num_remaining_jobs(self):
        return len(self.pending_jobs) + len(self.active_jobs)

    def num_active_jobs(self):
        return len(self.active_jobs)

    def num_pending_jobs(self):
        return len(self.pending_jobs)


    def clear_queue(self):
        """
        Abort all active and pending jobs
        """

        num_jobs = self.num_remaining_jobs()
        for job in self.active_jobs + self.pending_jobs:
            job.deallocate_pages()
        self.active_jobs.clear()
        self.pending_jobs.clear()
        if num_jobs and not self.num_remaining_jobs():
            self.pagetable.defrag()


    def enqueue(
        self,
        job: Job | list[Job]
    ) -> int | list[int]:
        """
        Adds a job or list of jobs to the queue.

        Each job is prepared against this generator before it is appended to the pending queue. Preparation assigns
        the current job_serial, hashes prompt pages for later cache lookup, and validates that the request can fit
        the configured cache and batch limits. The serial is then incremented and returned to the caller; it is also
        included in started, prefill and streaming result dictionaries so clients can correlate events with the
        logical request. Requeued jobs keep their original serial number internally so a long generation still
        appears as one stream.

        returns:
            int: (List of) unique serial number(s) for job(s)
        """

        if isinstance(job, list):
            serials = []
            for j in job:
                serials.append(self.enqueue(j))
            return serials

        job.prepare_for_queue(self, self.job_serial)
        self.job_serial += 1
        self.pending_jobs.append(job)
        return job.serial_number


    def cancel(
        self,
        job: Job
    ):
        """
        Cancel a single pending or active job.

        Pending jobs are removed before they ever allocate cache pages. Active jobs first release their page
        references and recurrent state, then leave active_jobs so future iterations stop sampling them. If the
        cancellation drains the generator completely, the page table is defragmented immediately because no active
        block tables still depend on the current physical page ordering.
        """

        num_jobs = self.num_remaining_jobs()

        if job in self.pending_jobs:
            self.pending_jobs.remove(job)
        elif job in self.active_jobs:
            job.deallocate_pages()
            self.active_jobs.remove(job)
        if num_jobs and not self.num_remaining_jobs():
            self.pagetable.defrag()


    @torch.inference_mode
    def iterate(self) -> list[dict]:
        """
        Performs inference on available jobs.

        :return:
            List of dicts:

            # Job has started
            {
                "job": Job  - reference to job
                "stage": "started"
                "identifier":  - optional identifier
                "serial": int  - job serial number
                "eos": bool  - always False at this stage
            }

            # Prefill is underway
            {
                "job": Job  - reference to job
                "stage": "prefill"
                "curr_progress": int  - prompt tokens ingested so far
                "max_progress": int  - total prompt tokens to ingest
                "identifier":  - optional identifier
                "serial": int   - job serial number
                "eos": bool  - always False at this stage
            }

            # Generation is underway
            {
                "job": Job  - reference to job
                "stage": "streaming"
                "identifier":  - optional identifier
                "serial": int   - job serial number
                "eos": bool  - True if stop condition has been met

                optional, if eos:
                    "eos_reason":  - one of:
                        "stop_token"  (stop token was reached)
                        "stop_string"  (stop string was completed)
                        "max_new_tokens"  (max_new_tokens reached)
                        "end_filter"  (filter reached end state with eos_after_completed = True)
                        "loop_detected"  (loop detection triggered)
                    optional, if "eos_reason" == "stop_token":
                        "eos_triggering_token_id": int
                        "eos_triggering_token_str": str
                    optional, if "eos_reason" == "stop_string":
                        "eos_triggering_string": str
                    "full_completion": str  - full text completion
                    "new_tokens": int  - number of tokens generated
                    "time_enqueued": float  - time from job was enqueued until it started, in seconds
                    "time_prefill": float  - time to first token, in seconds
                    "time_generate": float  - time to last token, in seconds
                    optional, if SD enabled:
                        "accepted_draft_tokens": int
                        "rejected_draft_tokens": int

                "text": str  - streamed text output. Does not include prefix from healed token, or stop string
                "token_ids": torch.Tensor  - output tokens, shape (1, n)
                "token_probs": torch.Tensor  - last sampling probability of output tokens, shape (1, n)
                "top_k_tokens": torch.Tensor  - shape (1, n, k)
                "top_k_probs": torch.Tensor  - shape (1, n, k)
                "logits": torch.Tensor  - shape (1, n, vocab_size)
            }
        """

        results = []
        self.iterate_start_jobs(results)

        # Perform one round of prefill
        for job in self.active_jobs:
            job.prefill(results)

        # Recurrent checkpoints
        if self.recurrent_cache is not None:
            self.recurrent_checkpoint()

        # Generation with draft model
        if self.draft_model:
            if self.dflash_draft:
                draft_tokens = self.iterate_draftmodel_dflash_gen(results)
                self.iterate_gen(results, draft_tokens)
            elif self.mtp_draft:
                draft_tokens = self.iterate_draftmodel_mtp_gen(results)
                self.iterate_gen(results, draft_tokens)
            else:
                draft_tokens = self.iterate_draftmodel_gen(results)
                self.iterate_gen(results, draft_tokens)

        # Generation with n-gram draft
        elif self.ngram_match_min:
            draft_tokens = self.iterate_ngram_gen(results)
            self.iterate_gen(results, draft_tokens)

        # Regular generation
        else:
            self.iterate_gen(results)

        # Visualization
        if self.visualizer:
            self.update_visualizer()

        # Finished iteration
        return results


    def recurrent_checkpoint(self):
        for job in self.active_jobs:
            job.maybe_stash_recurrent(self.recurrent_cache)


    def update_visualizer(self):
        chains = []
        for job in self.active_jobs:
            for seq in job.sequences:
                idx = job.serial_number
                chain = [page.page_index for page in seq.allocated_pages]
                chains.append((idx, chain))
        usage = [0] * self.pagetable.max_pages
        for page in self.pagetable.all_pages:
            usage[page.page_index] = page.kv_position / PAGE_SIZE
        self.visualizer.update(chains, usage)


    def iterate_draftmodel_gen(self, results: list):

        # Get shape of active batch
        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + self.num_draft_tokens + 1)
            batch_size += 1
        if batch_size == 0:
            return None

        # Create block index table for batch
        max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        block_index = torch.zeros((batch_size, max_pages_batch), dtype = torch.int32)
        cache_seqlens = torch.zeros((batch_size,), dtype = torch.int32)
        batch = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            for seq in job.sequences:
                seq_block_index = seq.block_index_tensor[:, :max_pages_batch]
                block_index[batch:batch+1, :seq_block_index.shape[-1]].copy_(seq_block_index)
                cache_seqlens[batch] = seq.kv_position
                batch += 1

        # Indexed embeddings not supported when drafting
        # TODO: Allow multimodal draft model, perhaps with dummy embeddings?
        for job in self.active_jobs:
            assert not job.embeddings, \
                "MM embeddings not supported while using draft model."

        # Collect input IDs
        input_ids_list = []
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            if job.time_first_token is None:
                cuda_sync_active()
                job.time_first_token = time.time()
            job_ids = job.get_input_ids_list()
            input_ids_list += job_ids
        batch_ids = self.draft_input_ids_pinned[:batch_size, :]
        batch_ids.copy_(torch.cat(input_ids_list, dim = 0))

        # Greedy sample num_draft_tokens batched tokens
        for idx in range(self.num_draft_tokens):
            batch_logits = self.draft_model.forward(
                input_ids = batch_ids,
                params = {
                    "attn_mode": "flash_attn",
                    "block_table": block_index,
                    "cache": self.draft_cache,
                    "cache_seqlens": cache_seqlens,
                }
            )
            new_ids = torch.argmax(batch_logits, dim = -1)
            self.draft_ids_pinned[:batch_size, idx:idx+1].copy_(new_ids)
            batch_ids.copy_(new_ids)
            cache_seqlens += 1

        self.draft_model.prefill(
            input_ids = batch_ids,
            params = {
                "attn_mode": "flash_attn",
                "block_table": block_index,
                "cache": self.draft_cache,
                "cache_seqlens": cache_seqlens
            }
        )

        return self.draft_ids_pinned


    def iterate_draftmodel_mtp_gen(self, results: list):

        # Get shape of active batch
        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + self.num_draft_tokens + 1)
            batch_size += 1
        if batch_size == 0:
            return None

        # Create block index table for batch
        max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        block_index = torch.zeros((batch_size, max_pages_batch), dtype = torch.int32)
        cache_seqlens = torch.zeros((batch_size,), dtype = torch.int32)
        batch = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            for seq in job.sequences:
                seq_block_index = seq.block_index_tensor[:, :max_pages_batch]
                block_index[batch:batch+1, :seq_block_index.shape[-1]].copy_(seq_block_index)
                cache_seqlens[batch] = seq.kv_position
                batch += 1

        # Collect input IDs
        input_ids_list = []
        mtp_hidden_list = []
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            assert len(job.sequences) == 1, "Qwen3.5 MTP drafting does not currently support CFG/multi-sequence jobs"
            if job.mtp_last_hidden is None:
                # A one-token prompt has no token to prefill before the generation input.
                # Run one normal target step first; iterate_gen() will initialize MTP state.
                return None
            if job.time_first_token is None:
                cuda_sync_active()
                job.time_first_token = time.time()
            job_ids = job.get_input_ids_list()
            input_ids_list += job_ids
            mtp_hidden_list.append(job.mtp_last_hidden)
        batch_ids = self.draft_input_ids_pinned[:batch_size, :]
        batch_ids.copy_(torch.cat(input_ids_list, dim = 0))
        temp_hidden = torch.cat(mtp_hidden_list, dim = 0)

        # Greedy sample num_draft_tokens batched tokens
        for idx in range(self.num_draft_tokens):
            params = {
                "target_hidden": temp_hidden,
                "attn_mode": "flash_attn",
                "block_table": block_index,
                "cache": self.draft_cache,
                "cache_seqlens": cache_seqlens,
            }
            batch_state = self.draft_model.forward(batch_ids, params)
            lm_head = self.model.modules[self.model.logit_layer_idx]
            batch_state = lm_head.prepare_for_device(batch_state, params)
            new_ids = self.draft_model.sample_from_state(batch_state, params)
            self.draft_ids_pinned[:batch_size, idx:idx+1].copy_(new_ids)
            batch_ids.copy_(new_ids)
            cache_seqlens += 1
            temp_hidden = batch_state


        return self.draft_ids_pinned


    # TODO: Refactor, share code with other draft fns
    def iterate_draftmodel_dflash_gen(self, results: list):

        # Get shape of active batch
        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(
                max_seq_len,
                job.get_max_seq_len() + self.num_draft_tokens + 1,
                job.get_max_seq_len() + self.draft_model.config.block_size + 1
            )
            batch_size += 1
        if batch_size == 0:
            return None

        # Create block index table for batch
        max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        block_index = torch.zeros((batch_size, max_pages_batch), dtype = torch.int32)
        cache_seqlens = torch.zeros((batch_size,), dtype = torch.int32)
        batch = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            for seq in job.sequences:
                seq_block_index = seq.block_index_tensor[:, :max_pages_batch]
                block_index[batch:batch+1, :seq_block_index.shape[-1]].copy_(seq_block_index)
                cache_seqlens[batch] = seq.kv_position
                batch += 1

        # Collect input IDs
        input_ids_list = []
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            if job.time_first_token is None:
                cuda_sync_active()
                job.time_first_token = time.time()
            job_ids = job.get_input_ids_list()
            input_ids_list += job_ids
        batch_ids = self.draft_input_ids_pinned[:batch_size, :]
        batch_ids.copy_(torch.cat(input_ids_list, dim = 0))

        # Run draft model
        params = {
            "attn_mode": "flash_attn",
            "block_table": block_index,
            "cache": self.draft_cache,
            "cache_seqlens": cache_seqlens,
        }
        out_state = self.draft_model.forward(
            input_ids = batch_ids,
            params = params,
        )
        new_ids = self.draft_model.sample_from_state(out_state, params)

        # Crop out the first token after sampling to keep batch contiguous for lm_head
        new_ids = new_ids[:, 1:]
        self.draft_ids_pinned[:batch_size, :self.num_draft_tokens].copy_(new_ids[:batch_size, :self.num_draft_tokens])
        return self.draft_ids_pinned


    def iterate_ngram_gen(self, results: list):

        # Get shape of active batch
        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + self.num_draft_tokens + 1)
            batch_size += 1
        if batch_size == 0:
            return None

        # Generate draft
        draft_ids = []
        min_len = self.num_draft_tokens
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            d = job.get_ngram_draft(self.num_draft_tokens)
            min_len = min(min_len, d.shape[-1])
            draft_ids.append(d)

        if min_len == 0:
            return None

        # Trim to minimum length in batch
        draft_ids = torch.cat([d[:, :min_len] for d in draft_ids], dim = 0)
        return draft_ids


    def _staging(self, name, rows: int, width: int | None = None, dtype = torch.int32):
        """
        Reusable pinned staging buffer, keyed by name and row width, grown by rows on demand.
        """
        key = (name, width)
        buf = self.staging_buffers.get(key)
        if buf is None or buf.shape[0] < rows:
            alloc_rows = max(rows, 32)
            shape = (alloc_rows,) if width is None else (alloc_rows, width)
            buf = torch.zeros(shape, dtype = dtype, pin_memory = True)
            self.staging_buffers[key] = buf
        return buf[:rows]


    def iterate_gen(self, results: list, draft_tokens: torch.Tensor | None = None):

        # Get shape of active batch
        # Only jobs that have finished prefill can participate in token generation. The maximum sequence length
        # determines how many cache pages the temporary block table needs for this iteration.
        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + self.num_draft_tokens)
            batch_size += len(job.sequences)
        if batch_size == 0:
            return
        if draft_tokens is not None:
            max_seq_len += draft_tokens.shape[-1]

        # Create block index table for batch
        # The model sees a compact batch, so build per-row mappings from logical page positions to physical cache
        # page indices, along with current cache lengths and optional MRoPE position offsets.
        # Block-table width is padded to a multiple of 16 pages so the pinned staging buffers
        # cover a few distinct widths only; the extra (zeroed) columns are never dereferenced
        # since the kernels bound their reads by the cache lengths
        max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        max_pages_batch = (max_pages_batch + 15) // 16 * 16
        block_index = self._staging("block_index", batch_size, max_pages_batch)
        block_index.zero_()
        cache_seqlens = self._staging("cache_seqlens", batch_size)
        batch = 0
        use_offsets = "mrope" in self.model.caps
        positions = self._staging("positions", batch_size) if use_offsets else None
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            for seq in job.sequences:
                seq_block_index = seq.block_index_tensor[:, :max_pages_batch]
                block_index[batch:batch+1, :seq_block_index.shape[-1]].copy_(seq_block_index)
                cache_seqlens[batch] = seq.kv_position
                if use_offsets:
                    positions[batch] = seq.kv_position + job.alt_rope_offset
                batch += 1

        # Collect input IDs and indexed embeddings
        # Jobs may contribute multiple sequence rows. logit_mapping records the slice of the model output that
        # belongs to each active job, while batch_jobs keeps the corresponding Job objects in compact-batch order.
        input_ids_list = []
        active_embeddings = []
        logit_mapping = []
        batch_jobs = []
        for job in self.active_jobs:
            logit_mapping.append(len(input_ids_list))
            if not job.is_prefill_done(): continue
            if job.time_first_token is None:
                cuda_sync_active()
                job.time_first_token = time.time()
            job_ids = job.get_input_ids_list(draft_tokens, len(input_ids_list), add_to_cache = True)
            input_ids_list += job_ids
            batch_jobs.append(job)
            active_embeddings += job.embeddings
        logit_mapping.append(len(input_ids_list))
        ids_width = input_ids_list[0].shape[-1]
        if all(ids.shape[-1] == ids_width for ids in input_ids_list):
            ids_staging = self._staging("batch_ids", len(input_ids_list), ids_width, torch.long)
            batch_ids = torch.cat(input_ids_list, dim = 0, out = ids_staging)
        else:
            batch_ids = torch.cat(input_ids_list, dim = 0)

        # Collect recurrent states for batch
        # Recurrent models carry mutable state beside the K/V cache; pass one state object per compact batch job so
        # the model can advance or rewind it consistently with accepted draft tokens.
        batch_states = None
        if self.recurrent_cache is not None:
            batch_states = [job.recurrent_state for job in batch_jobs]

        # GPU workload is scheduled here, so launch any sampling filters that can run in the background
        for job in batch_jobs:
            if job.new_tokens < 0: continue
            assert len(job.filter_futures) == 0
            for f in job.filters:
                if not f.is_active: continue
                if f.use_background_worker():
                    job.filter_futures.append(self.filter_pool.submit(f.get_next_logit_mask))
                else:
                    job.filter_futures.append(None)

        # Get logit batch from model. Forward writes new K/V entries into the cache for the supplied positions and
        # returns logits for either one target token or a target-plus-draft verification window.
        params = {
            "attn_mode": "flash_attn",
            "block_table": block_index,
            "cache": self.cache,
            "cache_seqlens": cache_seqlens,
            "recurrent_states": batch_states,
            "indexed_embeddings": active_embeddings,
            "positions": positions,
            "recurrent_history": draft_tokens is not None,
            "pinned_staging": True,
        }
        if self.draft_model:
            params.update(self.draft_model.draft_verifier_params)
        batch_logits = self.model.forward(
            input_ids = batch_ids,
            params = params,
        )

        # Keep only the fields needed below for draft-cache updates and drop the params dict so it cannot extend
        # references to recurrent state objects past this iteration.
        p_export_states = params.get("export_states")
        p_cache_seqlens = params.get("cache_seqlens")
        params = None

        # Background filters were launched before forward(); synchronous filters run now to overlap CPU work with
        # queued GPU execution as much as possible.
        for job in batch_jobs:
            if job.new_tokens < 0: continue
            assert len(job.logit_masks) == 0
            for f in job.filters:
                if not f.is_active: continue
                if f.use_background_worker():
                    job.logit_masks.append(None)
                else:
                    job.logit_masks.append(f.get_next_logit_mask())

        # Prepare past IDs (for sequences that need them for repetition penalty etc.) and logit masks where needed
        for job in batch_jobs:
            job.prepare_logit_mask()
            job.prepare_sampling_past_ids()

        # TODO: Batch sampling
        # Pass to jobs to sample
        # Sampling is still job-by-job. Each accepted token updates job state, may emit a stream result, may finish
        # the job, or may request requeueing once the configured per-job token budget is reached.
        completed_jobs = []
        requeuing_jobs = []
        accepted_lengths = []
        rewound_jobs = set()
        j = 0

        # Reject the trailing draft positions after the last accepted token at index i: count them, roll back the
        # job's recurrent state and return cache pages to the accepted position
        def reject_remainder(job_, j_, i_, batch_states_):
            num_rejected = batch_logits.shape[1] - 1 - i_
            if num_rejected == 0:
                return 0
            job_.rejected_draft_tokens += num_rejected

            # Rewind recurrent states
            if batch_states_ is not None:
                batch_states_[j_].rewind(num_rejected)

            # Rewind cache position (draft model cache layout is always the same as target)
            for seq_ in job_.sequences:
                r = num_rejected
                while r:
                    pos = seq_.kv_position + r
                    page = seq_.allocated_pages[(pos - 1) // PAGE_SIZE]
                    rp = min(page.kv_position, r)
                    page.kv_position -= rp
                    r -= rp
            return num_rejected

        # Without a draft, each job samples exactly one independent token, so all sampler chains
        # are launched first and the results collected in a second pass: the first collect
        # absorbs the whole GPU tail and the remaining reads are cheap, instead of one full
        # launch-to-readback round trip per job. Sampler launch order (and with it each job's
        # RNG draw) matches the serial loop exactly. Per-job constraints (filters, penalties)
        # are unaffected because no job's token depends on another job's result.
        if draft_tokens is None and batch_logits.shape[1] == 1:
            # Single-token results stage through one pinned buffer and one synchronize, so the
            # batch pays one launch-to-readback round trip instead of one per job
            pinned = self.sample_pinned
            if pinned is None or pinned.shape[0] < batch_size:
                self.sample_pinned = pinned = torch.empty(
                    (max(batch_size, 32), 1), dtype = torch.long, pin_memory = True
                )
            launched = []
            for job, a, b in zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:]):
                if a == b: continue
                token_logits = batch_logits[a:b, :, :]
                sampled = job.receive_logits(token_logits)
                next_token = sampled[0]
                if next_token.is_cuda and next_token.numel() == 1:
                    k = len(launched)
                    pinned[k:k + 1].copy_(next_token.view(1, 1), non_blocking = True)
                    sampled = (pinned[k:k + 1],) + sampled[1:]
                launched.append((job, token_logits, sampled))
            if launched:
                torch.cuda.synchronize(batch_logits.device)

            for job, token_logits, (next_token, next_k_tokens, next_k_probs, next_prob) in launched:
                eos, sampled_token, rq = job.receive_sample(
                    token_logits,
                    next_token,
                    next_k_tokens,
                    next_k_probs,
                    next_prob,
                    results,
                )
                # Single-token rewinds need no batch cleanup, but the post-batch draft carry update must still be
                # suppressed so it doesn't resurrect a stale MTP state
                if job.checkpoint_rewound:
                    job.checkpoint_rewound = False
                    rewound_jobs.add(id(job))

                # Requeue. Requeueing is only supported for single-sequence jobs because the replacement job
                # uses the full sequence generated so far as its next prompt.
                if len(job.sequences) == 1 and rq:
                    requeuing_jobs.append(job)
                elif eos:
                    completed_jobs.append(job)
                accepted_lengths.append(1)

        # With a draft, positions within a job are consumed strictly serially: acceptance of
        # position n gates position n+1, and constrained-decoding masks and sampling past IDs
        # must advance between positions.
        else:
            for idx, (job, a, b) in enumerate(zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:])):
                if a == b: continue
                job_logits = batch_logits[a:b, :, :]
                accepted_length = 1
                rejected = 0

                for i in range(batch_logits.shape[1]):
                    token_logits = job_logits[:, i:i + 1, :]
                    next_token, next_k_tokens, next_k_probs, next_prob = job.receive_logits(
                        token_logits,
                    )
                    eos, sampled_token, rq = job.receive_sample(
                        token_logits,
                        next_token,
                        next_k_tokens,
                        next_k_probs,
                        next_prob,
                        results,
                    )

                    # Requeue. Requeueing is only supported for single-sequence jobs because the replacement job
                    # uses the full sequence generated so far as its next prompt. Unconsumed draft positions must
                    # be rejected here so the recurrent state and page positions match the accepted sequence
                    # before the requeue stash.
                    if len(job.sequences) == 1 and rq:
                        if draft_tokens is not None:
                            rejected = reject_remainder(job, j, i, batch_states)
                        requeuing_jobs.append(job)
                        break

                    # EOS. Stop sampling this job immediately once a stop condition, filter condition or max
                    # token limit produces an EOS event.
                    if eos:
                        completed_jobs.append(job)
                        break

                    # A banned-string match inside receive_sample just rewound the job, resetting cache pages and
                    # rolling back or replacing any recurrent state. The remaining logit positions extend a
                    # sequence that no longer exists, so abandon them; everything is already consistent, and the
                    # usual rejection rollback and state normalization must not run on top of the rewind.
                    if job.checkpoint_rewound:
                        job.checkpoint_rewound = False
                        rewound_jobs.add(id(job))
                        if draft_tokens is not None:
                            job.rejected_draft_tokens += batch_logits.shape[1] - 1 - i
                            rejected = -1
                        break

                    # Continue sampling from logit batch as long as result matches draft, unless hitting
                    # checkpoint mark. For speculative decoding, consume additional logits only while the
                    # sampled target token matches the draft token. A recurrent checkpoint boundary also stops
                    # draft acceptance so state can be stashed at an exact page boundary.
                    if draft_tokens is not None and i < batch_logits.shape[1] - 1:
                        cp_boundary = batch_states is not None and job.is_checkpoint_boundary()
                        if draft_tokens[j, i].item() != sampled_token.item() or cp_boundary:
                            rejected = reject_remainder(job, j, i, batch_states)
                            break

                        # Accept draft token
                        else:
                            job.accepted_draft_tokens += 1
                            accepted_length += 1

                            # Advance filters
                            for f in job.filters:
                                if not f.is_active: continue
                                if f.use_background_worker():
                                    job.filter_futures.append(self.filter_pool.submit(f.get_next_logit_mask))
                                else:
                                    job.logit_masks.append(f.get_next_logit_mask())
                                    job.filter_futures.append(None)

                            # Update masks and past IDs
                            job.prepare_logit_mask()
                            job.prepare_sampling_past_ids()

                # Make sure outgoing state is valid if entire draft was accepted
                if batch_states and draft_tokens is not None and rejected == 0:
                    batch_states[j].rewind(0)

                accepted_lengths.append(accepted_length)
                j += 1

        # Accept new target_hidden if DFlash. DFlash draft models can update their cache from target-model hidden
        # states for the tokens accepted above, keeping draft and target cache layouts aligned.
        if self.dflash_draft:
            self.draft_model.update_kv_from_target(
                target_hidden = p_export_states,
                cache = self.draft_cache,
                lengths = accepted_lengths,
                params = {
                    "block_table": block_index,
                    "cache_seqlens": p_cache_seqlens,
                }
            )

        # Accept new target_hidden if MTP. MTP draft models can update their cache from target-model hidden
        # states for the tokens accepted above, keeping draft and target cache layouts aligned.
        if self.mtp_draft:
            target_hidden = p_export_states[-1]
            accepted_idx = 0
            for job, a_idx, b_idx in zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:]):
                if a_idx == b_idx:
                    continue
                accepted_length = accepted_lengths[accepted_idx]
                accepted_idx += 1

                # A banned-string rewind invalidated this job's carry; leave it unset so drafting pauses until the
                # next target forward provides a fresh one, and don't propagate hidden states from the abandoned
                # window into the draft cache
                if id(job) in rewound_jobs:
                    continue

                # Position K was drafted from the last target state already. Replace accepted
                # speculative positions K+1..K+A-1 with the corresponding target-state inputs.
                if accepted_length > 1:
                    self.draft_model.prefill(
                        batch_ids[a_idx:b_idx, 1:accepted_length],
                        {
                            "attn_mode": "flash_attn",
                            "block_table": block_index[a_idx:b_idx],
                            "cache": self.draft_cache,
                            "cache_seqlens": p_cache_seqlens[a_idx:b_idx] + 1,
                            "target_hidden": target_hidden[a_idx:b_idx, :accepted_length - 1, :],
                        },
                    )

                # The next unprocessed token is paired with the preceding target hidden state.
                job.mtp_last_hidden = target_hidden[
                    a_idx:b_idx, accepted_length - 1:accepted_length, :
                ].clone()

        # Release pages for completed jobs. Finished and requeued jobs no longer need their active page references.
        # Requeued recurrent jobs may stash the last checkpoint first so the next queued job can resume from cached
        # recurrent state.
        num_jobs = self.num_remaining_jobs()
        for job in completed_jobs + requeuing_jobs:
            if job in requeuing_jobs and self.recurrent_cache is not None:
                job.maybe_stash_recurrent(self.recurrent_cache, PAGE_SIZE)
            job.deallocate_pages()
            self.active_jobs.remove(job)

        # Requeue jobs. Puts replacement jobs at the front so long generations continue promptly after they yield
        # their cache pages.
        for job in requeuing_jobs:
            rq_job = job.prepare_for_requeue()
            self.pending_jobs.insert(0, rq_job)

        # Defrag. Physical page indices can only be compacted when no active block tables are using them.
        if num_jobs and not self.num_remaining_jobs():
            self.pagetable.defrag()


    def iterate_start_jobs(self, results: list):
        """
        Move pending jobs into the active set when batch and cache capacity allow.

        Jobs are considered in queue order, but a job can be skipped temporarily if it would exceed max_batch_size
        or needs more fresh pages than are currently unreferenced. Later jobs may start if they fit, which improves
        utilization, but each skipped job accumulates a skip count; once any skipped job reaches max_skips, startup
        stops for this iteration to preserve approximate queue fairness. Started jobs allocate their cache pages
        immediately and emit a "started" event.
        """

        # Get current max batch
        current_max_batch = 0
        for job in self.active_jobs:
            current_max_batch += len(job.sequences)

        # Start new jobs if possible
        if (self.pagetable.num_unreferenced_pages() and
            len(self.pending_jobs) and
            current_max_batch < self.max_batch_size):

            skipped_jobs = []
            for job in self.pending_jobs.copy():

                if (len(job.sequences) + current_max_batch > self.max_batch_size or
                        job.current_new_pages_required() > self.pagetable.num_unreferenced_pages()):
                    skipped_jobs.append(job)
                    continue

                # Make sure the job we're about to add doesn't skip a job that's been skipped too many times
                for j in skipped_jobs:
                    if j.skips >= j.max_skips:
                        return
                for j in skipped_jobs:
                    j.skips += 1

                # Add job to active list
                self.pending_jobs.remove(job)
                self.active_jobs.append(job)
                job.activate()

                # Allocate pages for job
                job.allocate_pages()
                current_max_batch += len(job.sequences)

                r = {
                    "job": job,
                    "stage": "started",
                    "eos": False,
                    "serial": job.serial_number,
                }
                if job.identifier is not None:
                    r.update({ "identifier": job.identifier })
                results.append(r)


    def generate(
        self,
        prompt: list[tuple] | list[str] | tuple | str,
        max_new_tokens: int | None = None,
        min_new_tokens: int = 0,
        seed: int | None = None,
        sampler: Sampler | list[Sampler] | None = None,
        token_healing: bool = False,
        encode_special_tokens: bool = False,
        decode_special_tokens: bool = False,
        stop_conditions: list[int | str] | None = None,
        add_bos: bool = False,
        abort_event: threading.Event | None = None,
        completion_only: bool = False,
        filters: list[list[Filter]] | list[Filter] | None = None,
        return_last_results: bool = False,
        embeddings: list[MMEmbedding] | list[list[MMEmbedding]] | None = None,
        max_rq_tokens: int | None = None,
        stop_on_loop: tuple[int, int] = None,
        **kwargs
    ):
        """
        This is a utility function for easily generating one or more completions from one or more prompt strings. For
        more versatility and streaming functionality, use the async wrapper or create Job objects directly, enqueue()
        them and call iterate() to receive one token at a time

        :param prompt:
            If this argument is a list, its length determines the batch size, and the output will be a list of strings
            as well. Each prompt is either a string or a pair of prompts for CFG sampling. If CFG is used, sampler
            settings must contain cfg_scale.

        :param sampler:
            Sampler stack settings for all prompts in batch or list of samplers for each prompt.

        :param max_new_tokens:
            Max number of tokens to generate.

        :param min_new_tokens:
            Minimum number of tokens to generate before stop tokens become active. Until this number have been
            sampled, stop tokens are suppressed but stop strings will still end response.

        :param seed:
            Seed for the sampling RNG. Doesn't guarantee perfect determinism from the implementation.

        :param token_healing:
            Apply token healing by regenerating the last token of the input sequence with prefix
            constraint.

        :param encode_special_tokens:
            Encode special tokens (BOS etc.) represented as text in the input. If False, special tokens are
            interpreted as text by the tokenizer.

        :param decode_special_tokens:
            Decode special tokens output by the model. If False, tokens marked as special in the tokenizer
            are decoded as empty strings.

        :param stop_conditions:
            List of strings and/or token IDs that will end generation. The stop condition is not included
            in the output.

        :param add_bos:
            Prepend the tokenizer's specified BOS token to the input.

        :param abort_event:
            Forwarded to the model during generation. Will abort prefill/context ingestion if triggered.

        :param completion_only:
            Only return completion. If False, returned string will include the input prompt.

        :param filters:
            (List of) list of Filters to apply during generation. Each prompt in a batch needs
            its own filter list, or a value of None to disable filters for individual prompts.

        :param return_last_results:
            If True, returns the last results dict for each job

        :param embeddings:
            Optional list of MMEmbeddings to use for, or list of lists for batched generation

        :param max_rq_tokens:
            Maximum number of tokens before job is requeued. Rounded to nearest page boundary. This limits how
            many new pages are allocated in the cache for the job in any one round and allows a single job to use
            the full cache size without limiting concurrency for other jobs.

        :param stop_on_loop:
            Tuple of (window_size: int, min_reps: int), or None. If enabled, generation will end if the last
            window_size tokens sampled make up >= min_reps consecutive instances of a looping string.

        :return:
            Completion(s): (str or list[str] depending on the type of the input prompt argument)
            Optionally, last results: (dict or list[dict] depending on the type of the input prompt argument)
        """

        order = {}
        if isinstance(prompt, list):
            prompts = prompt
        else:
            prompts = [prompt]
            filters = [filters]
            embeddings = [embeddings]

        if not filters:
            filters = [None] * len(prompts)
        else:
            assert len(filters) == len(prompts) and \
                all((f is None or isinstance(f, list)) for f in filters), \
                "If using filters, must provide one filter list (or None-value) per prompt."

        if not embeddings:
            embeddings = [None] * len(prompts)
        else:
            assert len(embeddings) == len(prompts) and all((isinstance(f, list) or not f) for f in embeddings), \
                "Must provide one list of embeddings per prompt."

        prompts = prompt if isinstance(prompt, list) else [prompt]
        batch_size = len(prompts)
        for idx, p in enumerate(prompts):
            if isinstance(p, str):
                input_ids = self.tokenizer.encode(
                    p,
                    encode_special_tokens = encode_special_tokens,
                    add_bos = add_bos,
                    embeddings = embeddings[idx]
                )
            elif isinstance(p, tuple):
                input_ids = [self.tokenizer.encode(
                    p_,
                    encode_special_tokens = encode_special_tokens,
                    add_bos = add_bos,
                    embeddings = embeddings[idx]
                ) for p_ in p]
            else:
                assert False, "Unexpected type in prompt"

            if sampler is None or isinstance(sampler, Sampler):
                p_sampler = sampler
            elif isinstance(sampler, list):
                assert len(sampler) == len(prompts)
                p_sampler = sampler[idx]
            else:
                assert False, "Unexpected sampler type"

            job = Job(
                input_ids = input_ids,
                max_new_tokens = max_new_tokens,
                min_new_tokens = min_new_tokens,
                seed = seed,
                stop_conditions = stop_conditions,
                sampler = p_sampler,
                filters = filters[idx] or [],
                token_healing = token_healing,
                decode_special_tokens = decode_special_tokens,
                embeddings = embeddings[idx] or [],
                max_rq_tokens = max_rq_tokens,
                stop_on_loop = stop_on_loop,
            )

            if seed is not None: seed += 1

            serial = self.enqueue(job)
            order[serial] = idx

        # Collect outputs until all jobs finish
        completions = [""] * batch_size
        last_results = [None] * batch_size

        while self.num_remaining_jobs():
            results = self.iterate()

            for r in results:
                idx = order[r["serial"]]
                if r["stage"] == "streaming":
                    text = r.get("text", "")
                    completions[idx] += text
                if r["eos"]:
                    last_results[idx] = r
            if abort_event is not None and abort_event.is_set():
                self.clear_queue()
                return None

        # Return results
        if not completion_only:
            completions = [(p if isinstance(p, str) else p[0]) + c for p, c in zip(prompts, completions)]

        if not isinstance(prompt, list):
            completions = completions[0]
            last_results = last_results[0]

        if return_last_results:
            return completions, last_results
        else:
            return completions
