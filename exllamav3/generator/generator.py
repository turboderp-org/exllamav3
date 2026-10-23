from __future__ import annotations
import logging
import torch
import os
from ..model.model import Model
from ..cache.cache import Cache
from ..cache.recurrent import RecurrentCache, PairedState, host_pool, mp_host_pool_release
from ..tokenizer.tokenizer import Tokenizer
from ..constants import PAGE_SIZE
from ..util import cuda_sync_active
from ..ext import exllamav3_ext as ext

logger = logging.getLogger(__name__)
from ..util.memory import malloc_trim
from .pagetable import PageTable, is_content_hash
from .cpu_cache import CPUPageCache
from .disk_cache import DiskPageCache
from .draft_confidence import DraftConfidenceCalibrator
from .job import Job
from .filter import Filter
from concurrent.futures import ThreadPoolExecutor
from .sampler import Sampler
from .visualizer import CacheVisualizer
import time
import threading
from ..tokenizer import MMEmbedding
from ..util import profile_opt

# Test hook: True forces the per-position serial speculative-window readback instead of the batched
# (launch-all-samplers, one pinned stage, one synchronize) path
_NO_BATCH_READBACK = False

class Generator:

    def __init__(
        self,
        model: Model,
        cache: Cache,
        tokenizer: Tokenizer,
        max_batch_size: int = 256,
        max_chunk_size: int = 2048,
        shared_chunk_size: int | None = None,
        prefill_cooldown: int | None = 4,
        max_q_size: int = 8,
        draft_model: Model | None = None,
        draft_cache: Cache | None = None,
        num_draft_tokens: int | None = None,
        show_visualizer: bool = False,
        enable_defrag: bool = True,
        cpu_cache_size: int = 0,
        disk_cache_dir: str | None = None,
        disk_cache_size: int = 50 * 1024**3,
        disk_cache_flush_on_idle: bool = True,
        disk_cache_checkpoint_interval: int = 4096,
        disk_cache_write_interval: float = 2.0,
        recurrent_cache_size: int = 4 * 1024**3,
        recurrent_checkpoint_interval: int = None,
        recurrent_checkpoint_interval_pp: int = 32768,
        ngram_match_min: int = 0,
        dynamic_draft_tokens: bool = False,
        draft_confidence: float = 0.4,
        record_draft_stats: bool = False,
        ngram_corpus: str | None = None,
        hybrid_draft_tokens: int = 0,
        draft_max_rows: int = 0,
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
            interruption to ongoing jobs when a new job is started. See shared_chunk_size for lowering it
            only while other jobs are generating.

        :param shared_chunk_size:
            Prefill chunk size to use while another active job is already generating, so that a new prompt
            interrupts ongoing generation for shorter stretches at a time. Must be a multiple of the page size
            and no larger than max_chunk_size. Prompt ingestion with no job generating keeps the full
            max_chunk_size, which remains the efficient choice for cold prefill. Default is None, which selects
            half of max_chunk_size rounded up to the page size. Set it to max_chunk_size to keep full chunks
            while other jobs generate

        :param prefill_cooldown:
            Number of iterations to wait between prefill passes while another active job is generating. Every
            call to iterate() still runs one generation step, so the generating jobs keep streaming at an even
            pace while prompt ingestion takes one chunk every prefill_cooldown + 1 calls. With several prompts
            waiting, they take turns, one chunk per pass. Default is 4. Set to 0 for a prefill pass on every
            iteration, as when no job is generating

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

        :param ngram_corpus:
            Optional frozen SAM file shared by n-gram jobs; requires ngram_match_min > 0.

        :param hybrid_draft_tokens:
            Long suffix-match drafting on top of whichever drafting mode is active (or on its own). Every
            round, each job's suffix automaton is consulted; when the longest match of the current suffix
            against the job's earlier context is at least this many tokens, the draft becomes the same number
            of tokens that followed that earlier occurrence, unless they disagree with the regular drafter's
            own guess. The intent is to catch verbatim repetition of context (reciting a drafted reply, editing
            source) and gamble a long verification pass on advancing many tokens at prefill speed. A recurrent
            model's cache needs max_history of at least this many tokens.

        :param draft_max_rows:
            Ceiling on the number of decoding rows in a speculative round, counting the verified draft
            positions, i.e. bsz * (1 + draft tokens) (0 disables the ceiling). Draft length is capped when
            possible to keep the batch from exceeding the ceiling.

        :param dynamic_draft_tokens:
            Adapt the per-round draft length to the workload. The draft is cut using a drafter-provided
            confidence score, calibrated online against observed acceptance rates (see draft_confidence). A
            DFlash drafter still runs at its fixed diffusion block size and only the verified window shrinks;
            AR and MTP drafters stop the drafting loop itself early, saving one drafter forward per pruned
            position. The acceptance behavior of the surviving positions is unchanged. Has no effect on n-gram
            drafting, whose draft length already adapts through the match length.

        :param draft_confidence:
            Used with dynamic_draft_tokens and a draft model: target acceptance probability for drafted
            positions, evaluated against an online mapping from drafter confidence scores to observed
            acceptance rates. Scale-free and portable across model pairs. For DFlash (block produced at fixed
            cost) the block is cut at the first position whose estimated conditional acceptance drops below
            the target. For AR and MTP drafters (one forward per position, and a position only pays off if
            everything before it is accepted too) drafting stops when the running product of conditional
            estimates - the probability that the next position actually contributes a token - falls below the
            target. Lower values keep longer drafts; higher values truncate more aggressively. Ignored for
            n-gram drafting.

        :param record_draft_stats:
            Append (position, window, accepted) per verification round to job.draft_stats, for analysis.

        :param show_visualizer:
            Open window to render visualization of cache (for debug/demonstration purposes)

        :param enable_defrag:
            Defragment cache periodically

        :param cpu_cache_size:
            Size in bytes of a second-tier page cache in pinned system memory, 0 (default) to disable. Complete
            K/V pages evicted from the GPU cache are stored there and restored on prompt-cache hits instead of
            being recomputed by prefill. In tensor-parallel mode each rank pins its own shard of the tier and
            the budget counts whole pages across all of them

        :param disk_cache_dir:
            Directory for a third, persistent tier behind the CPU page cache (requires cpu_cache_size > 0),
            None (default) to disable. Pages entering the CPU tier and recurrent checkpoints are written
            through to a per-model subdirectory (keyed by a fingerprint of the model, cache configuration and
            kernel sources) and read back, in this or a later process, when a prompt's pages are in no other
            tier. Page images and checkpoints are stored in a layout independent of the tensor-parallel
            split, so a directory is shared between TP loads with different splits and non-TP loads of the
            same model and cache configuration

        :param disk_cache_size:
            Budget in bytes for the whole disk_cache_dir, shared by every model and cache configuration
            stored under it (the least recently used of the others are removed whole when a load finds the
            root over budget). Default is 50 GB

        :param disk_cache_flush_on_idle:
            Also write every complete page still in VRAM (and every checkpoint) to the disk tier whenever the
            generator goes idle, so a context never has to be evicted first to be persisted. Default True;
            flush_cache_to_disk() does the same on demand

        :param disk_cache_checkpoint_interval:
            Minimum spacing, in tokens, between the recurrent checkpoints the disk tier keeps along any chain
            of cached pages, counted from the start of the chain; closer ones (and checkpoints whose page is
            gone) are pruned when the generator goes idle, so a resumed prompt replays at most this many
            tokens. Checkpoints take far more space than pages on hybrid models. Default 4096; 0 keeps all

        :param disk_cache_write_interval:
            While jobs run, each this many seconds, queue the complete pages of the active sequences for the disk
            tier (through the CPU tier). Without this, a page goes to disk only when it leaves VRAM or when the
            generator becomes idle, so a process that stops during a long prefill loses all of it. With this, it
            loses only the pages after the last checkpoint on disk. The last two pages before the position of
            each sequence are not queued: speculative decoding can still rewind them, and the MTP draft cache
            writes the last slot of a page one step later. 0 disables. Default 2.0

        :param recurrent_cache_size:
            Size of recurrent cache, in bytes. Recurrent cache resides in system RAM. Default is 4 GB.
            Ignored if model doesn't use recurrent states

        :param recurrent_checkpoint_interval:
            Minimum number of tokens between recurrent checkpoints in model output and the tail end of
            the prompt. Must be a multiple of the page size. Model architecture determines default

        :param recurrent_checkpoint_interval_pp:
            Minimum number of tokens between recurrent checkpoints during prompt ingestion. Must be a
            multiple of the page size. Checkpoints land on the first chunk boundary at least this far past the
            previous one, so the interval need not be a multiple of the chunk size. Default is 32768 tokens

        :param kwargs:
        """

        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        cfg = self.model.config
        self.padded_vocab_size = ((cfg.vocab_size + 31) // 32) * 32

        # Paging
        self.pagetable = PageTable(self, cache)
        # A cache has exactly one owning generator. Record the takeover and refuse to iterate a superseded generator
        cache.owner_serial = getattr(cache, "owner_serial", 0) + 1
        self.cache_owner_serial = cache.owner_serial
        self.max_total_tokens = PAGE_SIZE * self.pagetable.max_pages
        self._cache_stats_memo = (None, None)

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
            if draft_model.caps.get("recurrent_states"):
                # An autoregressive drafter with recurrent layers drafts through a token-by-token
                # speculative pass on its own state pool (iterate_draftmodel_gen); a DFlash-family
                # drafter's sliding-window rings take the projected rows directly. MTP heads with
                # recurrent layers have neither path yet
                assert not draft_model.caps.get("mtp_draft"), "Recurrent MTP draft models are not supported."
            if num_draft_tokens:
                self.num_draft_tokens = num_draft_tokens
            else:
                self.num_draft_tokens = draft_model.caps.get("default_draft_size", 4)
            depths = draft_model.caps.get("mtp_depths")
            if depths is not None and self.num_draft_tokens > depths:
                print(f" !! Warning: the MTP head has {depths} depth-specialized layers; draft positions past "
                      f"{depths} reuse the last one, with decreasing acceptance (num_draft_tokens = "
                      f"{self.num_draft_tokens})")
        elif ngram_match_min:
            self.num_draft_tokens = num_draft_tokens if num_draft_tokens is not None else 4
        else:
            self.num_draft_tokens = 0

        self.ngram_corpus = None
        if ngram_corpus is not None:
            if ngram_match_min <= 0:
                raise ValueError("ngram_corpus requires ngram_match_min > 0")
            from .ngram import NgramCorpus
            self.ngram_corpus = NgramCorpus(ngram_corpus, tokenizer)
        self.ngram_match_min = ngram_match_min
        # The widest verification pass any round can take: the regular drafter's window or a hybrid draft
        self.hybrid_draft_tokens = max(0, hybrid_draft_tokens or 0)
        self.max_draft_tokens = max(self.num_draft_tokens, self.hybrid_draft_tokens)
        self.dynamic_draft = dynamic_draft_tokens and self.num_draft_tokens > 0
        self.draft_max_rows = max(0, draft_max_rows or 0)
        self._draft_window_last = None
        self.record_draft_stats = record_draft_stats
        max_q_size = max(self.max_draft_tokens + 1, max_q_size)

        # Chunking/partitioning
        self.max_batch_size = max_batch_size
        self.max_chunk_size = max_chunk_size

        # Prompt ingestion alongside generating jobs, see iterate()
        if shared_chunk_size is None:
            shared_chunk_size = (max_chunk_size // 2 + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
        assert 0 < shared_chunk_size <= max_chunk_size and shared_chunk_size % PAGE_SIZE == 0, \
            "shared_chunk_size must be a multiple of the page size (256) and no larger than max_chunk_size"
        if prefill_cooldown is None:
            prefill_cooldown = 4
        assert prefill_cooldown >= 0, \
            "prefill_cooldown must not be negative"
        self.shared_chunk_size = shared_chunk_size
        self.prefill_cooldown = prefill_cooldown
        self.prefill_cooldown_left = 0
        self.prefill_rotation = 0

        # Job queues
        self.job_serial = 0
        self.pending_jobs = []
        self.active_jobs = []

        # Filter threads
        self.filter_pool = ThreadPoolExecutor(max_workers = 16)
        self.filter_queue = []

        # Rounds that verified a hybrid (long suffix-match) draft
        self.hybrid_rounds = 0

        # Pinned staging buffer for batched token readback in iterate_gen
        self.sample_pinned = None
        self.staging_buffers = {}
        self.stage_arenas = {}
        self.stage_devices = None

        # Buffers. Pinned: the draft input ids upload non-blocking from here every round (and
        # the DFlash2 selector reads its anchor from the same view), the drafted ids come back
        # into draft_ids_pinned
        if draft_model or ngram_match_min or self.hybrid_draft_tokens:
            self.draft_input_ids_pinned = torch.empty(
                (max_batch_size, 1),
                dtype = torch.long,
                pin_memory = True
            )
            self.draft_ids_pinned = torch.empty(
                (max_batch_size, self.max_draft_tokens),
                dtype = torch.long,
                pin_memory = True
            )

        # CPU page cache tier, and the disk tier behind it
        self.cpu_page_cache = None
        self.disk_page_cache = None
        self.disk_cache_flush_on_idle = disk_cache_flush_on_idle
        self.disk_cache_write_interval = disk_cache_write_interval
        self._write_behind_time = 0.0
        if cpu_cache_size:
            tier_caches = [cache] + ([draft_cache] if draft_cache is not None else [])
            self.cpu_page_cache = CPUPageCache(tier_caches, cpu_cache_size)
            self.cpu_page_cache.attach(self.pagetable)
            self.pagetable.cpu_tier = self.cpu_page_cache
            if disk_cache_dir:
                self.disk_page_cache = DiskPageCache(disk_cache_dir, tier_caches, self.cpu_page_cache, disk_cache_size,
                                                     min_state_distance = disk_cache_checkpoint_interval)
                self.cpu_page_cache.attach_disk(self.disk_page_cache)
        elif disk_cache_dir:
            raise ValueError("disk_cache_dir needs a CPU page cache tier to write through from (cpu_cache_size > 0)")

        # Visualizer
        if show_visualizer:
            self.visualizer = CacheVisualizer(self.pagetable.max_pages)
        else:
            self.visualizer = None

        # Defrag
        self.enable_defrag = enable_defrag

        # Recurrent cache. A job's recurrent state is a PairedState over the target and draft sides;
        # both sides checkpoint together into this one pool (see PairedState)
        self.recurrent_cache_size = recurrent_cache_size
        self.target_recurrent = bool(self.model.caps.get("recurrent_states"))
        self.draft_recurrent = bool(draft_model is not None and draft_model.caps.get("recurrent_states"))
        if self.target_recurrent:
            if self.max_draft_tokens > cache.max_history:
                raise ValueError(
                    f"Speculative passes of up to {self.max_draft_tokens} draft tokens need a cache with "
                    f"max_history >= {self.max_draft_tokens} on a recurrent model (this cache has "
                    f"max_history = {cache.max_history})"
                )
        if self.draft_recurrent and not draft_model.caps.get("dflash_draft") and self.num_draft_tokens > draft_cache.max_history:
            raise ValueError(
                f"A recurrent draft model drafting {self.num_draft_tokens} tokens needs a draft cache with "
                f"max_history >= {self.num_draft_tokens} (this cache has max_history = {draft_cache.max_history})"
            )
        if self.target_recurrent or self.draft_recurrent:
            self.recurrent_cache = RecurrentCache(self.model, recurrent_cache_size, draft_model)
            self.recurrent_cache.pagetable = self.pagetable
            self.recurrent_cache.disk_tier = self.disk_page_cache
            if self.disk_page_cache is not None:
                # Which model and cache stash under each side of a PairedState: a tensor-parallel side's
                # checkpoints are written and read by that model's ranks
                self.disk_page_cache.recurrent_sides = {
                    side: (m, c) for side, m, c, on in (
                        ("target", self.model, cache, self.target_recurrent),
                        ("draft", draft_model, draft_cache, self.draft_recurrent),
                    ) if on
                }
            # The new page table owns every page, so every state slot is ours too
            for c, recurrent in ((cache, self.target_recurrent), (draft_cache, self.draft_recurrent)):
                if recurrent:
                    c.reset_states()
                    # Limit batch size if cache has recurrent states
                    self.max_batch_size = min(self.max_batch_size, c.num_slots)
        else:
            self.recurrent_cache = None
        if recurrent_checkpoint_interval is None:
            # The target sets the checkpoint schedule; a recurrent draft under a non-recurrent target
            # lends it its own
            schedule_model = model if self.target_recurrent or not self.draft_recurrent else draft_model
            recurrent_checkpoint_interval = schedule_model.caps.get("default_recurrent_checkpoint_interval", 2048)

        assert recurrent_checkpoint_interval % PAGE_SIZE == 0 and recurrent_checkpoint_interval_pp % PAGE_SIZE == 0, \
            "checkpoint interval must be a multiple of the page size (256)"
        self.recurrent_checkpoint_interval = recurrent_checkpoint_interval
        self.recurrent_checkpoint_interval_pp = recurrent_checkpoint_interval_pp

        # Drafting mode
        if draft_model is not None and draft_model.caps.get("attach_target"):
            draft_model.attach_to(model)
        self.dflash_draft = self.draft_model is not None and self.draft_model.caps.get("dflash_draft", False)
        self.mtp_draft = self.draft_model is not None and self.draft_model.caps.get("mtp_draft", False)

        # Confidence-calibrated draft truncation (draft model + dynamic draft, any mode). For
        # DFlash the fixed-size drafted block is truncated before verification; for AR draft
        # models and MTP the drafting loop itself stops early, saving drafter forwards. Modes
        # whose TP lm_head path returns only the argmax simply never export confidence values,
        # leaving the calibrator inert
        self.draft_calibrator = None
        self._draft_conf_round = None
        if self.dynamic_draft and self.draft_model is not None:
            self.draft_calibrator = DraftConfidenceCalibrator(draft_confidence)


    def num_remaining_jobs(self):
        return len(self.pending_jobs) + len(self.active_jobs)

    def num_active_jobs(self):
        return len(self.active_jobs)

    def num_pending_jobs(self):
        return len(self.pending_jobs)

    def get_cache_stats(self) -> dict:
        """
        Snapshot of cache occupancy and prompt-cache efficiency, intended for status displays.

        Returns a dict with:
            page_size: tokens per page
            max_tokens: total cache capacity, in tokens
            used_tokens: tokens in pages referenced by active jobs
            cached_tokens: tokens in complete pages that a later or concurrent job could actually reuse: every
                page on an unbroken chain down to a root, whether or not an active job currently references it.
                Complete pages whose ancestors were evicted (orphans) don't count, since a prefix match walks
                the chain from the root
            free_tokens: tokens in unreferenced pages holding no reusable content
            tier_cached_tokens: reusable tokens held only in the system-memory tier (sysmem K/V cache), by the
                same chain rule; 0 without a tier
            tier_max_tokens: capacity of the system-memory tier, 0 without one
            disk_cached_tokens: reusable tokens held only in the disk tier, by the same chain rule (a chain
                may run through all three tiers; the disk tier is persistent, so these survive a restart);
                0 without a disk tier
            disk_pages, disk_states: pages and recurrent checkpoints the disk tier holds for this model
            disk_used_bytes, disk_max_bytes: the disk tier's occupancy and budget (the budget is for the whole
                directory, every model and cache configuration under it); 0 without a disk tier
            alloc_pages: cumulative pages claimed by started jobs, the pages reserved for generation included
            alloc_prompt_pages: of those, the complete prompt pages, i.e. the ones a prompt-cache hit can serve
            alloc_cached_pages: of those, pages that were reused from the prompt cache
            alloc_tier_pages: of those, pages restored from the system-memory tier (including pages that were
                promoted there from disk first)
            alloc_disk_pages: of those, pages that came from the disk tier, whether the enqueue-time prefetch
                or the allocation itself promoted them; a page counts here the first time it is restored after
                its promotion and as an ordinary tier page from then on
            hit_rate: alloc_cached_pages / alloc_prompt_pages, or None before any job has started. The
                denominator is the prompt pages only, so this matches the per-request cached fraction rather
                than being diluted by the pages each job reserves for its output
            tier_hit_rate: alloc_tier_pages / alloc_prompt_pages, the part of hit_rate served from the
                system-memory tier (disk promotions included), or None before any job has started
            disk_hit_rate: alloc_disk_pages / alloc_prompt_pages, the part of tier_hit_rate that was read from
                disk, or None before any job has started
            active_jobs, pending_jobs: current job counts

        The chain walk is memoized on a signature of the page tables, so repeated calls between changes are
        cheap.
        """
        pt = self.pagetable
        tier = pt.cpu_tier
        disk = self.disk_page_cache

        # The chain walk is linear in the number of pages, which can be large with a big system-memory
        # tier, so it only reruns when something that can change the outcome has moved
        signature = (
            len(pt.referenced_pages),
            len(pt.unreferenced_pages),
            sum(1 for h in pt.referenced_pages if is_content_hash(h)),
            pt.metrics["alloc_pages"],
            pt.metrics["evictions"],
            pt.last_defrag_serial,
            len(tier) if tier is not None else 0,
            (tier.metrics["pushes"], tier.metrics["evictions"]) if tier is not None else None,
            (len(disk), len(disk.states), disk.metrics["page_writes"], disk.metrics["evictions"]) if disk is not None else None,
        )
        memo_signature, memo = self._cache_stats_memo
        if signature == memo_signature:
            return {**memo, "active_jobs": len(self.active_jobs), "pending_jobs": len(self.pending_jobs)}

        # prev-hash links of every complete page, per tier
        gpu_links = {}
        for pages in (pt.referenced_pages, pt.unreferenced_pages):
            for h, page in pages.items():
                if is_content_hash(h):
                    gpu_links[h] = page.prev_hash
        tier_links = {h: e["prev_hash"] for h, e in tier.entries.items()} if tier is not None else {}
        if disk is not None:
            with disk._lock:
                disk_links = {h: m["prev_hash"] for h, m in disk.pages.items()}
        else:
            disk_links = {}

        # A page is reusable when its whole chain is present in either tier
        resumable = {}
        def is_resumable(h):
            chain = []
            while h is not None and h not in resumable:
                chain.append(h)
                if h in gpu_links:
                    h = gpu_links[h]
                elif h in tier_links:
                    h = tier_links[h]
                elif h in disk_links:
                    h = disk_links[h]
                else:
                    resumable[h] = False
                    break
            result = True if h is None else resumable[h]
            for c in chain:
                resumable[c] = result
            return result

        cached = sum(1 for h in gpu_links if is_resumable(h))
        tier_cached = sum(1 for h in tier_links if h not in gpu_links and is_resumable(h))
        disk_cached = sum(1 for h in disk_links if h not in gpu_links and h not in tier_links and is_resumable(h))
        used = len(pt.referenced_pages)
        free = sum(1 for h in pt.unreferenced_pages if not is_content_hash(h))
        alloc = pt.metrics["alloc_pages"]
        alloc_prompt = pt.metrics["alloc_prompt_pages"]
        alloc_cached = pt.metrics["alloc_cached_pages"]
        alloc_tier = pt.metrics["alloc_tier_pages"]
        alloc_disk = pt.metrics["alloc_disk_pages"]
        stats = {
            "page_size": PAGE_SIZE,
            "max_tokens": pt.max_pages * PAGE_SIZE,
            "used_tokens": used * PAGE_SIZE,
            "cached_tokens": cached * PAGE_SIZE,
            "free_tokens": free * PAGE_SIZE,
            "tier_cached_tokens": tier_cached * PAGE_SIZE,
            "tier_max_tokens": tier.max_slots * PAGE_SIZE if tier is not None else 0,
            "disk_cached_tokens": disk_cached * PAGE_SIZE,
            "disk_pages": len(disk) if disk is not None else 0,
            "disk_states": len(disk.states) if disk is not None else 0,
            "disk_used_bytes": disk.total_size if disk is not None else 0,
            "disk_max_bytes": disk.root_size if disk is not None else 0,
            "alloc_pages": alloc,
            "alloc_prompt_pages": alloc_prompt,
            "alloc_cached_pages": alloc_cached,
            "alloc_tier_pages": alloc_tier,
            "alloc_disk_pages": alloc_disk,
            "hit_rate": alloc_cached / alloc_prompt if alloc_prompt else None,
            "tier_hit_rate": alloc_tier / alloc_prompt if alloc_prompt else None,
            "disk_hit_rate": alloc_disk / alloc_prompt if alloc_prompt else None,
            "active_jobs": len(self.active_jobs),
            "pending_jobs": len(self.pending_jobs),
        }
        self._cache_stats_memo = (signature, stats)
        return stats


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
            self.on_queue_drained()


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
        if self.disk_page_cache is not None:
            # Start reading the prompt's pages that only the disk tier holds, so the allocation finds them in
            # the CPU tier (or in flight) rather than waiting on the disk itself
            for seq in job.sequences:
                hashes = seq.page_hashes if seq.max_cached_pages is None else seq.page_hashes[:seq.max_cached_pages]
                self.disk_page_cache.prefetch(hashes, self.pagetable.access_serial)
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
            self.on_queue_drained()


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

        assert self.cache.initialized, \
            "Cache tensors were never allocated. Construct the Cache BEFORE calling model.load()"
        owner_serial = getattr(self.cache, "owner_serial", None)
        if owner_serial is not None and owner_serial != getattr(self, "cache_owner_serial", owner_serial):
            # Superseded: drop every job without touching the cache (the new owner may already have
            # reused these pages and state slots) and refuse to run
            self.pending_jobs.clear()
            self.active_jobs.clear()
            raise RuntimeError(
                "This Generator no longer owns its Cache: a newer Generator was created over the same "
                "Cache. Close or discard the old Generator before creating a new one."
            )
        assert self.draft_cache is None or self.draft_cache.initialized, \
            "Draft cache tensors were never allocated. Construct the draft Cache BEFORE calling draft_model.load()"

        results = []
        self.iterate_start_jobs(results)

        # Perform one round of prefill. While some active job is already generating, prompt ingestion shares
        # the GPU with it: at most one job with prompt left to ingest runs a pass per iteration, with the
        # shared chunk size, and only once the cooldown since the previous pass has elapsed, so every call
        # still produces one generation step for the generating jobs. The other jobs' passes are deferred;
        # jobs with nothing left to ingest keep their call, which only does bookkeeping. With no job
        # generating, or with a full shared chunk and no cooldown, every job runs a full chunk on every
        # iteration
        prefilling = [job for job in self.active_jobs if not job.is_prefill_done()]
        chunk_size = None
        deferred = []
        if prefilling and len(prefilling) < len(self.active_jobs) and \
                (self.shared_chunk_size < self.max_chunk_size or self.prefill_cooldown > 0):
            if self.shared_chunk_size < self.max_chunk_size:
                chunk_size = self.shared_chunk_size
            if self.prefill_cooldown_left > 0:
                self.prefill_cooldown_left -= 1
                deferred = prefilling
            else:
                self.prefill_cooldown_left = self.prefill_cooldown
                selected = prefilling[self.prefill_rotation % len(prefilling)]
                self.prefill_rotation += 1
                deferred = [job for job in prefilling if job is not selected]
        else:
            self.prefill_cooldown_left = 0
        for job in list(self.active_jobs):
            if any(job is d for d in deferred):
                continue
            try:
                job.prefill(results, chunk_size)
            except Exception as e:
                self.reap_failed_job(job, e, results)

        # Recurrent checkpoints
        if self.recurrent_cache is not None:
            self.recurrent_checkpoint()

        # Generation with draft model, with the draft window trimmed to the row ceiling (draft_max_rows)
        draft_window = self.row_limited(self.num_draft_tokens) if self.draft_model else 0
        if self.draft_model and draft_window > 0:
            if self.dflash_draft:
                draft_tokens = self.iterate_draftmodel_dflash_gen(results, draft_window)
            elif self.mtp_draft:
                draft_tokens = self.iterate_draftmodel_mtp_gen(results, draft_window)
            else:
                draft_tokens = self.iterate_draftmodel_gen(results, draft_window)

        # Generation with n-gram draft
        elif self.ngram_match_min:
            draft_tokens = self.iterate_ngram_gen(results)

        # Regular generation
        else:
            draft_tokens = None

        # Long suffix-match draft over the regular one, when a job is repeating its context
        if self.hybrid_draft_tokens:
            draft_tokens = self.iterate_hybrid_gen(
                draft_tokens, self.row_limited(self.hybrid_draft_tokens)
            )

        self.iterate_gen(results, draft_tokens)

        # Visualization
        if self.visualizer:
            self.update_visualizer()

        # Disk tier: queue the complete pages of the active sequences (see disk_cache_write_interval)
        if self.disk_page_cache is not None and self.active_jobs and self.disk_cache_write_interval > 0:
            self.write_behind()

        # Finished iteration
        return results


    def close(self):
        """
        Release the host memory a retired generator would otherwise hold until it is garbage collected,
        which can be long after it stops being used: a generator sits in a reference cycle with its page
        table, and the jobs that failed on it keep it reachable through their exception tracebacks. The
        model and cache stay loaded for a replacement generator to use; this generator is not reusable
        afterwards. Safe to call more than once.

        Releases, in order of size: the recurrent checkpoint cache (up to recurrent_cache_size of system
        RAM, in the tensor-parallel ranks too), the CPU page cache tier (cpu_cache_size of pinned memory),
        the pinned staging and draft buffers, and the filter thread pool.
        """
        if self.disk_page_cache is not None:
            # Persist what is still only in VRAM or RAM, then let the writer drain before its sources go away.
            # After abort() (generator error), nothing is written
            if self.disk_cache_flush_on_idle and not self.disk_page_cache.write_disabled:
                self.flush_cache_to_disk()
            self.disk_page_cache.close()
            self.disk_page_cache = None
            if self.recurrent_cache is not None:
                self.recurrent_cache.disk_tier = None
        if self.recurrent_cache is not None:
            self.recurrent_cache.close()
        if self.cpu_page_cache is not None:
            self.cpu_page_cache.close()
            self.cpu_page_cache = None
            self.pagetable.cpu_tier = None
        self.sample_pinned = None
        self.staging_buffers = {}
        self.stage_arenas = {}
        self.draft_input_ids_pinned = None
        self.draft_ids_pinned = None
        self.filter_pool.shutdown(wait = False)


    @torch.inference_mode()
    def flush_cache_to_disk(self, wait: bool = False, max_pages: int | None = None) -> dict:
        """
        Queue every complete, content-hashed page that is in VRAM but in neither the CPU nor the disk tier for
        persistence (through the CPU tier, whose store is the write-through point), and every recurrent
        checkpoint not yet on disk. Returns counts of what was queued; with wait = True returns once the
        writer has finished.

        max_pages limits the number of pages queued in one call. Each page takes a CPU-tier slot until its
        write is complete. If the pages are more than the free slots, the tier must wait for the writer, and
        this call stops the generator for that time. With a limit, the call stops early and sets "more" in the
        result. The idle loop then calls again (see idle_work).
        """
        disk = self.disk_page_cache
        tier = self.cpu_page_cache
        queued = {"pages": 0, "states": 0, "more": False}
        if disk is None:
            return queued
        pt = self.pagetable
        pages = list(pt.unreferenced_pages.values()) + list(pt.referenced_pages.values())
        queued["pages"], queued["more"] = self._queue_pages_to_disk(pages, max_pages)
        rc = self.recurrent_cache
        if rc is not None and rc.disk_tier is not None:
            # Checkpoints follow their pages: every complete page is queued above, so an anchored checkpoint
            # is written now, the rest stay candidates. The last checkpoint of each chain (no other checkpoint
            # in RAM is after it on its chain) is written also when it is closer than min_state_distance to
            # the one before: a conversation continues from the end of its chain
            keys = [h for h in rc.keys() if is_content_hash(h)]
            keyset = set(keys)
            covered = set()
            limit = pt.max_pages + len(tier) + len(disk)
            for k in keys:
                h = disk.chain_prev(k)
                steps = 0
                while h is not None and steps <= limit and h not in covered:
                    if h in keyset:
                        covered.add(h)
                    h = disk.chain_prev(h)
                    steps += 1
            for h in keys:
                stashed = rc.get(h)
                if stashed is None or not (disk.has_page(h) or h in disk.pending_pages):
                    continue
                if disk.write_state(h, stashed, force = h not in covered):
                    queued["states"] += 1
        queued["states_pruned"] = disk.prune_states()
        self._disk_backlog = queued["more"]
        if wait:
            disk.flush()
        return queued


    def _queue_pages_to_disk(self, pages, budget: int | None) -> tuple[int, bool]:
        """
        Queue the complete, content-hashed, persistable pages among `pages` that are in VRAM but in neither the
        CPU nor the disk tier, through the CPU tier (its store is the write-through point; the copy to host
        memory goes on the current stream, after the kernels that wrote the page and before any that could
        write it again). Stops after `budget` pages. Returns (pages queued, stopped early).
        """
        disk, tier, pt = self.disk_page_cache, self.cpu_page_cache, self.pagetable
        queued = 0
        for page in pages:
            h = page.phash
            if page.kv_position != PAGE_SIZE or not is_content_hash(h) or not page.persistable:
                continue
            if h in tier or disk.has_page(h) or h in disk.pending_pages:
                continue
            if budget is not None and queued >= budget:
                return queued, True
            tier.store(page, pt.access_serial)
            queued += 1
        return queued, False


    @torch.inference_mode()
    def write_behind(self):
        """
        Queue the complete pages of the active sequences for the disk tier, at most once per
        disk_cache_write_interval seconds, so a process that dies during a long prefill loses only what came
        after the last page (and checkpoint) written. The generator does not wait for the copies or the disk.

        The last two pages before a sequence's position are not queued: speculative decoding can still rewind
        them, and the MTP draft cache writes the last slot of a page one step later. One call queues at most
        a quarter of the CPU tier, less the writes still queued, so the tier never has to wait for the writer
        to free a slot.
        """
        now = time.monotonic()
        if now - self._write_behind_time < self.disk_cache_write_interval:
            return
        self._write_behind_time = now
        disk = self.disk_page_cache
        if disk.write_disabled:
            return
        budget = self.cpu_page_cache.max_slots // 4 - len(disk.pending_pages)
        if budget <= 0:
            return
        pages = []
        for job in self.active_jobs:
            for seq in job.sequences:
                if seq.allocated_pages:
                    pages.extend(seq.allocated_pages[:max(0, seq.kv_position // PAGE_SIZE - 2)])
        self._queue_pages_to_disk(pages, budget)


    def _idle_flush_budget(self) -> int:
        """Pages for one idle flush step: half the CPU tier, less the writes still queued"""
        return max(1, self.cpu_page_cache.max_slots // 2 - len(self.disk_page_cache.pending_pages))


    def has_idle_work(self) -> bool:
        """True if the idle flush stopped early and has more pages to queue"""
        return self.disk_page_cache is not None and getattr(self, "_disk_backlog", False)


    @torch.inference_mode()
    def idle_work(self):
        """Do one more idle flush step, if no job is queued. The async generator calls this while idle"""
        if self.has_idle_work() and not self.num_remaining_jobs():
            self.flush_cache_to_disk(max_pages = self._idle_flush_budget())


    @torch.inference_mode()
    def on_queue_drained(self):
        """
        Idle-transition housekeeping: drop recurrent checkpoints stranded by KV eviction, then defragment the
        page table (both need/prefer a moment with no active block tables), and return whatever host memory
        glibc is retaining from stash/offload churn (issue #277) — sub-threshold residue would otherwise sit
        in the arenas for the whole idle period.
        """
        if self.disk_page_cache is not None and self.disk_cache_flush_on_idle and \
                not self.disk_page_cache.write_disabled:
            # Before pruning: a checkpoint stranded in VRAM/RAM terms may be anchored by pages on disk. One
            # step only; the async generator does the rest while idle (idle_work)
            self.flush_cache_to_disk(max_pages = self._idle_flush_budget())
        if self.recurrent_cache is not None:
            self.recurrent_cache.prune_stranded()
            # The pruned checkpoints' buffers went back to the stash pool; drop them so the RAM
            # is actually returned (the next stash reallocates once)
            host_pool.release()
            if self.model.loaded_tp:
                self.model.tp_dispatch_all(mp_host_pool_release, ())
        self.pagetable.defrag()
        # Dynamic expert placement: apply any pending swap sweep now, between generations —
        # a placement change perturbs the logits slightly (same expert, different device
        # numerics) and must never land mid-stream
        from ..modules.block_sparse_mlp_cpu import run_pending_swap_sweeps
        run_pending_swap_sweeps(self.model.config.infer_params)
        malloc_trim()


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


    def iterate_draftmodel_gen(self, results: list, window: int):

        self._draft_conf_round = None

        # Get shape of active batch
        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + window + 1)
            batch_size += 1
        if batch_size == 0:
            return None

        # Create block index table for batch
        max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        # Pinned staging (as iterate_gen), so the uploads are stream-ordered instead of blocking
        # pageable copies; the draft's own buffers, since the target's are still in flight
        max_pages_batch = (max_pages_batch + 15) // 16 * 16
        block_index = self._staging("draft_block_index", batch_size, max_pages_batch)
        block_index.zero_()
        cache_seqlens = self._staging("draft_cache_seqlens", batch_size)
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

        # Recurrent drafter: its states must stand exactly at the K/V position before the round (the
        # draft consumes every token it drafts, so this holds unless a hybrid round accepted past its
        # window); a lagging state first consumes the gap in a plain pass
        draft_states = None
        if self.draft_recurrent:
            draft_states = []
            for job in self.active_jobs:
                if not job.is_prefill_done(): continue
                seq = job.sequences[0]
                ds = job.recurrent_state.draft
                if ds.position < seq.kv_position:
                    gap = seq.sequence_ids.torch_slice(ds.position, seq.kv_position)
                    self.draft_model.prefill(gap, {
                        "attn_mode": "flash_attn",
                        "block_table": seq.block_index_tensor,
                        "cache": self.draft_cache,
                        "cache_seqlens": torch.tensor([ds.position], dtype = torch.int32),
                        "recurrent_states": [ds],
                    })
                assert ds.position == seq.kv_position, "recurrent draft state out of step with the K/V position"
                draft_states.append(ds)

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

        def draft_params(step):
            p = {
                "attn_mode": "flash_attn",
                "block_table": block_index,
                "cache": self.draft_cache,
                "cache_seqlens": cache_seqlens,
            }
            if draft_states is not None:
                # One speculative pass over the drafted tokens, fed one step at a time; the round's
                # verification settles it (rewind to the accepted prefix) in iterate_gen
                p.update({"recurrent_states": draft_states, "recurrent_history": True, "recurrent_spec_step": step})
            return p

        # Greedy sample batched draft tokens. With a confidence calibrator, drafting stops early
        cal = self.draft_calibrator
        conf_cols = []
        reach = None
        for idx in range(window):
            batch_logits = self.draft_model.forward(input_ids = batch_ids, params = draft_params(idx))
            if cal is not None:
                conf, new_ids = torch.max(batch_logits, dim = -1)
            else:
                new_ids = torch.argmax(batch_logits, dim = -1)
            self.draft_ids_pinned[:batch_size, idx:idx+1].copy_(new_ids)
            batch_ids.copy_(new_ids)
            cache_seqlens += 1
            if cal is not None:
                c = conf.float().cpu()
                conf_cols.append(c)
                est = [cal.estimate(v) for v in c.view(-1).tolist()]
                reach = est if reach is None else [r * e for r, e in zip(reach, est)]
                if idx + 1 < window and max(reach) < cal.confidence:
                    window = idx + 1
                    break

        # The last drafted token goes into the draft cache too, so a fully accepted round leaves the
        # drafter at the next unprocessed token
        self.draft_model.prefill(input_ids = batch_ids, params = draft_params(window))

        if conf_cols:
            self._draft_conf_round = {
                "ids": self.draft_ids_pinned[:batch_size, :window],
                "conf": torch.cat(conf_cols, dim = 1),
                "window": window,
            }

        return self.draft_ids_pinned[:, :window]


    def iterate_draftmodel_mtp_gen(self, results: list, window: int):

        self._draft_conf_round = None

        # Get shape of active batch
        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + window + 1)
            batch_size += 1
        if batch_size == 0:
            return None

        # Create block index table for batch
        max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        # Pinned staging (as iterate_gen), so the uploads are stream-ordered instead of blocking
        # pageable copies; the draft's own buffers, since the target's are still in flight
        max_pages_batch = (max_pages_batch + 15) // 16 * 16
        block_index = self._staging("draft_block_index", batch_size, max_pages_batch)
        block_index.zero_()
        cache_seqlens = self._staging("draft_cache_seqlens", batch_size)
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

        # Greedy sample batched draft tokens. As in iterate_draftmodel_gen, drafting stops once
        # every row's running product of estimated conditional acceptance probabilities falls
        # below the confidence target, keeping the first low-confidence token as the label probe
        cal = self.draft_calibrator
        conf_cols = []
        reach = None
        for idx in range(window):
            params = {
                "target_hidden": temp_hidden,
                "attn_mode": "flash_attn",
                "block_table": block_index,
                "cache": self.draft_cache,
                "cache_seqlens": cache_seqlens,
                "draft_step": idx,   # heads specialized per depth pick their head from this
            }
            if cal is not None:
                params["export_draft_conf"] = True
            batch_state = self.draft_model.forward(batch_ids, params)
            lm_head = self.model.modules[self.model.logit_layer_idx]
            batch_state = lm_head.prepare_for_device(batch_state, params)
            new_ids = self.draft_model.sample_from_state(batch_state, params)
            self.draft_ids_pinned[:batch_size, idx:idx+1].copy_(new_ids)
            batch_ids.copy_(new_ids)
            cache_seqlens += 1
            temp_hidden = batch_state
            draft_conf = params.get("draft_conf")
            if cal is not None and draft_conf is not None:
                c = draft_conf.float().cpu()
                conf_cols.append(c)
                est = [cal.estimate(v) for v in c.view(-1).tolist()]
                reach = est if reach is None else [r * e for r, e in zip(reach, est)]
                if idx + 1 < window and max(reach) < cal.confidence:
                    window = idx + 1
                    break

        if conf_cols and len(conf_cols) == window:
            self._draft_conf_round = {
                "ids": self.draft_ids_pinned[:batch_size, :window],
                "conf": torch.cat(conf_cols, dim = 1),
                "window": window,
            }

        return self.draft_ids_pinned[:, :window]


    # TODO: Refactor, share code with other draft fns
    def iterate_draftmodel_dflash_gen(self, results: list, window: int):

        self._draft_conf_round = None

        # Get shape of active batch
        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(
                max_seq_len,
                job.get_max_seq_len() + window + 1,
                job.get_max_seq_len() + self.draft_model.config.block_size + 1
            )
            batch_size += 1
        if batch_size == 0:
            return None

        # The diffusion drafter always runs at its fixed block size; the row ceiling and the dynamic
        # window only truncate the verified part of the drafted block

        # Create block index table for batch
        max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        # Pinned staging (as iterate_gen), so the uploads are stream-ordered instead of blocking
        # pageable copies; the draft's own buffers, since the target's are still in flight
        max_pages_batch = (max_pages_batch + 15) // 16 * 16
        block_index = self._staging("draft_block_index", batch_size, max_pages_batch)
        block_index.zero_()
        cache_seqlens = self._staging("draft_cache_seqlens", batch_size)
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

        # Run draft model. A drafter with sliding-window states appends its block to each row's
        # window ring; the rows the verification accepts are written over it afterwards
        # (update_kv_from_target below), which also rolls the state back to the K/V position
        params = {
            "attn_mode": "flash_attn",
            "block_table": block_index,
            "cache": self.draft_cache,
            "cache_seqlens": cache_seqlens,
        }
        if self.draft_recurrent:
            params["recurrent_states"] = [job.recurrent_state.draft for job in self.active_jobs if job.is_prefill_done()]
        if self.draft_calibrator is not None:
            params["export_draft_conf"] = True
        out_state = self.draft_model.forward(
            input_ids = batch_ids,
            params = params,
        )
        new_ids = self.draft_model.sample_from_state(out_state, params)

        # Draft models with a confidence head cap the usable draft length per round;
        # 0 means no draft position cleared the threshold, so skip drafting entirely
        conf_len = params.get("draft_confidence_len")
        if conf_len is not None:
            if conf_len == 0:
                return None
            window = min(window, conf_len)

        # Crop out the first token after sampling to keep batch contiguous for lm_head
        new_ids = new_ids[:, 1:]

        # Confidence-calibrated truncation: cut the batch window at the first position whose
        # drafter confidence falls below the calibrated threshold, taking the longest cut across
        # rows (jobs wanting less just reject earlier). The full block and
        # its scores are kept so iterate_gen can label the verified positions afterwards
        draft_conf = params.get("draft_conf")
        if self.draft_calibrator is not None and draft_conf is not None:
            conf = draft_conf[:batch_size, 1:].float().cpu()
            ids_full = new_ids[:batch_size, :].cpu()
            thr = self.draft_calibrator.threshold()
            below = conf[:, :window] < thr
            any_below = below.any(dim = 1)
            first_below = below.int().argmax(dim = 1)
            cuts = torch.where(any_below, first_below, torch.full_like(first_below, window))
            w_used = int(cuts.max().item())
            self._draft_conf_round = {"ids": ids_full, "conf": conf, "window": w_used}
            if w_used == 0:
                return None
            window = w_used

        self.draft_ids_pinned[:batch_size, :window].copy_(new_ids[:batch_size, :window])
        return self.draft_ids_pinned[:, :window]


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
        window = self.num_draft_tokens
        draft_ids = []
        min_len = window
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            d = job.get_ngram_draft(window)
            min_len = min(min_len, d.shape[-1])
            draft_ids.append(d)

        if min_len == 0:
            return None

        # Trim to minimum length in batch
        draft_ids = torch.cat([d[:, :min_len] for d in draft_ids], dim = 0)
        return draft_ids


    def new_recurrent_state(self) -> PairedState:
        """A fresh (cleared) pair of recurrent states for a new sequence, one side per recurrent model"""
        target = draft = None
        try:
            target = self.cache.get_new_state() if self.target_recurrent else None
            draft = self.draft_cache.get_new_state() if self.draft_recurrent else None
            return PairedState(target, draft)
        except BaseException:
            if draft is not None:
                draft.free()
            if target is not None:
                target.free()
            raise


    def recurrent_state_from_stashed(self, stashed: dict, position: int) -> PairedState:
        """The pair restored from a checkpoint stash (PairedState.stash) at `position`"""
        target = draft = None
        try:
            target = self.cache.new_from_stashed(stashed["target"], position) if self.target_recurrent else None
            draft = self.draft_cache.new_from_stashed(stashed["draft"], position) if self.draft_recurrent else None
            return PairedState(target, draft)
        except BaseException:
            if draft is not None:
                draft.free()
            if target is not None:
                target.free()
            raise


    def row_limited(self, window: int) -> int:
        """
        Trim a draft window so that bsz * (1 + window) rows stay under draft_max_rows (see the constructor).
        """
        if not self.draft_max_rows or not window:
            return window
        rows = 0
        for job in self.active_jobs:
            if job.is_prefill_done():
                rows += len(job.sequences)
        if rows < 1:
            return window
        limited = min(window, max(0, self.draft_max_rows // rows - 1))
        if window == self.num_draft_tokens and limited != self._draft_window_last:
            self._draft_window_last = limited
            logger.debug("draft_max_rows %d, %d rows: draft window %d",
                         self.draft_max_rows, rows, limited)
        return limited

    def iterate_hybrid_gen(self, draft_tokens: torch.Tensor | None, H: int):
        """
        Replace the round's draft with hybrid_draft_tokens tokens of suffix-match continuation when some job's
        current suffix matches at least that many tokens of its earlier context and the continuation agrees
        with the regular draft (the model's own short-range guess). The verification pass is rectangular, so
        when any row takes the long draft every row does: rows without a qualifying match keep their regular
        draft and are padded with whatever continuation their longest match offers, then with repeats of
        their last token, which verification rejects at the first mismatch. Returns the (rows, width) draft
        for iterate_gen, or the regular draft unchanged. H is that length for this round, already trimmed
        to the row ceiling.
        """
        jobs = [job for job in self.active_jobs if job.is_prefill_done()]
        if not jobs or not H:
            return draft_tokens
        rows = []
        fire = False
        for i, job in enumerate(jobs):
            short = draft_tokens[i : i + 1] if draft_tokens is not None else None
            long, matched = job.get_hybrid_draft(H)
            # A qualifying row: a match of at least H tokens, H tokens following it, and no disagreement with
            # the regular drafter over the positions it covers
            ok = matched >= H and long.shape[-1] == H
            if ok and short is not None and short.shape[-1] and not torch.equal(long[:, :short.shape[-1]], short):
                ok = False
            fire = fire or ok
            rows.append((ok, short, long))
        if not fire:
            return draft_tokens

        out = torch.empty((len(jobs), H), dtype = torch.long)
        for i, (ok, short, long) in enumerate(rows):
            if ok:
                out[i] = long[0]
                continue
            parts = []
            if short is not None and short.shape[-1]:
                parts.append(short[0])
            if long.shape[-1]:
                parts.append(long[0])
            fill = torch.cat(parts) if parts else jobs[i].sequences[0].sequence_ids.torch()[0, -1:]
            fill = fill[:H]
            out[i, : fill.shape[-1]] = fill
            out[i, fill.shape[-1] :] = fill[-1]
        # The regular drafter's confidence record no longer describes the verified window
        self._draft_conf_round = None
        self.hybrid_rounds += 1
        return out


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


    def _stage_arena(self, rows: int, width: int, with_positions: bool):
        """
        Pinned int32 arena holding one step's block index [rows, width], cache_seqlens [rows] and optionally
        positions [rows] back to back (the layout ext.generator_stage_decode fills), with its views and a device
        copy per attention device. Keyed by shape, so the views keep their identity from step to step.
        """
        key = (rows, width, with_positions)
        arena = self.stage_arenas.get(key)
        if arena is None:
            if len(self.stage_arenas) >= 32:
                self.stage_arenas.clear()
            n = rows * width + rows * (2 if with_positions else 1)
            host = torch.zeros((n,), dtype = torch.int32, pin_memory = True)
            arena = (host, self._arena_views(host, rows, width, with_positions), {})
            self.stage_arenas[key] = arena
        return arena


    @staticmethod
    def _arena_views(t: torch.Tensor, rows: int, width: int, with_positions: bool):
        nbi = rows * width
        return (
            t[:nbi].view(rows, width),
            t[nbi:nbi + rows],
            t[nbi + rows:nbi + 2 * rows] if with_positions else None,
        )


    def _upload_arena(self, arena, rows: int, width: int, with_positions: bool, params: dict):
        """
        Upload the staged tables to every attention device with one copy each, and seed the forward's per-device
        tensor cache (util.tensor.get_for_device) with the device views, so the layers find them there instead of
        uploading each table separately when first touched. The device arena is reused from step to step: its next
        upload is stream-ordered after the kernels reading this one.
        """
        if self.stage_devices is None:
            # TP forwards ship params to the workers themselves; there the tables stay host-side
            self.stage_devices = [] if self.model.loaded_tp else \
                sorted({torch.device(m.device) for m in self.model.get_cache_layers() if m.device is not None and
                        torch.device(m.device).type == "cuda"}, key = lambda d: d.index)
        if not self.stage_devices:
            return
        host, views, dev = arena
        dev_cache = {}
        for device in self.stage_devices:
            d = dev.get(device)
            if d is None:
                darena = torch.empty_like(host, device = device)
                d = (darena, self._arena_views(darena, rows, width, with_positions))
                dev[device] = d
            darena, dviews = d
            with torch.cuda.device(device):
                ext.cuda_memcpy_async(darena.data_ptr(), host.data_ptr(), host.numel() * 4, 1)
            for hv, dv in zip(views, dviews):
                if hv is not None:
                    dev_cache[(id(hv), device)] = (hv, dv)
        params["dev_cache"] = dev_cache


    def iterate_gen(self, results: list, draft_tokens: torch.Tensor | None = None):

        # Get shape of active batch
        # Only jobs that have finished prefill can participate in token generation. The maximum sequence length
        # determines how many cache pages the temporary block table needs for this iteration.
        batch_size = 0
        max_seq_len = 0
        draft_len = draft_tokens.shape[-1] if draft_tokens is not None else 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + draft_len)
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
        use_offsets = "mrope" in self.model.caps
        arena = self._stage_arena(batch_size, max_pages_batch, use_offsets)
        block_index, cache_seqlens, positions = arena[1]

        # Collect the rows' tables, and their input IDs and indexed embeddings. Jobs may contribute multiple
        # sequence rows. logit_mapping records the slice of the model output that belongs to each active job, while
        # batch_jobs keeps the corresponding Job objects in compact-batch order. Single-token decode rows (no draft)
        # have their input token gathered and recorded in its cache page by the staging call below, together with
        # the tables; any other row takes the general path through get_input_ids_list
        seq_bi, seq_kvp, seq_rope = [], [], []
        dec_ids, dec_pages, dec_offsets = [], [], []
        decode_rows = draft_tokens is None
        active_embeddings = []
        logit_mapping = []
        batch_jobs = []
        rows = 0
        for job in self.active_jobs:
            logit_mapping.append(rows)
            if not job.is_prefill_done(): continue
            if job.time_first_token is None:
                cuda_sync_active()
                job.time_first_token = time.time()
            for seq in job.sequences:
                kvp = seq.kv_position
                seq_bi.append(seq.block_index_tensor)
                seq_kvp.append(kvp)
                if use_offsets:
                    seq_rope.append(job.alt_rope_offset)
                if decode_rows:
                    decode_rows = len(seq.sequence_ids) - kvp == 1
            rows += len(job.sequences)
            batch_jobs.append(job)
            active_embeddings += job.embeddings
        logit_mapping.append(rows)

        if decode_rows:
            for job in batch_jobs:
                for seq in job.sequences:
                    page = seq.allocated_pages[seq.kv_position // PAGE_SIZE]
                    assert page.ref_count == 1
                    dec_ids.append(seq.sequence_ids.tensor)
                    dec_pages.append(page.sequence)
                    dec_offsets.append(page.kv_position)
                    page.kv_position += 1
                    page.can_revert = False
            batch_ids = self._staging("batch_ids", rows, 1, torch.long)
        else:
            batch_ids = None
        ext.generator_stage_decode(
            arena[0], max_pages_batch, seq_bi, seq_kvp, seq_rope, batch_ids, dec_ids, dec_pages, dec_offsets
        )
        if not decode_rows:
            input_ids_list = []
            for job in batch_jobs:
                input_ids_list += job.get_input_ids_list(draft_tokens, len(input_ids_list), add_to_cache = True)
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
        if self.target_recurrent:
            batch_states = [job.recurrent_state.target for job in batch_jobs]

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
        self._upload_arena(arena, batch_size, max_pages_batch, use_offsets, params)
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
                # The sampled tokens are the next forward's inputs
                self.model.prefetch_tokens([t for _, _, s in launched for t in s[0].view(-1).tolist()])

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
                W = batch_logits.shape[1]

                # Positions within a job are independent when no per-position constraint advances between
                # them: no active filter (masks are rebuilt per accepted position), no sampling past IDs
                # (repetition penalty), no banned-string checkpoint, and outside the min_new_tokens mask
                # window (token healing, new_tokens < 0, is covered by the last test). Then all samplers
                # launch up front in serial-loop order (each position's RNG draw is unchanged) and the W
                # sampled tokens read back through one pinned buffer and one synchronize, as in the
                # no-draft path, leaving a host-only accept/reject walk. Constrained jobs keep the serial
                # loop below.
                # A dormant triggered filter can activate on a token inside the window and constrain the
                # positions after it, and a pending constrain_output_now injection is consumed one token per
                # sampled position; both keep the serial loop
                batchable = (
                    not _NO_BATCH_READBACK
                    and b - a == 1
                    and job.checkpoint is None
                    and job.forced_ids is None
                    and job.new_tokens >= job.min_new_tokens
                    and not any(f.is_active or f.trigger_token is not None for f in job.filters)
                    and not job.sampler.reqs_past_ids
                )
                staged = None
                if batchable:
                    rng_state = job.rng.getstate()
                    staged = []
                    any_cuda = False
                    for i in range(W):
                        token_logits = job_logits[:, i:i + 1, :]
                        sampled = job.receive_logits(token_logits)
                        is_cuda = sampled[0].is_cuda and sampled[0].numel() == 1
                        staged.append((sampled, is_cuda))
                        any_cuda |= is_cuda
                    if any_cuda:
                        pinned_w = getattr(self, "draft_sample_pinned", None)
                        if pinned_w is None or pinned_w.shape[1] < W:
                            pinned_w = torch.empty((1, max(W, 8)), dtype = torch.long, pin_memory = True)
                            self.draft_sample_pinned = pinned_w
                        for i, (sampled, is_cuda) in enumerate(staged):
                            if is_cuda:
                                pinned_w[0:1, i:i + 1].copy_(sampled[0].view(1, 1), non_blocking = True)
                                staged[i] = ((pinned_w[0:1, i:i + 1],) + sampled[1:], False)
                        torch.cuda.synchronize(batch_logits.device)

                for i in range(W):
                    positions_used = i + 1
                    token_logits = job_logits[:, i:i + 1, :]
                    if staged is not None:
                        (next_token, next_k_tokens, next_k_probs, next_prob), _ = staged[i]
                    else:
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
                    # token limit produces an EOS event. Draft positions from the EOS position onward were fed
                    # to verification but never resolved; count them as rejected so accepted + rejected equals
                    # the number of drafted positions (no cache/state rollback needed, the pages are released).
                    # The EOS stream event was already built inside receive_sample, so patch its counter too
                    if eos:
                        num_unresolved = batch_logits.shape[1] - 1 - i if draft_tokens is not None else 0
                        if num_unresolved:
                            job.rejected_draft_tokens += num_unresolved
                            for r_ in reversed(results):
                                if r_.get("eos") and r_["job"] is job:
                                    if "rejected_draft_tokens" in r_:
                                        r_["rejected_draft_tokens"] = job.rejected_draft_tokens
                                    break
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
                                # Both lists take one entry per active filter (prepare_logit_mask indexes them
                                # in step)
                                if f.use_background_worker():
                                    job.filter_futures.append(self.filter_pool.submit(f.get_next_logit_mask))
                                    job.logit_masks.append(None)
                                else:
                                    job.logit_masks.append(f.get_next_logit_mask())
                                    job.filter_futures.append(None)

                            # Update masks and past IDs
                            job.prepare_logit_mask()
                            job.prepare_sampling_past_ids()

                if staged is not None and positions_used < W:
                    # The serial loop would have drawn one sampler seed per position it reached; pass 1 drew
                    # W. Rewind the job's RNG and replay only the draws that were used, so a seeded stream
                    # continues exactly as it would have on the serial loop
                    job.rng.setstate(rng_state)
                    for _ in range(positions_used):
                        job.rng.randint(0, (1 << 32) - 1)

                # Make sure outgoing state is valid if entire draft was accepted
                if batch_states and draft_tokens is not None and rejected == 0:
                    batch_states[j].rewind(0)

                # Record per-round draft stats. Skip abandoned windows (banned-string rewind);
                # checkpoint-boundary truncations are rare enough to count as ordinary rejections.
                if draft_tokens is not None and rejected != -1:
                    if self.record_draft_stats:
                        job.draft_stats.append((
                            job.new_tokens,
                            draft_tokens.shape[-1],
                            accepted_length - 1,
                        ))

                # The last sampled token leads the next forward
                self.model.prefetch_tokens([sampled_token.item()])

                accepted_lengths.append(accepted_length)
                j += 1

        # Update the draft-confidence calibration with this round's verification outcomes (any
        # draft-model mode)
        if self.draft_calibrator is not None and self._draft_conf_round is not None:
            st = self._draft_conf_round
            self._draft_conf_round = None
            cal = self.draft_calibrator
            row = 0
            for job, a_idx, b_idx in zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:]):
                if a_idx == b_idx: continue
                accepted_length = accepted_lengths[row]
                conf = st["conf"][row]
                ids_full = st["ids"][row]
                row += 1
                if id(job) in rewound_jobs:
                    continue
                a = accepted_length - 1
                for i in range(a):
                    cal.add_label(conf[i].item(), True)
                if a < conf.shape[-1]:
                    seq = job.sequences[0]
                    n = len(seq.sequence_ids)
                    tail = seq.sequence_ids.torch_slice(n - 1, n).item()
                    cal.add_label(conf[a].item(), ids_full[a].item() == tail)
            cal.decay_step()

        # Settle a recurrent drafter's speculative pass: it consumed the round's input token and every
        # drafted token (last_history of them); the rejected drafts are undone, leaving its state at
        # the accepted prefix, i.e. at the new K/V position (or behind it after a long hybrid round)
        if self.draft_recurrent and draft_tokens is not None:
            accepted_idx = 0
            for job, a_idx, b_idx in zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:]):
                if a_idx == b_idx:
                    continue
                accepted_drafts = accepted_lengths[accepted_idx] - 1
                accepted_idx += 1
                if id(job) in rewound_jobs or job.recurrent_state is None:
                    continue   # a checkpoint rewind replaced the pair
                ds = job.recurrent_state.draft
                ds.rewind(max(0, ds.last_history - accepted_drafts))

        # Accept new target_hidden if DFlash. DFlash draft models can update their cache from target-model hidden
        # states for the tokens accepted above, keeping draft and target cache layouts aligned.
        if self.dflash_draft:
            update_params = {
                "block_table": block_index,
                "cache_seqlens": p_cache_seqlens,
            }
            lengths = accepted_lengths
            if self.draft_recurrent:
                # A row whose pair a checkpoint rewind replaced gets its window from the replay
                # prefill instead; its length 0 skips the write (and the rollback)
                drafting = [job for job, a, b in zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:]) if a != b]
                update_params["recurrent_states"] = [job.recurrent_state.draft for job in drafting]
                lengths = [0 if id(job) in rewound_jobs or job.recurrent_state is None else n
                           for job, n in zip(drafting, accepted_lengths)]
            self.draft_model.update_kv_from_target(
                target_hidden = p_export_states,
                cache = self.draft_cache,
                lengths = lengths,
                params = update_params,
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
            self.on_queue_drained()


    def reap_failed_job(self, job, error, results: list):
        """
        Contain a per-job failure so the generator stays usable for other jobs: release the
        failed job's pages and recurrent state, drop it from the active set, and append an
        error result carrying the standard serial/stage/eos fields. The async wrapper
        delivers the raw exception to that job's consumer only (AsyncJob.__aiter__ re-raises
        queued exceptions) and the sync generate() API raises it, instead of the failure
        escaping to _run_iteration, which latches AsyncGenerator.error permanently and kills
        the iteration task for every job.
        """
        try:
            job.deallocate_pages()
        except Exception:
            logger.error(
                "reap_failed_job: deallocate_pages() raised while cleaning up failed "
                "job %s; pages or recurrent state may be stranded",
                job,
                exc_info = True,
            )
        if job in self.active_jobs:
            self.active_jobs.remove(job)
        results.append({
            "job": job,
            "serial": job.serial_number,
            "stage": "error",
            "eos": True,
            "error": error,
        })


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
                try:
                    job.allocate_pages()
                except Exception as e:
                    self.reap_failed_job(job, e, results)
                    continue
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
                if r["stage"] == "error":
                    # A per-job failure was contained by reap_failed_job; surface the
                    # original exception to this caller rather than returning a silently
                    # truncated completion. Jobs from the same batch that are still in
                    # flight are left to the generator's normal queue handling.
                    raise r["error"]
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
