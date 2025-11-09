from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING
from ..cache.cache import Cache
if TYPE_CHECKING:
    from .generator import Generator
from ..constants import PAGE_SIZE
import numpy as np
from .pagetable import CachePage, Sequence, tensor_hash_checksum, random_hash
from .filter import Filter
import random
from collections import deque
import time
from ..ext import exllamav3_ext as ext
from .sampler import Sampler, DefaultSampler
from ..util.tensor import SeqTensor
from ..util import profile_opt
from ..tokenizer import MMEmbedding
from functools import lru_cache
from ..util import profile_opt

# Convert list of strings to UTF32 format to pass by reference to partial matching function
@lru_cache(100)
def _strings_to_utf32(strings: tuple[str]) -> tuple[np.ndarray, np.ndarray] | None:
    if not strings: return bytearray(), None

    encoded_strings = [s.encode("utf-32-le") for s in strings]
    encoded_lengths = [len(s) for s in encoded_strings]
    offsets = [0] + encoded_lengths
    for i in range(1, len(offsets)):
        offsets[i] += offsets[i - 1]
    total_length = offsets[-1]
    concat_strings = bytearray(total_length)
    for s, offset in zip(encoded_strings, offsets[:-1]):
        concat_strings[offset:offset + len(s)] = s

    concat_strings = np.frombuffer(concat_strings, dtype = np.uint8)
    offsets = np.frombuffer(np.array(offsets, dtype = np.int32), dtype = np.uint8)
    return concat_strings, offsets


class Job:

    def __init__(
        self,
        input_ids: torch.Tensor | list[torch.Tensor],
        max_new_tokens: int | None = None,
        min_new_tokens: int = 0,
        max_skips: int | None = 4,
        sampler: Sampler | None = None,
        seed: int = None,
        stop_conditions: list | tuple | set | None = None,
        decode_special_tokens: bool = False,
        return_top_tokens: int = 0,
        return_logits: bool = False,
        return_probs: bool = False,
        filters: list[Filter] | None = None,
        token_healing: bool = False,
        identifier: object | None = None,
        banned_strings: list[str] | None = None,
        embeddings: list[MMEmbedding] | None = None,
        max_rq_tokens: int | None = None,
        rq_state: dict | None = None,
        **kwargs
    ):
        """
        Create new job.

        :param input_ids:
            Tokenized IDs of the input prompt, shape (1, n). Alternatively, list of tokenized IDs to inference on
            seperately but sample collectively (e.g. CFG prompt pair)

        :param max_new_tokens:
            Max no. output tokens to allow

        :param min_new_tokens:
            Minimum number of tokens to generate before stop tokens become active. Until this number have been
            sampled, stop tokens are suppressed but stop strings will still end response. May produce garbage output.

        :param max_skips:
            In the event that the job is too large to fit in the cache at any given moment but there are
            smaller jobs pending that would fit, those smaller jobs are started instead. This number
            specifies the maximum number of times a job can be skipped over in favor of a smaller job before it
            stalls the queue. After this, the job is guaranteed to be the next job started.

        :param sampler:
            Sampler

        :param seed:
            RNG seed (determinism is not guaranteed)

        :param stop_conditions:
            List of strings and/or token IDs that will trigger the EOS condition. If a stop condition is
            encountered it is not emitted as output. If the beginning of a stop string is sampled, stream output
            will be held until the stop condition can be resolved.

        :param decode_special_tokens:
            If True, special tokens like <|im_start|> etc. will be decoded and included in the text output.
            If False, special tokens will still be respected as stop conditions.

        :param return_top_tokens:
            Number of top tokens to return, along with their final sampling probabilities. There is some
            performance penalty for enabling this.

        :param return_logits:
            Return pre-sampling logits along with output tokens.

        :param return_probs:
            Return final sampling probability for each chosen token.

        :param filters:
            List of Filters to apply during generation.

        :param token_healing:
            Resample the last token of the input with a prefix constraint. E.g. if the last token is
            "_Hel", it is removed from the input and the first token of the output will be constrained to
            one of "_Hello", "_Help", "_Helium", etc. Only the added part of the healed token is emitted as
            text, i.e. "lo", "p", "ium" etc.

        :param identifier:
            Arbitrary object to return with every stream event relating to this job, e.g. an index to identify the
            output as belonging to a specific position in a batch

        :param embeddings:
            Optional list of MMEmbeddings to use, or list of lists for batched generation

        :param max_rq_tokens:
            Maximum number of tokens before job is requeued. Rounded to nearest page boundary. This limits how
            many new pages are allocated in the cache for the job in any one round and allows a single job to use
            the full cache size without limiting concurrency for other jobs.

        :param rq_state:
            Internal, passed when job requeues itself

        :param kwargs:
        """

        assert all(ids.device.type == "cpu" for ids in input_ids), \
            "input_ids must reside in system memory"

        if rq_state is None:
            rq_state = {}
            self.is_requeued = False
            self.rq_new_tokens = 0
        else:
            self.is_requeued = True
            self.rq_new_tokens = rq_state["rq_new_tokens"]

        self.generator = None
        self.pagetable = None
        self.serial_number = None
        self.identifier = identifier

        self.max_skips = max_skips
        self.skips = 0
        self.all_unique_hashes = None

        # Default sampler settings
        if sampler is None:
            sampler = DefaultSampler()

        # Sampling state
        self.held_text = rq_state.get("held_text", "")
        self.held_tokens = rq_state.get("held_tokens")
        self.held_k_tokens = rq_state.get("held_k_tokens")
        self.held_k_probs = rq_state.get("held_k_probs")
        self.held_probs = rq_state.get("held_probs")
        self.held_logits = rq_state.get("held_logits")
        self.full_completion = rq_state.get("full_completion", "")
        self.seed = seed

        # Prepare sequences
        if not isinstance(input_ids, list):
            input_ids = [input_ids]

        if token_healing and all(ids.shape[-1] > 1 for ids in input_ids):
            input_seq_ids = [ids[:, :-1] for ids in input_ids]
            self.prefix_token = torch.cat([ids[:, -1:] for ids in input_ids], dim = 0)
        else:
            input_seq_ids = input_ids
            self.prefix_token = None

        self.sequences = []
        for ids, seq_ids in zip(input_ids, input_seq_ids):
            assert ids.shape[-1] > 0, \
                "Input IDs cannot be empty."
            assert ids.shape[0] == 1, \
                "input_ids must be [1, seq_len] tensor or list of [1, seq_len] tensors"
            seq = Sequence(ids, seq_ids)
            self.sequences.append(seq)

        # Generation parameters
        self.max_new_tokens = max_new_tokens - 1 or 1
        self.min_new_tokens = min_new_tokens
        self.new_tokens = 0 if self.prefix_token is None else -1
        self.sampler = sampler
        self.rng = rq_state.get("rng")
        if self.rng is None:
            self.rng = random.Random() if seed is None else random.Random(seed)
        self.orig_max_rq_tokens = max_rq_tokens
        self.max_rq_tokens = max_rq_tokens

        # Output options
        self.decode_special_tokens = decode_special_tokens
        self.return_top_tokens = return_top_tokens
        self.return_logits = return_logits
        self.return_probs = return_probs

        # Stop conditions
        self.stop_strings = rq_state.get("stop_strings", set())
        self.stop_tokens = rq_state.get("stop_tokens", set())
        if stop_conditions is not None:
            for t in stop_conditions:
                if isinstance(t, int):
                    self.stop_tokens.add(t)
                elif isinstance(t, str):
                    self.stop_strings.add(t)
                else:
                    raise ValueError("Unsupported type in stop_conditions")
            self.stop_strings_utf32_buffer, self.stop_strings_utf32_offsets = \
                _strings_to_utf32(tuple(list(self.stop_strings)))
        else:
            self.stop_strings_utf32_buffer = rq_state.get("stop_strings_utf32_buffer")
            self.stop_strings_utf32_offsets = rq_state.get("stop_strings_utf32_offsets")

        self.stop_tokens_list = list(self.stop_tokens)
        self.stop_strings_list = list(self.stop_strings)

        # Banned strings
        if banned_strings:
            assert filters is None or len(filters) == 0, \
                "Cannot combine banned strings with filters"
            self.banned_strings = [s.lower() for s in banned_strings]
            self.banned_strings_utf32_buffer, self.banned_strings_utf32_offsets = \
                _strings_to_utf32(tuple(self.banned_strings))
        else:
            self.banned_strings = []
            self.banned_strings_utf32_buffer = None
            self.banned_strings_utf32_offsets = None

        self.checkpoint = None

        # Metrics
        self.time_enqueue = None
        self.time_first_prefill = None
        self.time_first_token = None
        self.time_last_token = None
        self.time_enqueued = rq_state.get("time_enqueued", 0.0)
        self.time_prefill = rq_state.get("time_prefill", 0.0)
        self.time_generate = rq_state.get("time_generate", 0.0)
        self.accepted_draft_tokens = 0
        self.rejected_draft_tokens = 0
        self.cached_pages = 0
        self.cached_tokens = 0
        self.is_finished = False
        self.non_sequential_pages = 0
        self.total_pages = 0

        # Filters
        self.filters = filters if filters is not None else []
        self.filter_futures = []
        self.logit_masks = []
        self.pinned_logit_mask = None
        self.device_logit_mask = None
        self.logits_device = None

        # Embeddings
        self.embeddings = embeddings or []
        self.alt_rope_freqs = None
        self.alt_rope_offset = 0

        # Pinned buffer for IDs during sampling
        self.current_pinned_ids = None
        self.pinned_ids = None  # Lazy alloc

        # Recurrent state
        self.recurrent_state = None


    def get_pinned_logit_mask(self):
        if self.pinned_logit_mask is None:
            self.pinned_logit_mask = torch.empty(
                (1, self.generator.padded_vocab_size),
                dtype = torch.half,
                device = "cpu",
                pin_memory = True
            )
        return self.pinned_logit_mask


    def __repr__(self):
        if self.serial_number is None:
            return "Generator job (new)"
        else:
            return f"Generator job #{self.serial_number}"


    def is_prefill_done(self):
        return all(seq.kv_position == len(seq.sequence_ids) - 1 for seq in self.sequences)


    def get_max_seq_len(self):
        if not self.is_prefill_done():
            return 0
        max_seq_len = 0
        for seq in self.sequences:
            if seq.kv_position == len(seq.sequence_ids) - 1:
                max_seq_len = max(max_seq_len, len(seq.sequence_ids))
        return max_seq_len


    def get_input_ids_list(
        self,
        draft_tokens: torch.Tensor | None = None,
        idx: int = 0,
        add_to_cache: bool = False
    ):
        input_ids_list = []
        for seq in self.sequences:
            ids = seq.sequence_ids.torch_slice(seq.kv_position, None)
            if draft_tokens is not None:
                ids = torch.cat((ids, draft_tokens[idx:idx + 1, :]), dim = -1)
            input_ids_list.append(ids)
            if add_to_cache:
                tokens_to_add = ids.shape[-1]
                skvp = seq.kv_position
                while tokens_to_add:
                    page = seq.allocated_pages[skvp // PAGE_SIZE]
                    assert page.ref_count == 1
                    tokens_page = min(tokens_to_add, PAGE_SIZE - page.kv_position)
                    page.sequence[:, page.kv_position:page.kv_position + tokens_page] = ids[:, :tokens_page]
                    page.kv_position += tokens_page
                    skvp += tokens_page
                    ids = ids[:, tokens_page:]
                    tokens_to_add -= tokens_page
                    page.can_revert = False
        return input_ids_list


    def prepare_logit_mask(self):

        logit_mask = None
        healing = self.prefix_token is not None and self.new_tokens == -1

        # Finish filters and compile logit mask, but delay to avoid conflict with token healing
        if not healing:
            f_idx = 0
            for f in self.filters:
                if not f.is_active:
                    continue
                if f.use_background_worker():
                    f_mask = self.filter_futures[f_idx].result()
                else:
                    f_mask = self.logit_masks[f_idx]
                if logit_mask is None:
                    logit_mask = self.get_pinned_logit_mask()
                    logit_mask.copy_(f_mask)
                else:
                    logit_mask += f_mask
                f_idx += 1
            self.filter_futures.clear()
            self.logit_masks.clear()

        # Add individually blocked tokens to mask
        blocked_tokens = []
        if self.checkpoint and self.checkpoint["offset"] == 0:
            blocked_tokens += self.checkpoint["explored_tokens"]
        if self.new_tokens < self.min_new_tokens:
            blocked_tokens = blocked_tokens + self.stop_tokens_list
        if blocked_tokens:
            if logit_mask is None:
                logit_mask = self.get_pinned_logit_mask()
                logit_mask.zero_()
            logit_mask[:, blocked_tokens] = float("-inf")

        # Mask out all but allowed tokens (token healing)
        allowed_tokens = []
        if healing:
            allowed_tokens += self.generator.tokenizer.get_tokens_with_prefix_id(self.prefix_token)
        if allowed_tokens:
            if logit_mask is None:
                logit_mask = self.get_pinned_logit_mask()
                logit_mask.fill_(float("-inf"))
                logit_mask[:, allowed_tokens] = 0
            else:
                inv_mask = torch.full_like(logit_mask, float("-inf"))
                inv_mask[:, allowed_tokens] = 0
                logit_mask += inv_mask

        # If any logits need masking, move mask to device.
        if logit_mask is not None:
            # logit_mask references the pinned host buffer for this job, which is correctly filled in at this point
            # Copy on the default stream so subsequent sampling blocks until mask is copied to device memory
            self.device_logit_mask = logit_mask.to(self.logits_device, non_blocking = True)
        else:
            self.device_logit_mask = None


    def receive_logits(
        self,
        logits: torch.Tensor,
    ):
        # TODO: (cfg)
        # assert logits.shape[0] == len(self.sequences) == (2 if self.gen_settings.cfg_scale is not None else 1)
        # assert logits.shape[0] == len(self.sequences)
        # assert self.is_prefill_done()
        # assert all(seq.live for seq in self.sequences)

        next_token = self.sampler.forward(
            logits,
            self.current_pinned_ids,
            self.rng.randint(0, (1<<32)-1),
            self.generator.tokenizer,
            logit_mask = self.device_logit_mask
        )

        next_prob, next_k_tokens, next_k_probs = None, None, None

        if self.return_probs or self.return_top_tokens > 0:
            probs = torch.softmax(logits.float(), dim = -1)

            if self.return_probs:
                next_prob = torch.gather(probs.squeeze(0), dim = 1, index = next_token)

            if self.return_top_tokens > 0:
                sorted_probs, sorted_indices = torch.sort(probs, dim = -1, descending = True)
                next_k_tokens = sorted_indices[:, :, :self.return_top_tokens]
                next_k_probs = sorted_probs[:, :, :self.return_top_tokens]

        return next_token, next_k_tokens, next_k_probs, next_prob


    def receive_sample(
        self,
        logits: torch.Tensor | None,
        next_token: torch.Tensor | None,
        next_k_tokens: torch.Tensor | None,
        next_k_probs: torch.Tensor | None,
        next_prob: torch.Tensor | None,
        results: list,
        first_sample_in_sd_batch: bool = True
    ):
        next_token = next_token.cpu()
        next_token_i = next_token.item()

        # Activate/advance filters if not healing
        filter_eos_condition = False
        if self.new_tokens >= 0:
            for f in self.filters:
                if not f.is_active and next_token_i == f.trigger_token:
                    f.is_active = True
                    f.reset()
                elif f.is_active:
                    f.accept_token(next_token_i)
                    if f.is_completed():
                        f.is_active = False
                        if f.eos_after_completed:
                            filter_eos_condition = True

        # Accept token
        self.new_tokens += 1
        requeue_now = self.new_tokens > self.max_rq_tokens - self.generator.num_draft_tokens

        for seq in self.sequences:

            # Accept new token
            seq.sequence_ids.append(next_token)
            page_before = seq.kv_position // PAGE_SIZE
            seq.kv_position += 1
            pos = seq.kv_position
            if self.checkpoint:
                pos -= self.checkpoint["offset"]
            page_after = pos // PAGE_SIZE

            # Hash completed page
            if page_after > page_before:
                assert page_after == page_before + 1

                page = seq.allocated_pages[page_before]

                if page_before > 0:
                    last_page = seq.allocated_pages[page_before - 1]
                    last_hash = last_page.phash
                else:
                    last_hash = None

                page_ids = seq.sequence_ids.torch_slice(page_before * PAGE_SIZE, page_after * PAGE_SIZE)
                new_hash = tensor_hash_checksum(page_ids, last_hash)

                # If another referenced page has the same hash, switch to referencing that instead
                if new_hash in self.pagetable.referenced_pages:
                    new_serial = page.access_serial
                    page.sub_ref()
                    page = self.pagetable.referenced_pages[new_hash]
                    assert page.kv_position == PAGE_SIZE
                    seq.allocated_pages[page_before] = page
                    seq.build_block_index_tensor()
                    page.add_ref(new_serial)

                else:
                    # If an unreferenced page has the same hash, clear that page
                    if new_hash in self.pagetable.unreferenced_pages:
                        up = self.pagetable.unreferenced_pages[new_hash]
                        up.clear()

                    # Update the hash
                    page.update_hash(new_hash)

                # Allow completing the final page without starting a new one (for requeue)
                if page_after < len(seq.allocated_pages):
                    page = seq.allocated_pages[page_after]
                    page.prev_hash = new_hash
                    page.can_revert = False

        # Stream output

        def emit(
            results_: list,
            emit_eos: bool = False,
            eos_reason: str = None,
            emit_held = False,
            suppressed_text = None,
            suppressed_tokens = None,
            stop_token: int = None,
            stop_string: str = None,
            rem_held_text: str = None
        ):
            nonlocal requeue_now

            r = {
                "job": self,
                "stage": "streaming",
                "eos": emit_eos,
                "serial": self.serial_number,
            }

            if eos_reason is not None:
                r.update({ "eos_reason": eos_reason })
                if eos_reason == "stop_token":
                    id_to_piece = self.generator.tokenizer.get_id_to_piece_list(True)
                    r.update({
                        "eos_triggering_token_id": stop_token,
                        "eos_triggering_token_str": id_to_piece[stop_token]
                    })
                    pass
                if eos_reason == "stop_string":
                    r.update({ "eos_triggering_string": stop_string })

            # Requeue if we reach max_rq_tokens
            if requeue_now:
                r.update({ "requeue": True })
                requeue_now = True

                # Can't revert to checkpoint after requeuing, so just emit whatever was being held
                if self.checkpoint is not None:
                    emit_held = True

            if emit_held:
                if self.held_text != "":
                    self.full_completion += self.held_text
                    r.update({ "text": self.held_text })
                    self.held_text = ""
                if self.held_tokens:
                    r.update({ "token_ids": self.held_tokens.torch().clone() })
                    self.held_tokens.clear()
                if self.held_probs:
                    r.update({ "token_probs": self.held_probs.torch().clone() })
                    self.held_probs.clear()
                if self.held_k_tokens:
                    r.update({ "top_k_tokens": self.held_k_tokens.torch().clone() })
                    r.update({ "top_k_probs": self.held_k_probs.torch().clone() })
                    self.held_k_tokens.clear()
                    self.held_k_probs.clear()
                if self.held_logits:
                    r.update({ "logits": self.held_logits.torch().clone() })
                    self.held_logits.clear()

            if suppressed_text:
                r.update({ "suppressed_text": suppressed_text })
                r.update({ "suppressed_tokens": suppressed_tokens.torch() })

            if emit_eos or requeue_now:
                self.time_last_token = time.time()
                self.time_enqueued += self.time_first_prefill - self.time_enqueue
                self.time_prefill += self.time_first_token - self.time_first_prefill
                self.time_generate += self.time_last_token - self.time_first_token

            if emit_eos:
                self.is_finished = True
                r.update({
                    "full_completion": self.full_completion,
                    "new_tokens": self.rq_new_tokens + self.new_tokens,
                    "prompt_tokens": len(self.sequences[0].input_ids),
                    "time_enqueued": self.time_enqueued,
                    "time_prefill": self.time_prefill,
                    "time_generate": self.time_generate,
                    "cached_pages": self.cached_pages // len(self.sequences),
                    "cached_tokens": (self.cached_pages * PAGE_SIZE + self.cached_tokens) // len(self.sequences),
                })
                if self.generator.draft_model:
                    r.update({
                        "accepted_draft_tokens": self.accepted_draft_tokens,
                        "rejected_draft_tokens": self.rejected_draft_tokens
                    })
                if eos_reason == "stop_string":
                    self.held_text = rem_held_text
                rh = {}
                if self.held_text:
                    rh.update({ "text": self.held_text })
                if self.held_tokens:
                    rh.update({ "token_ids": self.held_tokens.torch().clone() })
                if self.held_probs:
                    rh.update({ "token_probs": self.held_probs.torch().clone() })
                if self.held_k_tokens:
                    rh.update({ "top_k_tokens": self.held_k_tokens.torch().clone() })
                    rh.update({ "top_k_probs": self.held_k_probs.torch().clone() })
                if self.held_logits:
                    rh.update({ "logits": self.held_logits.torch().clone() })
                if rh:
                    r.update({ "held": rh })

            if self.identifier is not None:
                r.update({ "identifier": self.identifier })

            results_.append(r)
            return emit_eos, next_token, requeue_now

        # Decode and buffer output
        id_to_piece = self.generator.tokenizer.get_id_to_piece_list(self.decode_special_tokens)
        new_text = id_to_piece[next_token.item()]

        if self.new_tokens == 0:
            unhealed = id_to_piece[self.prefix_token[0].item()]
            new_text = new_text[len(unhealed):]

        self.held_text += new_text
        self.held_tokens.append(next_token)
        if self.return_probs:
            self.held_probs.append(next_prob)
        if self.return_top_tokens > 0:
            self.held_k_tokens.append(next_k_tokens)
            self.held_k_probs.append(next_k_probs)
        if self.return_logits:
            self.held_logits.append(logits[:1, :, :])

        # End on stop tokens
        if next_token.item() in self.stop_tokens:
            return emit(results, emit_eos = True, eos_reason = "stop_token", stop_token = next_token.item())

        # Stop if we reach max_new_tokens
        if self.new_tokens >= self.max_new_tokens - self.generator.num_draft_tokens:
            return emit(results, emit_eos = True, emit_held = True, eos_reason = "max_new_tokens")

        # End on filter completed
        if filter_eos_condition:
            return emit(results, emit_eos = True, emit_held = True, eos_reason = "end_filter")

        # Hold text if it contains an incomplete character
        if 1 <= self.held_text.count("�") < 5:
            test_decode = self.generator.tokenizer.decode(
                self.held_tokens.torch(),
                decode_special_tokens = self.decode_special_tokens
            )[0]
            if not "�" in test_decode:
                self.held_text = test_decode
            else:
                # Don't hold forever if a broken generation yields a replacement character but never completes
                # the Unicode symbol
                return emit(results, emit_held = (len(test_decode) > 20))

        # Hold text as long as it contains part of a banned string

        def unset_checkpoint():
            self.checkpoint = None

        def set_checkpoint():
            if self.checkpoint is None:
                self.checkpoint = {
                    "offset": 1,
                    "held_text": self.held_text[:-len(new_text)],
                    "held_tokens": self.held_tokens.clone(1),
                    "held_probs": self.held_probs.clone(1),
                    "held_k_tokens": self.held_k_tokens.clone(1),
                    "held_k_probs": self.held_k_probs.clone(1),
                    "held_logits": self.held_logits.clone(1),
                    "explored_tokens": [next_token.item()],
                }
            else:
                self.checkpoint["offset"] += 1
                if self.checkpoint["offset"] == 1:
                    self.checkpoint["explored_tokens"].append(next_token.item())

        def rewind_checkpoint():
            assert self.checkpoint is not None
            offset = self.checkpoint["offset"]
            self.new_tokens -= offset
            for seq in self.sequences:
                p_page = seq.kv_position // PAGE_SIZE
                seq.kv_position -= offset
                seq.sequence_ids.truncate(len(seq.sequence_ids) - offset)
                n_page = seq.kv_position // PAGE_SIZE
                for pi in range(n_page, p_page + 1):
                    page = seq.allocated_pages[pi]
                    page.can_revert = False
                    if page.kv_position == PAGE_SIZE:
                        page.update_hash(random_hash())
                    if pi == n_page:
                        page.kv_position = seq.kv_position - pi * PAGE_SIZE
                    else:
                        page.kv_position = 0
            off_tokens = self.held_tokens.slice(len(self.checkpoint["held_tokens"]), None)
            off_text = self.held_text[len(self.checkpoint["held_text"]):]
            self.held_text = self.checkpoint["held_text"]
            self.held_token = self.checkpoint["held_tokens"]
            self.held_probs = self.checkpoint["held_probs"]
            self.held_k_tokens = self.checkpoint["held_k_tokens"]
            self.held_k_probs = self.checkpoint["held_k_probs"]
            self.held_logits = self.checkpoint["held_logits"]
            self.checkpoint["offset"] = 0
            return off_tokens, off_text

        if requeue_now:
            unset_checkpoint()

        elif self.banned_strings_utf32_offsets is not None and self.new_tokens > 0:
            match = ext.partial_strings_match(
                np.frombuffer(self.held_text.lower().encode("utf-32-le"), dtype = np.uint8),
                self.banned_strings_utf32_offsets,
                self.banned_strings_utf32_buffer
            )
            if match >= 0:
                set_checkpoint()
                offending_tokens, offending_text = rewind_checkpoint()
                return emit(
                    results,
                    emit_held = True,
                    suppressed_text = offending_text,
                    suppressed_tokens = offending_tokens
                )
            elif match == -2:
                set_checkpoint()
                return emit(results)
            else:
                unset_checkpoint()

        # End on stop strings

        if self.stop_strings_utf32_offsets is not None:
            match = ext.partial_strings_match(
                np.frombuffer(self.held_text.encode("utf-32-le"), dtype = np.uint8),
                self.stop_strings_utf32_offsets,
                self.stop_strings_utf32_buffer
            )
            if match >= 0:
                held = self.held_text[match:]
                self.held_text = self.held_text[:match]
                for s in self.stop_strings:
                    if held.startswith(s):
                        return emit(
                            results,
                            emit_eos = True,
                            emit_held = True,
                            eos_reason = "stop_string",
                            stop_string = s,
                            rem_held_text = held
                        )
                assert False, "Detected stop string but couldn't identify it (logic error)"
            if match == -2:
                return emit(results)

        # Stream output

        return emit(results, emit_held = True)


    def prepare_for_requeue(self):
        assert len(self.sequences) == 1

        seq = self.sequences[0]
        last_completed_tokens = len(seq.sequence_ids) - len(seq.input_ids)
        new_input = seq.sequence_ids.torch()

        rq_state = {
            "rng": self.rng,
            "stop_strings": self.stop_strings,
            "stop_tokens": self.stop_tokens,
            "stop_strings_utf32_buffer": self.stop_strings_utf32_buffer,
            "stop_strings_utf32_offsets": self.stop_strings_utf32_offsets,
            "held_text": self.held_text,
            "held_tokens": self.held_tokens,
            "held_probs": self.held_probs,
            "held_k_tokens": self.held_k_tokens,
            "held_k_probs": self.held_k_probs,
            "held_logits": self.held_logits,
            "full_completion": self.full_completion,
            "time_enqueued": self.time_enqueued,
            "time_prefill": self.time_prefill,
            "time_generate": self.time_generate,
            "rq_new_tokens": self.new_tokens - 1
        }

        serial_number = self.serial_number
        generator = self.generator

        rq_job = self
        self.__init__(
            input_ids = new_input,
            max_new_tokens = self.max_new_tokens - last_completed_tokens,
            min_new_tokens = max(self.min_new_tokens - last_completed_tokens, 0),
            sampler = self.sampler,
            decode_special_tokens = self.decode_special_tokens,
            return_top_tokens = self.return_top_tokens,
            return_logits = self.return_logits,
            return_probs = self.return_probs,
            filters = self.filters,  # Carries over state
            token_healing = False,  # Token healed on first round
            identifier = self.identifier,
            banned_strings = self.banned_strings,
            embeddings = self.embeddings,
            max_rq_tokens = self.orig_max_rq_tokens,
            rq_state = rq_state,
        )

        rq_job.prepare_for_queue(generator, serial_number, rq = True)
        return rq_job


    def prepare_for_queue(self, generator, serial_number: int, rq: bool = False):

        # Attach to generator
        self.serial_number = serial_number
        self.generator = generator
        self.pagetable = generator.pagetable
        self.skips = 0

        # Align max_rq_tokens to page boundary or recurrent checkpoint
        if self.max_rq_tokens is not None:
            if len(self.sequences) == 1:
                boundary = self.generator.recurrent_checkpoint_interval \
                    if self.generator.recurrent_cache is not None else PAGE_SIZE
                x = len(self.sequences[0].input_ids)
                y = (x - 1 + self.max_rq_tokens + boundary - 1) // boundary * boundary
                self.max_rq_tokens = y - x
        else:
            self.max_rq_tokens = self.max_new_tokens + 1

        # Compatibility checks
        assert not self.banned_strings or self.generator.recurrent_cache is None, \
            "Cannot use banned strings on recurrent model"

        # Hash full pages of input IDs
        all_unique_hashes = set()
        all_unique_pages = 0
        for seq in self.sequences:
            unique_hashes, unique_pages = seq.prepare(self.prefix_token is not None, self.max_rq_tokens)
            all_unique_hashes |= unique_hashes
            all_unique_pages += unique_pages
        self.all_unique_hashes = list(all_unique_hashes)

        # Make sure the request can potentially fit
        total_pages = len(self.all_unique_hashes) + seq.new_unique_pages
        max_pages = self.pagetable.max_pages
        assert total_pages <= max_pages, \
            f"Job requires {total_pages} pages (only {max_pages} available) and cannot " + \
            f"be enqueued. Total cache allocated is {max_pages} * {PAGE_SIZE} = " + \
            f"{self.generator.max_total_tokens} tokens"
        assert len(self.sequences) <= self.generator.max_batch_size, \
            f"Job requires a minimum batch size of {len(self.sequences)}. Max supported batch size in" + \
            f"generator is {self.generator.max_batch_size}."

        # Initial conditions
        if not rq:
            self.held_text = ""
            self.held_tokens = SeqTensor((1, 0), dtype = torch.long, seq_dim = -1)
            self.held_k_tokens = SeqTensor((1, 0, self.return_top_tokens), dtype = torch.long, seq_dim = 1)
            self.held_k_probs = SeqTensor((1, 0, self.return_top_tokens), dtype = torch.float, seq_dim = 1)
            self.held_probs = SeqTensor((1, 0), dtype = torch.float, seq_dim = -1)
            self.held_logits = SeqTensor((1, 0, self.generator.padded_vocab_size), dtype = torch.float, seq_dim = 1)
            self.full_completion = ""

        self.time_enqueue = time.time()

        # Prepare MRoPE embeddings
        if self.embeddings and generator.model.caps.get("mrope"):
            ids = self.sequences[0].sequence_ids.torch()
            freqs, offset = generator.model.g_rope.get_mrope_freqs(
                ids,
                self.embeddings,
                ids.shape[-1]  # + self.max_new_tokens
            )
            self.alt_rope_freqs = freqs
            self.alt_rope_offset = offset - ids.shape[-1]
        else:
            self.alt_rope_freqs = None
            self.alt_rope_offset = 0


    def current_new_pages_required(self):
        new_pages = 0
        for h in self.all_unique_hashes:
            if h not in self.pagetable.referenced_pages:
                new_pages += 1
        for s in self.sequences:
            new_pages += s.new_unique_pages
        return new_pages


    def prefill(self, results: list):
        if self.time_first_prefill is None:
            self.time_first_prefill = time.time()

        progress = 0
        for seq in self.sequences:
            if seq.prefill_complete:
                continue

            cp_pos = next(iter(self.recurrent_state.values())).position if self.recurrent_state is not None else -1

            prefill_start = seq.kv_position
            prefill_end = seq.kv_position + self.generator.max_chunk_size
            prefill_end = (prefill_end // PAGE_SIZE) * PAGE_SIZE
            prefill_end = min(prefill_end, len(seq.sequence_ids) - 1)

            p0 = prefill_start // PAGE_SIZE
            p1 = (prefill_end + PAGE_SIZE - 1) // PAGE_SIZE
            for local_idx in range(p0, p1):
                if 0 <= cp_pos <= seq.kv_position:
                    break
                page = seq.allocated_pages[local_idx]
                if page.kv_position == PAGE_SIZE:
                    prefill_start = (local_idx + 1) * PAGE_SIZE
                    seq.kv_position = prefill_start
                    self.cached_pages += 1
                    page.can_revert = False
                else:
                    break

            p0 = prefill_start // PAGE_SIZE
            for local_idx in range(p0, p1):
                if 0 <= cp_pos <= seq.kv_position:
                    break
                page = seq.allocated_pages[local_idx]
                if page.kv_position == PAGE_SIZE:
                    prefill_end = local_idx * PAGE_SIZE
                    break

            if prefill_end <= prefill_start:
                continue

            assert prefill_start % 256 == 0
            prefill_ids = seq.sequence_ids.torch_slice(prefill_start, prefill_end)

            # Special case for partial last page, check if there's a page anywhere in the cache that
            # partially matches, then copy keys/values from there. Skip this step for recurrent models
            # since the recurrent checkpoint will always be on a page boundary
            p0 = prefill_start // PAGE_SIZE
            p1 = prefill_end // PAGE_SIZE
            if prefill_start == p0 * PAGE_SIZE and self.generator.recurrent_cache is None:
                prev_hash = None if p0 == 0 else seq.allocated_pages[p0 - 1].phash
                best_match = 0
                best_match_page = None
                for page in self.pagetable.all_pages:
                    if page.prev_hash != prev_hash or id(page) == id(seq.allocated_pages[p0]):
                        continue
                    match = ext.count_match_tensor(page.sequence, prefill_ids, page.kv_position)
                    if match > best_match:
                        best_match = match
                        best_match_page = page
                if best_match_page and best_match > 1:
                    page = seq.allocated_pages[p0]
                    for c in [self.generator.cache] if not self.generator.draft_model else \
                            [self.generator.cache, self.generator.draft_cache]:
                        c.copy_page(
                            c,
                            best_match_page.page_index,
                            page.page_index,
                            best_match,
                        )
                    page.prev_hash = best_match_page.prev_hash
                    page.sequence[:, :best_match].copy_(prefill_ids[:, :best_match])
                    prefill_ids = prefill_ids[:, best_match:]
                    prefill_start += best_match
                    seq.kv_position += best_match
                    page.kv_position = best_match
                    page.can_revert = False
                    self.cached_tokens += best_match
                    progress += best_match

            # Inference

            if prefill_end > prefill_start:

                if self.generator.draft_model:
                    self.generator.draft_model.prefill(
                        input_ids = prefill_ids,
                        params = {
                            "attn_mode": "flash_attn",
                            "block_table": seq.block_index_tensor,
                            "cache": self.generator.draft_cache,
                            "cache_seqlens": torch.tensor([prefill_start], dtype = torch.int32)
                        }
                    )

                self.generator.model.prefill(
                    input_ids = prefill_ids,
                    params = {
                        "attn_mode": "flash_attn",
                        "block_table": seq.block_index_tensor,
                        "cache": self.generator.cache,
                        "cache_seqlens": torch.tensor([prefill_start], dtype = torch.int32),
                        "recurrent_states": self.recurrent_state,
                        "indexed_embeddings": self.embeddings,
                        "inv_freq": self.alt_rope_freqs,
                    }
                )

                seq.kv_position = prefill_end

                p2 = min(p1 + 1, len(seq.allocated_pages))
                for local_idx in range(p0, p2):
                    page = seq.allocated_pages[local_idx]
                    page.kv_position = min(max(prefill_end - local_idx * PAGE_SIZE, 0), PAGE_SIZE)
                    if local_idx == 0:
                        page.prev_hash = None
                    else:
                        page.prev_hash = seq.allocated_pages[local_idx - 1].phash
                    pf_a = max(local_idx * PAGE_SIZE, prefill_start)
                    pf_b = min(local_idx * PAGE_SIZE + PAGE_SIZE, prefill_end)
                    pfp_a = pf_a - local_idx * PAGE_SIZE
                    pfp_b = pf_b - local_idx * PAGE_SIZE
                    page.sequence[:, pfp_a:pfp_b].copy_(seq.sequence_ids.torch_slice(pf_a, pf_b))
                    page.can_revert = False

                progress += prefill_end - prefill_start
                if progress >= len(seq.sequence_ids) - 1:
                    seq.prefill_complete = True

        if progress:
            r = {
                "job": self,
                "stage": "prefill",
                "eos": False,
                "curr_progress": sum(seq.kv_position for seq in self.sequences),
                "max_progress": sum(len(seq.sequence_ids) - 1 for seq in self.sequences),
                "serial": self.serial_number,
            }
            if self.identifier is not None:
                r.update({"identifier": self.identifier})
            results.append(r)


    def allocate_pages(self):
        for seq in self.sequences:
            allocated_pages, cached_pages, non_sequential_pages, recurrent_state = \
                seq.allocate_pages(self.pagetable, self.generator.recurrent_cache)

            self.recurrent_state = None
            if self.generator.recurrent_cache is not None:
                if recurrent_state is None:
                    self.recurrent_state = self.generator.recurrent_cache.get_empty_state()
                else:
                    self.recurrent_state = self.generator.recurrent_cache.get_unstashed(recurrent_state)

            # Metrics
            self.cached_pages += cached_pages
            self.total_pages += allocated_pages
            self.non_sequential_pages += non_sequential_pages


    def deallocate_pages(self):
        for seq in self.sequences:
            if seq.allocated_pages is not None:
                self.pagetable.deallocate_pages(seq.allocated_pages)
                seq.allocated_pages = []


    def prepare_sampling_past_ids(self):
        if not self.sampler.reqs_past_ids:
            return
        if self.pinned_ids is None:
            max_ids = max(len(seq.sequence_ids) for seq in self.sequences) + self.max_new_tokens + 8
            self.pinned_ids = torch.empty((1, max_ids), dtype = torch.long, pin_memory = True)
        self.current_pinned_ids = self.pinned_ids[:, :len(self.sequences[0].sequence_ids)]
        self.current_pinned_ids.copy_(self.sequences[0].sequence_ids.torch())


    def activate(self):
        self.logits_device = self.generator.model.output_device
        if not self.is_requeued:
            for f in self.filters:
                f.attach(self)
                f.reset()
                f.is_active = f.trigger_token is None

    def maybe_stash_recurrent(self, cache, interval):
        seq = self.sequences[0]
        if seq.kv_position % interval == 0:
            last_page = (seq.kv_position - 1) // PAGE_SIZE
            page = seq.allocated_pages[last_page]
            assert page.kv_position == PAGE_SIZE
            cache.stash(page.phash, self.recurrent_state)
