"""
Content-addressed conversation memory for MemT models (doc/memt_serving_plan.md, section 3).

The server in front of the generator (TabbyAPI, its web UI) is stateless: every request resends the full
message list as one rendered prompt, with no session id, and may rewrite history (edit, regenerate, a
different chat). A MemT commit is one-way, so memory cannot follow a session; it follows the content.

For each enqueued job this module
  1. splits the prompt at ``<|im_start|>`` into turns, grouped into commit units (one user turn with the
     replies/tool results that follow it) behind an optional system turn,
  2. keys every unit boundary by a blake2b chain over the exact token ids of the units before it,
  3. finds the longest keyed prefix with a stored snapshot, commits the uncommitted units one by one
     (storing a snapshot at each boundary), leaving the last ``context_turns`` completed units raw,
  4. rebases the job onto system turn + raw recent units + the live unit, reading the matched snapshot.

Keys are hashes of the client's own rendering, so template differences cannot break matching; an edited
history matches a shorter prefix and memory rolls back to that point. Snapshots live in RAM (fp32, 40 MiB
each for MemT-1.0-Nano) in an LRU capped in bytes; the zero state (no committed units) is implicit.
"""

from __future__ import annotations
import hashlib
import time
from collections import OrderedDict
import torch
from ..model.model import Model
from ..tokenizer.tokenizer import Tokenizer


def _chain(prev: bytes, ids: list[int]) -> bytes:
    h = hashlib.blake2b(digest_size = 16)
    h.update(prev)
    h.update(torch.tensor(ids, dtype = torch.int64).numpy().tobytes())
    return h.digest()


class MemTConversationMemory:

    def __init__(
        self,
        model: Model,
        tokenizer: Tokenizer,
        context_turns: int = 0,
        cache_size: int = 2 * 1024**3,
        verbose: bool = False,
    ):
        assert model.caps.get("memt"), "MemTConversationMemory needs a MemT model"
        self.model = model
        self.context_turns = max(0, context_turns)
        self.cache_size = cache_size
        self.verbose = verbose

        # Qwen chat format markers: <|im_start|>{role}\n ... <|im_end|>\n
        enc = lambda s: tokenizer.encode(s, add_bos = False, encode_special_tokens = True)[0].tolist()
        start = enc("<|im_start|>user\n")
        assert len(start) == 3, f"unexpected tokenization of the turn header: {start}"
        self.im_start = start[0]
        self.role_user = start[1]
        self.role_system = enc("<|im_start|>system\n")[1]
        self.role_assistant = enc("<|im_start|>assistant\n")[1]

        self.snapshots: OrderedDict[bytes, dict[int, torch.Tensor]] = OrderedDict()
        self.snapshot_bytes = sum(s.numel() * 4 for s in model.memt_zero_state().values())
        self.last = None   # stats of the last prepare(), for tests and logging

    # --- prompt structure ---------------------------------------------------

    def split_units(self, ids: list[int]) -> tuple[list[int], list[list[int]], list[int]]:
        """
        (system, units, live): the system turn (possibly empty), the completed commit units, and the live unit
        (the final user turn plus the generation prompt, or whatever follows the last user turn). A unit
        starts at a user turn and runs up to the next one.
        """
        starts = [i for i, t in enumerate(ids) if t == self.im_start]
        if not starts:
            return [], [], ids
        # Tokens before the first header (none for Qwen) stay with the first segment
        segs = [ids[a:b] for a, b in zip([0] + starts[1:], starts[1:] + [len(ids)])]
        system = []
        if len(segs[0]) > 1 and segs[0][1] == self.role_system:
            system = segs.pop(0)
        units = []
        for seg in segs:
            is_user = len(seg) > 1 and seg[1] == self.role_user
            if is_user or not units:
                units.append(list(seg))
            else:
                units[-1] += seg
        live = units.pop()
        return system, units, live

    @staticmethod
    def keys_of(system: list[int], units: list[list[int]]) -> list[bytes]:
        """keys[k] identifies the memory state after committing units[:k] behind this system turn"""
        keys = [_chain(b"", system)]
        for u in units:
            keys.append(_chain(keys[-1], u))
        return keys

    # --- snapshot store -----------------------------------------------------

    def _get(self, key: bytes):
        s = self.snapshots.get(key)
        if s is not None:
            self.snapshots.move_to_end(key)
        return s

    def _put(self, key: bytes, state: dict[int, torch.Tensor]):
        self.snapshots[key] = {i: s.detach().to("cpu", torch.float32) for i, s in state.items()}
        self.snapshots.move_to_end(key)
        while len(self.snapshots) * self.snapshot_bytes > self.cache_size and len(self.snapshots) > 1:
            self.snapshots.popitem(last = False)

    def clear(self):
        self.snapshots.clear()

    # --- the hook -----------------------------------------------------------

    def prepare(self, job) -> None:
        """
        Rebase a freshly created job (before prepare_for_queue) onto its short prompt and memory state,
        committing turns as needed. Runs model forwards; the caller guarantees no other forward is in flight.
        """
        ids = job.sequences[0].input_ids.torch()[0].tolist()
        system, units, live = self.split_units(ids)
        keys = self.keys_of(system, units)
        n = len(units)
        target = max(0, n - self.context_turns)   # units[:target] in memory, units[target:] raw in context

        # Longest keyed prefix with a snapshot, at or before target
        matched = 0
        state = None
        for k in range(target, 0, -1):
            state = self._get(keys[k])
            if state is not None:
                matched = k
                break
        if state is None:
            state = self.model.memt_zero_state()

        t0 = time.time()
        committed_tokens = 0
        for k in range(matched, target):
            unit = torch.tensor([units[k]], dtype = torch.long)
            state = self.model.memt_commit(unit, state)
            committed_tokens += len(units[k])
            self._put(keys[k + 1], state)
        commit_time = time.time() - t0

        prompt = list(system)
        for u in units[target:]:
            prompt += u
        prompt += live
        short = torch.tensor([prompt], dtype = torch.long)
        job.memt_rebase(short, state if target > 0 else None, keys[target])

        self.last = {
            "prompt_tokens": len(ids), "units": n, "matched": matched, "in_memory": target,
            "committed": target - matched, "committed_tokens": committed_tokens, "commit_time": commit_time,
            "short_prompt_tokens": len(prompt), "snapshots": len(self.snapshots),
        }
        if self.verbose:
            print(f" -- memt: {len(ids)} prompt tokens, {n} turns: {matched} matched, "
                  f"{target - matched} committed ({committed_tokens} tokens, {commit_time:.2f}s), "
                  f"{n - target} raw; prompt now {len(prompt)} tokens; {len(self.snapshots)} snapshots", flush = True)
