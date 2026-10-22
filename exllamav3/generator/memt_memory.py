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

Commit policy. By default every completed unit is committed (``commit_every`` = 1), minus the last
``context_turns``. Whatever is already in memory for the history stays there (memory is one-way). ``commit_every`` = N commits only once N more units have completed, so the recent turns
ride raw in the context until then; 0 never commits on its own. Two in-band controls override it, and are
stripped from what the model sees (keys are still taken over the client's untouched ids):
  * ``/commit`` on a line of its own in a user message commits everything before that message;
    ``/commit every N`` in a user or system message sets the period from there on. A message that is
    nothing but commands gets a fixed one-line reply without running the model, and that exchange is
    left out of memory and context later.
  * a ``<|im_start|>memt\n...<|im_end|>`` turn carrying the same lines, emitted by the chat template from the
    request's ``template_vars`` (``memt_commit``, ``memt_commit_every``), applies to that request only.

With ``commit_generated`` the server keeps the exact ids it generated for each reply (reasoning and stop token
included, as the release replays them) and commits those instead of the client's re-rendered copy whenever the
client's visible answer still matches what was generated; an edited reply falls back to the client's version.

Keys are hashes of the client's own rendering, so template differences cannot break matching; an edited
history matches a shorter prefix and memory rolls back to that point. Snapshots live in RAM (fp32, 40 MiB
each for MemT-1.0-Nano) in an LRU capped in bytes; the zero state (no committed units) is implicit. With a
``snapshot_dir`` every snapshot is also a file there (release layout), so memory survives a restart.
"""

from __future__ import annotations
import hashlib
import os
import re
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
        commit_every: int = 1,
        commit_generated: bool = False,
        cache_size: int = 2 * 1024**3,
        snapshot_dir: str | None = None,
        disk_size: int = 16 * 1024**3,
        verbose: bool = False,
    ):
        assert model.caps.get("memt"), "MemTConversationMemory needs a MemT model"
        self.model = model
        self.tokenizer = tokenizer
        self.context_turns = max(0, context_turns)
        self.commit_every = max(0, commit_every)
        self.commit_generated = commit_generated
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
        self.newline = start[2]
        self.im_end = enc("<|im_end|>")[0]

        self.snapshots: OrderedDict[bytes, dict[int, torch.Tensor]] = OrderedDict()
        self.disk_hits = 0
        self.disk_size = disk_size
        self.snapshot_dir = None
        if snapshot_dir:
            self.set_snapshot_dir(snapshot_dir)
        # (user-turn hash, visible-answer hash) -> exact unit ids as generated, for commit_generated. Keyed on the
        # turn alone, not the history, so a reply stays findable after an earlier turn is edited
        self.generated: OrderedDict[tuple[bytes, bytes], list[int]] = OrderedDict()
        self.generated_max = 4096
        self.snapshot_bytes = sum(s.numel() * 4 for s in model.memt_zero_state().values())
        self.last = None   # stats of the last prepare(), for tests and logging
        self._warm_chunked_scan()
        self.last_prompt = None   # token ids the last job actually ran on

    def _warm_chunked_scan(self):
        """Compile fla's chunked scan now (random inputs, no model forward) so the first long commit does
        not pay the ~4 s Triton build in the middle of a chat."""
        from ..training import memt as m
        if not (m._memt_chunked_enable and torch.cuda.is_available()):
            return
        try:
            layer = next(iter(self.model.memt_zero_state().values()))
            spec = self.model.memt_layout.spec
            dev = next((blk.attn.memt.device for blk in self.model.modules
                        if getattr(getattr(blk, "attn", None), "memt", None) is not None), None)
            if dev is None:   # tensor parallel: the parent's modules are unloaded; compile on the output device
                dev = getattr(self.model, "tp_output_device", None) or "cuda:0"
            t = m.MEMT_CHUNKED_MIN_TOKENS
            r = lambda: torch.randn(1, t, spec.num_heads, spec.head_dim, device = dev)
            m.memt_update(r(), r(), r().abs(), r().abs(), -r().abs() * 0.01, layer.to(dev), spec)
        except Exception as e:
            print(f" !! memt: chunked-scan warmup failed, long commits will compile on first use: {e}", flush = True)

    # --- in-band commands ---------------------------------------------------

    COMMAND = re.compile(r"^[ \t]*/commit(?:[ \t]+every[ \t]+(\d+))?[ \t]*$", re.IGNORECASE | re.MULTILINE)

    def _parse_commands(self, text: str):
        """(commit_now, every, stripped_text); every is None when not set"""
        now, every = False, None
        for m in self.COMMAND.finditer(text):
            if m.group(1) is None:
                now = True
            else:
                every = int(m.group(1))
        if not now and every is None:
            return False, None, text
        stripped = "\n".join(l for l in text.split("\n") if not self.COMMAND.fullmatch(l)).strip()
        return now, every, stripped

    def _role_of(self, seg: list[int]) -> str:
        if not seg or seg[0] != self.im_start or self.newline not in seg:
            return ""
        return self.tokenizer.decode(torch.tensor(seg[1:seg.index(self.newline)])).strip()

    def _scan_unit(self, unit: list[int], roles: tuple[str, ...]):
        """
        Read /commit commands from the turns of ``roles`` inside one unit. Returns (commit_now, every,
        unit_without_the_command_lines_and_without_memt_turns, bare). Turns without commands keep their exact
        ids; bare is True when a user turn held commands and nothing else.
        """
        starts = [i for i, t in enumerate(unit) if t == self.im_start] or [0]
        segs = [unit[a:b] for a, b in zip(starts, starts[1:] + [len(unit)])]
        if starts[0] > 0:
            segs.insert(0, unit[:starts[0]])
        now, every, out, bare = False, None, [], False
        for seg in segs:
            role = self._role_of(seg)
            if role not in roles or self.newline not in seg:
                out += seg
                continue
            a = seg.index(self.newline) + 1
            b = seg.index(self.im_end) if self.im_end in seg else len(seg)
            text = self.tokenizer.decode(torch.tensor(seg[a:b])) if b > a else ""
            n, e, stripped = self._parse_commands(text)
            if role == "memt":
                now, every = now or n, e if e is not None else every
                continue   # the whole turn is control, the model never sees it
            if not n and e is None:
                out += seg
                continue
            now, every = now or n, e if e is not None else every
            bare = bare or (role == "user" and not stripped)
            body = self.tokenizer.encode(stripped, add_bos = False, encode_special_tokens = True)[0].tolist() if stripped else []
            out += seg[:a] + body + seg[b:]
        return now, every, out, bare

    # --- generated-reply store (commit_generated) ------------------------------

    def _segments(self, unit: list[int]) -> list[list[int]]:
        starts = [i for i, t in enumerate(unit) if t == self.im_start]
        if not starts:
            return [unit]
        segs = [unit[a:b] for a, b in zip(starts, starts[1:] + [len(unit)])]
        if starts[0] > 0:
            segs.insert(0, unit[:starts[0]])
        return segs

    def _user_portion(self, unit: list[int]) -> list[int]:
        """the leading user turn(s) of a unit, before its first reply: what the live prompt and the later
        unit have in common, so the generated-reply key is stable across the two"""
        out = []
        for seg in self._segments(unit):
            if self._role_of(seg) != "user":
                break
            out += seg
        return out

    @staticmethod
    def _visible(text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags = re.DOTALL)
        return text.strip()

    def _reply_visible(self, unit: list[int]) -> str | None:
        """visible text of the unit's first assistant turn as the client rendered it, None if there is none"""
        for seg in self._segments(unit):
            if self._role_of(seg) == "assistant":
                a = seg.index(self.newline) + 1
                b = seg.index(self.im_end) if self.im_end in seg else len(seg)
                return self._visible(self.tokenizer.decode(torch.tensor(seg[a:b]), decode_special_tokens = True)) if b > a else ""
        return None

    @staticmethod
    def _vis_hash(text: str) -> bytes:
        return hashlib.blake2b(text.encode(), digest_size = 16).digest()

    def record(self, job) -> None:
        """
        Called by the generator when a job with a pending entry finishes: store the unit as it was actually
        generated (live user turn as run + generated ids incl. the stop token + newline) under its pending key.
        """
        pending = job.memt_pending
        job.memt_pending = None
        if pending is None:
            return
        gen = job.sequences[0].sequence_ids.torch()[0].tolist()[pending["prompt_len"]:]
        if not gen:
            return
        body = gen[:-1] if gen[-1] in (self.im_end, self.tokenizer.eos_token_id) else gen
        text = self.tokenizer.decode(torch.tensor(body), decode_special_tokens = True)
        if gen[-1] != self.im_end:
            gen = body + [self.im_end]
        unit = pending["live"] + gen + [self.newline]
        self.generated[(pending["key"], self._vis_hash(self._visible(text)))] = unit
        while len(self.generated) > self.generated_max:
            self.generated.popitem(last = False)
        self._save_generated()

    @staticmethod
    def canned_reply(committed: int, in_memory: int, every: int | None) -> str:
        """What a message consisting only of commands gets back, without running the model"""
        parts = []
        if committed:
            parts.append(f"Memorized {committed} exchange{'s' if committed != 1 else ''}.")
        elif every is None:
            parts.append(f"Nothing new to memorize ({in_memory} exchange{'s' if in_memory != 1 else ''} in memory).")
        if every is not None:
            parts.append(f"Memorizing every {every} exchanges from now." if every else "Automatic memorizing is off.")
        return " ".join(parts)

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
        if self.snapshot_dir is not None:
            path = self._snapshot_path(key)
            if os.path.exists(path):
                try:
                    s = self.model.memt_load_state(path)
                except Exception as e:
                    print(f" !! memt: unreadable snapshot {path}: {e}", flush = True)
                    return None
                os.utime(path)   # most recently used, for the disk LRU
                self.snapshots[key] = s
                self._trim_ram()
                self.disk_hits += 1
                return s
        return None

    def _put(self, key: bytes, state: dict[int, torch.Tensor]):
        self.snapshots[key] = {i: s.detach().to("cpu", torch.float32) for i, s in state.items()}
        self.snapshots.move_to_end(key)
        self._trim_ram()
        if self.snapshot_dir is not None:
            path = self._snapshot_path(key)
            if not os.path.exists(path):
                self.model.memt_save_state(self.snapshots[key], path + ".tmp")
                os.replace(path + ".tmp", path)
                self._trim_disk()

    def _trim_ram(self):
        while len(self.snapshots) * self.snapshot_bytes > self.cache_size and len(self.snapshots) > 1:
            self.snapshots.popitem(last = False)

    def clear(self):
        """forget the RAM copies (and the generated-reply store); snapshots on disk stay"""
        self.snapshots.clear()
        self.generated.clear()

    # --- disk persistence ---------------------------------------------------

    def set_snapshot_dir(self, snapshot_dir: str):
        """
        Keep every snapshot as a safetensors file (the release's layout) under snapshot_dir/<model>/, so memory
        survives a restart: RAM misses are filled from disk, the disk is an LRU by mtime capped at disk_size.
        The generated-reply store (commit_generated) is kept there too.
        """
        tag = hashlib.blake2b(self.model.config.directory.encode(), digest_size = 4).hexdigest()
        self.snapshot_dir = os.path.join(snapshot_dir, f"{os.path.basename(self.model.config.directory.rstrip('/'))}-{tag}")
        os.makedirs(self.snapshot_dir, exist_ok = True)
        self._load_generated()

    def _snapshot_path(self, key: bytes) -> str:
        return os.path.join(self.snapshot_dir, key.hex() + ".safetensors")

    def _trim_disk(self):
        files = [(os.path.getmtime(p), p) for p in
                 (os.path.join(self.snapshot_dir, f) for f in os.listdir(self.snapshot_dir)) if p.endswith(".safetensors")]
        total = len(files) * self.snapshot_bytes
        for _, p in sorted(files):
            if total <= self.disk_size:
                break
            os.remove(p)
            total -= self.snapshot_bytes

    def _generated_path(self) -> str:
        return os.path.join(self.snapshot_dir, "generated.pt")

    def _load_generated(self):
        path = self._generated_path()
        if os.path.exists(path):
            try:
                self.generated = OrderedDict(torch.load(path, weights_only = True))
            except Exception as e:
                print(f" !! memt: unreadable {path}: {e}", flush = True)

    def _save_generated(self):
        if self.snapshot_dir is None:
            return
        path = self._generated_path()
        torch.save(dict(self.generated), path + ".tmp")
        os.replace(path + ".tmp", path)

    # --- the hook -----------------------------------------------------------

    def prepare(self, job) -> None:
        """
        Rebase a freshly created job (before prepare_for_queue) onto its short prompt and memory state,
        committing turns as needed. Runs model forwards; the caller guarantees no other forward is in flight.
        """
        ids = job.sequences[0].input_ids.torch()[0].tolist()
        system, units, live = self.split_units(ids)
        keys = self.keys_of(system, units)   # over the client's ids, before any command is stripped
        n = len(units)
        client_units, client_live = units, live

        # Commit policy: env default, then commands in the system turn, past user turns, the live turn,
        # and finally the request's own memt turn (from template_vars), each overriding the one before
        every = self.commit_every
        floor = 0   # units[:floor] must be in memory because a /commit was issued after them
        _, e, system, _ = self._scan_unit(system, ("system",))
        every = e if e is not None else every
        stripped_units, bare = [], []
        for k, u in enumerate(units):
            now, e, u, b = self._scan_unit(u, ("user",))
            stripped_units.append(u)
            bare.append(b)
            every = e if e is not None else every
            if now:
                floor = k
        units = stripped_units
        now, e, live, live_bare = self._scan_unit(live, ("user", "memt"))
        live_every = e
        every = e if e is not None else every
        if now:
            floor = n
        auto = max(0, n - self.context_turns)
        auto = (auto // every) * every if every > 0 else 0

        # Longest keyed prefix with a snapshot. Memory is one-way: what was committed stays committed, so
        # the target is the furthest of the policy, a /commit floor and what is already in memory.
        matched = 0
        state = None
        for k in range(n, 0, -1):
            state = self._get(keys[k])
            if state is not None:
                matched = k
                break
        if state is None:
            state = self.model.memt_zero_state()
        target = max(floor, auto, matched)   # units[:target] in memory, units[target:] raw in context

        t0 = time.time()
        committed_tokens = 0
        skipped = 0
        generated = 0   # units committed from the server's own generated ids rather than the client's copy
        for k in range(matched, target):
            if bare[k]:
                skipped += 1   # a command-only message and its canned reply: not worth remembering
            else:
                unit = units[k]
                if self.commit_generated:
                    vis = self._reply_visible(client_units[k])
                    if vis is not None:
                        g = self.generated.get((_chain(b"", self._user_portion(client_units[k])), self._vis_hash(vis)))
                        if g is not None:
                            unit = g
                            generated += 1
                state = self.model.memt_commit(torch.tensor([unit], dtype = torch.long), state)
                committed_tokens += len(unit)
            self._put(keys[k + 1], state)
        commit_time = time.time() - t0

        prompt = list(system)
        for k in range(target, n):
            if not bare[k]:
                prompt += units[k]
        prompt += live
        short = torch.tensor([prompt], dtype = torch.long)
        job.memt_rebase(short, state if target > 0 else None, keys[target])
        if live_bare:
            job.memt_canned = self.canned_reply(target - matched - skipped, target, live_every)
        elif self.commit_generated:
            job.memt_pending = {"key": _chain(b"", self._user_portion(client_live)), "live": list(live), "prompt_len": len(prompt)}

        self.last_prompt = prompt
        self.last = {
            "prompt_tokens": len(ids), "units": n, "matched": matched, "in_memory": target,
            "commit_every": every, "commit_floor": floor,
            "committed": target - matched, "committed_tokens": committed_tokens, "commit_time": commit_time,
            "skipped": skipped, "canned": job.memt_canned, "generated": generated, "disk_hits": self.disk_hits,
            "short_prompt_tokens": len(prompt), "snapshots": len(self.snapshots),
        }
        if self.verbose:
            print(f" -- memt: {len(ids)} prompt tokens, {n} turns: {matched} matched, "
                  f"{target - matched} committed ({committed_tokens} tokens, {commit_time:.2f}s"
                  f"{f', {generated} as generated' if generated else ''}), "
                  f"{n - target} raw (every {every}, floor {floor}); prompt now {len(prompt)} tokens; "
                  f"{len(self.snapshots)} snapshots", flush = True)
