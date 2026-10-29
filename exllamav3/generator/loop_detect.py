"""
Streaming detection of generation loops: the model repeating the same span of text, exactly or nearly, with nothing
else happening in between, for as long as it is allowed to run.

Approach
--------
A repetition is found from recurring k-grams (k consecutive tokens) rather than by testing every possible period.
Each token's k-gram is hashed (rolling hash) and looked up in a table of recent k-gram positions; the distance to an
earlier occurrence is a candidate period p. A *run* tracks one candidate period: token t matches the run when the same
k-gram also ended at t - m*p for some m in 1..max_mult, with p allowed to drift by tol*p (insertions and deletions in
the copies). Matching against up to max_mult copies back lets an edit in the previous copy be bridged by the copy
before it. A run ends after `gap` consecutive unmatched tokens, so new text between repetitions (commentary, other
work) of more than `gap` tokens breaks it.

A run is reported as a loop when all of these hold:
  - it spans min_reps periods and min_len tokens (exact repetition: exact_reps periods and exact_len tokens)
  - at least min_frac of its tokens matched, and its recent match rate (exponential average over about one period)
    is at least min_recent, so a run that has turned into mostly new content stops qualifying
  - it is not transcribing distinct prompt items (see prime()): when the differences between copies are prompt
    content that has not been generated before, the copies are e.g. answer options being compared, not a loop. The
    exemption lapses two periods after the last such token, so cycling through the same items again is a loop
  - over a tiny vocabulary (fewer than DIVERSE_MIN distinct tokens recently: DNA written one base per token, mostly
    empty boards), k-grams recur at short distances by structure alone; only exact repetition, or near-exact
    repetition sustained for lowdiv_len tokens, counts there

What it does not distinguish: a long list of nearly identical items whose only difference is an index (boilerplate
code, JSON objects sharing a long field) has the same form as a runaway enumeration. The defaults require eight
repetitions spanning at least 8192 tokens for anything short of exact repetition, which leaves room for such lists
at the cost of detecting real loops later.

Cost
----
O(1) per token: one hash update, a table lookup and at most max_runs * max_mult * ways_pos comparisons. All storage
is preallocated typed arrays sized from max_period (about 3.5 MiB at the default 32768); the detector creates no
per-token objects the garbage collector tracks.
"""

from array import array

import torch

_MOD = (1 << 61) - 1                # rolling hash modulus (Mersenne prime)
_BASE = 1_000_003                   # rolling hash base

EXACT_RATE = 0.995                  # recent match rate counted as exact repetition
NEAR_EXACT_RATE = 0.97              # ... as near-exact (tiny-vocabulary text)
DIVERSE_WINDOW = 256                # tokens examined for vocabulary diversity (at most)
DIVERSE_MIN = 12                    # distinct tokens for text to count as varied
PROMPT_BLOOM_BITS = 16              # Bloom filter bits per prompt token
PROMPT_SOURCED_MIN = 4              # prompt-sourced differences before a run counts as transcription


class LoopDetector:

    def __init__(
        self,
        k: int = 6,
        max_period: int = 32768,
        min_reps: int = 8,
        min_len: int = 8192,
        exact_reps: int = 4,
        exact_len: int = 1024,
        tol: float = 0.05,
        gap: int = 24,
        max_mult: int = 3,
        min_frac: float = 0.5,
        min_recent: float = 0.75,
        lowdiv_len: int = 2400,
        prompt_frac: float = 0.25,
        max_runs: int = 8,
        ways_pos: int = 4,
    ):
        """
        :param k:
            k-gram length in tokens. Matches are judged on k-grams, so an edit makes up to k tokens unmatched
        :param max_period:
            Longest repeating unit, in tokens, that can be detected
        :param min_reps:
            Repetitions (periods spanned) before an approximate loop is reported
        :param min_len:
            Minimum length in tokens of an approximate loop before it is reported
        :param exact_reps:
            Repetitions before an exact loop (recent match rate >= EXACT_RATE) is reported
        :param exact_len:
            Minimum length in tokens of an exact loop before it is reported
        :param tol:
            Allowed drift of the period between copies, as a fraction of the period
        :param gap:
            Consecutive unmatched tokens that end a run (new text between repetitions)
        :param max_mult:
            Number of copies back a k-gram may match
        :param min_frac:
            Minimum fraction of matched tokens over the run
        :param min_recent:
            Minimum recent match rate (exponential average over about one period)
        :param lowdiv_len:
            Length a near-exact loop over a tiny vocabulary must reach before it is reported
        :param prompt_frac:
            Fraction of a run's unmatched tokens that must be new prompt content for it to count as transcription
        :param max_runs:
            Candidate periods tracked at once
        :param ways_pos:
            Recent positions remembered per k-gram
        """
        self.k = k
        self.max_period = max_period
        self.min_reps = min_reps
        self.min_len = min_len
        self.exact_reps = exact_reps
        self.exact_len = exact_len
        self.tol = tol
        self.gap = gap
        self.max_mult = max_mult
        self.min_frac = min_frac
        self.min_recent = min_recent
        self.lowdiv_len = lowdiv_len
        self.prompt_frac = prompt_frac
        self.max_runs = max_runs
        self.ways_pos = ways_pos
        self._kwargs = dict(
            k = k, max_period = max_period, min_reps = min_reps, min_len = min_len, exact_reps = exact_reps,
            exact_len = exact_len, tol = tol, gap = gap, max_mult = max_mult, min_frac = min_frac,
            min_recent = min_recent, lowdiv_len = lowdiv_len, prompt_frac = prompt_frac, max_runs = max_runs,
            ways_pos = ways_pos,
        )

        # k-gram table: 2-way buckets of (hash, newest position, ring of the last ways_pos positions). A missing
        # k-gram evicts the bucket entry seen least recently, so stale history is overwritten rather than pruned
        slots = 1
        while slots < 2 * max_period:
            slots <<= 1
        self._mask = slots - 1
        self._key = array("q", [0]) * slots
        self._newest = array("q", [-1]) * slots
        self._pos = array("q", [-1]) * (slots * ways_pos)
        self._head = array("b", [0]) * slots

        # Rolling hash state, recent k-gram hashes (exact-offset matching) and recent tokens (vocabulary check)
        self._ring = array("q", [0]) * k
        self._bk = pow(_BASE, k, _MOD)
        self._h = 0
        self._hashes = array("q", [0]) * (max_period * max_mult + 1)
        self._tokens = array("q", [-1]) * DIVERSE_WINDOW
        self._t = -1

        # Runs, as parallel fixed-size lists
        self._on = [False] * max_runs           # active
        self._start = [0] * max_runs            # position of the first copy
        self._p = [0] * max_runs                # period
        self._last = [0] * max_runs             # last matched position
        self._matched = [0] * max_runs          # matched tokens
        self._unmatched = [0] * max_runs        # unmatched tokens (within the gap allowance)
        self._sourced = [0] * max_runs          # ... whose k-gram is prompt content not generated before
        self._sourced_last = [0] * max_runs     # position of the last such token
        self._exempt = [False] * max_runs       # run has shown prompt-sourced differences
        self._recent = [1.0] * max_runs         # recent match rate

        self._bloom = None
        self._bloom_mask = 0
        self.detected_at = None
        self.period = None

    @property
    def detected(self) -> bool:
        """True once a loop has been reported (detection latches until reset())"""
        return self.detected_at is not None

    def reset(self):
        """Clear all state, keeping the configuration (the prompt must be primed again)"""
        self.__init__(**self._kwargs)

    def _kgram_hashes(self, ids):
        """Rolling hashes of every k-gram in a token list, computed as in feed()"""
        k, h, ring = self.k, 0, [0] * self.k
        for t, token in enumerate(ids):
            i = t % k
            out = ring[i]
            ring[i] = token + 1
            h = ((h * _BASE - out * self._bk + token + 1) % _MOD) if t >= k else ((h * _BASE + token + 1) % _MOD)
            if t >= k - 1:
                yield h or 1

    def prime(self, prompt_ids: list[int] | torch.Tensor):
        """
        Index the prompt's k-grams (Bloom filter, three probes). Call once before feeding generated tokens. Without a
        prompt, the transcription exemption is inactive and everything else works as usual.

        :param prompt_ids:
            Token IDs of the context the generation continues from
        """
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.flatten().tolist()
        n = 1 << max(10, (max(len(prompt_ids), 1) * PROMPT_BLOOM_BITS - 1).bit_length())
        self._bloom_mask = n - 1
        self._bloom = bytearray(n >> 3)
        for h in self._kgram_hashes(prompt_ids):
            for x in (h, h >> 21, h >> 42):
                b = x & self._bloom_mask
                self._bloom[b >> 3] |= 1 << (b & 7)

    def _in_prompt(self, h: int) -> bool:
        if self._bloom is None:
            return False
        for x in (h, h >> 21, h >> 42):
            b = x & self._bloom_mask
            if not (self._bloom[b >> 3] >> (b & 7)) & 1:
                return False
        return True

    def _diverse(self, t: int, p: int) -> bool:
        """Whether the latest max(p, 64) tokens (at most DIVERSE_WINDOW) contain DIVERSE_MIN distinct tokens"""
        n = min(max(p, 64), DIVERSE_WINDOW, t + 1)
        if n < 32:
            return True
        return len({self._tokens[(t - j) % DIVERSE_WINDOW] for j in range(n)}) >= DIVERSE_MIN

    def feed(self, token: int) -> bool:
        """
        Observe one generated token.

        :return:
            True if a loop has been detected (at this token or earlier)
        """
        self._t += 1
        t, k = self._t, self.k

        # Rolling hash of the k-gram ending at t
        i = t % k
        out = self._ring[i]
        self._ring[i] = token + 1
        self._h = ((self._h * _BASE - out * self._bk + token + 1) % _MOD) if t >= k \
            else ((self._h * _BASE + token + 1) % _MOD)
        self._tokens[t % DIVERSE_WINDOW] = token
        if t < k - 1:
            return self.detected_at is not None
        h = self._h or 1

        # Earlier occurrences of this k-gram within max_period, as distances
        b = h & self._mask & ~1
        e = -1
        for w in (b, b + 1):
            if self._key[w] == h and self._newest[w] >= t - self.max_period:
                e = w
                break
        npos = self.ways_pos
        dists = []
        if e >= 0:
            base = e * npos
            for j in range(npos):
                x = self._pos[base + j]
                if x >= 0 and t - x <= self.max_period:
                    dists.append(t - x)
        else:
            e = b if self._newest[b] <= self._newest[b + 1] else b + 1
            self._key[e] = h
            base = e * npos
            for j in range(npos):
                self._pos[base + j] = -1
            self._head[e] = 0

        # A k-gram never generated before that occurs in the prompt
        new_prompt_content = not dists and self._in_prompt(h)

        # Update runs
        nh = len(self._hashes)
        for r in range(self.max_runs):
            if not self._on[r]:
                continue
            p = self._p[r]
            tl = int(p * self.tol)

            # Same k-gram m periods back, within the period tolerance (stored positions)
            hit_m = hit_q = 0
            for m in range(1, self.max_mult + 1):
                lo, hi = m * (p - tl), m * (p + tl)
                for q in dists:
                    if lo <= q <= hi:
                        hit_m, hit_q = m, q
                        break
                if hit_m:
                    break

            # The few stored positions can all lie inside the current copy when the repeating unit itself repeats
            # phrases (similar lines of code); compare the k-gram hash at exactly m periods back
            if not hit_m:
                for m in range(1, self.max_mult + 1):
                    s = t - m * p
                    if s >= 0 and t - s < nh and self._hashes[s % nh] == h:
                        hit_m, hit_q = m, m * p
                        break

            tau = max(32, p)
            if hit_m:
                if hit_m == 1 and hit_q != p:
                    self._p[r] = hit_q          # follow insertions/deletions
                self._last[r] = t
                self._matched[r] += 1
                self._recent[r] += (1.0 - self._recent[r]) / tau
            elif t - self._last[r] > self.gap:
                self._on[r] = False
                continue
            else:
                self._recent[r] -= self._recent[r] / tau
                self._unmatched[r] += 1
                if new_prompt_content:
                    self._sourced[r] += 1
                    self._sourced_last[r] = t

            if self.detected_at is None and self._is_loop(r, t):
                self.detected_at = t
                self.period = self._p[r]

        # Start runs for distances no live run explains
        for q in dists:
            explained = False
            for r in range(self.max_runs):
                if self._on[r]:
                    p = self._p[r]
                    tl = int(p * self.tol)
                    for m in range(1, self.max_mult + 1):
                        if abs(q - m * p) <= m * tl:
                            explained = True
                            break
                    if explained:
                        break
            if explained:
                continue
            free = -1
            for r in range(self.max_runs):
                if not self._on[r]:
                    free = r
                    break
            if free < 0:
                # Replace a live run that has not matched beyond its first token, otherwise drop the candidate
                free = min(range(self.max_runs), key = self._matched.__getitem__)
                if self._matched[free] > 1:
                    continue
            self._on[free] = True
            self._start[free] = t - q
            self._p[free] = q
            self._last[free] = t
            self._matched[free] = 1
            self._unmatched[free] = self._sourced[free] = self._sourced_last[free] = 0
            self._exempt[free] = False
            self._recent[free] = 1.0

        # Record this k-gram
        self._hashes[t % nh] = h
        j = self._head[e]
        self._pos[e * npos + j] = t
        self._head[e] = (j + 1) % npos
        self._newest[e] = t
        return self.detected_at is not None

    def _is_loop(self, r: int, t: int) -> bool:
        """Whether run r, just updated at position t, qualifies as a loop"""
        p = self._p[r]
        span = t - self._start[r]
        recent = self._recent[r]

        # Transcription of distinct prompt items (sticky once established, lapses two periods after the last new
        # prompt content)
        if self._sourced[r] >= PROMPT_SOURCED_MIN and self._sourced[r] >= self.prompt_frac * self._unmatched[r]:
            self._exempt[r] = True
        if self._exempt[r] and t - self._sourced_last[r] <= 2 * p:
            return False

        exact = recent >= EXACT_RATE
        reps, length = (self.exact_reps, self.exact_len) if exact else (self.min_reps, self.min_len)
        if span < reps * p or span < length:
            return False
        if self._matched[r] < self.min_frac * max(span - p, 1) or recent < self.min_recent:
            return False

        # Tiny vocabulary: structure alone makes k-grams recur, so require (near-)exact repetition
        if not self._diverse(t, p):
            return exact or (recent >= NEAR_EXACT_RATE and span >= self.lowdiv_len)
        return True

    def feed_many(self, tokens: list[int] | torch.Tensor) -> list[int] | None:
        """
        Observe several generated tokens.

        :return:
            [index of the token at which the loop was detected, counted over all tokens fed], or None
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.flatten().tolist()
        was = self.detected_at
        for token in tokens:
            if self.feed(token):
                break
        if self.detected_at is None:
            return None
        return [self.detected_at] if was is None else [was]
