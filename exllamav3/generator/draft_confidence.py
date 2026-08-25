from __future__ import annotations
import math


class DraftConfidenceCalibrator:
    """
    Online mapping from a drafter's per-position confidence score (argmax logit) to an observed
    acceptance probability, used to truncate draft blocks at the first position whose estimated
    acceptance falls below a target confidence.

    Scores are collected into fixed-width bins of decayed (tested, accepted) counts. Labels come
    only from positions the verifier actually tested: accepted positions, the first mismatch, and
    the bonus token sampled right after a fully accepted (possibly truncated) window, which
    labels the first position below the current threshold.

    The threshold is the lower edge of the lowest bin such that it and every populated bin above
    it show an acceptance rate >= confidence. Sparse bins (effective count < min_count) are
    skipped rather than trusted. Until burn_in labels have been seen, the threshold is -inf
    (full windows).
    """

    def __init__(
        self,
        confidence: float,
        bin_width: float = 1.0,
        decay: float = 0.995,
        min_count: float = 8.0,
        burn_in: float = 64.0,
    ):
        assert 0.0 < confidence < 1.0, "draft_confidence must be in (0, 1)"
        self.confidence = confidence
        self.bin_width = bin_width
        self.decay = decay
        self.min_count = min_count
        self.burn_in = burn_in

        self.bins = {}  # bin index -> [tested, accepted], exponentially decayed
        self.total = 0.0
        self.cached_threshold = None


    def add_label(self, score: float, accepted: bool):
        idx = math.floor(score / self.bin_width)
        b = self.bins.get(idx)
        if b is None:
            b = self.bins[idx] = [0.0, 0.0]
        b[0] += 1.0
        if accepted:
            b[1] += 1.0
        self.total += 1.0
        self.cached_threshold = None


    def decay_step(self):
        """
        Age the statistics; call once per verification round so the mapping tracks drift in
        output style (prose vs code etc.) over a few hundred rounds.
        """
        for b in self.bins.values():
            b[0] *= self.decay
            b[1] *= self.decay
        self.total *= self.decay
        self.cached_threshold = None

    def threshold(self) -> float:
        """
        Confidence score below which draft positions should be cut. -inf while insufficient
        data has been collected; +inf if even the most confident bins fall short of the target
        (recovery then relies on the bonus-token labels of empty windows).
        """
        if self.cached_threshold is not None:
            return self.cached_threshold
        if self.total < self.burn_in:
            return -math.inf  # not cached; keeps returning -inf only until burn-in completes
        thr = math.inf
        any_populated = False
        for idx in sorted(self.bins.keys(), reverse = True):
            tested, accepted = self.bins[idx]
            if tested < self.min_count:
                continue
            any_populated = True
            if accepted / tested >= self.confidence:
                thr = idx * self.bin_width
            else:
                break
        # No bin has accumulated enough labels to judge: still learning, keep full windows so
        # labels keep flowing (cutting everything would starve the statistics instead). +inf is
        # reserved for the case where populated top bins genuinely fail the target
        if not any_populated:
            thr = -math.inf
        self.cached_threshold = thr
        return thr

    def estimate(self, score: float) -> float:
        """
        Estimated conditional acceptance probability for a drafted position with the given
        confidence score, from the nearest populated bin at or below it (falling back to the
        nearest above). Optimistic 1.0 while no statistics are available, so sequential
        drafting keeps producing full windows (and labels) during the learning phase.
        """
        if self.total < self.burn_in or not self.bins:
            return 1.0
        idx = math.floor(score / self.bin_width)
        populated = sorted(k for k, v in self.bins.items() if v[0] >= self.min_count)
        if not populated:
            return 1.0
        below = [k for k in populated if k <= idx]
        k = below[-1] if below else populated[0]
        tested, accepted = self.bins[k]
        return accepted / tested
