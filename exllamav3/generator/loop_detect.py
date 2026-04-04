import heapq
import torch

class LoopDetector:
    """
    Flat-latency streaming loop detector. Observes streaming tokens and detects repeating sequences.
    Triggering condition is when the _entire_ observed window contains only repeating sequences.
    """

    def __init__(self, window_size: int = 1000, max_period: int = None):
        """
        :param window_size:
            Window size, in tokens

        :param max_period:
            Longest repetition to scan for, capped at window_size // 2 (no longer period is possible)
            Default is window_size // 3
        """

        self.W = window_size
        self.max_period = min(max_period or window_size // 3, window_size // 2)

        # Circular buffer
        self._buf = [None] * self.W
        self._total = 0  # Total tokens received (monotonic)

        # Per-detector state
        self._streak = [0] * (self.max_period + 1)  # Consecutive matches

        # Scheduling heap: (wake_time, period, generation)
        self._heap: list[tuple[int, int, int]] = []
        self._gen = [0] * (self.max_period + 1)

        # Schedule initial wake-up as soon as buffer is full
        for p in range(1, self.max_period + 1):
            wake = self.W
            heapq.heappush(self._heap, (wake, p, 0))

        self._detected_period: int | None = None

    @property
    def detected(self) -> bool:
        return self._detected_period is not None

    @property
    def period(self) -> int | None:
        return self._detected_period

    def _schedule(self, p: int, wake_time: int):
        # Schedule detector p to wake at wake_time, invalidating prior entry
        self._gen[p] += 1
        heapq.heappush(self._heap, (wake_time, p, self._gen[p]))

    def _backlog_scan(self, p: int) -> int:
        """
        Scan backward through the buffer to count how many consecutive positions (starting from the newest)
        satisfy s[i] == s[i - p]. Returns the streak count (0 to W - p).
        Worst case O(W), typically O(1).
        """
        t = self._total
        streak = 0
        max_check = self.W - p

        for k in range(max_check):
            idx_a = (t - 1 - k) % self.W
            idx_b = (t - 1 - k - p) % self.W
            if self._buf[idx_a] == self._buf[idx_b]:
                streak += 1
            else:
                break

        return streak

    def feed(self, token: str) -> bool:
        """
        Feed one token. Returns True if a loop is detected
        """
        pos = self._total % self.W
        self._buf[pos] = token
        self._total += 1

        if self._total < self.W:
            return False

        t = self._total
        detected = False

        while self._heap and self._heap[0][0] <= t:
            wake_time, p, gen = heapq.heappop(self._heap)

            if gen != self._gen[p]:
                continue  # Stale entry (might never happen?)

            if self._streak[p] > 0:
                # Active detector: check the newest token
                if self._buf[(t - 1) % self.W] == self._buf[(t - 1 - p) % self.W]:
                    self._streak[p] += 1
                    if self._streak[p] >= self.W - p:
                        self._detected_period = p
                        detected = True
                    self._schedule(p, t + 1)
                else:
                    # Streak broken: sleep
                    self._streak[p] = 0
                    if self._detected_period == p:
                        self._detected_period = None
                    self._schedule(p, t + self.W)  # The mismatch at buf[(t-1)%W] leaves after W more tokens
            else:
                # Waking from sleep: scan backlog
                streak = self._backlog_scan(p)
                self._streak[p] = streak

                if streak >= self.W - p:
                    self._detected_period = p
                    detected = True
                    self._schedule(p, t + 1)
                elif streak > 0:
                    self._schedule(p, t + 1)  # Partial streak: stay active to extend it
                else:
                    self._schedule(p, t + self.W)  # Immediate mismatch: back to sleep

        return detected or self._detected_period is not None

    def feed_many(self, tokens: list[int] | torch.Tensor) -> int | None:
        # Returns global index of (last token of) first loop
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.flatten().tolist()
        det = []
        for i, token in enumerate(tokens):
            if self.feed(token):
                det.append(self._total - 1)
        return det or None

    def reset(self):
        self.__init__(window_size = self.W, max_period = self.max_period)

"""
detector = LoopDetector(100, 50)

seqs = [
    range(25),
    [0, 2, 3, 4] * 24,
    [0, 2, 3, 4, 0],
    [0],
    [1, 2, 3, 4, 5],
    [1, 2, 3] * 7,
]
for seq in seqs:
    loops = detector.feed_many(seq)
    if loops:
        print("->", loops)
"""