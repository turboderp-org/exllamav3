from collections import OrderedDict

class RecurrentCache(OrderedDict):
    def __init__(
        self,
        model,
        max_size: int = 4 * 1024**3,
    ):
        super().__init__()
        self.max_size = max_size
        self.current_size = 0
        self.model = model


    def get_stashed(self, key, default = None):
        """
        Fetch state from cache and move it to the end of the queue
        """
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default


    def put(self, key, state):
        """
        Add state to cache
        """
        if key in self:
            self.move_to_end(key)
        else:
            stashed_state = state.stash()
            state_size = stashed_state["checkpoint_size"]
            while self.update_total_size() + state_size > self.max_size:
                assert self.current_size >= 0, "Not enough space in cache for single state"
                self.popitem(last = False)
            self[key] = stashed_state
            self.update_total_size()


    def update_total_size(self):
        seen = set()
        total = 0
        for v in self.values():
            if id(v) in seen:
                continue
            seen.add(id(v))
            total += v["checkpoint_size"]
        self.current_size = total
        return total
