from collections import OrderedDict
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


class CacheableState(ABC):
    def __init__(self):
        self.checkpoint = None

    @abstractmethod
    def stash(self):
        """
        Return CPU copy of state
        """
        pass

    @abstractmethod
    def unstash(self, device):
        """
        Return GPU copy of state
        """
        pass

    @abstractmethod
    def get_size(self):
        """
        Return size of state in bytes (in system memory after stashing)
        """
        pass


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

        # Get device map needed for unstashing
        self.rl = model.get_recurrent_layers()
        self.device_map = {m.layer_idx: m.device for m in self.rl}

    def get(self, key, default = None):
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
            old_state_size = self.get_size(self[key])
            del self[key]
            self.current_size -= old_state_size
        state_size = self.get_size(state)
        while self.current_size + state_size > self.max_size:
            assert self.current_size > 0, "Not enough space in cache for single state"
            _, oldest_state = self.popitem(last = False)
            oldest_state_size = self.get_size(oldest_state)
            self.current_size -= oldest_state_size
        self[key] = state
        self.current_size += state_size

    def get_size(self, state):
        elems = state if isinstance(state, list) else state.values()
        total = sum(e.get_size() for e in elems)
        return total

    def stash(self, key, state):
        """
        Save state checkpoint to system memory
        """
        if key in self:
            self.move_to_end(key)
        else:
            stashed_state = {k: v.stash() for k, v in state.items()}
            self.put(key, stashed_state)

    def get_unstashed(self, stashed_state):
        """
        Retrieve state checkpoint from system memory to device memory, device-mapped according to
        loaded model.
        """
        unstashed_state = {k: v.unstash(self.device_map[k]) for k, v in stashed_state.items()}
        return unstashed_state

    def get_empty_state(self):
        """
        Create an empty recurrent state compatible with the associated model
        """
        new_state = {m.layer_idx: m.new_recurrent_state() for m in self.rl}
        return new_state
