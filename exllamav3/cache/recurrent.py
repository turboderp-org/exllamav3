from collections import OrderedDict
from abc import ABC, abstractmethod

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
    def unstash(self, device, trim_position):
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

    @abstractmethod
    def get_cachable_interval(self):
        """
        Return number of past positions from which state can be trivially reconstructed
        """

    @abstractmethod
    def reset(self):
        """
        Clear this recurrent state (used during autosplit load)
        """

    @abstractmethod
    def force_position(self, position: int):
        """
        For testing/benchmark purposes, modify the state to mapping to a specific position. Actual state
        value will be incorrect but state will be valid.
        """

    @abstractmethod
    def clone(self):
        """
        Return a fully materialized clone of this state
        """

    @abstractmethod
    def rewind(self, count: int):
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
        self.model = model
        self.rl = model.get_recurrent_layers()
        self.device_map = {}
        for m in self.rl:
            for instance in self.model.get_layer_instances(m.layer_idx):
                self.device_map[instance] =  m.device


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
            del self[key]
        if state not in self.values():
            state_size = self.get_size(state)
            while self.update_total_size() + state_size > self.max_size:
                assert self.current_size > 0, "Not enough space in cache for single state"
                _, oldest_state = self.popitem(last = False)
        self[key] = state
        self.update_total_size()

    def update_total_size(self):
        seen = set()
        total = 0
        for v in self.values():
            if id(v) in seen:
                continue
            seen.add(id(v))
            total += self.get_size(v)
        self.current_size = total
        return total

    def get_size(self, state):
        elems = state if isinstance(state, list) else state.values()
        total = sum(e.get_size() for e in elems)
        return total

    def stash(self, keys, state):
        """
        Save state checkpoint to system memory
        """
        stashed_state = None
        for offset, key in keys:
            if key in self:
                self.move_to_end(key)
            else:
                stashed_state = stashed_state or self.get_stashed(state)
                self.put(key, stashed_state)

    def get_stashed(self, state):
        """
        Move checkpoint to system RAM and prepare to store in cache
        """
        return {k: v.stash() for k, v in state.items()}

    def get_unstashed(self, stashed_state, trim_position):
        """
        Retrieve state checkpoint from system memory to device memory, device-mapped according to
        loaded model.
        """
        unstashed_state = {k: v.unstash(self.device_map[k], trim_position) for k, v in stashed_state.items()}
        return unstashed_state

    def get_empty_state(self):
        """
        Create an empty recurrent state compatible with the associated model
        """
        return self.model.get_empty_state()
