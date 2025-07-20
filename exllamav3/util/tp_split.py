from .misc import ratio_split
import heapq

def top_k_mask_(lst, k):
    assert 0 < k <= len(lst)
    idx_k_largest = [i for i, _ in heapq.nlargest(k, enumerate(lst), key=lambda t: t[1])]
    keep = set(idx_k_largest)
    for i in range(len(lst)):
        if i not in keep:
            lst[i] = 0


class TPAllocation:

    def __init__(
        self,
        key: str,
        channel_width: int = None,
        channel_unit: str = None,
        storage_per_device: int = 0,
        storage_to_split: int = 0,
        overhead_per_device: int = 0,  # per token
        overhead_to_split: int = 0,  # per token
        recons_temp: int = 0,
        channels_to_split: int = 1,
        limit_key: str = None,
        max_devices: int = None
    ):
        self.key = key
        self.channel_width = channel_width
        self.channel_unit = channel_unit
        self.limit_key = limit_key

        self.storage_per_device = storage_per_device
        self.storage_to_split = storage_to_split
        self.overhead_per_device = overhead_per_device
        self.overhead_to_split = overhead_to_split
        self.recons_temp = recons_temp
        self.channels_to_split = channels_to_split
        self.max_devices = max_devices

        self.current_split = []


class TPAllocator:

    def __init__(
        self,
        components: list[TPAllocation],
        num_tokens: int,
        output_num_tokens: int,
        dev_limits: dict = None,
    ):
        self.components = components
        self.current_split = None
        self.current_usage = None
        self.num_tokens = num_tokens
        self.output_num_tokens = output_num_tokens
        self.dev_limits = dev_limits or {}
        self.estimate = None
        self.num_devices = None
        self.plan = None


    def initial_split(
        self,
        max_mem: list[int],
    ):
        self.num_devices = len(max_mem)
        storage_sum = [0] * self.num_devices
        overhead_max = [0] * self.num_devices

        for c in self.components:
            rem_mem_s = [max(0, mm - ss - om) for mm, ss, om in zip(max_mem, storage_sum, overhead_max)]
            if c.max_devices is not None or c.limit_key:
                dev_limit = self.dev_limits.get(c.limit_key, c.max_devices)
                if dev_limit is not None:
                    top_k_mask_(rem_mem_s, dev_limit)
            mask = [m > 0 for m in rem_mem_s]
            tokens = self.output_num_tokens if c is self.components[-1] else self.num_tokens
            channels = c.channels_to_split
            split = ratio_split(channels, rem_mem_s, chunk_size = 1)
            c.current_split = split
            storage = [
                (c.storage_per_device if m else 0)
                + c.storage_to_split * s // channels
                for s, m in zip(split, mask)
            ]
            overhead = [
                (c.overhead_per_device if m else 0)
                + tokens * c.overhead_to_split * s // channels
                + c.recons_temp * s // channels
                for s, m in zip(split, mask)
            ]
            storage_sum = [ss + s for ss, s in zip(storage_sum, storage)]
            overhead_max = [max(om, o) for om, o in zip(overhead_max, overhead)]

        self.estimate = [ss + om for ss, om in zip(storage_sum, overhead_max)]
        return self.estimate


    def print_split(self):
        for c in self.components:
            if c.channel_unit is None:
                continue
            print(f"{c.key:50}", end = "")
            print(f"{c.channel_unit:12}", end = "")
            for s in c.current_split:
                print(f"{s * c.channel_width: 10}", end = "")
            print()
        print("-" * 100)
        print(f"{'Total':50}", end = "")
        print(f"{'GB':12}", end = "")
        for e in self.estimate:
            print(f"{e / 1024**3:10.2f}", end = "")
        print()


    def compile_tp_plan(self):
        plan = []
        for _ in range(self.num_devices):
            plan.append({})
        for c in self.components:
            key = c.key
            idx_end = 0
            for dev in range(self.num_devices):
                idx_beg = idx_end
                idx_end += c.current_split[dev]
                plan[dev][key] = (idx_beg, idx_end)
        self.plan = plan
        return self.plan