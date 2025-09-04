from ..util.misc import ratio_split
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
        self.estimate_total = None
        self.estimate_storage = None
        self.estimate_overhead = None
        self.num_devices = None
        self.plan = None


    def initial_split(
        self,
        max_mem: list[int],
    ):
        self.num_devices = len(max_mem)
        active_devices = [i for i in range(self.num_devices) if max_mem[i] > 0]
        storage_sum = [0] * self.num_devices
        overhead_max = [0] * self.num_devices

        for c in self.components:

            # Remaining computed space per device. If space runs out completely, increase maximum by 10%
            while True:
                rem_mem_s = [max(0, mm - ss - om) for mm, ss, om in zip(max_mem, storage_sum, overhead_max)]
                if sum(rem_mem_s) == 0:
                    max_mem = [m * 11 // 10 for m in max_mem]
                else:
                    break

            # Mask out devices to satisy max split per component type
            if c.max_devices is not None or c.limit_key:
                dev_limit = self.dev_limits.get(c.limit_key, c.max_devices)
                if dev_limit is not None:
                    dev_limit = min(dev_limit, len(active_devices))
                if dev_limit is not None:
                    top_k_mask_(rem_mem_s, dev_limit)

            # Active devices on layer
            mask = [m > 0 for m in rem_mem_s]

            # Perform split
            channels = c.channels_to_split
            split = ratio_split(channels, rem_mem_s, chunk_size = 1)
            c.current_split = split

            # Compute storage and overhead given layer and split
            tokens = self.output_num_tokens if c is self.components[-1] else self.num_tokens
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

            # Compute overall usage
            storage_sum = [ss + s for ss, s in zip(storage_sum, storage)]
            overhead_max = [max(om, o) for om, o in zip(overhead_max, overhead)]

        self.estimate_storage = [ss for ss, om in zip(storage_sum, overhead_max)]
        self.estimate_overhead = [om for ss, om in zip(storage_sum, overhead_max)]
        self.estimate_total = [ss + om for ss, om in zip(storage_sum, overhead_max)]
        return self.estimate_total, self.estimate_storage, self.estimate_overhead


    def print_split(self):
        n_columns = len(self.estimate_total)
        def _divider():
            nonlocal n_columns
            print("    " + "-" * (62 + 10 * n_columns))
        def _columns(t, u, d):
            print(f"    {t:<50}{u:<12}" + "".join([f"{d_:>10}" for d_ in d]))

        print(" -- Model split:")
        _divider()
        _columns("", "Units", [f"CUDA:{i}" for i in range(n_columns)])
        _divider()
        for c in (c for c in self.components if c.channel_unit):
            _columns(c.key, c.channel_unit, [f"{s * c.channel_width}" for s in c.current_split])
        _divider()
        _columns("Storage", "GB", [f"{e / 1024**3:10.2f}" for e in self.estimate_storage])
        _columns("Overhead", "GB", [f"{e / 1024**3:10.2f}" for e in self.estimate_overhead])
        _columns("Total", "GB", [f"{e / 1024**3:10.2f}" for e in self.estimate_total])


    def compile_tp_plan(self):
        plan = []
        for _ in range(self.num_devices):
            plan.append({})
        for c in self.components:
            key = c.key
            idx_end = 0
            cw = c.channel_width or 1
            for dev in range(self.num_devices):
                idx_beg = idx_end
                idx_end += c.current_split[dev]
                plan[dev][key] = (idx_beg * cw, idx_end * cw, c.channel_unit)
        self.plan = plan
        return self.plan