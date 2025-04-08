from __future__ import annotations
import torch
import os, glob
import numpy as np
import json
import mmap
from ..util import Timer, cuda_sync_active
from ..ext import exllamav3_ext as ext
from functools import lru_cache

def convert_dtype(dt: str):
    if dt == "I32": return torch.int, np.int32, 4
    elif dt == "I16": return torch.short, np.int16, 2
    elif dt == "F16": return torch.float16, np.float16, 2
    elif dt == "BF16": return torch.bfloat16, np.float16, 2
    elif dt == "F32": return torch.float, np.float32, 4
    else:
        raise ValueError(f"Unknown dtype {dt}")


def read_header(filename: str) -> dict:
    with open(filename, "rb") as fp:
        header_size = np.fromfile(fp, dtype = np.int64, count = 1).item()
        header_json = fp.read(header_size)
        header = json.loads(header_json.decode("utf-8"))
        header["_header_offset"] = fp.tell()
        return header


class SafetensorsCollection:

    def __init__(
        self,
        directory: str,
        load_method: str | None = None
    ):
        """
        Scan directory for .safetensors files and build collection, preparing to load tensors indexed by key.

        :param directory:
            Directory to scan.

        :param load_method:
            - "mt_fread": multithreaded C++ loader using fread
            - "python": use fp.seek() and fp.read() to load tensor data via bytearray and torch.frombuffer
        """

        self.directory = directory
        self.tensor_file_map = {}
        self.file_headers = {}
        self.handles: dict[str, list | None] = {}
        self.load_method = load_method or "mt_fread"

        self.bytes_loaded = 0
        self.time_elapsed = 0

        self.tensor_files = []
        self.add_tensor_files(directory)


    def add_tensor_files(
        self,
        directory: str,
        warn_if_override: bool = True
    ):
        st_pattern = os.path.join(directory, "*.safetensors")
        new_tensor_files = glob.glob(st_pattern)
        self.tensor_files += new_tensor_files

        overrides = 0
        for st_file in new_tensor_files:
            self.handles[st_file] = None
            header = read_header(st_file)
            self.file_headers[st_file] = header
            for key in header.keys():
                if key in ["__metadata__", "_header_offset"]:
                    continue
                if key in self.tensor_file_map and warn_if_override:
                    # print(f" !! Overriding {key} from {self.tensor_file_map[key]} with f{st_file}")
                    overrides += 1
                self.tensor_file_map[key] = st_file
        if overrides:
            print(f" !! Replaced {overrides} tensors from {directory}")


    def has_tensor(
        self,
        key: str,
    ):
        return key in self.tensor_file_map


    def has_tensor_group(
        self,
        key: str,
        subkeys: list,
    ):
        return all(
            (
                f"{key}.{subkey}" in self.tensor_file_map if isinstance(subkey, str) else
                any(f"{key}.{sk}" in self.tensor_file_map for sk in subkey)
            ) for subkey in subkeys
        )


    def get_tensor_sizes(
        self,
        prefix: str,
    ):
        keys = [
            key for key in self.tensor_file_map.keys()
            if key == prefix or key.startswith(prefix + ".")
        ]
        sizes = [self.get_tensor_size(key) for key in keys]
        return sizes


    def get_tensor_size(
        self,
        key: str,
        optional: bool = False
    ):
        if not key in self.tensor_file_map:
            if not optional:
                raise ValueError(f"Required tensor {key} not found in any *.safetensors file in {self.directory}")
            else:
                return 0

        filename = self.tensor_file_map[key]
        header = self.file_headers[filename]
        h = header[key]
        # _, _, esize = convert_dtype(h["dtype"])
        # bytesize = np.prod(h["shape"]) * esize
        beg, end = h["data_offsets"]
        bytesize = end - beg
        return bytesize


    def list_tensors(
        self,
        prefix: str,
    ) -> dict:
        keys = [
            key for key in self.tensor_file_map.keys()
            if key == prefix or key.startswith(prefix + ".")
        ]
        results = {}
        for key in keys:
            filename = self.tensor_file_map[key]
            header = self.file_headers[filename]
            h = header[key]
            dtype, np_dtype, esize = convert_dtype(h["dtype"])
            beg, end = h["data_offsets"]
            results[key] = {
                "shape": h["shape"],
                "n_bytes": end - beg,
                "dtype": str(dtype),
            }
        return results


    def get_tensors(
        self,
        prefix: str,
        device: torch.device | None = None,
        allow_bf16: bool = False,
    ) -> dict:
        keys = [
            key for key in self.tensor_file_map.keys()
            if key == prefix or key.startswith(prefix + ".")
        ]
        result = {key: self.get_tensor(key, device, allow_bf16 = allow_bf16) for key in keys}
        return result


    def get_tensor(
        self,
        key: str,
        device: torch.device | None = None,
        optional: bool = False,
        allow_bf16: bool = False
    ) -> torch.Tensor | None:

        if not key in self.tensor_file_map:
            if not optional:
                raise ValueError(f"Required tensor {key} not found in any *.safetensors file in {self.directory}")
            else:
                return None

        if device is None:
            device = torch.device("cpu")

        filename = self.tensor_file_map[key]
        header = self.file_headers[filename]
        h = header[key]
        offset = header["_header_offset"]

        dtype, np_dtype, esize = convert_dtype(h["dtype"])
        beg, end = h["data_offsets"]
        bytesize = end - beg
        shape = h["shape"]
        numel = np.prod(shape)
        assert numel * esize == bytesize, \
            f"Incorrect size of {key} in {filename}"

        load_method = self.load_method

        with Timer() as timer:
            match load_method:
                case "mt_fread":
                    h = self.handles[filename]
                    if not h:
                        h = ext.stloader_open_file(filename)
                        self.handles[filename] = h
                    tensor = torch.empty(shape, dtype = dtype, device = device)
                    if device != "cpu":
                        cuda_sync_active()
                    assert tensor.is_contiguous()
                    ext.stloader_read(
                        h,
                        offset + beg,
                        bytesize,
                        tensor,
                    )
                case "python":
                    with open(filename, "rb") as fp:
                        fp.seek(offset + beg)
                        buffer = bytearray(fp.read(bytesize))
                        tensor = torch.frombuffer(buffer, dtype = dtype, count = numel).reshape(shape)
                        tensor = tensor.to(device)
                case _:
                    raise ValueError(f"Invalid load_method: {load_method}")

        self.bytes_loaded += bytesize
        self.time_elapsed += timer.interval

        if tensor.dtype == torch.bfloat16 and not allow_bf16:
            tensor = tensor.to(torch.float16)

        return tensor


    def close(self):
        for filename, h in self.handles.items():
            if h:
                ext.stloader_close_file(h)
                self.handles[filename] = None


    def get_metrics(self):
        bandwidth = self.bytes_loaded / (1024**3) / self.time_elapsed
        return self.bytes_loaded, self.time_elapsed, bandwidth


    @lru_cache
    def max_key_len(self):
        l = max(len(k) for k in self.tensor_file_map.keys())
        return l


class VariantSafetensorsCollection:

    def __init__(
        self,
        tensor_map: dict[str, str],
        **kwargs
    ):
        self.tensor_map = None
        self.tensor_map_sort = None
        self.all_dirs = None
        self.stcs = {}
        self.kwargs = kwargs
        self.update_map(tensor_map)


    def update_map(
        self,
        tensor_map: dict[str, str]
    ):
        self.tensor_map = tensor_map
        self.tensor_map_sort = sorted(tensor_map.items(), key = lambda kv: len(kv[0]), reverse = True)
        all_dirs = list(set(tensor_map.values()))

        for d in all_dirs:
            if d not in self.stcs:
                self.stcs[d] = SafetensorsCollection(directory = d, **self.kwargs)


    def has_tensor(
        self,
        key: str,
    ):
        return any(key in stc.tensor_file_map for stc in self.stcs.values())


    def has_tensor_group(
        self,
        key: str,
        subkeys: list[str],
    ):
        return all(
            any(f"{key}.{subkey}" in stc.tensor_file_map for stc in self.stcs.values())
            for subkey in subkeys
        )


    def get_tensor(
        self,
        key: str,
        device: torch.device | None = None,
        optional: bool = False,
        allow_bf16: bool = False
    ) -> torch.Tensor | None:

        file = None
        for k, v in self.tensor_map_sort:
            if key.startswith(k):
                file = v
                break
        if file is None:
            if not optional:
                raise ValueError(f"No prefix found in variants map with the matching key: {key}")
            else:
                return None

        return self.stcs[file].get_tensor(key, device, optional, allow_bf16)


    def close(self):
        for stc in self.stcs.values():
            stc.close()


    def get_metrics(self):
        res = [stc.get_metrics() for stc in self.stcs.values()]
        bytes_loaded = sum(r[0] for r in res)
        time_elapsed = sum(r[1] for r in res)
        bandwidth = bytes_loaded / (1024**3) / time_elapsed
        return bytes_loaded, time_elapsed, bandwidth