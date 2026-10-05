// Minimal at::Tensor stub for building moe_mul1.cpp without torch (bench only)
#pragma once
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace at {

using Half = c10::Half;

enum ScalarType { kByte, kChar, kShort, kInt, kLong, kHalf, kFloat, kDouble };

struct Device {
    bool is_cpu() const { return true; }
};

struct Tensor {
    void* p = nullptr;
    int64_t dims[3] = {0, 0, 0};
    int nd = 0;
    ScalarType st = kFloat;

    Device device() const { return Device(); }
    bool is_contiguous() const { return true; }
    int dim() const { return nd; }
    int64_t size(int i) const { return dims[i]; }
    ScalarType scalar_type() const { return st; }
    void* data_ptr() const { return p; }
    template <typename T> T* data_ptr() const { return static_cast<T*>(p); }
};

} // namespace at

#define TORCH_CHECK(cond, ...) do { if (!(cond)) throw std::runtime_error("TORCH_CHECK failed"); } while (0)