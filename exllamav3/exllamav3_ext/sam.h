#pragma once

#include <vector>
#include <cstdint>
#include <unordered_map>
#include <pybind11/pybind11.h>
namespace py = pybind11;

struct BC_SAM
{
private:
    std::vector<int32_t> link_;
    std::vector<int32_t> max_len_;
    std::vector<int32_t> min_end_;
    std::vector<int32_t> first_edge_;

    std::vector<int32_t> edge_token_;
    std::vector<int32_t> edge_to_;
    std::vector<int32_t> edge_next_;

    // Counts saturate at the promotion threshold; small lists stay cheap.
    static constexpr uint8_t indexed_degree = 64;
    static constexpr uint32_t dense_root_limit = 1 << 20;
    std::vector<uint8_t> degree_;
    std::vector<int32_t> root_edge_;
    // Edge IDs survive clone redirects, which only change edge_to_.
    std::unordered_map<uint64_t, int32_t> indexed_edges_;

    static uint64_t edge_key(int32_t state, int32_t token)
    {
        return (uint64_t(uint32_t(state)) << 32) | uint32_t(token);
    }

    std::int32_t last_ = 0;
    std::int32_t match_state_ = 0;
    std::int32_t match_len_ = 0;
    std::int64_t pos_ = 0;

    int32_t new_state(int32_t max_len, int32_t link, int32_t min_end);
    void add_edge(int32_t from, int32_t token, int32_t to);
    int32_t find_edge(int32_t state, int32_t token);
    std::pair<int32_t, int32_t> advance_match(int32_t token);
    void extend(int32_t token);

public:
    BC_SAM();
    void reset(int64_t reserve_tokens = 0);
    std::pair<int64_t, int64_t> accept(int64_t token);
    std::pair<int64_t, int64_t> accept_tensor(const at::Tensor& tokens);
    // Pointer-free immutable graph: links, lengths, ends, CSR offsets/tokens/
    // destinations, and a dense root destination table. All tensors own copies.
    std::vector<at::Tensor> export_csr() const;

    int64_t length() { return pos_; }
};

// Each cursor owns references to shared, immutable byte-plane section storage.
// Validation of the on-disk graph is performed once by the Python loader.
struct FrozenSAMCursor
{
    std::vector<at::Tensor> arrays;
    std::vector<int64_t> counts;
    int32_t state = 0, matched = 0;
    int64_t pos = 0;
    FrozenSAMCursor(std::vector<at::Tensor> sections);
    int32_t word(int section, int64_t index) const;
    int32_t next(int32_t s, int32_t token) const;
    std::pair<int64_t, at::Tensor> draft(const at::Tensor& history, int64_t minimum, int64_t length);
};
