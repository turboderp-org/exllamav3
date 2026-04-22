#pragma once

#include <vector>
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

    int64_t length() { return pos_; }
};