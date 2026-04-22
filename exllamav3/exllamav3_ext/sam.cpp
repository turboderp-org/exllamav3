#include <Python.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include "sam.h"
#include "util.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

BC_SAM::BC_SAM()
{
    reset();
}

void BC_SAM::reset(int64_t reserve_tokens)
{
    TORCH_CHECK(reserve_tokens >= 0, "reserve_tokens must be >= 0");
    const size_t r = (size_t) reserve_tokens;

    link_.clear();
    max_len_.clear();
    min_end_.clear();
    first_edge_.clear();

    edge_token_.clear();
    edge_to_.clear();
    edge_next_.clear();

    const size_t state_cap = r > 0 ? (2 * r + 1) : 1;
    const size_t edge_cap = r > 0 ? (3 * r + 1) : 0;

    link_.reserve(state_cap);
    max_len_.reserve(state_cap);
    min_end_.reserve(state_cap);
    first_edge_.reserve(state_cap);

    edge_token_.reserve(edge_cap);
    edge_to_.reserve(edge_cap);
    edge_next_.reserve(edge_cap);

    // root
    link_.push_back(-1);
    max_len_.push_back(0);
    min_end_.push_back(0x7fffffff);
    first_edge_.push_back(-1);

    last_ = 0;
    match_state_ = 0;
    match_len_ = 0;
    pos_ = 0;
}

std::pair<int64_t, int64_t> BC_SAM::accept(int64_t token)
{
    const auto [state, match_len] = advance_match(token);

    int64_t start = -1;
    int64_t end = -1;  // exclusive
    if (match_len > 0)
    {
        const int32_t source_end = min_end_[state];
        start = (int64_t) source_end - (int64_t) match_len + 1;
        end = (int64_t) source_end + 1;
    }

    extend(token);
    return { start, end };
}

std::pair<int64_t, int64_t> BC_SAM::accept_tensor(const at::Tensor& tokens)
{
    TORCH_CHECK_DTYPE(tokens, kLong);
    TORCH_CHECK(tokens.is_contiguous(), "tokens must be contiguous");
    if (tokens.dim() == 2)
        TORCH_CHECK(tokens.size(0) == 1, "2D tokens must have bsz 1");

    size_t len = tokens.size(-1);
    size_t offset = length();
    len -= offset;
    if (len < 1) return { -1, -1 };

    const int64_t* tokens_ptr = (const int64_t*) tokens.data_ptr();
    tokens_ptr += offset;

    for (int i = 0; i < len - 1; ++i)
    {
        advance_match(tokens_ptr[i]);
        extend(tokens_ptr[i]);
    }
    const auto [state, match_len] = advance_match(tokens_ptr[len - 1]);
    extend(tokens_ptr[len - 1]);

    int64_t start = -1;
    int64_t end = -1;  // exclusive
    if (match_len > 0)
    {
        const int32_t source_end = min_end_[state];
        start = (int64_t) source_end - (int64_t) match_len + 1;
        end = (int64_t) source_end + 1;
    }

    return { start, end };
}

int32_t BC_SAM::new_state(int32_t max_len, int32_t link, int32_t min_end)
{
    const int32_t idx = (int32_t) link_.size();
    link_.push_back(link);
    max_len_.push_back(max_len);
    min_end_.push_back(min_end);
    first_edge_.push_back(-1);
    return idx;
}

void BC_SAM::add_edge(int32_t from, int32_t token, int32_t to)
{
    edge_token_.push_back(token);
    edge_to_.push_back(to);
    edge_next_.push_back(first_edge_[from]);
    first_edge_[from] = (int32_t) edge_to_.size() - 1;
}

int32_t BC_SAM::find_edge(int32_t state, int32_t token)
{
    for (int32_t e = first_edge_[state]; e != -1; e = edge_next_[e])
    {
        if (edge_token_[e] == token) return e;
    }
    return -1;
}

std::pair<int32_t, int32_t> BC_SAM::advance_match(int32_t token)
{
    int32_t state = match_state_;
    int32_t length = match_len_;

    int32_t edge = find_edge(state, token);
    while (state != 0 && edge == -1)
    {
        state = link_[state];
        length = std::min(length, max_len_[state]);
        edge = find_edge(state, token);
    }

    if (edge != -1) { state = edge_to_[edge]; ++length; }
    else            { state = 0; length = 0; }

    match_state_ = state;
    match_len_ = length;
    return { state, length };
}

void BC_SAM::extend(int32_t token)
{
    const int32_t pos32 = (int32_t) pos_;

    const int32_t cur = new_state(max_len_[last_] + 1, 0, pos32);
    int32_t p = last_;

    while (p != -1 && find_edge(p, token) == -1)
    {
        add_edge(p, token, cur);
        p = link_[p];
    }

    if (p == -1) link_[cur] = 0;
    else
    {
        const int32_t e = find_edge(p, token);
        const int32_t q = edge_to_[e];
        if (max_len_[p] + 1 == max_len_[q]) link_[cur] = q;
        else
        {
            const int32_t clone = new_state(max_len_[p] + 1, link_[q], min_end_[q]);

            // Copy q's outgoing transitions into clone.
            for (int32_t ee = first_edge_[q]; ee != -1; ee = edge_next_[ee])
                add_edge(clone, edge_token_[ee], edge_to_[ee]);

            while (p != -1)
            {
                const int32_t pe = find_edge(p, token);
                if (pe == -1 || edge_to_[pe] != q) break;
                edge_to_[pe] = clone;
                p = link_[p];
            }

            link_[q] = clone;
            link_[cur] = clone;
        }
    }

    last_ = cur;
    ++pos_;
}
