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

std::vector<at::Tensor> BC_SAM::export_csr() const
{
    TORCH_CHECK(link_.size() <= size_t(INT32_MAX) && edge_token_.size() <= size_t(INT32_MAX),
                "SAM exceeds int32 frozen format limits");
    auto options = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
    auto copy = [&](const std::vector<int32_t>& src)
    {
        auto dst = at::empty({int64_t(src.size())}, options);
        if (!src.empty()) std::copy(src.begin(), src.end(), dst.data_ptr<int32_t>());
        return dst;
    };
    auto offsets = at::empty({int64_t(link_.size() + 1)}, options);
    auto tokens = at::empty({int64_t(edge_token_.size())}, options);
    auto destinations = at::empty_like(tokens);
    auto root = at::full({int64_t(root_edge_.size())}, -1, options);
    auto off = offsets.data_ptr<int32_t>();
    auto tok = tokens.data_ptr<int32_t>();
    auto dst = destinations.data_ptr<int32_t>();
    auto rt = root.data_ptr<int32_t>();
    for (size_t i = 0; i < root_edge_.size(); ++i)
        if (root_edge_[i] != -1) rt[i] = edge_to_[root_edge_[i]];
    std::vector<std::pair<int32_t, int32_t>> edges;
    int64_t cursor = 0;
    for (size_t s = 0; s < link_.size(); ++s)
    {
        off[s] = int32_t(cursor);
        edges.clear();
        for (int32_t e = first_edge_[s]; e != -1; e = edge_next_[e])
            edges.emplace_back(edge_token_[e], edge_to_[e]);
        std::sort(edges.begin(), edges.end());
        for (const auto& edge : edges)
        {
            tok[cursor] = edge.first;
            dst[cursor++] = edge.second;
        }
    }
    off[link_.size()] = int32_t(cursor);
    TORCH_CHECK(size_t(cursor) == edge_token_.size(), "Invalid SAM edge lists");
    return {copy(link_), copy(max_len_), copy(min_end_), offsets, tokens, destinations, root};
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
    degree_.clear();
    std::fill(root_edge_.begin(), root_edge_.end(), -1);
    indexed_edges_.clear();

    const size_t state_cap = r > 0 ? (2 * r + 1) : 1;
    const size_t edge_cap = r > 0 ? (3 * r + 1) : 0;

    link_.reserve(state_cap);
    max_len_.reserve(state_cap);
    min_end_.reserve(state_cap);
    first_edge_.reserve(state_cap);
    degree_.reserve(state_cap);

    edge_token_.reserve(edge_cap);
    edge_to_.reserve(edge_cap);
    edge_next_.reserve(edge_cap);

    // root
    link_.push_back(-1);
    max_len_.push_back(0);
    min_end_.push_back(0x7fffffff);
    first_edge_.push_back(-1);
    degree_.push_back(0);

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

    // The sequence shrinks when the job rewinds (banned-string suppression); a suffix automaton cannot
    // un-accept tokens, so rebuild it from the truncated sequence. Without this the unsigned length
    // difference below underflows and the loop reads far out of bounds.
    int64_t total = tokens.size(-1);
    if (total < length())
        reset(total);

    size_t offset = length();
    size_t len = (size_t) total - offset;
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
    degree_.push_back(0);
    return idx;
}

void BC_SAM::add_edge(int32_t from, int32_t token, int32_t to)
{
    edge_token_.push_back(token);
    edge_to_.push_back(to);
    edge_next_.push_back(first_edge_[from]);
    first_edge_[from] = (int32_t) edge_to_.size() - 1;

    if (from == 0)
    {
        // Bound dense storage for sparse/negative IDs, retaining the int32 domain.
        if (uint32_t(token) < dense_root_limit)
        {
            if (size_t(token) >= root_edge_.size())
                root_edge_.resize(std::min(size_t(dense_root_limit),
                    std::max(size_t(token) + 1,
                        std::max(size_t(256), root_edge_.size() * 2))), -1);
            root_edge_[token] = first_edge_[from];
        }
        else indexed_edges_.emplace(edge_key(from, token), first_edge_[from]);
    }
    else if (degree_[from] < indexed_degree)
    {
        if (++degree_[from] == indexed_degree)
            for (int32_t e = first_edge_[from]; e != -1; e = edge_next_[e])
                indexed_edges_.emplace(edge_key(from, edge_token_[e]), e);
    }
    else indexed_edges_.emplace(edge_key(from, token), first_edge_[from]);
}

int32_t BC_SAM::find_edge(int32_t state, int32_t token)
{
    if (state == 0 && uint32_t(token) < dense_root_limit)
        return size_t(token) < root_edge_.size() ? root_edge_[token] : -1;
    if (state == 0 || degree_[state] == indexed_degree)
    {
        auto it = indexed_edges_.find(edge_key(state, token));
        return it == indexed_edges_.end() ? -1 : it->second;
    }
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

FrozenSAMCursor::FrozenSAMCursor(std::vector<at::Tensor> sections): arrays(std::move(sections))
{
    TORCH_CHECK(arrays.size() == 9, "Expected nine frozen SAM sections");
    for (const auto& a : arrays)
    {
        TORCH_CHECK(a.device().is_cpu() && a.scalar_type() == at::kByte && a.is_contiguous()
                    && a.dim() == 1 && a.numel() % 4 == 0, "Invalid frozen SAM section");
        counts.push_back(a.numel() / 4);
    }
}

int32_t FrozenSAMCursor::word(int section, int64_t index) const
{
    const auto p = arrays[section].data_ptr<uint8_t>();
    const auto n = counts[section];
    return int32_t(uint32_t(p[index]) | (uint32_t(p[n + index]) << 8)
        | (uint32_t(p[2 * n + index]) << 16) | (uint32_t(p[3 * n + index]) << 24));
}

int32_t FrozenSAMCursor::next(int32_t s, int32_t token) const
{
    if (s == 0 && token >= 0 && token < counts[6]) return word(6, token);
    int64_t lo = word(3, s), hi = word(3, s + 1);
    if (hi - lo < 16)
    {
        for (auto e = lo; e < hi; ++e) if (word(4, e) == token) return word(5, e);
        return -1;
    }
    const auto end = hi;
    while (lo < hi)
    {
        auto mid = (lo + hi) / 2;
        if (word(4, mid) < token) lo = mid + 1; else hi = mid;
    }
    return lo < end && word(4, lo) == token ? word(5, lo) : -1;
}

std::pair<int64_t, at::Tensor> FrozenSAMCursor::draft(const at::Tensor& history, int64_t minimum, int64_t length)
{
    TORCH_CHECK(history.device().is_cpu() && history.scalar_type() == at::kLong && history.is_contiguous(),
                "History must be contiguous CPU int64");
    TORCH_CHECK(minimum > 0 && length >= 0, "Invalid draft limits");
    const auto size = history.numel();
    if (size < pos) { pos = 0; state = matched = 0; }
    const auto ids = history.data_ptr<int64_t>();
    for (; pos < size; ++pos)
    {
        if (ids[pos] < 0 || ids[pos] > INT32_MAX) { state = matched = 0; continue; }
        int32_t token = int32_t(ids[pos]), to = next(state, token);
        while (to == -1 && state)
        {
            state = word(0, state);
            matched = std::min(matched, word(1, state));
            to = next(state, token);
        }
        if (to == -1) { state = matched = 0; }
        else { state = to; ++matched; }
    }
    int64_t end = matched ? int64_t(word(2, state)) + 1 : 0;
    int64_t count = 0;
    if (matched >= minimum)
    {
        // Corpus -1 separators are authoritative, including end-of-corpus.
        while (count < length && end + count < counts[7] && word(7, end + count) >= 0) ++count;
    }
    auto result = at::empty({1, count}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    for (int64_t i = 0; i < count; ++i) result.data_ptr<int64_t>()[i] = word(7, end + i);
    return {matched, result};
}
