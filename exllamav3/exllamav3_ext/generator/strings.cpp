#include "strings.h"
#include "../util.h"

// Compare string Q against list of strings S, utf-32 encoded and packed in byte array.
//
// Returns:
// -1: No matches
// -2: Partial match; some string in S overlaps the right-hand end of Q, starting before any full
//     match, so a later completion could produce an earlier stop than any full match found so far
// >= 0: Index into Q of the EARLIEST full match with any string in S
//
// All strings are scanned before deciding: an earlier-listed string's trailing partial match must
// not hide a later-listed string's full match at a smaller index (the caller truncates at the
// returned index, so a hidden full match would be overrun)

int partial_strings_match
(
    py::buffer match,
    py::buffer offsets,
    py::buffer strings
)
{
    py::buffer_info info;

    info = match.request();
    uint32_t* q = static_cast<uint32_t*>(info.ptr);
    int q_len = info.size / 4;

    info = offsets.request();
    uint32_t* offsets_int = static_cast<uint32_t*>(info.ptr);
    int num_strings = info.size / 4 - 1;

    info = strings.request();
    uint32_t* strings_utf32 = static_cast<uint32_t*>(info.ptr);
    int s_total = info.size / 4;

    const int none = q_len + 1;
    int best_full = none;       // earliest full match over all strings
    int best_partial = none;    // earliest start of a match that runs off the end of Q

    for (int i = 0; i < num_strings; ++i)
    {
        int beg = offsets_int[i] / 4;
        int s_len = offsets_int[i + 1] / 4 - beg;
        uint32_t* s = strings_utf32 + beg;

        if (s_len <= 0 || beg + s_len > s_total) continue;

        // Only starts before the best candidates so far can change the outcome
        int limit = best_full < best_partial ? best_full : best_partial;
        for (int a0 = 0; a0 < q_len && a0 < limit; ++a0)
        {
            int k = 0;
            while (a0 + k < q_len && k < s_len && q[a0 + k] == s[k]) ++k;
            if (k == s_len) { best_full = a0; break; }               // earliest full match of this string
            if (a0 + k == q_len) { best_partial = a0; break; }       // ran off the end of Q while matching
        }
    }

    if (best_partial < best_full) return -2;
    if (best_full < none) return best_full;
    return -1;
}

int count_match_tensor
(
    at::Tensor a,
    at::Tensor b,
    int max_a
)
{
    uint64_t* pa = (uint64_t*) a.data_ptr();
    uint64_t* pb = (uint64_t*) b.data_ptr();
    int max_b = b.size(1);
    if (max_b < max_a) max_a = max_b;

    int match = 0;
    while (match < max_a && *pa++ == *pb++)
        match++;

    return match;
}