#ifdef _WIN32

// Windows implementation of ngram_gather_cpu. The gather is thousands of tiny scattered reads
// (82-320 bytes each) whose cold latency only amortizes at high queue depth. The file HANDLE
// (opened by DiskTensorHandle with FILE_FLAG_OVERLAPPED) takes ReadFile calls with per-call
// OVERLAPPED offsets. One thread holds many NVMe queue slots.
//
// The handle arrives opened with FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING, so reads go
// through 4K-aligned spans into per-thread bounce slots, with the payload copied out on
// completion.
//
//   - Small gathers (decode: <= 64 runs) issue every run asynchronously from the calling
//     thread and then wait: all cold reads in flight at once with no thread handoff.
//   - Large gathers (prefill) go to a persistent worker pool like the Linux path, each worker
//     keeping WG_WORKER_DEPTH reads of its span in flight, for pool-width x depth (~200)
//     outstanding NVMe commands.

#include "ngram.cuh"
#include <pybind11/pybind11.h>
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;

#define WG_MAX_DEPTH 64                 // caller-inline async depth (WaitForMultipleObjects limit)
#define WG_WORKER_DEPTH 8               // per-worker async depth in the pooled path
#define WG_SECTOR 4096                  // covers 512e/4Kn drives; NO_BUFFERING alignment unit
#define WG_BOUNCE_BYTES (64 * 1024)     // per-slot bounce size (VirtualAlloc base is 64K-aligned)

namespace
{

struct GatherCtx
{
    HANDLE h_rd;             // overlapped + unbuffered handle, as passed in
    int64_t base_offset;
    int64_t row_bytes;
    int64_t uid_base;
    const int64_t* up;
    uint8_t* op;
    std::atomic<bool> failed { false };
    // EXL3_NGRAM_GATHER_PROF: reads completing synchronously in ReadFile vs left pending, and
    // tasks drained by pool workers vs the calling thread
    std::atomic<long> n_sync { 0 };
    std::atomic<long> n_pend { 0 };
    std::atomic<long> t_worker { 0 };
    std::atomic<long> t_caller { 0 };
};

static bool gather_prof()
{
    static bool p = getenv("EXL3_NGRAM_GATHER_PROF") != nullptr;
    return p;
}

// Per-thread reusable auto-reset events for the OVERLAPPED slots: created lazily, closed at
// thread exit (pool workers never exit, callers only ever close their own set). Auto-reset is
// safe across slot reuse because NtReadFile clears the event when the next operation is issued.
struct EventSet
{
    HANDLE ev[WG_MAX_DEPTH] = {};
    int n = 0;
    HANDLE get(int i)
    {
        while (n <= i)
        {
            ev[n] = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!ev[n]) return nullptr;
            ++n;
        }
        return ev[i];
    }
    ~EventSet() { for (int i = 0; i < n; ++i) CloseHandle(ev[i]); }
};
thread_local EventSet t_events;

// Per-thread bounce slots for the unbuffered path, allocated lazily one slot at a time
struct BounceSet
{
    uint8_t* ptr[WG_MAX_DEPTH] = {};
    uint8_t* slot(int i)
    {
        if (!ptr[i])
            ptr[i] = (uint8_t*) VirtualAlloc(nullptr, WG_BOUNCE_BYTES,
                                             MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        return ptr[i];
    }
    ~BounceSet() { for (auto p : ptr) if (p) VirtualFree(p, 0, MEM_RELEASE); }
};
thread_local BounceSet t_bounce;

// Read the coalesced runs of uid span [i0, i1) with up to `depth` ReadFile calls in flight,
// harvesting completions as they arrive
static void read_span_async(GatherCtx* c, int64_t i0, int64_t i1, int depth)
{
    if (depth > WG_MAX_DEPTH) depth = WG_MAX_DEPTH;
    if (depth < 1) depth = 1;

    OVERLAPPED ovs[WG_MAX_DEPTH];
    bool busy[WG_MAX_DEPTH] = {};
    DWORD expect[WG_MAX_DEPTH];       // minimum acceptable byte count
    uint8_t* dst[WG_MAX_DEPTH];       // payload destination...
    DWORD pay_off[WG_MAX_DEPTH];      // ...offset of the payload inside the bounce slot...
    DWORD pay_len[WG_MAX_DEPTH];      // ...and its length
    int pending = 0;
    bool failing = false;
    int64_t i = i0;

    // cap each read so its aligned span fits the bounce slot
    const int64_t max_payload = WG_BOUNCE_BYTES - 2 * WG_SECTOR;

    // complete slot s: verify byte count, copy the payload out of the bounce slot
    auto complete = [&](int s) -> bool
    {
        DWORD got = 0;
        if (!GetOverlappedResult(c->h_rd, &ovs[s], &got, FALSE) || got < expect[s])
            return false;
        memcpy(dst[s], t_bounce.slot(s) + pay_off[s], pay_len[s]);
        return true;
    };

    while (!failing && (i < i1 || pending > 0))
    {
        while (!failing && i < i1 && pending < depth)
        {
            int s = 0;
            while (busy[s]) ++s;
            HANDLE ev = t_events.get(s);
            if (!ev) { failing = true; break; }

            // coalesce a run of consecutive rows into one read
            int64_t j = i + 1;
            while (j < i1 && c->up[j] == c->up[j - 1] + 1 &&
                   (j - i) * c->row_bytes < max_payload) ++j;
            uint64_t off = (uint64_t) (c->base_offset + (c->up[i] - c->uid_base) * c->row_bytes);
            int64_t payload = (j - i) * c->row_bytes;

            uint8_t* buf = t_bounce.slot(s);
            if (!buf) { failing = true; break; }
            uint64_t rd_off = off & ~((uint64_t) WG_SECTOR - 1);
            DWORD delta = (DWORD) (off - rd_off);
            DWORD rd_len = (DWORD) ((delta + payload + WG_SECTOR - 1) & ~(WG_SECTOR - 1));
            dst[s] = c->op + i * c->row_bytes;
            pay_off[s] = delta;
            pay_len[s] = (DWORD) payload;
            expect[s] = delta + (DWORD) payload;   // EOF may trim the tail padding

            OVERLAPPED* ov = &ovs[s];
            memset(ov, 0, sizeof(OVERLAPPED));
            ov->Offset = (DWORD) (rd_off & 0xffffffffull);
            ov->OffsetHigh = (DWORD) (rd_off >> 32);
            ov->hEvent = ev;

            BOOL ok = ReadFile(c->h_rd, buf, rd_len, nullptr, ov);
            if (ok)
            {
                // completed synchronously inside ReadFile: finish and recycle the slot now
                if (!complete(s)) failing = true;
                if (gather_prof()) c->n_sync.fetch_add(1, std::memory_order_relaxed);
            }
            else if (GetLastError() == ERROR_IO_PENDING)
            {
                busy[s] = true;
                ++pending;
                if (gather_prof()) c->n_pend.fetch_add(1, std::memory_order_relaxed);
            }
            else failing = true;
            i = j;
        }
        if (failing || pending == 0) continue;

        HANDLE hs[WG_MAX_DEPTH];
        int smap[WG_MAX_DEPTH];
        DWORD nh = 0;
        for (int s = 0; s < depth; ++s)
            if (busy[s]) { smap[nh] = s; hs[nh++] = ovs[s].hEvent; }
        DWORD w = WaitForMultipleObjects(nh, hs, FALSE, INFINITE);
        if (w < nh)
        {
            int s = smap[w];
            if (!complete(s)) failing = true;
            busy[s] = false;
            --pending;
        }
        else failing = true;
    }

    if (failing)
    {
        c->failed.store(true);
        if (pending > 0)
        {
            // CancelIoEx hits every operation on the handle, concurrent workers included
            CancelIoEx(c->h_rd, nullptr);
            for (int s = 0; s < depth; ++s)
            {
                if (!busy[s]) continue;
                DWORD got = 0;
                GetOverlappedResult(c->h_rd, &ovs[s], &got, TRUE);
            }
        }
    }
}

// Persistent worker pool for prefill-sized gathers, mirroring the Linux GatherPool: lazily
// started detached workers pull uid-index spans from a queue, the caller drains the queue too,
// and the object is intentionally leaked (joining or destroying a static condvar with waiters at
// process exit deadlocks; see the stloader shutdown notes).
struct GatherPool
{
    std::mutex mx;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    GatherCtx* ctx = nullptr;
    std::vector<std::pair<int64_t, int64_t>> tasks;   // uid-index spans [i0, i1)
    size_t next = 0;
    size_t pending = 0;
    bool started = false;
    std::mutex call_mx;

    void worker()
    {
        std::unique_lock<std::mutex> lk(mx);
        while (true)
        {
            cv_work.wait(lk, [&] { return next < tasks.size(); });
            auto span = tasks[next++];
            GatherCtx* c = ctx;
            lk.unlock();
            if (gather_prof()) c->t_worker.fetch_add(1, std::memory_order_relaxed);
            read_span_async(c, span.first, span.second, WG_WORKER_DEPTH);
            lk.lock();
            if (--pending == 0)
                cv_done.notify_all();
        }
    }

    void run(GatherCtx* c, std::vector<std::pair<int64_t, int64_t>>& new_tasks)
    {
        std::lock_guard<std::mutex> call_lock(call_mx);
        std::unique_lock<std::mutex> lk(mx);
        if (!started)
        {
            started = true;
            int n = (int) std::min(32u, std::max(4u, std::thread::hardware_concurrency()));
            for (int i = 0; i < n; ++i)
                std::thread(&GatherPool::worker, this).detach();
        }
        ctx = c;
        tasks.swap(new_tasks);
        next = 0;
        pending = tasks.size();
        cv_work.notify_all();
        while (next < tasks.size())
        {
            auto span = tasks[next++];
            lk.unlock();
            if (gather_prof()) c->t_caller.fetch_add(1, std::memory_order_relaxed);
            read_span_async(c, span.first, span.second, WG_WORKER_DEPTH);
            lk.lock();
            if (--pending == 0)
                cv_done.notify_all();
        }
        cv_done.wait(lk, [&] { return pending == 0; });
        tasks.clear();
        ctx = nullptr;
    }
};

GatherPool* gather_pool()
{
    static GatherPool* pool = new GatherPool();   // leaked: see note above
    return pool;
}

}  // namespace

void ngram_gather_cpu
(
    int64_t fd,                      // HANDLE with FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING
    int64_t base_offset,
    int64_t row_bytes,
    const at::Tensor& uids,          // (U) int64 CPU, sorted ascending
    int64_t uid_base,                // subtracted from uids (shard-local row index)
    at::Tensor out                   // (U, row_bytes / itemsize) CPU, contiguous
)
{
    TORCH_CHECK(uids.device().is_cpu() && uids.dtype() == at::kLong && uids.is_contiguous(),
                "ngram_gather_cpu: uids must be contiguous int64 CPU");
    TORCH_CHECK(out.device().is_cpu() && out.is_contiguous(), "ngram_gather_cpu: bad out");
    int64_t U = uids.numel();
    TORCH_CHECK(out.size(0) >= U && out.size(1) * out.element_size() == row_bytes,
                "ngram_gather_cpu: out shape");
    if (!U) return;
    const int64_t* up = (const int64_t*) uids.data_ptr();
    uint8_t* op = (uint8_t*) out.data_ptr();

    py::gil_scoped_release release;

    GatherCtx ctx { (HANDLE) (intptr_t) fd, base_offset, row_bytes, uid_base, up, op };

    // Coalesce consecutive rows into runs
    std::vector<std::pair<int64_t, int64_t>> runs;   // (first index, count)
    runs.reserve(64);
    for (int64_t i = 0; i < U; )
    {
        int64_t j = i + 1;
        while (j < U && up[j] == up[j - 1] + 1) ++j;
        runs.emplace_back(i, j - i);
        i = j;
    }
    size_t n_runs = runs.size();

    if (n_runs <= WG_MAX_DEPTH)
    {
        // decode-sized gather: all runs in flight from this thread, no pool round trip
        read_span_async(&ctx, 0, U, (int) n_runs);
    }
    else
    {
        // prefill-sized gather: grouped into <= 64 span tasks for the pool; a worker walks its
        // group with WG_WORKER_DEPTH reads in flight, so pool-width x depth reads stay
        // outstanding
        std::vector<std::pair<int64_t, int64_t>> tasks;
        size_t group = (n_runs + 63) / 64;
        tasks.reserve((n_runs + group - 1) / group);
        for (size_t r = 0; r < n_runs; r += group)
        {
            size_t r1 = std::min(n_runs, r + group);
            tasks.emplace_back(runs[r].first, runs[r1 - 1].first + runs[r1 - 1].second);
        }
        gather_pool()->run(&ctx, tasks);
    }
    if (gather_prof())
        printf(" -- gather prof: %lld uids %zu runs | sync %ld pend %ld | tasks worker %ld caller %ld\n",
               (long long) U, n_runs,
               ctx.n_sync.load(), ctx.n_pend.load(), ctx.t_worker.load(), ctx.t_caller.load());
    TORCH_CHECK(!ctx.failed.load(), "ngram_gather_cpu: read failed");
}

#endif
