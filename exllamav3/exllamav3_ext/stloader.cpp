#include "stloader.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <condition_variable>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <Python.h>
#include "util.h"
#include "stloader_cu.cuh"
#include <cerrno>
#include <cstring>

// A run is one contiguous file read covering one or more jobs. Small gaps between jobs (tensors
// in the same region that load elsewhere, e.g. to a different device) are read through and
// discarded rather than breaking the run: a few KB of dead read is much cheaper than losing
// sequential access. Batches with cross-tensor parallelism (deferred loads) use slot-size runs;
// single-tensor reads use smaller runs so one tensor still spreads across all workers.
#define STLOADER_MAX_MERGE_GAP (512*1024)
#define STLOADER_DIRECT_RUN (1024*1024)

struct LoadRun
{
    size_t first_job;
    size_t num_jobs;
    size_t file_base;
    size_t span;
};

// ---------------------------------------------------------------------------------------------
// Shared state for the persistent CUDA load pool. The engine mutex serializes engine calls (the
// pinned pool is not partitionable between callers, and concurrent loads would just thrash the
// disk anyway), so the per-batch globals below are only ever owned by one call at a time.
// Workers are spawned once and kept (detached) for the process lifetime; each keeps its slot
// ring state, streams and events across batches, avoiding thread spawn/join and stream/event
// churn per call.

// The sync primitives are deliberately leaked: detached workers wait on them for the process
// lifetime, and destroying a condition variable with waiters at static-destruction time
// deadlocks process exit
static std::mutex& engine_mutex = *(new std::mutex());

// Pinned staging pool: STLOADER_THREADS rings of STLOADER_SLOTS_PER_THREAD slots. Allocated
// lazily on first CUDA load, freed when file handles are closed (workers are idle then).
static uint8_t* pinned_pool = nullptr;

static std::mutex& pool_mtx = *(new std::mutex());
static std::condition_variable& pool_cv_work = *(new std::condition_variable());
static std::condition_variable& pool_cv_done = *(new std::condition_variable());
static bool pool_started = false;
static uint64_t pool_generation = 0;
static int pool_workers_done = 0;

// Per-batch state, owned by the engine call
static const std::vector<TensorLoadJob>* batch_jobs = nullptr;
static const std::vector<LoadRun>* batch_runs = nullptr;
static std::atomic<size_t> batch_next_run;
static std::atomic<bool> batch_failed;
static std::mutex& batch_err_mtx = *(new std::mutex());
static std::string& batch_err = *(new std::string());

static void batch_fail(std::string msg)
{
    std::lock_guard<std::mutex> lock(batch_err_mtx);
    if (batch_err.empty()) batch_err = std::move(msg);
    batch_failed.store(true, std::memory_order_release);
}

// Read a contiguous file range into buf with the platform's positioned-read primitive. Returns
// false and sets errnum on failure.
static bool read_range(FILE* file, size_t file_offset, size_t bytesize, uint8_t* buf, int& errnum)
{
    #ifdef __linux__
        ssize_t br = pread(fileno(file), buf, bytesize, file_offset);
        if (br != (ssize_t) bytesize) { errnum = errno; return false; }
    #else
        int sr = _fseeki64(file, static_cast<__int64>(file_offset), SEEK_SET);
        if (sr) { errnum = errno; return false; }
        size_t br = fread(buf, 1, bytesize, file);
        if (br != bytesize) { errnum = errno; return false; }
    #endif
    return true;
}

// Worker loop: cycle through the thread's ring of pinned slots. Wait on the slot's event (only
// that slot's copy, not the whole device), read up to a slot's worth of adjacent jobs in one
// pread/fread, issue async H2D copies and any bf16 conversions on the worker's own non-blocking
// stream, record the slot event, move on. The next read overlaps the previous slots' copies.
// Each worker drains its streams before reporting done, so batch completion implies all copies
// have landed.
static void pool_worker_main(int thread_idx)
{
    cudaEvent_t events[STLOADER_SLOTS_PER_THREAD] = {};
    int event_device[STLOADER_SLOTS_PER_THREAD] = {};
    bool pending[STLOADER_SLOTS_PER_THREAD] = {};
    std::vector<std::pair<int, cudaStream_t>> streams;
    int cur_slot = 0;
    int cur_device = -1;
    uint64_t seen_generation = 0;

    while (true)
    {
        const std::vector<TensorLoadJob>* jobs;
        const std::vector<LoadRun>* runs;
        {
            std::unique_lock<std::mutex> lock(pool_mtx);
            pool_cv_work.wait(lock, [&] { return pool_generation != seen_generation; });
            seen_generation = pool_generation;
            jobs = batch_jobs;
            runs = batch_runs;
        }

        uint8_t* ring = pinned_pool + (size_t) thread_idx * STLOADER_SLOTS_PER_THREAD * STLOADER_SLOT_SIZE;
        cudaError_t cr;

        size_t run_i;
        while (!batch_failed.load(std::memory_order_acquire) && (run_i = batch_next_run.fetch_add(1)) < runs->size())
        {
            const LoadRun& run = (*runs)[run_i];
            const TensorLoadJob& job0 = (*jobs)[run.first_job];
            uint8_t* buf = ring + (size_t) cur_slot * STLOADER_SLOT_SIZE;

            // The slot's previous copy must land before the buffer is overwritten
            if (pending[cur_slot])
            {
                cr = cudaEventSynchronize(events[cur_slot]);
                if (cr != cudaSuccess) { batch_fail(std::string("stloader: event sync failed: ") + cudaGetErrorString(cr)); break; }
                pending[cur_slot] = false;
            }

            FILE* file = reinterpret_cast<FILE*>(job0.handles[thread_idx]);
            int errnum = 0;
            if (!read_range(file, run.file_base, run.span, buf, errnum))
            {
                batch_fail(std::string("stloader: error reading file: ") + std::strerror(errnum) +
                           " (errno=" + std::to_string(errnum) + ")");
                break;
            }

            if (job0.device_id != cur_device)
            {
                cudaSetDevice(job0.device_id);
                cur_device = job0.device_id;
            }

            cudaStream_t stream = nullptr;
            for (auto& p : streams)
                if (p.first == cur_device) { stream = p.second; break; }
            if (!stream)
            {
                cr = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
                if (cr != cudaSuccess) { batch_fail(std::string("stloader: stream create failed: ") + cudaGetErrorString(cr)); break; }
                streams.push_back({cur_device, stream});
            }

            for (size_t ji = run.first_job; ji < run.first_job + run.num_jobs; ++ji)
            {
                const TensorLoadJob& job = (*jobs)[ji];
                cr = cudaMemcpyAsync
                (
                    reinterpret_cast<void*>(job.destination),
                    buf + (job.file_offset - run.file_base),
                    job.bytesize,
                    cudaMemcpyHostToDevice,
                    stream
                );
                if (cr == cudaSuccess && job.bf16_to_fp16)
                    cr = inplace_bf16_to_fp16_cuda(reinterpret_cast<void*>(job.destination), job.bytesize / 2, stream);
                if (cr != cudaSuccess)
                {
                    batch_fail(std::string("stloader: H2D copy failed: ") + cudaGetErrorString(cr));
                    break;
                }
            }
            if (batch_failed.load(std::memory_order_acquire)) break;

            // Events are device-bound: recreate on device change (in practice a batch stays on
            // one device)
            if (events[cur_slot] && event_device[cur_slot] != cur_device)
            {
                cudaEventDestroy(events[cur_slot]);
                events[cur_slot] = nullptr;
            }
            if (!events[cur_slot])
            {
                cr = cudaEventCreateWithFlags(&events[cur_slot], cudaEventDisableTiming);
                if (cr != cudaSuccess) { batch_fail(std::string("stloader: event create failed: ") + cudaGetErrorString(cr)); break; }
                event_device[cur_slot] = cur_device;
            }
            cr = cudaEventRecord(events[cur_slot], stream);
            if (cr != cudaSuccess) { batch_fail(std::string("stloader: event record failed: ") + cudaGetErrorString(cr)); break; }
            pending[cur_slot] = true;
            cur_slot = (cur_slot + 1) % STLOADER_SLOTS_PER_THREAD;
        }

        // Drain this worker's outstanding copies and conversions before reporting done
        for (auto& p : streams)
        {
            cudaSetDevice(p.first);
            cr = cudaStreamSynchronize(p.second);
            if (cr != cudaSuccess)
                batch_fail(std::string("stloader: stream sync failed: ") + cudaGetErrorString(cr));
        }
        for (int i = 0; i < STLOADER_SLOTS_PER_THREAD; ++i)
            pending[i] = false;

        {
            std::lock_guard<std::mutex> lock(pool_mtx);
            pool_workers_done++;
        }
        pool_cv_done.notify_all();
    }
}

// Core CUDA load engine. Jobs must arrive sorted by file offset (per file) so adjacent tensors
// and chunks of split tensors coalesce into single reads of up to max_run_bytes.
//
// Runs without the GIL and must not throw: returns an error message, empty on success.
static std::string stloader_cuda_engine(std::vector<TensorLoadJob> const& jobs, size_t max_run_bytes)
{
    std::lock_guard<std::mutex> guard(engine_mutex);

    if (jobs.empty())
        return "";

    if (max_run_bytes > STLOADER_SLOT_SIZE)
        max_run_bytes = STLOADER_SLOT_SIZE;

    if (!pinned_pool)
    {
        cudaError_t cr = cudaMallocHost
        (
            (void**) &pinned_pool,
            (size_t) STLOADER_THREADS * STLOADER_SLOTS_PER_THREAD * STLOADER_SLOT_SIZE
        );
        if (cr != cudaSuccess)
        {
            pinned_pool = nullptr;
            return std::string("stloader: failed to allocate pinned pool: ") + cudaGetErrorString(cr);
        }
    }

    // Coalesce jobs into runs
    std::vector<LoadRun> runs;
    for (size_t i = 0; i < jobs.size(); ++i)
    {
        const TensorLoadJob& job = jobs[i];
        if (job.bytesize > STLOADER_SLOT_SIZE)
            return "stloader: job larger than staging slot";
        bool merge = false;
        if (!runs.empty())
        {
            const TensorLoadJob& prev = jobs[i - 1];
            const LoadRun& r = runs.back();
            size_t prev_end = prev.file_offset + prev.bytesize;
            merge = prev.handles[0] == job.handles[0]
                && prev.device_id == job.device_id
                && job.file_offset >= prev_end
                && job.file_offset - prev_end <= STLOADER_MAX_MERGE_GAP
                && job.file_offset + job.bytesize - r.file_base <= max_run_bytes;
        }
        if (merge)
        {
            LoadRun& r = runs.back();
            r.num_jobs++;
            r.span = job.file_offset + job.bytesize - r.file_base;
        }
        else runs.push_back({i, 1, job.file_offset, job.bytesize});
    }

    if (getenv("EXL3_STLOADER_DEBUG"))
    {
        size_t bytes = 0, span = 0;
        for (const TensorLoadJob& job : jobs) bytes += job.bytesize;
        for (auto& r : runs) span += r.span;
        fprintf(stderr, "stloader: %zu jobs -> %zu runs, %zu bytes read for %zu wanted (%.2f%% amplification)\n",
                jobs.size(), runs.size(), span, bytes, 100.0 * (double)(span - bytes) / (double)bytes);
    }

    // Destination tensors were allocated stream-ordered by the caller's framework; sync each
    // involved device once so they are safe to write from the loader's own streams
    int restore_device = -1;
    cudaGetDevice(&restore_device);
    std::vector<int> devices;
    for (const TensorLoadJob& job : jobs)
        if (std::find(devices.begin(), devices.end(), job.device_id) == devices.end())
            devices.push_back(job.device_id);
    for (int dev : devices)
    {
        cudaSetDevice(dev);
        cudaError_t cr = cudaDeviceSynchronize();
        if (cr != cudaSuccess)
        {
            cudaSetDevice(restore_device);
            return std::string("stloader: device sync failed: ") + cudaGetErrorString(cr);
        }
    }
    cudaSetDevice(restore_device);

    // Reset batch state and wake the pool
    batch_failed.store(false, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(batch_err_mtx);
        batch_err.clear();
    }
    batch_next_run.store(0, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(pool_mtx);
        batch_jobs = &jobs;
        batch_runs = &runs;
        pool_workers_done = 0;
        if (!pool_started)
        {
            for (int i = 0; i < STLOADER_THREADS; ++i)
                std::thread(pool_worker_main, i).detach();
            pool_started = true;
        }
        pool_generation++;
    }
    pool_cv_work.notify_all();

    // Wait for all workers to finish and drain
    {
        std::unique_lock<std::mutex> lock(pool_mtx);
        pool_cv_done.wait(lock, [] { return pool_workers_done == STLOADER_THREADS; });
    }

    if (batch_failed.load(std::memory_order_acquire))
    {
        std::lock_guard<std::mutex> lock(batch_err_mtx);
        return batch_err.empty() ? "stloader: unknown error" : batch_err;
    }
    return "";
}

void stloader_read
(
    std::vector<uintptr_t> handles,
    size_t offset,
    size_t size,
    at::Tensor target
)
{
    at::Device device = target.device();
    TORCH_CHECK(target.is_contiguous(), "target must be contiguous");
    TORCH_CHECK(target.nbytes() >= size, "target too small for requested read");

    // CUDA target: chunk into jobs and run through the pinned-ring engine. Runs are capped
    // below slot size here so a single medium-size tensor still spreads across the worker pool
    if (!device.is_cpu())
    {
        std::vector<TensorLoadJob> jobs;
        uint8_t* dst = (uint8_t*) target.data_ptr();
        for (size_t pos = 0; pos < size; pos += STLOADER_DIRECT_RUN)
        {
            size_t chunk = std::min((size_t) STLOADER_DIRECT_RUN, size - pos);
            jobs.push_back(TensorLoadJob
            {
                handles,
                offset + pos,
                chunk,
                reinterpret_cast<uintptr_t>(dst + pos),
                false,
                false,
                true,
                (int) device.index()
            });
        }
        std::string err;
        Py_BEGIN_ALLOW_THREADS
        err = stloader_cuda_engine(jobs, STLOADER_DIRECT_RUN);
        Py_END_ALLOW_THREADS
        TORCH_CHECK(err.empty(), err);
        return;
    }

    // CPU target: threaded strided reads directly into the destination buffer
    uint8_t* load_buffer = (uint8_t*) target.data_ptr();

    volatile bool load_failed = false;
    volatile int load_errnum = 0;

    Py_BEGIN_ALLOW_THREADS

    auto load_worker = [&] (size_t pos_a, int thread_idx)
    {
        FILE* file = reinterpret_cast<FILE*>(handles[thread_idx]);

        while (pos_a < size && !load_failed)
        {
            size_t pos_b = pos_a + STLOADER_BLOCK_SIZE;
            if (pos_b > size) pos_b = size;

            int errnum = 0;
            if (!read_range(file, offset + pos_a, pos_b - pos_a, load_buffer + pos_a, errnum))
            {
                load_errnum = errnum;
                load_failed = true;
                return;
            }

            pos_a += STLOADER_THREADS * STLOADER_BLOCK_SIZE;
        }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < STLOADER_THREADS && i * STLOADER_BLOCK_SIZE < size; ++i)
        threads.emplace_back(load_worker, i * STLOADER_BLOCK_SIZE, i);
    for (auto& thread : threads)
        thread.join();

    Py_END_ALLOW_THREADS

    if (load_failed)
        TORCH_CHECK
        (
            false, " ## Error reading file: ", std::strerror(load_errnum),
            " (errno=", load_errnum, ")"
        );
}

std::vector<uintptr_t> stloader_open_file(const char* filename)
{
    std::vector<uintptr_t> handles;
    for (int i = 0; i < STLOADER_THREADS; ++i)
    {
        FILE* file = fopen(filename, "rb");
        if (!file)
        {
            int errnum = errno;
            TORCH_CHECK(
                false, " ## Error opening file '", filename, "': ",
                std::strerror(errnum), " (errno=", errnum, ")"
            );
        }
        handles.push_back(reinterpret_cast<uintptr_t>(file));
    }
    return handles;
}

void stloader_close_file(std::vector<uintptr_t> handles)
{
    for (size_t i = 0; i < handles.size(); ++i)
    {
        FILE* file = reinterpret_cast<FILE*>(handles[i]);
        fclose(file);
    }

    // Workers are idle between engine calls, so the staging pool can be released here; it is
    // re-allocated lazily by the next CUDA load
    std::lock_guard<std::mutex> guard(engine_mutex);
    if (pinned_pool)
    {
        cudaFreeHost(pinned_pool);
        pinned_pool = nullptr;
    }
}

void stloader_deferred_cpu(std::vector<TensorLoadJob> const& jobs)
{
    volatile bool load_failed = false;
    volatile int load_errnum = 0;

    Py_BEGIN_ALLOW_THREADS

    auto load_worker = [&] (int base_index)
    {
        size_t index = base_index;
        while (index < jobs.size() && !load_failed)
        {
            TensorLoadJob const& job = jobs[index];
            FILE* file = reinterpret_cast<FILE*>(job.handles[base_index]);
            uint8_t* dest = reinterpret_cast<uint8_t*>(job.destination);

            int errnum = 0;
            if (!read_range(file, job.file_offset, job.bytesize, dest, errnum))
            {
                load_errnum = errnum;
                load_failed = true;
                return;
            }

            if (job.bf16_to_fp16)
                inplace_bf16_to_fp16_cpu(dest, job.bytesize / 2);

            index += STLOADER_THREADS;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < STLOADER_THREADS && (size_t) i < jobs.size(); ++i)
        threads.emplace_back(load_worker, i);
    for (auto& thread : threads)
        thread.join();

    Py_END_ALLOW_THREADS

    if (load_failed)
        TORCH_CHECK
        (
            false, " ## Error reading file: ", std::strerror(load_errnum),
            " (errno=", load_errnum, ")"
        );
}

// TODO: GPUDirect option
void stloader_deferred_cuda(std::vector<TensorLoadJob> const& jobs, size_t max_chunk_size)
{
    TORCH_CHECK(max_chunk_size <= STLOADER_SLOT_SIZE, "stloader: max_chunk_size exceeds staging slot size");
    std::string err;
    Py_BEGIN_ALLOW_THREADS
    err = stloader_cuda_engine(jobs, STLOADER_SLOT_SIZE);
    Py_END_ALLOW_THREADS
    TORCH_CHECK(err.empty(), err);
}
