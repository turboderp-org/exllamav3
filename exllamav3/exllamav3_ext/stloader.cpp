#include "stloader.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <Python.h>
#include "util.h"
#include "stloader_cu.cuh"
#include <cerrno>
#include <cstring>

uint8_t* host_buffers[STLOADER_THREADS] = {nullptr};

void stloader_read
(
    std::vector<uintptr_t> handles,
    size_t offset,
    size_t size,
    at::Tensor target
)
{
    at::Device device = target.device();
    bool target_cpu = device.is_cpu();
    cudaStream_t stream;

    // Buffers

    uint8_t* load_buffer;
    uint8_t* cuda_buffer;
    if (target_cpu)
    {
        load_buffer = (uint8_t*) target.data_ptr();
        cuda_buffer = nullptr;
    }
    else
    {
        load_buffer = (uint8_t*) malloc(size);
        TORCH_CHECK(load_buffer, "Can't allocate buffer for tensor");
        cuda_buffer = (uint8_t*) target.data_ptr();
        cudaSetDevice(device.index());
        stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
    }

    // Synchronization

    Py_BEGIN_ALLOW_THREADS

    volatile bool load_failed = false;
    volatile int load_errnum = 0;
    std::mutex mtx;
    std::deque<std::pair<size_t, size_t>> dq;
    std::condition_variable cv;

    // Load chunks

    auto load_worker = [&] (size_t pos_a, int thread_idx)
    {
        FILE* file = reinterpret_cast<FILE*>(handles[thread_idx]);

        while (pos_a < size && !load_failed)
        {
            size_t pos_b = pos_a + STLOADER_BLOCK_SIZE;
            if (pos_b > size) pos_b = size;

            #ifdef __linux__
                ssize_t br = pread(fileno(file), load_buffer + pos_a, pos_b - pos_a, offset + pos_a);
                if (br != pos_b - pos_a) goto error;
            #else
                int sr = _fseeki64(file, static_cast<__int64>(offset + pos_a), SEEK_SET);
                if (sr) goto error;
                size_t br = fread(load_buffer + pos_a, 1, pos_b - pos_a, file);
                if (br != pos_b - pos_a) goto error;
            #endif

            {
                std::lock_guard<std::mutex> lock(mtx);
                dq.push_back(std::pair<size_t, size_t>(pos_a, pos_b));
                cv.notify_one();
            }

            pos_a += STLOADER_THREADS * STLOADER_BLOCK_SIZE;
        }

        return;

        error:
        if (file && ferror(file))
            printf(" ## Error reading file: %s (errno: %d)\n", strerror(errno), errno);
        load_errnum = errno;
        load_failed = true;
    };

    // Copy chunks to device

    auto copy_worker = [&] ()
    {
        cudaSetDevice(device.index());

        size_t total_blocks = CEIL_DIVIDE(size, STLOADER_BLOCK_SIZE);
        while (total_blocks && !load_failed)
        {
            size_t pos_a, pos_b;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&dq] { return !dq.empty(); });

                auto pop = dq.front();
                dq.pop_front();
                total_blocks--;
                pos_a = std::get<0>(pop);
                pos_b = std::get<1>(pop);

                while (!dq.empty() && std::get<0>(dq.front()) == pos_b)
                {
                    pop = dq.front();
                    dq.pop_front();
                    pos_b = std::get<1>(pop);
                    total_blocks--;
                }
            }

            cudaError_t cr = cudaMemcpyAsync
            (
                cuda_buffer + pos_a,
                load_buffer + pos_a,
                pos_b - pos_a,
                cudaMemcpyHostToDevice,
                stream
            );

            if (cr != cudaSuccess)
            {
                fprintf(stderr, " ## GPUassert: %s\n", cudaGetErrorString(cr));
                goto error;
            }
        }
        return;

        error:
        load_errnum = errno;
        load_failed = true;
    };

    std::vector<std::thread> threads;
    if (cuda_buffer)
        threads.emplace_back(copy_worker);
    for (size_t i = 0; i < STLOADER_THREADS && i * STLOADER_BLOCK_SIZE < size; ++i)
        threads.emplace_back(load_worker, i * STLOADER_BLOCK_SIZE, i);
    for (auto& thread : threads)
        thread.join();

    if (load_failed)
        TORCH_CHECK
        (
            false, " ## Error reading file: ", std::strerror(load_errnum),
            " (errno=", load_errnum, ")"
        );

    if (!target_cpu)
    {
        cudaDeviceSynchronize();
        free(load_buffer);
    }

    Py_END_ALLOW_THREADS
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
    for (int i = 0; i < handles.size(); ++i)
    {
        FILE* file = reinterpret_cast<FILE*>(handles[i]);
        fclose(file);

        if (host_buffers[i])
        {
            cudaFreeHost(host_buffers[i]);
            host_buffers[i] = nullptr;
        }
    }

}

void stloader_deferred_cpu(std::vector<TensorLoadJob> const& jobs)
{
    Py_BEGIN_ALLOW_THREADS
    volatile bool load_failed = false;
    volatile int load_errnum = 0;

    auto load_worker = [&] (int base_index)
    {
        int index = base_index;
        while (index < jobs.size())
        {
            TensorLoadJob const& job = jobs[index];
            FILE* file = reinterpret_cast<FILE*>(job.handles[base_index]);
            uint8_t* dest = reinterpret_cast<uint8_t*>(job.destination);

            #ifdef __linux__
                ssize_t br = pread(fileno(file), dest, job.bytesize, job.file_offset);
                if (br != job.bytesize) goto error;
            #else
                int sr = _fseeki64(file, static_cast<__int64>(job.file_offset), SEEK_SET);
                if (sr) goto error;
                size_t br = fread(dest, 1, job.bytesize, file);
                if (br != job.bytesize) goto error;
            #endif

            if (job.bf16_to_fp16)
                inplace_bf16_to_fp16_cpu(dest, job.bytesize / 2);

            index += STLOADER_THREADS;
        }

        return;

        error:
        load_errnum = errno;
        load_failed = true;
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < STLOADER_THREADS && i < jobs.size(); ++i)
        threads.emplace_back(load_worker, i);
    for (auto& thread : threads)
        thread.join();

    TORCH_CHECK(!load_failed, "I/O error reading tensor");

    Py_END_ALLOW_THREADS
}

// TODO: GPUDirect option
void stloader_deferred_cuda(std::vector<TensorLoadJob> const& jobs, size_t max_chunk_size)
{
    Py_BEGIN_ALLOW_THREADS
    volatile bool load_failed = false;
    volatile int load_errnum = 0;

    auto load_worker = [&] (int base_index)
    {
        cudaError_t cr;

        if (!host_buffers[base_index])
            cudaMallocHost((void **) &host_buffers[base_index], max_chunk_size);
        uint8_t* temp = host_buffers[base_index];

        int index = base_index;
        while (index < jobs.size())
        {
            TensorLoadJob const& job = jobs[index];
            FILE* file = reinterpret_cast<FILE*>(job.handles[base_index]);
            uint8_t* dest = reinterpret_cast<uint8_t*>(job.destination);

            #ifdef __linux__
                ssize_t br = pread(fileno(file), temp, job.bytesize, job.file_offset);
                if (br != job.bytesize) goto error;
            #else
                int sr = _fseeki64(file, static_cast<__int64>(job.file_offset), SEEK_SET);
                if (sr) goto error;
                size_t br = fread(temp, 1, job.bytesize, file);
                if (br != job.bytesize) goto error;
            #endif

            cudaSetDevice(job.device_id);

            cudaError_t cr = cudaMemcpyAsync(dest, temp, job.bytesize, cudaMemcpyDefault);
            cudaDeviceSynchronize();
            if (cr != cudaSuccess)
            {
                fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(cr));
                goto error;
            }

            if (job.bf16_to_fp16)
                inplace_bf16_to_fp16_cuda(dest, job.bytesize / 2);

            index += STLOADER_THREADS;
        }

        return;

        error:
        load_failed = true;
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < STLOADER_THREADS && i < jobs.size(); ++i)
        threads.emplace_back(load_worker, i);
    for (auto& thread : threads)
        thread.join();

    if (load_failed)
        TORCH_CHECK
        (
            false, " ## Error reading file: ", std::strerror(load_errnum),
            " (errno=", load_errnum, ")"
        );

    Py_END_ALLOW_THREADS
}

