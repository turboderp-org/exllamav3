#include <Python.h>
#include <cstring>
#include "graph.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
//#include <torch/extension.h>
#include "util.h"
#include "util.cuh"
#include "cuda_drv.h"
#include "quant/exl3_devctx.cuh"
#include "cublas_handle.cuh"

//#define GRAPHDEBUG 1

// EXL3_GRAPHS=0 disables graph capture everywhere: every graphed site then runs its C++ launch sequence
// eagerly on each call (the same kernels in the same order, as on the uncaptured first run). Default off
// on ROCm: the HIP runtime launches a graph node no faster than a plain kernel, so replay only saves the
// per-kernel host submission, which is within noise of the per-graph submission; meanwhile every
// instantiated exec holds a 2 MB device kernarg pool that grows with each parameter update until the
// exec is destroyed. On CUDA the graphs are kept, where they are measurably faster
static bool graphs_disabled()
{
    static const bool v = [] {
        const char* e = getenv("EXL3_GRAPHS");
#if defined(USE_ROCM)
        return !(e && *e == '1');
#else
        return e && *e == '0';
#endif
    }();
    return v;
}

Graph::Graph()
{
    ready = false;
    ready_to_record = false;
    disabled = graphs_disabled();
    capture_stream = nullptr;
    capture_active = false;
    graph = NULL;
    graph_exec = NULL;
    need_cublas = false;
}

Graph::~Graph()
{
    capture_abort();
}

cudaStream_t Graph::capture_begin()
{
    TORCH_CHECK(!capture_stream && !capture_active, "Graph: capture already active");
    // Surface an earlier asynchronous fault before starting a new capture.
    AT_CUDA_CHECK(cudaDeviceSynchronize());
    try
    {
        AT_CUDA_CHECK(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking));
        AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal));
        capture_active = true;
        return capture_stream;
    }
    catch (...)
    {
        capture_abort();
        throw;
    }
}

void Graph::capture_abort() noexcept
{
    // End even an invalidated capture. Preserve the original exception if cleanup fails.
    if (capture_stream)
    {
        if (capture_active)
        {
            cudaGraph_t abandoned = nullptr;
            (void) cudaStreamEndCapture(capture_stream, &abandoned);
            capture_active = false;
            if (abandoned && abandoned != graph) (void) cudaGraphDestroy(abandoned);
        }
        (void) cudaStreamDestroy(capture_stream);
        capture_stream = nullptr;
    }
    if (graph_exec) (void) cudaGraphExecDestroy(graph_exec);
    if (graph) (void) cudaGraphDestroy(graph);
    graph_exec = nullptr;
    graph = nullptr;
    ready = false;
    ready_to_record = false;
    need_cublas = false;
    graph_sites.clear();
    graph_node_sites.clear();
    nodes.clear();
    node_params.clear();
    node_params_drv.clear();
    node_is_driver.clear();
    current_values.clear();
    node_needs_update.clear();
}

void Graph::capture_end()
{
    TORCH_CHECK(capture_stream && capture_active, "Graph: no active capture");
    try
    {
        // End capture. An invalidated capture is still ended by this call.
        cudaError_t end_status = cudaStreamEndCapture(capture_stream, &graph);
        capture_active = false;
        AT_CUDA_CHECK(end_status);
        AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        //inspect_graph();

        // Get graph nodes
        size_t num_nodes;
        AT_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &num_nodes));
        nodes.resize(num_nodes);
        AT_CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &num_nodes));

        // Store copies of all node param structures
        node_params.resize(num_nodes);
        node_params_drv.resize(num_nodes);
        node_is_driver.resize(num_nodes);
        node_needs_update.resize(num_nodes);
        for (int i = 0; i < num_nodes; ++i)
            node_needs_update[i] = false;

        int n = 0;
        int c = 0;
        while (c < graph_sites.size())
        {
            TORCH_CHECK(n < num_nodes, "Graph recording failed");
            cudaGraphNodeType t{};
            AT_CUDA_CHECK(cudaGraphNodeGetType(nodes[n], &t));

            // Node type: kernel
            if (t == cudaGraphNodeTypeKernel)
            {
                // Nodes captured from driver-API launches (Triton cubins) can't be read through the
                // runtime API; fall back to the driver API for those. The func handle recorded via
                // record_param is the CUfunction in that case, so matching works the same way
                void* node_func;
                node_is_driver[n] = 0;
                cudaError_t e = cudaGraphKernelNodeGetParams(nodes[n], &node_params[n]);
                if (e == cudaSuccess)
                    node_func = (void*) node_params[n].func;
                else
                {
                    (void) cudaGetLastError();
                    cuda_check_drv(CudaDrv::instance().graph_kernel_node_get_params((CUgraphNode) nodes[n], &node_params_drv[n]));
                    node_is_driver[n] = 1;
                    node_func = (void*) node_params_drv[n].func;
                }

                for(; c < graph_sites.size(); c++)
                {
                    void* func = std::get<0>(graph_sites[c]);
                    if (func != node_func) break;

                    int param_id     = std::get<1>(graph_sites[c]);
                    int param_offset = std::get<2>(graph_sites[c]);
                    int param_size   = std::get<3>(graph_sites[c]);

                    graph_node_sites.push_back(std::tuple<int, int, int, int>(n, param_id, param_offset, param_size));
                    if (param_id == GP_end) { c++; break; }
                }
            }

            n++;
            if (c == graph_sites.size()) break;
            if (n == num_nodes) TORCH_CHECK(false, "Graph recording failed");
        };

        // Destroy capture stream
        AT_CUDA_CHECK(cudaStreamDestroy(capture_stream));
        capture_stream = nullptr;

        // Graph is ready
        ready = true;

        #ifdef GRAPHDEBUG
            printf("End graph capture, num_nodes=%d, graph_sites.size()=%d\n", num_nodes, graph_sites.size());
        #endif
    }
    catch (...)
    {
        capture_abort();
        throw;
    }
}

void Graph::record_param(void* kernel, int param_id, int param_offset, int size)
{
    graph_sites.push_back(std::tuple<void*, int, int, int>(kernel, param_id, param_offset, size));
}

void Graph::launch(std::vector<PPTR> params, cudaStream_t stream)
{
    if (need_cublas)
    {
        (void) exl3_cublas_handle(stream);
    }

    int p = 0;
    int n = 0;
    while (true)
    {
        if (std::get<1>(graph_node_sites[n]) == std::get<0>(params[p]))
        {
            if (std::get<0>(params[p]) != GP_end)
            {
                void* new_value  = std::get<1>(params[p]);
                int node_idx     = std::get<0>(graph_node_sites[n]);
                int param_offset = std::get<2>(graph_node_sites[n]);
                int param_size   = std::get<3>(graph_node_sites[n]);

                void** kernel_params = node_is_driver[node_idx] ?
                    node_params_drv[node_idx].kernelParams :
                    node_params[node_idx].kernelParams;
                void* p_old_value = kernel_params[param_offset];
                if (memcmp(p_old_value, &new_value, param_size))
                {
                    memcpy(p_old_value, &new_value, param_size);
                    node_needs_update[node_idx] = true;
                }
            }
            p++;
        }

        n++;
        if (p == params.size()) break;
        if (n == graph_node_sites.size()) TORCH_CHECK(false, "Graph update failed");
    }

    for (int n = 0; n < nodes.size(); ++n)
    {
        if (!node_needs_update[n]) continue;
        if (node_is_driver[n])
        {
            CUresult r = CudaDrv::instance().graph_exec_kernel_node_set_params((CUgraphExec) graph_exec, (CUgraphNode) nodes[n], &node_params_drv[n]);
            TORCH_CHECK(r == CUDA_SUCCESS, "Graph node parameter update failed (driver), CUresult ", (int) r);
        }
        else
            cuda_check(cudaGraphExecKernelNodeSetParams(graph_exec, nodes[n], &node_params[n]));
        node_needs_update[n] = false;
    }

    cuda_check(cudaGraphLaunch(graph_exec, stream));
}

void Graph::inspect_graph()
{
    // Get the number of nodes in the graph
    size_t numNodes;
    cudaGraphGetNodes(graph, nullptr, &numNodes);

    // Get the nodes in the graph
    std::vector<cudaGraphNode_t> nodes(numNodes);
    cudaGraphGetNodes(graph, nodes.data(), &numNodes);
    DBGI(nodes.size());

    // Inspect each node
    for (size_t i = 0; i < numNodes; ++i)
    {
        cudaGraphNodeType nodeType;
        cudaGraphNodeGetType(nodes[i], &nodeType);

        if (nodeType == cudaGraphNodeTypeKernel)
        {
            cudaKernelNodeParams nodeParams;
            cudaGraphKernelNodeGetParams(nodes[i], &nodeParams);
            std::cout << "Kernel node " << i << ":" << std::endl;
            std::cout << "  Function pointer: " << nodeParams.func << std::endl;
            std::cout << "  Grid dimensions: (" << nodeParams.gridDim.x << ", " << nodeParams.gridDim.y << ", " << nodeParams.gridDim.z << ")" << std::endl;
            std::cout << "  Block dimensions: (" << nodeParams.blockDim.x << ", " << nodeParams.blockDim.y << ", " << nodeParams.blockDim.z << ")" << std::endl;
            std::cout << "  Shared memory: " << nodeParams.sharedMemBytes << " bytes" << std::endl;

        } else {
            std::cout << "Node " << i << " is not a kernel node." << std::endl;
        }
    }
}

