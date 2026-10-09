"""CPU fault injection for the actual isolated Graph/helper/BC capture code.

This compiles selected production C++ methods against a fake CUDA runtime.
It verifies ordering and exception cleanup, not CUDA kernels or ABI compatibility.
"""
import pathlib
import os
import subprocess
import tempfile
import unittest

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.nogpu

ROOT = pathlib.Path(__file__).resolve().parent
EXT = pathlib.Path(os.environ.get("EXLLAMAV3_ENGINE_ROOT", pathlib.Path(__file__).resolve().parents[2])).resolve() / "exllamav3/exllamav3_ext"


def braced(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 1
    pos = opening + 1
    while depth:
        depth += (source[pos] == "{") - (source[pos] == "}")
        pos += 1
    return source[start:pos]


PRELUDE = r'''
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <map>
#include <thread>
#include <cstdlib>
using cudaError_t=int;
using cudaStream_t=void*;
using cudaGraph_t=void*;
using cudaGraphExec_t=void*;
using cudaGraphNode_t=void*;
using CUgraphNode=void*;
using cublasHandle_t=void*;
using PPTR=std::tuple<int,void*>;
enum cudaGraphNodeType {cudaGraphNodeTypeKernel};
struct cudaKernelNodeParams {void* func; void** kernelParams;};
struct CUDA_KERNEL_NODE_PARAMS {void* func; void** kernelParams;};
constexpr int cudaSuccess=0, CUDA_SUCCESS=0, CUBLAS_STATUS_SUCCESS=0;
constexpr int cudaStreamNonBlocking=1,cudaStreamCaptureModeThreadLocal=1;
constexpr int CUBLAS_POINTER_MODE_HOST=0,CUBLAS_TF32_TENSOR_OP_MATH=3,CUBLAS_DEFAULT_MATH=0;
constexpr int GP_end=0;
constexpr size_t WORKSPACE_SIZE=16*1024*1024;
std::string fail_at;
std::vector<std::string> events;
bool capturing=false, warmed=false, tf32=false, no_tf32=false;
int streams=0,graphs=0,execs=0,torch_workspaces=0,current_device=1;
void* kernel=(void*)0x80;
int call(const char* name) {events.push_back(name);if(fail_at==name){fail_at.clear();return 99;}return 0;}
#define AT_CUDA_CHECK(expr) do {int x=(expr);if(x)throw std::runtime_error(#expr);}while(0)
#define TORCH_CUDABLAS_CHECK(expr) AT_CUDA_CHECK(expr)
#define TORCH_CHECK(test,...) do {if(!(test))throw std::runtime_error("TORCH_CHECK");}while(0)
#define cuda_check_drv(expr) AT_CUDA_CHECK(expr)
int cudaDeviceSynchronize(){return call("sync");}
int cudaGetDevice(int* d){int r=call("get_device");*d=current_device;return r;}
int cudaStreamCreateWithFlags(void** p,int){int r=call("create_stream");if(!r){*p=(void*)0x10;streams++;}return r;}
int cudaStreamBeginCapture(void*,int){int r=call("begin");if(!r)capturing=true;return r;}
int cudaStreamEndCapture(void*,void** g){int r=call("end");capturing=false;if(!r){*g=(void*)0x20;graphs++;}else *g=nullptr;return r;}
int cudaStreamDestroy(void*){int r=call("destroy_stream");if(!r)streams--;return r;}
int cudaGraphDestroy(void*){graphs--;return call("destroy_graph");}
int cudaGraphExecDestroy(void*){execs--;return call("destroy_exec");}
int cudaGraphInstantiate(void** e,void*,void*,void*,int){int r=call("instantiate");if(!r){*e=(void*)0x30;execs++;}return r;}
int cudaGraphGetNodes(void*,void** n,size_t* count){int r=call(n?"nodes_fill":"nodes_count");if(!r){*count=1;if(n)n[0]=(void*)0x40;}return r;}
int cudaGraphNodeGetType(void*,cudaGraphNodeType* t){*t=cudaGraphNodeTypeKernel;return call("node_type");}
int cudaGraphKernelNodeGetParams(void*,cudaKernelNodeParams* p){p->func=kernel;return call("node_params");}
int cudaGetLastError(){return 0;}
struct CudaDrv {
  static CudaDrv& instance(){static CudaDrv d;return d;}
  int graph_kernel_node_get_params(void*,CUDA_KERNEL_NODE_PARAMS* p){p->func=kernel;return call("driver_params");}
};
struct DevCtx {
  static DevCtx& instance(){static DevCtx c;return c;}
  void* get_ws(int d){assert(d==current_device);if(call("workspace"))throw std::runtime_error("workspace");return (void*)0x60;}
};
namespace at {
enum class Float32Backend {CUDA}; enum class Float32Op {MATMUL}; enum class Float32Precision {TF32,NONE};
struct NoTF32Guard {static bool should_disable_tf32(){return no_tf32;}};
struct Context {Float32Precision float32Precision(Float32Backend,Float32Op){return tf32?Float32Precision::TF32:Float32Precision::NONE;}};
Context& globalContext(){static Context c;return c;}
namespace cuda {
void* getCurrentCUDABlasHandle(bool setup=true){
  if(call("reserve_handle"))throw std::runtime_error("reserve_handle");
  if(!warmed){if(capturing)throw std::runtime_error("cublasCreate inside capture");warmed=true;}
  if(setup)torch_workspaces++;
  return (void*)0x50;
}
}}
int math_mode=-1;
int cublasSetMathMode(void*,int m){math_mode=m;return call("math_mode");}
int cublasSetStream(void*,void*){return call("set_stream");}
int cublasSetPointerMode(void*,int){return call("pointer_mode");}
int cublasSetWorkspace(void*,void*,size_t sz){assert(sz==WORKSPACE_SIZE);return call("set_workspace");}
'''

TESTS = r'''
void clean(Graph& g){assert(!capturing);assert(!g.capture_stream);assert(!g.capture_active);assert(!g.ready);assert(!g.graph);assert(!g.graph_exec);assert(streams==0&&graphs==0&&execs==0);}
void reset(){assert(streams==0&&graphs==0&&execs==0);events.clear();fail_at.clear();capturing=false;warmed=false;torch_workspaces=0;tf32=false;no_tf32=false;}
template<class F>void throws(F f){bool saw=false;try{f();}catch(const std::runtime_error&){saw=true;}assert(saw);}
void record(Graph& g){g.graph_sites.emplace_back(kernel,GP_end,0,8);}
int main(int argc,char** argv){
  if(argc>1){Graph g;assert(g.disabled==(std::string(argv[1])=="1"));std::cout<<"Graph switch checked\n";return 0;}
  int checks=0;
  for(const char* stage:{"sync","create_stream","begin"}){
    reset();Graph g;fail_at=stage;throws([&]{g.capture_begin();});clean(g);checks++;
  }
  reset();{Graph g;g.capture_begin();throws([&]{g.capture_begin();});g.capture_abort();clean(g);checks++;}
  for(const char* stage:{"end","instantiate","nodes_count","nodes_fill","node_type","destroy_stream"}){
    reset();Graph g;g.capture_begin();record(g);fail_at=stage;throws([&]{g.capture_end();});clean(g);checks++;
  }
  reset();{Graph g;g.capture_begin();record(g);fail_at="node_params";g.capture_end();assert(g.ready&&!g.capture_stream&&!g.capture_active);g.capture_abort();clean(g);checks++;}
  reset();{Graph g;g.capture_begin();g.capture_end();assert(g.ready);g.capture_abort();clean(g);checks++;}
  reset();{Graph g;g.capture_begin();g.graph_sites.emplace_back((void*)0x999,GP_end,0,8);throws([&]{g.capture_end();});clean(g);checks++;}
  reset();{Graph g;g.capture_begin();record(g);g.capture_abort();g.capture_abort();clean(g);g.capture_begin();record(g);g.capture_end();assert(g.ready);checks++;}
  reset();{Graph g;try {GraphCapture guard(g);throw std::runtime_error("kernel");}catch(const std::runtime_error&){}clean(g);checks++;}
  reset();{Graph g;{GraphCapture guard(g);record(g);guard.finish();}assert(g.ready);g.capture_abort();clean(g);checks++;}
  std::cout<<checks<<" CPU C++ fault-injection checks passed\n";
}
'''


class NativeHardeningTests(unittest.TestCase):
    def test_all_capture_call_sites_have_scopes(self):
        callers = list((EXT / "libtorch").glob("*.cpp"))
        scopes = 0
        for path in callers:
            text = path.read_text()
            self.assertNotIn(".capture_begin(", text, path.name)
            self.assertNotIn("->capture_begin(", text, path.name)
            count = text.count("GraphCapture capture(")
            self.assertEqual(count, text.count("capture.finish();"), path.name)
            scopes += count
        self.assertEqual(scopes, 9)

    def test_actual_cpp_methods_against_fake_cuda(self):
        source = (EXT / "graph.cu").read_text()
        header = (EXT / "graph.cuh").read_text()
        declaration = header[header.index("class Graph"):]
        methods = source[source.index("Graph::Graph()") : source.index("void Graph::launch")]
        if "static bool graphs_disabled()" in source:
            methods = braced(source, "static bool graphs_disabled()") + "\n" + methods
        helper = ""
        branch = ""
        mock_run = ""
        program = PRELUDE + declaration + helper + methods + mock_run + TESTS
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory = pathlib.Path(directory)
            cpp = directory / "fault_injection.cpp"
            cpp.write_text(program)
            built = subprocess.run([os.environ.get("CXX", "clang++"), "-std=c++17", "-Wall", "-Wextra", str(cpp), "-o", str(directory / "fault_injection")], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            result = subprocess.run([str(directory / "fault_injection")], check=True, capture_output=True, text=True)
            print(result.stdout.strip())
            if "static bool graphs_disabled()" in source:
                for flag, expected in (("0", "1"), ("1", "0")):
                    subprocess.run([str(directory / "fault_injection"), expected], env={**os.environ, "EXL3_GRAPHS": flag}, check=True, capture_output=True, text=True)
                print("2 independent-process EXL3_GRAPHS switch checks passed")


if __name__ == "__main__":
    unittest.main()
