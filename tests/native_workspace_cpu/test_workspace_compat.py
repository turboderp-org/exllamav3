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
  for(const char* stage:{"reserve_handle","set_stream","pointer_mode","get_device","workspace","set_workspace"}){
    reset();fail_at=stage;throws([&]{exl3_cublas_handle((void*)0x10);});assert(!capturing);checks++;
  }
  reset();exl3_cublas_handle((void*)0x10);assert(torch_workspaces==EXPECTED_WORKSPACES);checks++;
  if (EXPECTED_WORKSPACES==0) {fail_at="math_mode";throws([&]{exl3_cublas_handle((void*)0x10);});checks++;}
#if FAST_HANDLE
  tf32=true;exl3_cublas_handle((void*)0x10);assert(math_mode==CUBLAS_TF32_TENSOR_OP_MATH);no_tf32=true;exl3_cublas_handle((void*)0x10);assert(math_mode==CUBLAS_DEFAULT_MATH);checks++;
#endif
  reset();{Slot s;capture_bc(s,1);assert(warmed&&s.runs==2&&s.graph->ready);assert(events[0]=="reserve_handle");checks++;}
  reset();{Slot s;fail_at="reserve_handle";throws([&]{capture_bc(s,1);});clean(*s.graph);assert(s.runs==1);checks++;}
  reset();{Slot s;throw_run=true;throws([&]{capture_bc(s,1);});throw_run=false;clean(*s.graph);assert(s.runs==1);checks++;}
  reset();{Slot s;capture_bc(s,0);assert(!warmed&&s.runs==2);checks++;}
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
        helper = (EXT / "cublas_handle.cuh").read_text()
        helper = helper[helper.index("inline cublasHandle_t"):]
        branch = braced((EXT / "libtorch/mla_attention.cpp").read_text(), "if (!s.graph->ready)")
        mock_run = r'''
struct Slot {std::shared_ptr<Graph> graph=std::make_shared<Graph>();int runs=1;};
bool throw_run=false;
template<class... Args>void run_gr(Args...){if(throw_run)throw std::runtime_error("kernel launch failed");}
void capture_bc(Slot& s,int idx_mode){
int bsz=1,q_len=1,x=0,y=0,cache_seqlens=0,block_table=0,position=0,positions=0,position_ids=0,regime=0,t_total=0,ext_indices=0;
cudaStream_t stream=(void*)0x70;
'''
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            for major, minor, rocm in ((2,6,False),(2,11,False),(2,12,False),(2,14,False),(2,15,False),(3,0,False),(2,14,True)):
                fast = major==2 and 12<=minor<=14 and not rocm
                prelude = PRELUDE
                if not fast:
                    # The fallback's headers intentionally omit all new precision API names.
                    start = prelude.index("enum class Float32Backend")
                    end = prelude.index("namespace cuda {", start)
                    prelude = prelude[:start]+prelude[end:]
                    prelude = prelude.replace("getCurrentCUDABlasHandle(bool setup=true)", "getCurrentCUDABlasHandle()")
                    prelude = prelude.replace("if(setup)torch_workspaces++;", "torch_workspaces++;")
                flags = f"#define TORCH_VERSION_MAJOR {major}\n#define TORCH_VERSION_MINOR {minor}\n#define EXPECTED_WORKSPACES {0 if fast else 1}\n#define FAST_HANDLE {1 if fast else 0}\n"
                if rocm: flags += "#define USE_ROCM 1\n"
                program = flags + prelude + declaration + helper + methods + mock_run + branch + "\n}\n" + TESTS
                cpp = directory / "fault_injection.cpp"
                cpp.write_text(program)
                built = subprocess.run([os.environ.get("CXX", "clang++"), "-std=c++17", "-Wall", "-Wextra", str(cpp), "-o", str(directory/"probe")], capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, f"Torch {major}.{minor} ROCm={rocm}: {built.stderr}")
                result = subprocess.run([str(directory/"probe")], check=True, capture_output=True, text=True)
                print(f"Torch {major}.{minor} ROCm={rocm}: {result.stdout.strip()}")
                for flag, expected in (("0", "1"), ("1", "0")):
                    subprocess.run([str(directory/"probe"), expected], env={**os.environ,"EXL3_GRAPHS":flag}, check=True, capture_output=True, text=True)
                env = {k:v for k,v in os.environ.items() if k!="EXL3_GRAPHS"}
                subprocess.run([str(directory/"probe"), "1" if rocm else "0"], env=env, check=True, capture_output=True, text=True)



if __name__ == "__main__":
    unittest.main()
