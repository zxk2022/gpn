#include "gabriel_graph_gpu.h"

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor gabriel_graph_wrapper(torch::Tensor points, int max_edges) {
    CHECK_INPUT(points);
    return gabriel_graph_launcher(points, max_edges);
}
