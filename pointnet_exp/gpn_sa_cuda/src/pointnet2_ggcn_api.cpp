#include <torch/extension.h>

#include "gabriel_graph_gpu.h"
#include "message_passing_sum_gpu.h"
#include "aggregate_sum_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gabriel_graph_wrapper", &gabriel_graph_wrapper, "Gabriel Graph Generate (CUDA)");
    m.def("mp_sum_wrapper", &message_passing_sum_wrapper, "Message Passing (CUDA)");
    m.def("mp_sum_grad_wrapper", &message_passing_sum_grad_wrapper, "Message Passing Gradient (CUDA)");
    m.def("aggregate_sum_wrapper", &aggregate_sum_wrapper, "Aggregate Sum (CUDA)");
    m.def("aggregate_sum_grad_wrapper", &aggregate_sum_grad_wrapper, "Aggregate Sum Gradient (CUDA)");
}
