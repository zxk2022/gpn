#include "message_passing_sum_gpu.h"

#define CHECK_CUDA(x) do { \
    if (!x.is_cuda()) { \
        fprintf(stderr, "%s must be a CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
        exit(-1); \
    } \
} while (0)

#define CHECK_CONTIGUOUS(x) do { \
    if (!x.is_contiguous()) { \
        fprintf(stderr, "%s must be a contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
        exit(-1); \
    } \
} while (0)

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor message_passing_sum_wrapper(at::Tensor input, at::Tensor edge_index) {
    CHECK_INPUT(input);
    CHECK_INPUT(edge_index);
    
    const auto batch_size = input.size(0);
    const auto num_graphs_per_batch = input.size(1);
    const auto num_nodes = input.size(2);
    const auto feature_channels = input.size(3);

    const auto num_edges_max = edge_index.size(2);

    // Allocate output tensor
    at::Tensor output_tensor = at::empty({batch_size, num_graphs_per_batch, num_nodes, feature_channels}, input.options());

    // Convert Tensors to raw pointers and launch CUDA kernel
    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output_tensor.data_ptr<float>();
    const int* edge_index_ptr = edge_index.data_ptr<int>();

    message_passing_sum_launcher(output_ptr, input_ptr, edge_index_ptr,
                             batch_size, num_graphs_per_batch, num_nodes, feature_channels, num_edges_max);

    return output_tensor;  // Return success
}

at::Tensor message_passing_sum_grad_wrapper(at::Tensor grad_output, at::Tensor edge_index) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(edge_index);

    const auto batch_size = grad_output.size(0);
    const auto num_graphs_per_batch = grad_output.size(1);
    const auto num_nodes = grad_output.size(2);
    const auto feature_channels = grad_output.size(3);

    const auto num_edges_max = edge_index.size(2);

    // Allocate gradient input tensor
    at::Tensor grad_input_tensor = at::empty({batch_size, num_graphs_per_batch, num_nodes, feature_channels}, grad_output.options());

    // Convert Tensors to raw pointers and launch CUDA kernel
    float* grad_input_ptr = grad_input_tensor.data_ptr<float>();
    const float* grad_output_ptr = grad_output.data_ptr<float>();
    const int* edge_index_ptr = edge_index.data_ptr<int>();

    message_passing_sum_grad_launcher(grad_input_ptr, grad_output_ptr, edge_index_ptr,
                                  batch_size, num_graphs_per_batch, num_nodes, feature_channels, num_edges_max);

    return grad_input_tensor;  // Return success
}