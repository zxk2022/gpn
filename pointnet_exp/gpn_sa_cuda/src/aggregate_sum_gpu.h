#ifndef AGGREGATE_SUM_GPU_H
#define AGGREGATE_SUM_GPU_H

#include <torch/torch.h>
#include <torch/extension.h>

#include <vector>

#include <cuda_runtime.h>

// CPP function declarations
at::Tensor aggregate_sum_wrapper(at::Tensor input, at::Tensor edge_index);

at::Tensor aggregate_sum_grad_wrapper(at::Tensor grad_output, at::Tensor edge_index);

// CUDA kernel function declarations
void aggregate_sum_launcher(float* output, const float* input, const int* edge_index,
                 const int batch_size, const int num_graphs_per_batch, const int num_nodes, const int feature_channels, const int num_edges_max);

void aggregate_sum_grad_launcher(float* grad_input, const float* grad_output, const int* edge_index,
                  const int batch_size, const int num_graphs_per_batch, const int num_nodes, const int feature_channels, const int num_edges_max);

#endif // aggregate_GPU_H