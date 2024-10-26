#ifndef _GABRIEL_GRAPH_GPU_H
#define _GABRIEL_GRAPH_GPU_H

#include <torch/torch.h>  // 包含用于序列化张量的PyTorch头文件
#include <vector>  // 包含C++标准库中的vector容器
#include <cuda_runtime_api.h>  // 包含CUDA运行时API头文件

// CUDA forward declarations
torch::Tensor gabriel_graph_wrapper(torch::Tensor points, int max_edges);

torch::Tensor gabriel_graph_launcher(torch::Tensor points, int max_edges);

#endif