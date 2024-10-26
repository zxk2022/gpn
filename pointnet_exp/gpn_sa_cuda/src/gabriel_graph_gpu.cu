#include <torch/extension.h>        // 包含 PyTorch C++ 扩展所需的头文件
#include <cuda.h>                   // 包含 CUDA 的头文件
#include <cuda_runtime.h>           // 包含 CUDA 运行时的头文件

#include "gabriel_graph_gpu.h"

#include "cuda_utils.h"

// 计算两个三维点之间的欧几里得距离的平方
__device__ float euclidean_dist_squared(float3 p1, float3 p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) +
           (p1.y - p2.y) * (p1.y - p2.y) +
           (p1.z - p2.z) * (p1.z - p2.z);
}

// CUDA 核函数，用于计算 Gabriel 图的边
__global__ void gabriel_graph_kernel(const float* __restrict__ points, int* __restrict__ edge_index,
                                     int B, int G, int N, int D, int max_edges) {
    int batch_idx = blockIdx.y;        // 获取当前批次的索引
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;  // 获取当前组的索引

    if (batch_idx >= B || group_idx >= G) return;        // 如果点的索引超出范围，则退出

    int edge_count = 0;  // 初始化边计数器

    for (int point_idx = 0; point_idx < N; ++point_idx){
        float3 p1 = make_float3(points[(batch_idx * G * N * D) + (group_idx * N * D) + (point_idx * D) + 0],
                                points[(batch_idx * G * N * D) + (group_idx * N * D) + (point_idx * D) + 1],
                                points[(batch_idx * G * N * D) + (group_idx * N * D) + (point_idx * D) + 2]);

        for (int i = 0; i < N; ++i) {      // 遍历所有点
            // if (i == point_idx) continue;  // 跳过自身

            float3 p2 = make_float3(points[(batch_idx * G * N * D) + (group_idx * N * D) + (i * D) + 0],
                                    points[(batch_idx * G * N * D) + (group_idx * N * D) + (i * D) + 1],
                                    points[(batch_idx * G * N * D) + (group_idx * N * D) + (i * D) + 2]);

            // 计算两个点之间的欧几里得距离的平方
            float dist_squared = euclidean_dist_squared(p1, p2);
            // 计算两个点的中心点坐标
            float3 c = make_float3((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2);
            bool is_gabriel_edge = true;   // 初始化为 Gabriel 边的标志

            // 遍历所有点，检查是否有其他点在 p1 和 p2 的中间
            for (int j = 0; j < N; ++j) {
                if (j == point_idx || j == i) continue;  // 跳过自身和另一个点

                float3 p3 = make_float3(points[(batch_idx * G * N * D) + (group_idx * N * D) + (j * D) + 0],
                                        points[(batch_idx * G * N * D) + (group_idx * N * D) + (j * D) + 1],
                                        points[(batch_idx * G * N * D) + (group_idx * N * D) + (j * D) + 2]);
                // 如果有点与中心点坐标的距离小于半径，则不是 Gabriel 边
                if (euclidean_dist_squared(c, p3) <= dist_squared / 4) {
                    is_gabriel_edge = false;
                    break;
                }
            }
            
            // 如果是 Gabriel 边，并且边计数器没有超出最大值，则记录边
            if (is_gabriel_edge) {
                int index = edge_count;
                edge_count++;
                if (index < max_edges) {
                    edge_index[(batch_idx * G * max_edges * 2) + (group_idx * max_edges * 2) + (index * 2) + 0] = point_idx;
                    edge_index[(batch_idx * G * max_edges * 2) + (group_idx * max_edges * 2) + (index * 2) + 1] = i;
                }
            }
        }
    }
}

// 用于调用 CUDA 核函数并返回计算得到的 Gabriel 图的边索引
torch::Tensor gabriel_graph_launcher(torch::Tensor points, int max_edges) {
    const auto B = points.size(0);  // 获取批次数
    const auto G = points.size(1);  // 获取组数
    const auto N = points.size(2);  // 获取点数
    const auto D = points.size(3);  // 获取维度数

    // 创建一个用于存储边索引的张量，初始化为零
    auto edge_index = torch::zeros({B, G, max_edges, 2}, torch::TensorOptions().dtype(torch::kInt32).device(points.device()));

    dim3 blocks(DIVUP(G, THREADS_PER_BLOCK), B);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    // 调用 CUDA 核函数
    gabriel_graph_kernel<<<blocks, threads>>>(points.data_ptr<float>(), edge_index.data_ptr<int>(), B, G, N, D, max_edges);

    return edge_index;  // 返回边索引张量
}
