#include "aggregate_sum_gpu.h"

#include "cuda_utils.h"

__global__ void aggregate_sum_kernel(
    float* output, const float* input, const int* edge_index,
    const int batch_size, const int num_graphs_per_batch, const int num_nodes, const int feature_channels, const int num_edges_max)
{
    // 线程索引
    int batch_idx = blockIdx.y;
    int graph_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || graph_idx >= num_graphs_per_batch) return;

    for (int node_idx = 0; node_idx < num_nodes; ++node_idx) {

        // 取空数组
        float* node_output = &output[batch_idx * num_graphs_per_batch * num_nodes * feature_channels + graph_idx * num_nodes * feature_channels + node_idx * feature_channels];
        for (int c = 0; c < feature_channels; ++c) {
            node_output[c] = 0.0f;
        }

        // break signal for find the end of the edge_index
        int i = 1;
        // 消息传递
        for (int e = 0; e < num_edges_max; ++e) {
            int src_node = edge_index[batch_idx * num_graphs_per_batch * num_edges_max * 2 + graph_idx * num_edges_max * 2 + e * 2];
            int dst_node = edge_index[batch_idx * num_graphs_per_batch * num_edges_max * 2 + graph_idx * num_edges_max * 2 + e * 2 + 1];
            if (src_node == dst_node && src_node == 0){
                i = i - 1;
                if (i == -1){
                    break;
                }
            }
            if (dst_node == node_idx) {
                const float* src_node_features = &input[batch_idx * num_graphs_per_batch * num_nodes * feature_channels + graph_idx * num_nodes * feature_channels + src_node * feature_channels];
                for (int c = 0; c < feature_channels; ++c) {
                    // 此处求和聚合
                    node_output[c] += src_node_features[c];
                }
            }
        }
    }  
}

void aggregate_sum_launcher(float* output, const float* input, const int* edge_index,
                 const int batch_size, const int num_graphs_per_batch, const int num_nodes, const int feature_channels, const int num_edges_max)
{
    dim3 blocks(DIVUP(num_graphs_per_batch, THREADS_PER_BLOCK), batch_size);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    // 启动核函数
    aggregate_sum_kernel<<<blocks, threads>>>(output, input, edge_index, batch_size, num_graphs_per_batch, num_nodes, feature_channels, num_edges_max);

    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void aggregate_sum_grad_kernel(
    float* d_input, const float* d_output, const int* edge_index,
    const int batch_size, const int num_graphs_per_batch, const int num_nodes, const int feature_channels, const int num_edges_max)
{
    // 线程索引
    int batch_idx = blockIdx.y;
    int graph_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || graph_idx >= num_graphs_per_batch) return;

    for (int node_idx = 0; node_idx < num_nodes; ++node_idx) {

        // 初始化节点梯度
        float* d_node_input = &d_input[batch_idx * num_graphs_per_batch * num_nodes * feature_channels + graph_idx * num_nodes * feature_channels + node_idx * feature_channels];
        for (int c = 0; c < feature_channels; ++c) {
            d_node_input[c] = 0.0f;
        }

        // break signal for find the end of the edge_index
        int i = 1;
        // 反向传播
        for (int e = 0; e < num_edges_max; ++e) {
            int src_node = edge_index[batch_idx * num_graphs_per_batch * num_edges_max * 2 + graph_idx * num_edges_max * 2 + e * 2];
            int dst_node = edge_index[batch_idx * num_graphs_per_batch * num_edges_max * 2 + graph_idx * num_edges_max * 2 + e * 2 + 1];
            if (src_node == dst_node && src_node == 0){
                i = i - 1;
                if (i == -1){
                    break;
                }
            }
            // 如果 dst_node 是 node_idx，那么 src_node 的输入梯度应该加上 dst_node 输出梯度的贡献
            if (dst_node == node_idx) {
                const float* d_dst_output = &d_output[batch_idx * num_graphs_per_batch * num_nodes * feature_channels + graph_idx * num_nodes * feature_channels + dst_node * feature_channels];
                float* d_src_input = &d_input[batch_idx * num_graphs_per_batch * num_nodes * feature_channels + graph_idx * num_nodes * feature_channels + src_node * feature_channels];
                for (int c = 0; c < feature_channels; ++c) {
                    // 注意：此处使用加法聚合
                    d_src_input[c] += d_dst_output[c];
                }
            }
        }
    }
}

void aggregate_sum_grad_launcher(float* grad_input, const float* grad_output, const int* edge_index,
                  const int batch_size, const int num_graphs_per_batch, const int num_nodes, const int feature_channels, const int num_edges_max)
{
    dim3 blocks(DIVUP(num_graphs_per_batch, THREADS_PER_BLOCK), batch_size);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    // 启动核函数
    aggregate_sum_grad_kernel<<<blocks, threads>>>(grad_input, grad_output, edge_index, batch_size, num_graphs_per_batch, num_nodes, feature_channels, num_edges_max);

    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}
