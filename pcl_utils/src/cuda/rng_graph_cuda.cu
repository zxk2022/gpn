#pragma nv_diag_suppress 20012

#include "pcl_utils/cuda/rng_graph_cuda.hpp"

namespace pcl_utils_cuda {

__global__ void build_rng_graph_kernel(const PointXYZ* points, int num_points, Edge* edges, int* num_edges) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= num_points || j >= num_points || i == j || i >= j) return;

    PointXYZ point_i = points[i];
    PointXYZ point_j = points[j];

    float dist_ij = (point_i - point_j).norm();

    bool is_rng_edge = true;

    for (int k = 0; k < num_points; ++k) {
        if (k != i && k != j) {
            PointXYZ point_k = points[k];
            float dist_ik = (point_i - point_k).norm();
            float dist_jk = (point_j - point_k).norm();

            if (dist_ik < dist_ij && dist_jk < dist_ij) {
                is_rng_edge = false;
                break;
            }
        }
    }

    if (is_rng_edge) {
        int idx = atomicAdd(num_edges, 1);
        edges[idx] = {i, j};
    }
}

}

void build_rng_graph_kernel_launcher(const pcl::PointCloud<pcl::PointXYZ>& points, std::vector<pcl_utils::Edge>& edges) {
    int num_points = points.size();

    // 将点云数据传输到GPU
    thrust::device_vector<pcl_utils_cuda::PointXYZ> d_points(num_points);
    for (int i = 0; i < num_points; ++i) {
        d_points[i] = pcl_utils_cuda::PointXYZ(points[i].x, points[i].y, points[i].z);
    }

    // 分配GPU内存
    thrust::device_vector<pcl_utils_cuda::Edge> d_edges(num_points * (num_points - 1) / 2);
    int* d_num_edges;
    cudaMalloc(&d_num_edges, sizeof(int));
    cudaMemset(d_num_edges, 0, sizeof(int)); // 初始化为0

    // 定义网格和块大小
    dim3 blockSize(16, 16);
    dim3 gridSize((num_points + blockSize.x - 1) / blockSize.x, (num_points + blockSize.y - 1) / blockSize.y);

    // 启动CUDA内核
    build_rng_graph_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_points.data()), num_points, thrust::raw_pointer_cast(d_edges.data()), d_num_edges);

    // 获取结果
    int num_edges;
    cudaMemcpy(&num_edges, d_num_edges, sizeof(int), cudaMemcpyDeviceToHost);
    edges.resize(num_edges);

    // 将结果从设备复制到主机
    thrust::host_vector<pcl_utils_cuda::Edge> h_edges(num_edges);
    thrust::copy(d_edges.begin(), d_edges.begin() + num_edges, h_edges.begin());

    // 显式类型转换，pcl_utils_cuda::Edge -> pcl_utils::Edge(此处两者结构体定义相同，否则需要定义thrust::transform)
    for (int i = 0; i < num_edges; ++i) {
        edges[i] = reinterpret_cast<const pcl_utils::Edge&>(h_edges[i]);
    }

    // 释放GPU内存
    cudaFree(d_num_edges);
}