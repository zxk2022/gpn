#ifndef PCL_UTILS_GABRIEL_GRAPH_CUDA_HPP
#define PCL_UTILS_GABRIEL_GRAPH_CUDA_HPP

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "pcl_utils/cuda/pointxyz_cuda.hpp"
#include "pcl_utils/cuda/edge_cuda.hpp"
#include "pcl_utils/edge.hpp"


void build_gabriel_graph_kernel_launcher(const pcl::PointCloud<pcl::PointXYZ>& points, std::vector<pcl_utils::Edge>& edges);


#endif  // PCL_UTILS_GABRIEL_GRAPH_CUDA_HPP