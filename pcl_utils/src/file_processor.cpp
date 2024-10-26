#include "pcl_utils/file_processor.hpp"
#include "pcl_utils/edge.hpp"
#include "pcl_utils/cuda/gabriel_graph_cuda.hpp"
#include "pcl_utils/cuda/rng_graph_cuda.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


// 构建Gabriel图
std::vector<pcl_utils::Edge> build_gabriel_graph(pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree, const pcl::PointCloud<pcl::PointXYZ>& points) {
    using namespace pcl_utils;
    std::vector<Edge> edges;
    size_t num_points = points.size();
    
    for (size_t i = 0; i < num_points; ++i) {
        for (size_t j = i + 1; j < num_points; ++j) {
            pcl::PointXYZ point_i = points[i];
            pcl::PointXYZ point_j = points[j];
            
            float dist_ij = (point_i.getVector3fMap() - point_j.getVector3fMap()).norm();
            
            pcl::PointXYZ mid_point;
            mid_point.getVector3fMap() = (point_i.getVector3fMap() + point_j.getVector3fMap()) / 2.0f;
            
            std::vector<int> indices;
            std::vector<float> dists;
            kdtree.radiusSearch(mid_point, dist_ij / 2.0f, indices, dists);
            
            bool is_gabriel_edge = true;
            for (size_t k = 0; k < indices.size(); ++k) {
                if (indices[k] != i && indices[k] != j) {
                    is_gabriel_edge = false;
                    break;
                }
            }
            
            if (is_gabriel_edge) {
                edges.push_back({static_cast<int>(i), static_cast<int>(j)});
            }
        }
    }
    
    return edges;
}

// 构建RNG图
std::vector<pcl_utils::Edge> build_rng_graph(pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree, const pcl::PointCloud<pcl::PointXYZ>& points) {
    using namespace pcl_utils;
    std::vector<Edge> edges;
    size_t num_points = points.size();

    for (size_t i = 0; i < num_points; ++i) {
        for (size_t j = i + 1; j < num_points; ++j) {
            pcl::PointXYZ point_i = points[i];
            pcl::PointXYZ point_j = points[j];

            float dist_ij = (point_i.getVector3fMap() - point_j.getVector3fMap()).norm();

            // Check if there exists any point k such that dist(i,k) < dist(i,j) and dist(j,k) < dist(i,j)
            bool is_rng_edge = true;

            for (size_t k = 0; k < num_points; ++k) {
                if (k == i || k == j) continue;
                pcl::PointXYZ point_k = points[k];
                float dist_ik = (point_i.getVector3fMap() - point_k.getVector3fMap()).norm();
                float dist_jk = (point_j.getVector3fMap() - point_k.getVector3fMap()).norm();

                if (dist_ik < dist_ij && dist_jk < dist_ij) {
                    is_rng_edge = false;
                    break;
                }
            }

            if (is_rng_edge) {
                edges.push_back({static_cast<int>(i), static_cast<int>(j)});
            }
        }
    }

    return edges;
}

// 完全图 (Complete Graph)
std::vector<pcl_utils::Edge> build_complete_graph(const pcl::PointCloud<pcl::PointXYZ>& points) {
    using namespace pcl_utils;
    std::vector<Edge> edges;
    size_t num_points = points.size();

    for (size_t i = 0; i < num_points; ++i) {
        for (size_t j = i + 1; j < num_points; ++j) {
            edges.push_back({static_cast<int>(i), static_cast<int>(j)});
        }
    }

    return edges;
}

// 寻找质心
int find_central_point(const pcl::PointCloud<pcl::PointXYZ>& points) {
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    size_t num_points = points.size();

    // 计算质心
    for (const auto& point : points) {
        centroid += point.getVector3fMap();
    }
    centroid /= static_cast<float>(num_points);

    // 计算每个点到质心的距离
    float min_distance = std::numeric_limits<float>::max();
    int central_point_index = -1;

    for (size_t i = 0; i < num_points; ++i) {
        float distance = (points[i].getVector3fMap() - centroid).norm();
        if (distance < min_distance) {
            min_distance = distance;
            central_point_index = static_cast<int>(i);
        }
    }

    return central_point_index;
}


// 构建knn图
std::vector<pcl_utils::Edge> build_local_knn_graph(const pcl::PointCloud<pcl::PointXYZ>& points) {
    using namespace pcl_utils;
    std::vector<Edge> edges;
    int central_point_index = find_central_point(points);
    size_t num_points = points.size();

    for (size_t i = 0; i < num_points; ++i) {
        if (i != central_point_index) {
            edges.push_back({static_cast<int>(i), central_point_index});
        }
    }

    return edges;
}

// 构建Gabriel图(cuda)
std::vector<pcl_utils::Edge> build_gabriel_graph_cuda(const pcl::PointCloud<pcl::PointXYZ>& points) {
    std::vector<pcl_utils::Edge> edges;
    build_gabriel_graph_kernel_launcher(points, edges);
    pcl_utils::remove_invalid_edges(edges);
    return edges;
}

// 构建RNG图(cuda)
std::vector<pcl_utils::Edge> build_rng_graph_cuda(const pcl::PointCloud<pcl::PointXYZ>& points) {
    std::vector<pcl_utils::Edge> edges;
    build_rng_graph_kernel_launcher(points, edges);
    pcl_utils::remove_invalid_edges(edges);
    return edges;
}

namespace pcl_utils {

void process_pcd_file(const fs::path& pcd_file_path, const fs::path& output_dir, std::atomic_size_t& progress_counter, size_t total_files, const std::string& graph_type, bool use_cuda, std::promise<void> promise) {
    try {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file_path.string(), *cloud) == -1) {
            PCL_ERROR("Couldn't read file %s\n", pcd_file_path.c_str());
            return;
        }

        // 构建图
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<Edge> edges;

        if (use_cuda) {
            if (graph_type == "gabriel") {
                edges = build_gabriel_graph_cuda(*cloud);
            } else if (graph_type == "rng") {
                edges = build_rng_graph_cuda(*cloud);
            } else {
                throw std::invalid_argument("Invalid graph type: " + graph_type + ". It must be 'gabriel' or 'rng'.");
            }
        }else {
            // 构建KDTree
            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            kdtree.setInputCloud(cloud);

            if (graph_type == "gabriel") {
                edges = build_gabriel_graph(kdtree, *cloud);
            } else if (graph_type == "rng") {
                edges = build_rng_graph(kdtree, *cloud);
            } else if (graph_type == "complete") {
                edges = build_complete_graph(*cloud); // 调用完全图生成函数
            } else if (graph_type == "knn") {
                edges = build_local_knn_graph(*cloud); // 调用局部K近邻图生成函数
            } else {
                throw std::invalid_argument("Invalid graph type: " + graph_type + ". It must be 'gabriel', 'rng', 'complete', or 'knn'.");
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << graph_type + "图构建耗时: " << duration << " ms" << std::endl;

        // 生成输出文件路径
        fs::path output_file_path = output_dir / (pcd_file_path.stem().string() + "_" + graph_type + ".obj");

        // 保存点云及边到OBJ文件
        std::ofstream obj_file(output_file_path);
        
        // 写入点
        for (const auto& point : cloud->points) {
            obj_file << "v " << point.x << " " << point.y << " " << point.z << "\n";
        }

        // 写入边
        for (const auto& edge : edges) {
            obj_file << "l " << edge.index1 + 1 << " " << edge.index2 + 1 << "\n"; // OBJ索引从1开始
        }

        obj_file.close();
        std::cout << "点云及边已保存至 '" << output_file_path << "'" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error processing file " << pcd_file_path << ": " << e.what() << std::endl;
    }

    // 更新进度计数器
    size_t current_progress = ++progress_counter;
    if (current_progress <= total_files) { // 防止溢出
        std::cout << "Processed file " << current_progress << " of " << total_files << std::endl;
    }

    // 任务完成
    promise.set_value();
}

}