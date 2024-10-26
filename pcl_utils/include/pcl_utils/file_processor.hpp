#ifndef FILE_PROCESSOR_HPP
#define FILE_PROCESSOR_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <atomic>
#include <string>
#include <filesystem>
#include <future>

namespace fs = std::filesystem;

namespace pcl_utils {

void process_pcd_file(const fs::path& pcd_file_path, const fs::path& output_dir, std::atomic_size_t& progress_counter, size_t total_files, const std::string& graph_type, bool use_cuda, std::promise<void> promise);

}

#endif // FILE_PROCESSOR_HPP