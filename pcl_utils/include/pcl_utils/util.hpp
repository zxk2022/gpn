#ifndef UTIL_HPP
#define UTIL_HPP

#include <filesystem>
#include <vector>
#include <string>

namespace fs = std::filesystem;

namespace pcl_utils {

std::vector<fs::path> scan_directory_for_pcd_files(const fs::path& directory_path);

}

#endif // UTIL_HPP