#include "pcl_utils/util.hpp"

namespace pcl_utils {

std::vector<fs::path> scan_directory_for_pcd_files(const fs::path& directory_path) {
    std::vector<fs::path> pcd_files;
    for (const auto& entry : fs::recursive_directory_iterator(directory_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".pcd") {
            pcd_files.push_back(entry.path());
        }
    }
    return pcd_files;
}

}