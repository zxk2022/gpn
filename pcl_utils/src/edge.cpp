#include "pcl_utils/edge.hpp"

void pcl_utils::remove_invalid_edges(std::vector<Edge>& edges) {
    // 使用 erase-remove idiom 移除所有无效的边
    edges.erase(
        std::remove_if(edges.begin(), edges.end(), [](const Edge& e) {
            return e.index1 == 0 && e.index2 == 0; // 检查是否为无效边
        }),
        edges.end()
    );
}