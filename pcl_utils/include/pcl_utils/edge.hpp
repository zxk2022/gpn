#ifndef EDGE_HPP
#define EDGE_HPP

#include <vector>
#include <algorithm>
#include <iostream>

namespace pcl_utils {

    struct Edge {
        int index1;
        int index2;
    };

    void remove_invalid_edges(std::vector<Edge>& edges);
}

#endif // EDGE_HPP