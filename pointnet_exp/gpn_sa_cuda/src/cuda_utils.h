/**
 * @file cuda_utils.h
 * @brief Utility functions and macros for CUDA operations.
 * 
 * This header file contains utility functions and macros to facilitate CUDA programming.
 * 
 * @details
 * - TOTAL_THREADS: Defines the total number of threads.
 * - THREADS_PER_BLOCK: Defines the number of threads per block.
 * - DIVUP(m, n): Macro to compute the ceiling of the division of m by n.
 * - opt_n_threads(int work_size): Function to compute the optimal number of threads for a given work size.
 * 
 * @note
 * The code is adapted from the repository: https://github.com/guochengqian/openpoints/blob/master/cpp/pointnet2_batch/src/cuda_utils.h
 * 
 * @param work_size The size of the work to be done.
 * @return The optimal number of threads.
 */
#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <cmath>

#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return max(min(1 << pow_2, TOTAL_THREADS), 1);
}
#endif
