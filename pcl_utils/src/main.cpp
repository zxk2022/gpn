#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <future>
#include <chrono>
#include <atomic>
#include <filesystem>
#include "pcl_utils/file_processor.hpp"
#include "pcl_utils/thread_pool.hpp"
#include "pcl_utils/util.hpp"


using namespace pcl_utils;
namespace fs = std::filesystem;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " directory_path graph_type [thread_count] [--use_cuda]" << std::endl;
    std::cout << "graph_type must be 'gabriel', 'rng', 'complete', or 'knn'" << std::endl;
    std::cout << "If --use_cuda is specified, the program will use CUDA and run in single-threaded mode." << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        print_usage(argv[0]);
        return -1;
    }

    fs::path directory_path = argv[1];
    if (!fs::exists(directory_path) || !fs::is_directory(directory_path)) {
        std::cerr << "Directory does not exist or is not a directory: " << directory_path << std::endl;
        return -1;
    }

    std::string graph_type = argv[2];
    if (graph_type != "gabriel" && graph_type != "rng" && graph_type != "complete" && graph_type != "knn") {
        std::cerr << "Invalid graph type: " << graph_type << ". Graph type must be 'gabriel', 'rng', 'complete', or 'knn'" << std::endl;
        return -1;
    }

    size_t thread_count = 1;
    bool use_cuda = false;

    if (argc >= 4) {
        if (std::string(argv[3]) == "--use_cuda") {
            use_cuda = true;
        } else {
            try {
                thread_count = std::stoul(argv[3]);
                if (thread_count == 0) {
                    throw std::invalid_argument("Thread count must be greater than 0");
                }
            } catch (const std::exception& e) {
                std::cerr << "Invalid thread count: " << e.what() << ". Using default single-threaded mode." << std::endl;
                thread_count = 1;
            }
        }
    }

    if (argc == 5) {
        if (std::string(argv[4]) != "--use_cuda") {
            std::cerr << "Invalid argument: " << argv[4] << ". Expected '--use_cuda'." << std::endl;
            return -1;
        }
        use_cuda = true;
    }

    std::vector<fs::path> pcd_files = scan_directory_for_pcd_files(directory_path);

    std::cout << "Found " << pcd_files.size() << " PCD files in the directory tree." << std::endl;

    if (use_cuda) {
        std::atomic_size_t progress_counter(0);

        for (const auto& pcd_file : pcd_files) {
            std::promise<void> promise;
            auto future = promise.get_future();
            process_pcd_file(pcd_file, pcd_file.parent_path(), std::ref(progress_counter), pcd_files.size(), graph_type, true, std::move(promise));
            future.get(); // Wait for the task to complete
        }
    } else {
        ThreadPool pool(thread_count);
        std::atomic_size_t progress_counter(0);

        std::vector<std::future<void>> futures;

        for (const auto& pcd_file : pcd_files) {
            std::promise<void> promise;
            futures.push_back(promise.get_future());
            pool.enqueue([pcd_file, output_dir = pcd_file.parent_path(), &progress_counter, total_files = pcd_files.size(), graph_type, promise = std::move(promise)] () mutable {
                process_pcd_file(pcd_file, output_dir, std::ref(progress_counter), total_files, graph_type, false, std::move(promise));
            });
        }

        for (auto& future : futures) {
            future.get();
        }
    }

    std::cout << std::endl << "All PCD files processed successfully." << std::endl;

    return 0;
}