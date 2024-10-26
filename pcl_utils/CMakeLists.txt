cmake_minimum_required(VERSION 3.12)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 vcpkg 工具链文件
set(CMAKE_TOOLCHAIN_FILE $ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

# 设置Boost版本警告
set(Boost_NO_WARN_NEW_VERSIONS TRUE)

# 指定CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 86)

# 设置资源文件
set(RESOURCE_FILES 
    src/main.cpp
    src/edge.cpp
    src/file_processor.cpp
    src/thread_pool.cpp
    src/util.cpp
    src/cuda/gabriel_graph_cuda.cu
    src/cuda/rng_graph_cuda.cu
)

# 设置生成目标的名称
set(TARGET_NAME pcl_utils)

# 项目名称
project(${TARGET_NAME} LANGUAGES CXX CUDA)

# 打印路径信息
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

# 查找pcl库
find_package(PCL CONFIG REQUIRED)

# 包含PCL的头文件目录
include_directories(${PCL_INCLUDE_DIRS})

# 包含CUDA的头文件目录
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# 包含项目头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 添加可执行文件
add_executable(${TARGET_NAME} ${RESOURCE_FILES})

# 链接 PCL 库
target_link_libraries(${TARGET_NAME} PRIVATE ${PCL_LIBRARIES})

# 链接 CUDA 库
target_link_libraries(${TARGET_NAME} PRIVATE cudart)

# # 配置安装选项
# install(TARGETS ${TARGET_NAME}
#         RUNTIME DESTINATION bin
#         LIBRARY DESTINATION lib
#         ARCHIVE DESTINATION lib)

# # 配置打包选项
# set(VERSION_MAJOR 1)
# set(VERSION_MINOR 0)

# include(InstallRequiredSystemLibraries)
# set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
# set(CPACK_PACKAGE_VERSION_MAJOR "${VERSION_MAJOR}")
# set(CPACK_PACKAGE_VERSION_MINOR "${VERSION_MINOR}")
# set(CPACK_GENERATOR "TGZ")
# set(CPACK_SOURCE_GENERATOR "TGZ")
# include(CPack)