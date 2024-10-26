#ifndef PCL_UTILS_POINT_XYZ_HPP
#define PCL_UTILS_POINT_XYZ_HPP

#include <cmath>

namespace pcl_utils_cuda {

    struct PointXYZ {
        float x, y, z;

        // 无参数的默认构造函数
        __host__ __device__ PointXYZ() : x(0), y(0), z(0) {}

        // 带参数的构造函数
        __host__ __device__ PointXYZ(float x, float y, float z) : x(x), y(y), z(z) {}

        // 计算范数
        __host__ __device__ float norm() const {
            return sqrt(x * x + y * y + z * z);
        }

        // 减法运算符重载
        __host__ __device__ PointXYZ operator-(const PointXYZ& other) const {
            return PointXYZ(x - other.x, y - other.y, z - other.z);
        }

        // 加法运算符重载
        __host__ __device__ PointXYZ operator+(const PointXYZ& other) const {
            return PointXYZ(x + other.x, y + other.y, z + other.z);
        }

        // 除法运算符重载
        __host__ __device__ PointXYZ operator/(float scalar) const {
            return PointXYZ(x / scalar, y / scalar, z / scalar);
        }
    };

} // namespace pcl_utils_cuda

#endif  // PCL_UTILS_POINT_XYZ_HPP