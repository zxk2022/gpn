import os
import time
import torch
from torch_cluster import knn
import open3d as o3d
import numpy as np

# 读取PCD文件
def read_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points

# 将点云和k-近邻关系写入OBJ文件
def write_obj(file_path, points, row, col):
    with open(file_path, 'w') as f:
        # 写入顶点
        for point in points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")
        
        # 写入边
        for i in range(len(row)):
            f.write(f"l {row[i] + 1} {col[i] + 1}\n")

# 处理单个PCD文件
def process_pcd_file(file_path, file_index, total_files, k):
    start_time = time.time()
    
    points = read_pcd(file_path)
    
    # 转换为PyTorch张量
    points_tensor = torch.tensor(points, dtype=torch.float32)
    
    # 使用knn函数计算每个点的k个最近邻居
    batch_x = torch.zeros(points_tensor.shape[0], dtype=torch.long)  # 所有点属于同一个batch
    batch_y = batch_x  # 查询点与数据点属于同一个batch
    row, col = knn(points_tensor, points_tensor, k, batch_x, batch_y)
    
    # 构造输出文件路径
    output_file_path = os.path.splitext(file_path)[0] + '_knn.obj'
    
    # 将结果写入OBJ文件
    write_obj(output_file_path, points, row.numpy(), col.numpy())
    
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒
    print(f"knn图构建耗时: {elapsed_time:.1f} ms")
    print(f"点云及边已保存至 '{output_file_path}'")
    print(f"Processed file {file_index} of {total_files}")
    print()

# 遍历目录及其子目录中的所有PCD文件
def process_directory(root_dir, k):
    pcd_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pcd'):
                pcd_files.append(os.path.join(dirpath, filename))
    
    total_files = len(pcd_files)
    print(f"Found {total_files} PCD files in the directory tree.")
    
    for index, file_path in enumerate(pcd_files, start=1):
        process_pcd_file(file_path, index, total_files, k)

# 主函数
def main():
    root_dir = 'data/ScanObjectNN'  # 指定要处理的根目录
    k = 40
    process_directory(root_dir, k)

if __name__ == "__main__":
    main()