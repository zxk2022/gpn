import os
import h5py
import numpy as np

def save_point_cloud_as_pcd(point_cloud, label, output_dir, index):
    """
    Save a single point cloud as a .pcd file.

    Args:
        point_cloud (numpy.ndarray): The point cloud data.
        label (int or str): The label associated with the point cloud.
        output_dir (str): Directory where to save the .pcd files.
        index (int): Index used to name the file uniquely.
    """
    # 创建标签对应的子文件夹
    label_dir = os.path.join(output_dir, str(label))
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    filename = os.path.join(label_dir, f'{index}.pcd')
    
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {len(point_cloud)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(point_cloud)}
DATA ascii
"""
    
    with open(filename, 'w') as f:
        f.write(header)
        for vertex in point_cloud:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

def main():
    # 指定 HDF5 文件的路径
    file_path = 'data/h5_files/main_split/test_objectdataset.h5'
    
    # 指定输出目录
    output_dir = 'pcd_test'

    # 打开HDF5文件
    with h5py.File(file_path, 'r') as f:
        # 获取数据集
        data = f['data'][:]
        labels = f['label'][:]
        
        # 遍历所有点云并将它们保存为 .pcd 文件
        for i, (point_cloud, label) in enumerate(zip(data, labels)):
            save_point_cloud_as_pcd(point_cloud, label, output_dir, i)

if __name__ == "__main__":
    main()