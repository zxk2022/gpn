import os
import pickle
import numpy as np

def load_dat_file(file_path):
    """
    Load preprocessed data from a .dat file.

    Args:
        file_path (str): Path to the .dat file.

    Returns:
        tuple: A tuple containing the list of point clouds and the list of labels.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'rb') as f:
        list_of_points, list_of_labels = pickle.load(f)

    return list_of_points, list_of_labels

def save_point_cloud_as_pcd(point_cloud, label, output_dir, index):
    """
    Save a single point cloud as a .pcd file.

    Args:
        point_cloud (numpy.ndarray): The point cloud data.
        label (int or str): The label associated with the point cloud.
        output_dir (str): Directory where to save the .pcd files.
        index (int): Index used to name the file uniquely.
    """
    # 确保标签是纯数字
    label = label.item()
    
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
    # 指定 .dat 文件的路径
    file_path = 'data/ModelNet10/raw/modelnet10_test_1024pts_fps.dat'
    
    # 指定输出目录
    output_dir = 'output_pcd'

    # 加载数据
    list_of_points, list_of_labels = load_dat_file(file_path)
    
    # 遍历所有点云并将它们保存为 .pcd 文件
    for i, (point_cloud, label) in enumerate(zip(list_of_points, list_of_labels)):
        save_point_cloud_as_pcd(point_cloud, label, output_dir, i)

if __name__ == "__main__":
    main()