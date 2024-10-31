import os
import numpy as np

def read_off_file(filepath):
    """
    Read a single .off file and return its point cloud data.

    Args:
        filepath (str): Path to the .off file.

    Returns:
        numpy.ndarray: The point cloud data.
    """
    with open(filepath, 'r') as f:
        if f.readline().strip() != 'OFF':
            raise ValueError("Not a valid OFF file")
        
        n_vertices, _, _ = map(int, f.readline().split())
        vertices = []
        
        for _ in range(n_vertices):
            vertex = list(map(float, f.readline().split()))
            vertices.append(vertex)
        
    return np.array(vertices)

def save_point_cloud_as_pcd(point_cloud, output_filepath):
    """
    Save a single point cloud as a .pcd file.

    Args:
        point_cloud (numpy.ndarray): The point cloud data.
        output_filepath (str): Path to save the .pcd file.
    """
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
    
    with open(output_filepath, 'w') as f:
        f.write(header)
        for vertex in point_cloud:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

def main():
    # 指定输入文件夹
    input_dir = 'data/SHAPENET_SKEL/raw'

    # 遍历输入文件夹中的所有 .off 文件
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('center.off'):
                filepath = os.path.join(root, filename)
                point_cloud = read_off_file(filepath)
                
                # 构建输出文件路径
                output_filepath = os.path.join(root, os.path.splitext(filename)[0] + '.pcd')
                
                save_point_cloud_as_pcd(point_cloud, output_filepath)

if __name__ == "__main__":
    main()