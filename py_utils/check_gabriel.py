import math
from tqdm import tqdm

def read_obj_file(filename):
    vertices = []
    edges = []
    
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):  # 顶点
                coords = line.split()[1:]
                vertices.append(tuple(map(float, coords)))
            elif line.startswith('l '):  # 边
                edge = tuple(sorted(map(int, line.split()[1:])))  # 排序以确保边的一致性
                edges.append(edge)
                
    return vertices, edges

def distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def is_gabriel_edge(vertices, edge):
    p1, p2 = [vertices[i-1] for i in edge]  # 调整索引，因为OBJ文件中的索引是从1开始的
    mid = [(a + b) / 2 for a, b in zip(p1, p2)]
    radius = distance(p1, p2) / 2
    
    for v in vertices:
        if v not in (p1, p2) and distance(v, mid) < radius:
            return False
    return True

def check_gabriel_edges(filename):
    vertices, edges = read_obj_file(filename)
    gabriel_edges = set()  # 使用集合来避免重复边
    non_gabriel_edges = []
    duplicate_edges = set()
    
    total_edges = len(edges)
    
    for edge in tqdm(edges, desc='Checking Edges', unit='edge'):
        if edge in gabriel_edges:  # 检查是否是重复边
            duplicate_edges.add(edge)
        elif is_gabriel_edge(vertices, edge):
            gabriel_edges.add(edge)
        else:
            non_gabriel_edges.append(edge)
    
    print("Total number of edges input to the program:", total_edges)
    print("Number of Gabriel edges:", len(gabriel_edges))
    print("Number of Non-Gabriel edges:", len(non_gabriel_edges))
    print("Number of duplicate edges:", len(duplicate_edges))

# 使用函数
filename = 'output_pcd/[0]/0_rng.obj'  # 替换为你的文件路径
check_gabriel_edges(filename)