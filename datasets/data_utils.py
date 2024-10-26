import torch

def parse_obj(file_path):
    vertices = []
    edges = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
            elif line.startswith('l '):
                edge = list(map(int, line.strip().split()[1:]))  # 转换为0-based索引
                edges.extend([(min(e1, e2), max(e1, e2)) for e1, e2 in zip(edge[:-1], edge[1:])])
                # 处理线的最后一段
                if len(edge) > 2:
                    edges.append((edge[-1], edge[0]))

    # 构建边索引
    edge_index = torch.zeros((2, len(edges)), dtype=torch.long)
    for i, (u, v) in enumerate(edges):
        edge_index[0, i] = u - 1  # 调整为0-based索引
        edge_index[1, i] = v - 1

    # 构建顶点位置张量
    pos = torch.tensor(vertices, dtype=torch.float)

    return pos, edge_index
