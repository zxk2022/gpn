import torch
from pointnet2_ggcn import gabriel_graph_wrapper, mp_sum_wrapper, mp_sum_grad_wrapper
from torch.autograd import Function
from typing import Tuple
import time
import matplotlib.pyplot as plt
import networkx as nx

class GabrielGraph(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, num_edges_max: int) -> torch.Tensor:
        assert xyz.is_contiguous()
        return gabriel_graph_wrapper(xyz, num_edges_max)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor = None) -> Tuple[None, ...]:
        return (None,) * 4

class MessagePassing(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        assert input.is_contiguous()
        assert edge_index.is_contiguous()

        output = mp_sum_wrapper(input, edge_index)
        ctx.save_for_backward(edge_index)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        from time import time
        time_start = time()
        edge_index, = ctx.saved_tensors
        grad_input = mp_sum_grad_wrapper(grad_output, edge_index)
        print("     grad_input耗时：", time()-time_start)
        return grad_input, None

gabriel_graph = GabrielGraph.apply
message_passing = MessagePassing.apply

def test1():
    # 设置随机种子以获得可复现的结果
    torch.manual_seed(0)
    # 随机生成一组点云数据
    xyz = torch.randn(8, 6000, 32, 3).cuda()
    xyz = torch.tensor([[[[1, 1, 0],
                            [2, 2, 0],
                            [3, 3, 0],
                            [2, 3, 0]]]], dtype=torch.float32, device='cuda', requires_grad=False)
    # 生成Gabriel图的边索引
    edge_index = gabriel_graph(xyz, 512)

    points = xyz[0, 0].cpu().numpy()
    poitns_edge = edge_index[0, 0].cpu().numpy()

    # 画图
    G = nx.Graph()
    for i in range(points.shape[0]):
        G.add_node(i, pos=(points[i][0], points[i][1]))
    for i in range(poitns_edge.shape[0]):
        G.add_edge(poitns_edge[i][0], poitns_edge[i][1])

    # 绘制Gabriel图（3D）
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

    # 绘制边
    for edge in G.edges():
        p1 = points[edge[0]]
        p2 = points[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-')

    # 设置坐标轴范围
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_zlim(0, 4)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

from torch_scatter import scatter
def test2():
    # 假设 edge_index 是一个形状为 [2, E] 的 PyTorch 张量，其中E是边的数量
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],  # source nodes
                            [1, 0, 2, 1, 3, 2]], dtype=torch.long)  # target nodes

    # 使用 scatter 函数计算邻居数量
    # 这里我们使用 ones_like(edge_index[0]) 作为权重，因为所有邻居都被计为1
    neighbor_counts = scatter(torch.ones_like(edge_index[0]), edge_index[0], dim_size=edge_index.max().item() + 1, reduce='add')

    print("每个节点的邻居数量:", neighbor_counts)

test2()