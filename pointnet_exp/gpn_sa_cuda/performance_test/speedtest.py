import torch
from pointnet2_ggcn import gabriel_graph_wrapper, message_passing_wrapper, message_passing_grad_wrapper
from torch.autograd import Function
from typing import Tuple
import time

from torch_geometric.nn import MessagePassing

class GabrielGraph(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, num_edges_max: int) -> torch.Tensor:
        assert xyz.is_contiguous()
        return gabriel_graph_wrapper(xyz, num_edges_max)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor = None) -> Tuple[None, ...]:
        return (None,) * 4

class SMessagePassing(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        assert input.is_contiguous()
        assert edge_index.is_contiguous()

        output = message_passing_wrapper(input, edge_index)
        ctx.save_for_backward(edge_index)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        from time import time
        time_start = time()
        edge_index, = ctx.saved_tensors
        grad_input = message_passing_grad_wrapper(grad_output, edge_index)
        print("     grad_input耗时：", time()-time_start)
        return grad_input, None

gabriel_graph = GabrielGraph.apply
message_passing = MessagePassing.apply

def test_1():
    # 设置随机种子以获得可复现的结果
    torch.manual_seed(0)

    # 随机生成一组特征数据
    features = torch.randn(8, 6000, 32, 64, requires_grad=True).cuda()
    features.retain_grad()

    # 随机生成一组点云数据
    xyz = torch.randn(8, 6000, 32, 3).cuda()
    # 生成Gabriel图的边索引
    edge_index = gabriel_graph(xyz, 496)
    # 消息传递操作
    output = message_passing(features, edge_index)
    print("Output shape:", output.shape)

    # 定义一个损失函数，例如 L2 损失
    target = torch.randn(8, 6000, 32, 64).cuda()  # 假设的目标值
    loss = torch.nn.functional.mse_loss(output, target)
    print("Loss:", loss.item())

    # 反向传播
    loss.backward()

    # 打印特征的梯度
    print("Gradient of features:", features.grad.shape)