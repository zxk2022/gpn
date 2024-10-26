import torch
import torch.nn as nn
from pointnet2_ggcn import gabriel_graph_wrapper, mp_sum_wrapper, mp_sum_grad_wrapper
from torch.autograd import Function
from typing import Tuple
import time

from torch_geometric.nn import GCNConv

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
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        assert input.is_contiguous()
        assert edge_index.is_contiguous()

        output = mp_sum_wrapper(input, edge_index)
        ctx.save_for_backward(edge_index)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_output = grad_output.contiguous()
        edge_index, = ctx.saved_tensors
        grad_input = mp_sum_grad_wrapper(grad_output, edge_index)
        return grad_input, None

gabriel_graph = GabrielGraph.apply
message_passing = MessagePassing.apply

class MessagePassingLayer(nn.Module):
    def __init__(self):
        super(MessagePassingLayer, self).__init__()

    # 输入是[8, 64, 6000, 32], 中间是[8, 6000, 32, 64], 输出是[8, 64, 6000, 32]
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = message_passing(x, edge_index)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
class CustomSequential(nn.Sequential):
    def __init__(self, *args):
        super(CustomSequential, self).__init__(*args)

    def forward(self, input, edge_index):
        for module in self:
            if isinstance(module, MessagePassingLayer) or isinstance(module, CustomSequential) or isinstance(module, CustomGCN):
                input = module(input, edge_index)
            else:
                input = module(input)
        return input

class CustomGCN(nn.Module):
    def __init__(self):
        super(CustomGCN, self).__init__()
        self.conv = GCNConv(64, 64)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x    


def test_1():
    x = torch.randn(8, 64, 6000, 32, requires_grad=True).cuda()
    y = torch.randn(8, 64, 6000, 32, requires_grad=True).cuda()

    edge_index = gabriel_graph(x.permute(0, 2, 3, 1).contiguous(), 512)

    model = CustomSequential(
        # nn.Conv2d(64, 64, 1, 1),
        nn.BatchNorm2d(64),
        MessagePassingLayer(),
        nn.ReLU(),
    )
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    for i in range(100):
        time_start = time.time()
        y_pred = model(x, edge_index)
        print("MPforward耗时：", time.time()-time_start)
        loss = criterion(y_pred, y)
        time_start = time.time()
        loss.backward()
        print("MPbackward耗时：", time.time()-time_start)

def test_2():
    x = torch.randn(8, 64, 6000, 32, requires_grad=True).cuda()
    y = torch.randn(8, 64, 6000, 32, requires_grad=True).cuda()

    # 设置张量的形状
    shape = (2, 9600000)

    # 设置最大值
    max_value = 1536000

    # 使用torch.randint函数生成张量
    edge_index = torch.randint(0, max_value, shape, dtype=torch.long).cuda()

    x = x.view(1536000, 64, 1, 1).squeeze()
    y = y.view(1536000, 64, 1, 1).squeeze()

    edge_index = edge_index.view(2, 9600000)

    model = CustomSequential(
        nn.BatchNorm1d(64),
        CustomGCN(),
        nn.ReLU(),
    )
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    for i in range(100):
        time_start = time.time()
        y_pred = model(x, edge_index)
        print("GCNforward耗时：", time.time()-time_start)
        loss = criterion(y_pred, y)
        time_start = time.time()
        loss.backward()
        print("GCNbackward耗时：", time.time()-time_start)

test_2()