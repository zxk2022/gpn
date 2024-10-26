from typing import Tuple
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F

from pointnet2_ggcn import gabriel_graph_wrapper, mp_sum_wrapper, mp_sum_grad_wrapper, aggregate_sum_wrapper, aggregate_sum_grad_wrapper

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

class AggregateSum(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        assert input.is_contiguous()
        assert edge_index.is_contiguous()

        output = aggregate_sum_wrapper(input, edge_index)
        ctx.save_for_backward(edge_index)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_output = grad_output.contiguous()
        edge_index, = ctx.saved_tensors
        grad_input = aggregate_sum_grad_wrapper(grad_output, edge_index)
        return grad_input, None

gabriel_graph = GabrielGraph.apply
message_passing = MessagePassing.apply
aggregate_sum = AggregateSum.apply

class AlphaLayer(nn.Module):
    def __init__(self):
        super(AlphaLayer, self).__init__()
        # 将 alpha 参数注册到模型中
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, m):
        # 在前向传播中使用 alpha * m
        return x * (1 + self.alpha) + m

class MessagePassingLayer(nn.Module):
    def __init__(self, use_mp='mp_sum', out_channels: int = 64):
        super(MessagePassingLayer, self).__init__()
        self.use_mp = use_mp
        if self.use_mp == 'ag_sum_ud_alpha':
            self.alpha_layer = AlphaLayer()
        if self.use_mp == 'ag_sum_ud_concatandlin':
            self.conv = nn.Conv2d(2*out_channels, out_channels, 1)
        if self.use_mp == 'ag_sum_ms_ud_lin':
            self.conv = nn.Conv2d(out_channels, out_channels, 1)

    # 输入是[8, 64, 6000, 32], 中间是[8, 6000, 32, 64], 输出是[8, 64, 6000, 32]
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.use_mp == 'mp_sum':
            x = x.permute(0, 2, 3, 1).contiguous()
            x = message_passing(x, edge_index)
            x = x.permute(0, 3, 1, 2).contiguous()
            return x
        elif self.use_mp == 'ag_sum_ud_alpha':
            m = x.permute(0, 2, 3, 1).contiguous()
            m = aggregate_sum(m, edge_index)
            m = m.permute(0, 3, 1, 2).contiguous()
            return self.alpha_layer(x, m)
        elif self.use_mp == 'ag_sum_ms_ud_lin':
            m = x.permute(0, 2, 3, 1).contiguous()
            m = aggregate_sum(m, edge_index)
            m = m.permute(0, 3, 1, 2).contiguous()
            return x + self.conv(m)
        elif self.use_mp == 'ag_sum_ud_concatandlin':
            m = x.permute(0, 2, 3, 1).contiguous()
            m = aggregate_sum(m, edge_index)
            m = m.permute(0, 3, 1, 2).contiguous()
            return self.conv(torch.cat([x, m], dim=1))
        else:
            raise ValueError(f'Unknown message passing method: {self.use_mp}')
        