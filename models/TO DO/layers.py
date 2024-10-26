from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn import GINConv
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)

class MyCustomGINConv(GINConv):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False, **kwargs):
        super().__init__(nn, eps, train_eps, **kwargs)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
        use_mp: bool = True,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        # 通过新参数use_mp控制是否使用message-passing
        if use_mp:
            out = self.propagate(edge_index, x=x, size=size)
            x_r = x[1]
            if x_r is not None:
                out = out + (1 + self.eps) * x_r
        else:
            out = x[0]

        if use_mp:
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=x)
        else:
            out = x[0]

        return self.nn(out)

# 示例用法

if __name__ == '__main__':
    # 假设我们有一个神经网络模块nn和一个新的参数值use_mp
    nn = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 30))

    # 创建MyCustomGINConv的实例
    custom_gin_conv = MyCustomGINConv(nn)

    # 使用自定义的forward方法
    x = torch.randn(100, 10)  # 假设的节点特征
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)  # 假设的边索引

    out = custom_gin_conv(x, edge_index, use_mp=True)
    print(out.size())  # torch.Size([100, 30])



