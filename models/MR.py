import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, PairTensor
from typing import Callable, Union
from torch_geometric.nn.inits import reset
import torch.nn as nn
from torch_geometric.nn import global_max_pool
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv
from torch.nn import Linear, ReLU, ModuleList, Sequential
from torch_geometric.utils import to_undirected

def reset(nn: Callable):
    if hasattr(nn, 'reset_parameters'):
        nn.reset_parameters()

class MRConv(MessagePassing):
    r"""The Max-Relative Graph Convolution operator.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        # Compute the difference between each node and its neighbors.
        diff = x_j - x_i
        # Apply the neural network to the difference vector.
        return diff

    def update(self, aggr_out: Tensor, x: PairTensor) -> Tensor:
        # Concatenate the aggregated max value with the original feature.
        return self.nn(torch.cat([x[0], aggr_out], dim=-1))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

class MLP(nn.Module):
    def __init__(self, nn: Callable):
        super(MLP, self).__init__()
        self.nn = nn
    
    def forward(self, x, edge_index):
        return self.nn(x)


class MRNet(nn.Module):
    def __init__(self, num_features, num_classes, config):
        super(MRNet, self).__init__()
        
        # 根据layers参数动态创建GIN卷积层
        self.convs = ModuleList()
        in_channels = num_features
        for layer in config:
            if layer[0] == 'relu':
                self.convs.append(MLP(Sequential(ReLU())))
                continue
            if layer[0] == 'mlp':
                conv_layers = []
                for i in range(len(layer) - 2):
                    conv_layers.append(Linear(in_channels, layer[i+1]))
                    conv_layers.append(ReLU())
                    in_channels = layer[i+1]
                conv_layers.append(Linear(in_channels, layer[-1]))
                self.convs.append(MLP(Sequential(*conv_layers)))
                in_channels = layer[-1]
            elif layer[0] == 'mr':
                conv_layers = []
                for i in range(len(layer) - 3):
                    if i == 0:
                        conv_layers.append(Linear(in_channels*2, layer[i+2]))
                    else:
                        conv_layers.append(Linear(in_channels, layer[i+2]))
                    conv_layers.append(ReLU())
                    in_channels = layer[i+2]
                if len(layer) == 3:
                    conv_layers.append(Linear(in_channels*2, layer[-1]))
                else:
                    conv_layers.append(Linear(in_channels, layer[-1]))
                self.convs.append(MRConv(Sequential(*conv_layers), aggr=layer[1]))
                in_channels = layer[-1]
        
        # 用于图级分类的全连接层
        self.fc = Sequential(
            Linear(in_channels, 64),
            ReLU(),
            Linear(64, num_classes)
        )
        
    def forward(self, data, use_mp=True, keys='pos', undirected=True):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if undirected:
            edge_index = to_undirected(edge_index)
        if keys == '1':
            x = torch.ones_like(x).to(x.device)
        
        # 通过所有Edge卷积层传播信息
        for conv in self.convs:
            x = conv(x, edge_index)
        
        # 使用全局池化函数
        x = global_max_pool(x, batch)
        
        # 通过全连接层进行分类
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    # 创建模型
    model = MRNet(10, 2, [['mr', 'max', 16], ['relu'], ['mr', 'max', 64], ['relu'], ['mr', 'max', 256]])
    print(model)
