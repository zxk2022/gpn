import torch.nn as nn
from torch_geometric.nn import global_max_pool
import torch.nn.functional as F
import torch

from torch_geometric.nn import EdgeConv
from torch.nn import Linear, ReLU, ModuleList, Sequential
from torch_geometric.utils import to_undirected

from typing import Callable

class MLP(nn.Module):
    def __init__(self, nn: Callable):
        super(MLP, self).__init__()
        self.nn = nn
    
    def forward(self, x, edge_index):
        return self.nn(x)


class EdgeNet(nn.Module):
    def __init__(self, num_features, num_classes, config):
        super(EdgeNet, self).__init__()
        
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
            elif layer[0] == 'edge':
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
                self.convs.append(EdgeConv(Sequential(*conv_layers), aggr=layer[1]))
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
    model = EdgeNet(10, 2, [['edge', 'max', 32], ['relu'], ['edge', 'max', 64], ['relu'], ['edge', 'max', 128]])
    print(model)
