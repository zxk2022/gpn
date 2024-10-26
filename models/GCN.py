import torch.nn as nn
from torch_geometric.nn import global_max_pool
import torch.nn.functional as F
import torch

from torch_geometric.nn import GCNConv
from torch.nn import Linear, ReLU, ModuleList, Sequential
from torch_geometric.utils import to_undirected

class GCNNet(nn.Module):
    def __init__(self, num_feature, num_classes, config):
        super(GCNNet, self).__init__()

        # 根据layers参数动态创建GCN卷积层
        self.convs = ModuleList()
        in_channels = num_feature
        for layer in config:
            if layer[0] == 'relu':
                self.convs.append(ReLU())
                continue
            elif layer[0] == 'gcn':
                for i in range(len(layer) - 2):
                    self.convs.append(GCNConv(in_channels, layer[i+1]))
                    self.convs.append(ReLU())
                    in_channels = layer[i+1]
                self.convs.append(GCNConv(in_channels, layer[-1]))                
                in_channels = layer[-1]
            elif layer[0] == 'mlp':
                for i in range(len(layer) - 2):
                    self.convs.append(Linear(in_channels, layer[i+1]))
                    self.convs.append(ReLU())
                    in_channels = layer[i+1]
                self.convs.append(Linear(in_channels, layer[-1]))                
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

        # 通过所有GCN卷积层传播信息
        for conv in self.convs:
            if isinstance(conv, GCNConv):
                x = conv(x, edge_index)
            else:
                x = conv(x)
        
        # 使用全局池化函数
        x = global_max_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
if __name__ == '__main__':
    model = GCNNet(3, 40, [['gcn', 32], ['relu'], ['gcn', 64], ['relu'], ['gcn', 128]])
    print(model)