import torch.nn as nn
from torch_geometric.nn import global_max_pool
import torch.nn.functional as F
import torch
from .layers import MyCustomGINConv
from torch.nn import Linear, ReLU, ModuleList, Sequential

class GINNet(nn.Module):
    def __init__(self, num_features, num_classes, mlps):
        super(GINNet, self).__init__()
        
        # 根据layers参数动态创建GIN卷积层
        self.convs = ModuleList()
        in_channels = num_features
        for layer in mlps:
            conv_layers = []
            for i in range(len(layer) - 1):
                conv_layers.append(Linear(in_channels, layer[i]))
                conv_layers.append(ReLU())
                in_channels = layer[i]
            conv_layers.append(Linear(in_channels, layer[-1]))
            self.convs.append(MyCustomGINConv(Sequential(*conv_layers)))
            in_channels = layer[-1]
        
        # 用于图级分类的全连接层
        self.fc = Sequential(
            Linear(in_channels, 64),
            ReLU(),
            Linear(64, num_classes)
        )
        
    def forward(self, data, use_mp=True, keys='pos'):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if keys == '1':
            x = torch.ones_like(x).to(x.device)
        
        # 通过所有GIN卷积层传播信息
        for conv in self.convs:
            x = conv(x, edge_index)
        
        # 使用全局池化函数
        x = global_max_pool(x, batch)
        
        # 通过全连接层进行分类
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
