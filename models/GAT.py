import torch
from torch.nn import ModuleList, Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, global_max_pool
from torch.nn.functional import log_softmax

from torch_geometric.utils import to_undirected

class GATNet(torch.nn.Module):
    def __init__(self, num_feature, num_classes, config):
        super(GATNet, self).__init__()

        # 根据layers参数动态创建GAT卷积层
        self.convs = ModuleList()
        in_channels = num_feature
        for layer in config:
            if layer[0] == 'relu':
                self.convs.append(ReLU())
                continue
            elif layer[0] == 'gat':
                heads = layer[1]  # 注意力头的数量
                for i in range(len(layer) - 3):
                    self.convs.append(GATConv(in_channels, layer[i+2], heads=heads))
                    self.convs.append(ReLU())
                    in_channels = layer[i+2] * heads  # 注意力头的输出特征维度
                self.convs.append(GATConv(in_channels, layer[-1], heads=1, concat=False))  # 最后一层不拼接
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

        # 通过所有GAT卷积层传播信息
        for conv in self.convs:
            if isinstance(conv, GATConv):
                x = conv(x, edge_index)
            else:
                x = conv(x)
        
        # 使用全局池化函数
        x = global_max_pool(x, batch)
        x = self.fc(x)
        return log_softmax(x, dim=1)
    
if __name__ == '__main__':
    model = GATNet(3, 40, [['gat', 4, 32, 64, 128]])  # 注意力头数量为8
    print(model)