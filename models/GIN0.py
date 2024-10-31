import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

class GIN0Net(torch.nn.Module):
    def __init__(self):
        super(GIN0Net, self).__init__()
        
        # 创建三个不包含任何参数的 GINConv 层
        self.conv1 = GINConv(nn=torch.nn.Identity())
        self.conv2 = GINConv(nn=torch.nn.Identity())
        self.conv3 = GINConv(nn=torch.nn.Identity())

    def forward(self, data, use_mp=True, keys='pos', undirected=True):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if undirected:
            edge_index = to_undirected(edge_index)
        if keys == '1':
            x = torch.ones_like(x).to(x.device)
        # 第一层消息传递
        x = self.conv1(x, edge_index)        
        # 第二层消息传递
        x = self.conv2(x, edge_index)        
        # 第三层消息传递
        x = self.conv3(x, edge_index)        
        return x
