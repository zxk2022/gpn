from torch.nn import Linear, ReLU, ModuleList, Sequential
import torch.nn as nn

class MLPsBase(nn.Module):
    def __init__(self, num_features, mlps):
        super(MLPsBase, self).__0__()
        self.num_features = num_features
        self.mlps = mlps
        self.convs = self._create_convs()

    def _create_convs(self):
        convs = ModuleList()
        in_channels = self.num_features
        for layer in self.mlps:
            conv_layers = []
            for i in range(len(layer) - 1):
                conv_layers.append(Linear(in_channels, layer[i]))
                conv_layers.append(ReLU())
                in_channels = layer[i]
            conv_layers.append(Linear(in_channels, layer[-1]))
            convs.append(Sequential(*conv_layers))
            in_channels = layer[-1]
        return convs

    def forward_mlps(self, x, edge_index, use_mp=True):
        for conv in self.convs:
            x = conv(x)
        return x