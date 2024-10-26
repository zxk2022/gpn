from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .conv import create_convblock2d, create_convblock1d, MultiInputSequential
from .activation import create_act
from .group import create_grouper, get_aggregation_feautres


CHANNEL_MAP = {
    'fj': lambda x: x,
    'df': lambda x: x,
    'assa': lambda x: x * 3,
    'assa_dp': lambda x: x * 3 + 3,
    'dp_fj': lambda x: 3 + x,
    'pj': lambda x: x,
    'dp': lambda x: 3,
    'pi_dp': lambda x: x + 3,
    'pj_dp': lambda x: x + 3,
    'dp_fj_df': lambda x: x*2 + 3,
    'dp_fi_df': lambda x: x*2 + 3,
    'pi_dp_fj_df': lambda x: x*2 + 6,
    'pj_dp_fj_df': lambda x: x*2 + 6,
    'pj_dp_df': lambda x: x + 6,
    'dp_df': lambda x: x + 3,
}


class ASSA(nn.Module):
    def __init__(self,
                 channels: List[int],
                 conv_args=None,
                 norm_args=None,
                 act_args=None,
                 group_args=None,
                 feature_type='dp_fj',
                 reduction='mean',
                 use_res=True,
                 use_inverted_dims=False,
                 ):
        """Separable depthwise convolution with aggregation . 
        Args:
            channels (List[int]): [description]
            conv_args ([type], optional): [description]. Defaults to None.
            norm_args ([type], optional): [description]. Defaults to None.
            act_args ([type], optional): [description]. Defaults to None.
            group_args ([type], optional): [description]. Defaults to None.
            feature_type (str, optional): [description]. Defaults to 'dp_fj'.
            reduction (str, optional): [description]. Defaults to 'mean'.
            layers (int, optional): [description]. Defaults to 1.
            use_res (bool, optional): [use residual connection or not ]. Defaults to False.
            use_depth (bool, optional): [use depwise convo connection or not ]. Defaults to False.

        Raises:
            NotImplementedError: [description]
        """
        super(ASSA, self).__init__()
        self.feature_type = feature_type
        self.use_res = use_res
        convs = []

        # pointwise convolution before reduction
        num_preconv = int(np.ceil((len(channels) - 1) / 2))
        self.num_preconv = num_preconv
        if self.feature_type == 'assa' and not use_inverted_dims:
            channels[num_preconv] = int(np.ceil(channels[num_preconv] / 3.0))
        for i in range(num_preconv):  # #layers in each blocks
            convs.append(create_convblock1d(channels[i], channels[i + 1],
                                            norm_args=norm_args, act_args=act_args,
                                            **conv_args)
                         )

        # pointwise convolution after reduction
        skip_channels = channels[num_preconv]
        mid_conv_in_channel = CHANNEL_MAP[self.feature_type](
            channels[num_preconv])
        channels[num_preconv] = mid_conv_in_channel
        for i in range(num_preconv, len(channels) - 1):
            convs.append(create_convblock1d(channels[i], channels[i + 1],
                                            norm_args=norm_args,
                                            act_args=None if use_res and i == len(
                                                channels)-2 else act_args,
                                            **conv_args)
                         )
        self.act = create_act(act_args)
        self.convs = nn.Sequential(*convs)

        # residual connection
        if use_res:
            self.skip_layer = nn.Identity() if skip_channels == channels[-1] else nn.Conv1d(
                skip_channels, channels[-1], 1, bias=False)

        # grouping and reduction 
        self.grouper = create_grouper(group_args)
        if reduction == 'max':
            self.reduction_layer = lambda x: torch.max(
                x, dim=-1, keepdim=False)[0]
        elif reduction == 'avg' or reduction == 'mean':
            self.reduction_layer = lambda x: torch.mean(
                x, dim=-1, keepdim=False)
        elif reduction == 'sum':
            self.reduction_layer = lambda x: torch.sum(
                x, dim=-1, keepdim=False)
        else:
            raise NotImplementedError(
                f'reduction {self.reduction} not implemented')

    def forward(self, query_xyz, support_xyz, features, query_idx=None):
        """
        Args:
            features: support features
        Returns:
           output features of query points: [B, C_out, 3]
        """
        features = self.convs[:self.num_preconv](features)
        
        # grouping 
        dp, fj = self.grouper(query_xyz, support_xyz, features)
        if self.use_res and query_idx is not None:
            features = torch.gather(
                features, -1, query_idx.unsqueeze(1).expand(-1, features.shape[1], -1))

        # reduction layer
        B, C, npoint, nsample = fj.shape
        fj = fj.unsqueeze(1).expand(-1, 3, -1, -1, -1) \
            * dp.unsqueeze(2)
        fj = fj.view(B, -1, npoint, nsample)
        out_features = self.reduction_layer(fj)
        
        # pointwise convolution
        out_features = self.convs[self.num_preconv:](out_features)
        
        if self.use_res:
            out_features = self.act(out_features + self.skip_layer(features))
        return out_features


class ConvPool(nn.Module):
    def __init__(self,
                 channels: List[int],
                 conv_args=None,
                 norm_args=None,
                 act_args=None,
                 group_args=None,
                 feature_type='dp_fj',
                 reduction='mean',
                 use_res=False,
                 use_pooled_as_identity=False,
                 **kwargs
                 ):
        """基于常规共享卷积和聚合的局部聚合。
        参数：
            channels (List[int]): 通道列表。
            conv_args ([type], optional): 卷积参数。默认为 None。
            norm_args ([type], optional): 归一化参数。默认为 None。
            act_args ([type], optional): 激活函数参数。默认为 None。
            group_args ([type], optional): 分组参数。默认为 None。
            feature_type (str, optional): 特征类型。默认为 'dp_fj'。
            reduction (str, optional): 归约类型。默认为 'mean'。
            use_res (bool, optional): 是否使用残差连接。默认为 False。
            use_pooled_as_identity (bool, optional): 是否使用池化作为身份。默认为 False。

        抛出：
            NotImplementedError: 未实现的功能。
        """
        super(ConvPool, self).__init__()
        skip_channel = channels[0]
        self.use_res = use_res
        self.use_pooled_as_identity = use_pooled_as_identity

        # 如果使用残差连接，则初始化残差卷积层
        if use_res:
            self.skipconv = create_convblock1d(skip_channel, channels[-1], norm_args=None, act_args=None,
                                               **conv_args) if skip_channel != channels[-1] else nn.Identity()

        self.feature_type = feature_type

        # 获取特征类型对应的输入通道
        channel_in = CHANNEL_MAP[feature_type](channels[0])
        channels[0] = channel_in

        # 初始化卷积层
        convs = []
        for i in range(len(channels) - 2):  # 每个块中的层数
            convs.append(create_convblock2d(channels[i], channels[i + 1], norm_args=norm_args, act_args=act_args,
                                            **conv_args)
                         )
        convs.append(create_convblock2d(channels[-2], channels[-1], norm_args=norm_args,
                                        act_args=None if use_res else act_args,
                                        **conv_args))
        self.act = create_act(act_args)
        self.convs = MultiInputSequential(*convs)

        # 初始化分组操作
        self.grouper = create_grouper(group_args)
        # 设置归约层
        if reduction == 'max':
            self.reduction_layer = lambda x: torch.max(
                x, dim=-1, keepdim=False)[0]
        elif reduction == 'avg' or reduction == 'mean':
            self.reduction_layer = lambda x: torch.mean(
                x, dim=-1, keepdim=False)
        elif reduction == 'sum':
            self.reduction_layer = lambda x: torch.sum(
                x, dim=-1, keepdim=False)
        else:
            raise NotImplementedError(
                f'reduction {self.reduction} not implemented')

    def forward(self, query_xyz, support_xyz, features, query_idx=None):
        """
        前向传播函数。

        参数：
            query_xyz: 查询点的坐标 (B, N, 3)
            support_xyz: 支持点的坐标 (B, M, 3)
            features: 输入特征 (B, C, M)
            query_idx: 查询点的索引 (B, N), 可选

        返回：
           查询点的输出特征: [B, C_out, 3]
        """
        # 通过分组操作获取距离矩阵 dp 和聚合特征 fj
        dp, fj, edge_index = self.grouper(query_xyz, support_xyz, features)

        # # debug判断是否有重复的坐标
        # tensor = dp[0, :, 1, :].permute(1, 0)
        # # 将张量转换为NumPy数组
        # np_array = tensor.cpu().numpy()

        # # 查找唯一的行
        # unique_rows = np.unique(np_array, axis=0)

        # # 比较行数
        # if unique_rows.shape[0] != np_array.shape[0]:
        #     print("张量中有重复的坐标")
        # else:
        #     print("张量中没有重复的坐标")

        neighbor_dim = 3
        # 检查特征类型和是否使用残差连接
        if 'df' in self.feature_type or self.use_res:
            if self.use_pooled_as_identity:
                features = torch.max(fj, dim=-1, keepdim=False)[0]
            elif query_idx is not None:
                # 如果提供了查询索引，根据索引调整特征
                if query_xyz.shape[1] != support_xyz.shape[1]:
                    features = torch.gather(
                        features, -1, query_idx.unsqueeze(1).expand(-1, features.shape[1], -1))
            elif dp.shape[2] == 1:
                # 如果 dp 的维度为 1, 则设置邻居维度为 2
                neighbor_dim = 2    # this means the current layer is a aggragation all lauyers
            if self.use_res and neighbor_dim != 2:
                # 如果使用残差连接且邻居维度不为 2, 通过残差卷积处理特征
                identity = self.skipconv(features)
            else:
                identity = 0

        # """ Debug neighbor numbers. """
        # if hasattr(self.grouper, 'radius'):
        #     radius = self.grouper.radius
        #     dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
        #     points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
        #     print(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
        # """End of debug"""
        
        # 获取聚合特征
        fj = get_aggregation_feautres(query_xyz, dp, features, fj, feature_type=self.feature_type)

        # 通过卷积层和规约层处理特征
        out_features = self.reduction_layer(self.convs(fj, edge_index))

        # 如果使用残差连接，应用激活函数s
        if self.use_res:
            out_features = self.act(out_features + identity)
        return out_features


# 1 local aggregation = res block in resnet.
class LocalAggregation(nn.Module):
    def __init__(self,
                 channels: List[int],
                 aggr_args: dict,
                 conv_args=None,
                 norm_args=None,
                 act_args=None,
                 group_args=None,
                 use_res=False,
                 ):
        """LocalAggregation operators
        Args:
            config: config file
        """
        super(LocalAggregation, self).__init__()
        aggr_type = aggr_args.get('NAME', 'convpool')
        feature_type = aggr_args.get('feature_type', 'dp_fj')
        reduction = aggr_args.get('reduction', 'max')
        use_inverted_dims = aggr_args.get('use_inverted_dims', False)
        use_pooled_as_identity = aggr_args.get('use_pooled_as_identity', False)
        # num_preconv = aggr_args.get('num_preconv', 1)
        # num_posconv = aggr_args.get('num_posconv', 1)

        if aggr_type.lower() == 'convpool':
            self.SA_CONFIG_operator = ConvPool(channels, conv_args, norm_args, act_args,
                                               group_args, feature_type, reduction, use_res, use_pooled_as_identity)
        elif aggr_type.lower() == 'assa':
            self.SA_CONFIG_operator = ASSA(channels, conv_args, norm_args, act_args,
                                           group_args, feature_type, reduction, use_res, use_inverted_dims)

        else:
            raise NotImplementedError(
                f'LocalAggregation {aggr_type.lower()} not implemented')

    def forward(self, query_xyz, support_xyz, support_features, query_idx=None):
        """
        Args:
        Returns:
           output features of query points: [B, C_out, 3]
        """
        return self.SA_CONFIG_operator(query_xyz, support_xyz, support_features, query_idx)
