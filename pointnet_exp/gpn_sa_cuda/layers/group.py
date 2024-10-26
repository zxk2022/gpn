# group layer: find neighbors for each point
# knn, knn_sparse, ball query

# gather layer, gather features by index
from typing import Tuple
import copy, logging
import torch
import torch.nn as nn
from torch.autograd import Function
from openpoints.cpp import pointnet2_cuda

from .ggcn import gabriel_graph

class KNN(nn.Module):
    def __init__(self, neighbors, transpose_mode=True):
        super(KNN, self).__init__()
        self.neighbors = neighbors

    @torch.no_grad()
    def forward(self, support, query):
        """
        Args:
            support ([tensor]): [B, N, C]
            query ([tensor]): [B, M, C]
        Returns:
            [int]: neighbor idx. [B, M, K]
        """
        dist = torch.cdist(support, query)
        k_dist = dist.topk(k=self.neighbors, dim=1, largest=False)
        return k_dist.values, k_dist.indices.transpose(1, 2).contiguous().int()

# dilated knn
class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    index: (B, npoint, nsample)
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, randnum]
            else:
                edge_index = edge_index[:, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, ::self.dilation]
        return edge_index.contiguous()


class DilatedKNN(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DilatedKNN, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        self.knn = KNN(k * self.dilation, transpose_mode=True)

    def forward(self, query):
        _, idx = self.knn(query, query)
        return self._dilated(idx)


class GroupingOperation(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，用于根据给定索引对特征进行分组。
        
        参数:
            ctx: 上下文对象，用于保存反向传播需要的信息。
            features (torch.Tensor): 需要分组的特征张量，形状为 (B, C, N)。
            idx (torch.Tensor): 包含分组索引的张量，形状为 (B, npoint, nsample)。

        返回:
            output (torch.Tensor): 分组后的特征张量，形状为 (B, C, npoint, nsample)。
        """
        
        # 确保张量是连续的
        assert features.is_contiguous()
        assert idx.is_contiguous()

        # 获取批量大小B, 特征数量 nfeatures 和采样数 nsample
        B, nfeatures, nsample = idx.size()
        # # 获取特征通道数 C 和支持点数量 N
        _, C, N = features.size()
        # # 初始化输出张量 output，形状为 (B, C, nfeatures, nsample)
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample, device=features.device)

        # # 调用 CUDA 内核函数进行分组操作
        pointnet2_cuda.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        # 保存反向传播需要的信息
        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        反向传播函数，用于计算输入特征的梯度。
        
        参数:
            ctx: 上下文对象，包含前向传播保存的信息。
            grad_out (torch.Tensor): 前向传播输出的梯度，形状为 (B, C, npoint, nsample)。

        返回:
            grad_features (torch.Tensor): 输入特征的梯度，形状为 (B, C, N)。
        """

        # 从上下文对象中获取前向传播保存的信息
        idx, N = ctx.for_backwards

        # 获取批量大小 B、特征通道数 C、查询点数量 npoint 和采样数 nsample
        B, C, npoint, nsample = grad_out.size()
        # 初始化输入特征的梯度张量 grad_features，形状为 (B, C, N)
        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        # 确保 grad_out 数据是连续的
        grad_out_data = grad_out.data.contiguous()
        # 调用 CUDA 内核函数计算输入特征的梯度
        pointnet2_cuda.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


def torch_grouping_operation(features, idx):
    r"""from torch points kernels
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of features to group
    idx : torch.Tensor
        (B, npoint, nsample) tensor containing the indicies of features to group with

    Returns
    -------
    torch.Tensor
        (B, C, npoint, nsample) tensor
    """
    all_idx = idx.reshape(idx.shape[0], -1)
    all_idx = all_idx.unsqueeze(1).repeat(1, features.shape[1], 1)
    grouped_features = features.gather(2, all_idx)
    return grouped_features.reshape(idx.shape[0], features.shape[1], idx.shape[1], idx.shape[2])


class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint, device=features.device)

        pointnet2_cuda.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        BallQuery 类用于实现球查询操作，它在给定半径范围内查找点云的邻居点。

        参数:
            ctx: 上下文对象，用于保存反向传播需要的信息（本方法中未使用）。
            radius (float): 球的半径，用于确定查询范围。
            nsample (int): 球内要采样的最大特征数量。
            xyz (torch.Tensor): 支持点的坐标，形状为 (B, N, 3)，其中 B 是批量大小，N 是支持点的数量，3 表示坐标维度。
            new_xyz (torch.Tensor): 查询点的坐标，形状为 (B, npoint, 3)，其中 B 是批量大小，npoint 是查询点的数量，3 表示坐标维度。

        返回:
            idx (torch.Tensor): 形成查询球的特征索引张量，形状为 (B, npoint, nsample)。
        """

        # 确保张量是连续的
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        # 获取批量大小B和支持点的输了N
        B, N, _ = xyz.size()
        # 获取查询点数量npoint
        npoint = new_xyz.size(1)
        # 初始化索引张量idx, 形状为(B, npoint, nsample)
        idx = torch.cuda.IntTensor(B, npoint, nsample, device=xyz.device).zero_()
        # 调用cuda内核函数进行球半径查询
        pointnet2_cuda.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 normalize_by_std=False,
                 normalize_by_allstd=False,
                 normalize_by_allstd2=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        """
        QueryAndGroup 类用于在给定半径范围内对点云进行查询和分组。
        
        参数:
            radius (float): 球体的半径，用于确定查询范围。
            nsample (int): 球体内要采样的最大特征数量。
            relative_xyz (bool, optional): 是否返回相对坐标。默认为 True。
            normalize_dp (bool, optional): 是否归一化相对坐标。默认为 False。
            normalize_by_std (bool, optional): 是否通过标准差归一化。默认为 False。
            normalize_by_allstd (bool, optional): 是否通过所有标准差归一化。默认为 False。
            normalize_by_allstd2 (bool, optional): 是否通过另一种所有标准差归一化。默认为 False。
            return_only_idx (bool, optional): 是否仅返回索引。默认为 False。
        """
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.normalize_dp = normalize_dp
        self.normalize_by_std = normalize_by_std
        self.normalize_by_allstd = normalize_by_allstd
        self.normalize_by_allstd2 = normalize_by_allstd2
        assert self.normalize_dp + self.normalize_by_std + self.normalize_by_allstd < 2   # only nomalize by one method
        self.relative_xyz = relative_xyz
        self.return_only_idx = return_only_idx

        self.edge_generate = kwargs.pop('edge_generate', None)
        print('edge_generate:', self.edge_generate)

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[
        torch.Tensor]:
        """
        前向传播函数。
        
        参数:
            query_xyz (torch.Tensor): 查询点的坐标，形状为 (B, npoint, 3)。
            support_xyz (torch.Tensor): 支持点的坐标，形状为 (B, N, 3)。
            features (torch.Tensor, optional): 支持点的特征描述，形状为 (B, C, N)。默认为 None。
        
        返回:
            grouped_xyz (torch.Tensor): 分组后的点坐标，形状为 (B, 3, npoint, nsample)。
            grouped_features (torch.Tensor): 分组后的点特征，形状为 (B, C, npoint, nsample)。
        """
        # 使用球查询算法获取每个查询点在支持点中的邻居索引
        idx = ball_query(self.radius, self.nsample, support_xyz, query_xyz)

        if self.return_only_idx:
            return idx
        # 转置支持点坐标以便于分组操作
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        # 使用分组操作获取邻居点坐标
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)

        if self.edge_generate == 'gg':
            xyz_edge = grouped_xyz.permute(0, 2, 3, 1).contiguous()
            max_num_edges = (xyz_edge.shape[2] * xyz_edge.shape[2]) // 2
            edge_index = gabriel_graph(xyz_edge, max_num_edges)
        else:
            edge_index = None
        # # DBUG
        # points = xyz_edge[0, 0].cpu().numpy()
        # poitns_edge = edge_index[0, 0].cpu().numpy()
        # # 画图
        # import networkx as nx
        # import matplotlib.pyplot as plt
        # G = nx.Graph()
        # for i in range(points.shape[0]):
        #     G.add_node(i, pos=(points[i][0], points[i][1]))
        # for i in range(poitns_edge.shape[0]):
        #     G.add_edge(poitns_edge[i][0], poitns_edge[i][1])
        # # 绘制Gabriel图（3D）
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # 绘制点
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

        # # 绘制边
        # for edge in G.edges():
        #     p1 = points[edge[0]]
        #     p2 = points[edge[1]]
        #     ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()
    
        if self.relative_xyz:
            # 计算相对坐标
            grouped_xyz = grouped_xyz - query_xyz.transpose(1, 2).unsqueeze(-1)  # relative position
            if self.normalize_dp:
                # 归一化相对坐标
                grouped_xyz /= self.radius
        # 如果提供了特征，则进行特征分组
        grouped_features = grouping_operation(features, idx) if features is not None else None
        return grouped_xyz, grouped_features, edge_index


class GroupAll(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, new_xyz: torch.Tensor, xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)

        xyz_edge = grouped_xyz.permute(0, 2, 3, 1).contiguous()

        max_num_edges = (xyz_edge.shape[2] * xyz_edge.shape[2]) // 2

        edge_index = gabriel_graph(xyz_edge, max_num_edges)

        grouped_features = features.unsqueeze(2) if features is not None else None
        return grouped_xyz, grouped_features, edge_index


class KNNGroup(nn.Module):
    def __init__(self, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        """
        KNNGroup 类用于从支持点云中提取给定查询点的邻居点。
        
        参数:
            nsample (int): 要收集的最大邻居点数量。
            relative_xyz (bool, optional): 是否返回相对坐标。默认为 True。
            normalize_dp (bool, optional): 是否归一化相对坐标。默认为 False。
            return_only_idx (bool, optional): 是否仅返回邻居点索引。默认为 False。
        """
        super().__init__()
        self.nsample = nsample
        self.knn = KNN(nsample, transpose_mode=True)
        self.relative_xyz = relative_xyz
        self.normalize_dp = normalize_dp
        self.return_only_idx = return_only_idx

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[
        torch.Tensor]:
        """
        前向传播函数，通过 KNN 查找邻居点并进行相应的处理。

        参数:
            query_xyz (torch.Tensor): 查询点的坐标，形状为 (B, N, 3)，其中 B 是批次大小, N 是查询点数量, 3 是坐标维度。
            support_xyz (torch.Tensor): 支持点的坐标，形状为 (B, npoint, 3)，其中 B 是批次大小, npoint 是支持点数量, 3 是坐标维度。
            features (torch.Tensor, optional): 查询点的特征描述，形状为 (B, C, N)，其中 B 是批次大小, C 是特征维度, N 是查询点数量。

        返回:
            如果 return_only_idx 为 True, 则返回邻居点索引 (B, N, nsample)。
            否则返回：
                - grouped_xyz: 组内点的相对或绝对坐标，形状为 (B, 3, npoint, nsample)。
                - grouped_features: 组内点的特征，形状为 (B, C, npoint, nsample)。
        """
        # 使用 KNN 查找邻居点索引
        _, idx = self.knn(support_xyz, query_xyz)
        if self.return_only_idx:
            return idx
        idx = idx.int()
        # 转置支持点的坐标以便进行操作
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        # 通过索引进行分组操作
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)

        xyz_edge = grouped_xyz.permute(0, 2, 3, 1).contiguous()

        max_num_edges = (xyz_edge.shape[2] * xyz_edge.shape[2]) // 2

        edge_index = gabriel_graph(xyz_edge, max_num_edges)

        # 如果需要相对坐标，则计算相对位置
        if self.relative_xyz:
            grouped_xyz -= query_xyz.transpose(1, 2).unsqueeze(-1)  # relative position

        # 如果需要归一化，则进行归一化操作
        if self.normalize_dp:
            grouped_xyz /= torch.amax(torch.sqrt(torch.sum(grouped_xyz**2, dim=1)), dim=(1, 2)).view(-1, 1, 1, 1)

        # 如果提供了特征描述，则进行特征分组
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            return grouped_xyz, grouped_features, edge_index
        else:
            return grouped_xyz, None, edge_index


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


def create_grouper(group_args):
    group_args_copy = copy.deepcopy(group_args)
    method = group_args_copy.pop('NAME', 'ballquery')
    radius = group_args_copy.pop('radius', 0.1)
    nsample = group_args_copy.pop('nsample', 20)

    logging.info(group_args)
    if nsample is not None:
        if method == 'ballquery':
            grouper = QueryAndGroup(radius, nsample, **group_args_copy)
        elif method == 'knn':
            grouper = KNNGroup(nsample,  **group_args_copy)
    else:
        grouper = GroupAll()
    return grouper


if __name__ == "__main__":
    import time

    B, C, N = 2, 3, 40960
    K = 16
    device = 'cuda'
    points = torch.randn([B, N, C], device=device, dtype=torch.float)
    print(points.shape, '\n', points)

    # --------------- debug downsampling
    from openpoints.models.layers.layer3d import RandomSample, random_sample, furthest_point_sample

    npoints = 10000
    # rs = RandomSample(num_to_sample=npoints)
    # query, _= rs(points)
    idx = random_sample(points, npoints)
    # torch gather is faster then operation gather. 
    query = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    print(query.shape, '\n', query)

    idx = furthest_point_sample(points, npoints).to(torch.int64)
    query = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    print(query.shape, '\n', query)

    # # --------------- debug KNN
    # knn = KNN(k=K, transpose_mode=True)
    # # knn to get the neighborhood

    # # compare time usage.
    # st = time.time()
    # for _ in range(100):
    #     _, knnidx = knn(points, query) # B G M
    #     idx_base = torch.arange(0, B, device=points.device).view(-1, 1, 1) * N
    #     idx = knnidx + idx_base
    #     idx = idx.view(-1)
    #     neighborhood = points.view(B * N, -1)[idx, :]
    #     neighborhood = neighborhood.view(B, npoints, K, 3).contiguous()
    #     # normalize
    #     neighborhood1 = neighborhood - query.unsqueeze(2)
    # print(time.time() - st)
    # # print(neighborhood1.shape, '\n', neighborhood1)

    # knngroup = KNNGroup(K)
    # # KNN Group is faster then above torch indexing when warpped in class.  
    # st = time.time()
    # for _ in range(100):
    #     neighborhood2 = knngroup(query, points)
    # print(time.time() - st)
    # # print(neighborhood2.shape, '\n', neighborhood2)
    # flag = torch.allclose(neighborhood1, neighborhood2.permute(0, 2, 3, 1))
    # print(flag)

    # ------------- debug ball query
    query_group = QueryAndGroup(0.1, K)

    st = time.time()
    for _ in range(100):
        # ball querying is 40 times faster then KNN 
        features = query_group(query, points)
    print(time.time() - st)
    print(features.shape)
