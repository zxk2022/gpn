import torch
from pointnet2_ggcn import gabriel_graph_wrapper, mp_sum_wrapper, mp_sum_grad_wrapper, aggregate_sum_wrapper, aggregate_sum_grad_wrapper

from torch.autograd import Function
from typing import Tuple
import time
class GabrielGraph(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, num_edges_max: int) -> torch.Tensor:
        assert xyz.is_contiguous()
        return gabriel_graph_wrapper(xyz, num_edges_max)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor = None) -> Tuple[None, ...]:
        return (None,) * 4

class MessagePassing(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        assert input.is_contiguous()
        assert edge_index.is_contiguous()

        output = mp_sum_wrapper(input, edge_index)
        ctx.save_for_backward(edge_index)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_output = grad_output.contiguous()
        edge_index, = ctx.saved_tensors
        grad_input = mp_sum_grad_wrapper(grad_output, edge_index)
        return grad_input, None

class AggregateSum(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        assert input.is_contiguous()
        assert edge_index.is_contiguous()

        output = aggregate_sum_wrapper(input, edge_index)
        ctx.save_for_backward(edge_index)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_output = grad_output.contiguous()
        edge_index, = ctx.saved_tensors
        grad_input = aggregate_sum_grad_wrapper(grad_output, edge_index)
        return grad_input, None

gabriel_graph = GabrielGraph.apply
message_passing = MessagePassing.apply
aggregate_sum = AggregateSum.apply

def test_1():
    # 设置随机种子以获得可复现的结果
    torch.manual_seed(0)

    # 随机生成一组特征数据
    features = torch.randn(8, 6000, 32, 64, requires_grad=True).cuda()
    features.retain_grad()

    # 随机生成一组点云数据
    xyz = torch.randn(8, 6000, 32, 3).cuda()
    # 生成Gabriel图的边索引
    edge_index = gabriel_graph(xyz, 496)
    # 消息传递操作
    output = message_passing(features, edge_index)
    print("Output shape:", output.shape)

    # 定义一个损失函数，例如 L2 损失
    target = torch.randn(8, 6000, 32, 64).cuda()  # 假设的目标值
    loss = torch.nn.functional.mse_loss(output, target)
    print("Loss:", loss.item())

    # 反向传播
    loss.backward()

    # 打印特征的梯度
    print("Gradient of features:", features.grad.shape)

def test_2():
    # 设置随机种子以获得可复现的结果
    torch.manual_seed(0)

    # 随机生成一组特征数据
    features = torch.randn(8, 6000, 32, 64, requires_grad=True).cuda()
    features = torch.tensor([[[[1, 1, 1, 1],
                            [2, 2, 2, 2]]]], dtype=torch.float32, device='cuda', requires_grad=True)

    features.retain_grad()

    # 随机生成一组点云数据
    xyz = torch.randn(8, 6000, 32, 3).cuda()
    xyz = torch.tensor([[[[1, 1, 1],
                            [2, 2, 2]]]], dtype=torch.float32, device='cuda', requires_grad=False)
    # 生成Gabriel图的边索引
    edge_index = gabriel_graph(xyz, 496)
    # 消息传递操作
    output = message_passing(features, edge_index)
    print("Output shape:", output.shape)

    # 定义一个损失函数，例如 L2 损失
    target = torch.randn(8, 6000, 32, 64).cuda()  # 假设的目标值
    target = torch.tensor([[[[1, 1, 1, 1],
                            [2, 2, 2, 2]]]], dtype=torch.float32, device='cuda', requires_grad=True)
    loss = torch.nn.functional.mse_loss(output, target)
    print("Loss:", loss.item())

    # 反向传播
    loss.backward()

    # 打印特征的梯度
    print("Gradient of features:", features.grad.shape)

def test_3():
    # 设置随机种子以获得可复现的结果
    torch.manual_seed(0)

    # 随机生成一组特征数据
    features = torch.randn(8, 6000, 32, 64, requires_grad=True).cuda()

    # 随机生成一组点云数据
    xyz = torch.randn(8, 6000, 32, 3).cuda()

    # 生成Gabriel图的边索引
    print("edge_index生成10次耗时")
    start = time()
    for i in range(10):
        edge_index = gabriel_graph(xyz, 496)
    print("耗时：", time()-start)

    # 消息传递操作
    print("message_passing10次耗时")
    output = message_passing(features, edge_index)
    start = time()
    for i in range(10):
        output = message_passing(features, edge_index)
    print("耗时：", time()-start)

    features = features.permute(0, 3, 1, 2)

    # 卷积操作10次耗时
    print("卷积操作10次耗时")
    start = time()
    for i in range(10):
        output = torch.nn.functional.conv2d(features, torch.randn(64, 64, 1, 1).cuda())
    print("耗时：", time()-start)

    # permute 10次耗时
    print("permute 1000次耗时")
    start = time()
    for i in range(100):
        # 顺序变换
        features = features.permute(0, 3, 1, 2).contiguous()
    print("耗时：", time()-start)

    # # 定义一个损失函数，例如 L2 损失
    # target = torch.randn(8, 6000, 32, 64).cuda()  # 假设的目标值
    
    # loss = torch.nn.functional.mse_loss(output, target)
    # print("Loss:", loss.item())

    # # 反向传播
    # loss.backward()

    # # 打印特征的梯度
    # print("Gradient of features:", features.grad.shape)

def test_4():
    features = torch.tensor([[[[1, 2, 4, 3],
                            [5, 2, 6, 2]],
                            [[7, 9, 4, 3],
                            [5, 8, 8, 2]]],
                            
                            [[[8, 5, 5, 3],
                            [11, 2, 4, 52]],
                            [[43, 66, 44, 3],
                            [95, 78, 48, 52]]]], dtype=torch.float32, device='cuda', requires_grad=True)
    print(features)
    features = features.permute(0, 2, 3, 1).contiguous()
    print(features)
    features = features.permute(0, 3, 1, 2).contiguous()
    print(features)

def test_5():
    batch_size = 8
    num_points = 6000
    max_edges = 512
    channel = 64
    mp = 1
    
    # 设置随机种子以获得可复现的结果
    torch.manual_seed(0)

    # 随机生成一组特征数据
    x = torch.randn(batch_size, num_points, 32, channel, requires_grad=True).cuda()

    # 随机生成一组点云数据
    xyz = torch.randn(batch_size, num_points, 32, 3).cuda()


    time_start = time.time()
    x = x.permute(0, 3, 1, 2)

    # 卷积操作
    time_start = time.time()
    conv_weights = torch.randn(channel, channel, 1, 1, requires_grad=True).cuda()
    x = torch.nn.functional.conv2d(x, conv_weights)
    print("第一次卷积操作耗时：", time.time() - time_start)

    x = x.permute(0, 2, 3, 1)

    # 生成Gabriel图的边索引
    time_start = time.time()
    edge_index = gabriel_graph(xyz, max_edges)
    print("边索引生成耗时：", time.time() - time_start)

    if mp:
        # 消息传递操作
        time_start = time.time()
        x = aggregate_sum(x, edge_index)
        print("消息传递耗时：", time.time() - time_start)

    x = x.permute(0, 3, 1, 2)

    # 卷积操作
    time_start = time.time()
    conv_weights2 = torch.randn(channel, channel, 1, 1, requires_grad=True).cuda()
    x = torch.nn.functional.conv2d(x, conv_weights2)
    print("第二次卷积操作耗时：", time.time() - time_start)

    x = x.permute(0, 2, 3, 1)

    x = x.permute(0, 3, 1, 2)

    # 卷积操作
    time_start = time.time()
    conv_weights2 = torch.randn(channel, channel, 1, 1, requires_grad=True).cuda()
    x = torch.nn.functional.conv2d(x, conv_weights2)
    print("第三次卷积操作耗时：", time.time() - time_start)

    x = x.permute(0, 2, 3, 1)

    # 定义一个损失函数，例如 L2 损失
    target = torch.randn(batch_size, num_points, 32, channel).cuda()  # 假设的目标值
    time_start = time.time()
    loss = torch.nn.functional.mse_loss(x, target)
    print("计算损失耗时：", time.time() - time_start)

    time_start = time.time()
    # 反向传播
    loss.backward()
    print("反向传播耗时：", time.time() - time_start)
    
    # 计算第二次卷积操作的梯度
    # grad_outputs = torch.ones_like(loss)  # 创建与loss相同形状的张量，用于传递到autograd.grad
    # gradients_conv2 = torch.autograd.grad(outputs=loss, inputs=conv_weights2, grad_outputs=grad_outputs, retain_graph=True)
    # print("第二次卷积操作梯度计算耗时：", time.time() - time_start)

    # # 如果有消息传递，则计算消息传递部分的梯度
    # if mp:
    #     time_start = time.time()
    #     gradients_message_passing = torch.autograd.grad(outputs=loss, inputs=x, grad_outputs=grad_outputs, retain_graph=True)
    #     print("消息传递梯度计算耗时：", time.time() - time_start)

    # # 计算第一次卷积操作的梯度
    # time_start = time.time()
    # gradients_conv1 = torch.autograd.grad(outputs=loss, inputs=conv_weights, grad_outputs=grad_outputs, retain_graph=True)
    # print("第一次卷积操作梯度计算耗时：", time.time() - time_start)

for i in range(100):
    print("---------第", i, "次---------")
    test_5()
