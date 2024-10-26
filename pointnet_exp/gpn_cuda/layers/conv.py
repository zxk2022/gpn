from distutils.command.build import build
import torch.nn as nn

from .activation import create_act
from .norm import create_norm, create_norm

from .ggcn import MessagePassingLayer


class Conv2d(nn.Conv2d):
    """
    这是一个继承自 nn.Conv2d 的卷积层类。

    初始化方法根据传入参数的数量和类型来决定如何调用父类的构造函数。

    参数:
        *args: 位置参数，通常包括输入通道数和输出通道数。
        **kwargs: 关键字参数，可以包括卷积层的其他配置参数。

    如果传入的参数数量为2且关键字参数中不包含 'kernel_size'，则默认使用 (1, 1) 作为卷积核大小。
    否则，直接传递所有参数给父类的构造函数。
    """
    def __init__(self, *args, **kwargs):
        if len(args) == 2 and 'kernel_size' not in kwargs.keys():
            super(Conv2d, self).__init__(*args, (1, 1), **kwargs)
        else:
            super(Conv2d, self).__init__(*args, **kwargs)


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        if len(args) == 2 and 'kernel_size' not in kwargs.keys():
            super(Conv1d, self).__init__(*args, 1, **kwargs)
        else:
            super(Conv1d, self).__init__(*args, **kwargs)

class MultiInputSequential(nn.Sequential):
    def __init__(self, *args):
        super(MultiInputSequential, self).__init__(*args)

    def forward(self, input, edge_index):
        for module in self:
            if isinstance(module, MessagePassingLayer) or isinstance(module, MultiInputSequential):
                input = module(input, edge_index)
            else:
                input = module(input)
        return input

def create_convblock2d(*args,
                       norm_args=None, act_args=None, order='conv-norm-act', **kwargs):
    # 定义一个函数 create_convblock2d，它接受不定数量的位置参数(*args)，以及关键字参数 norm_args, act_args, order 和任意数量的关键字变量(**kwargs)。
    in_channels = args[0]
    out_channels = args[1]
    # 从位置参数中提取输入通道数和输出通道数，这两个参数通常用于定义卷积层的输入和输出特征图的通道数。
    use_mp = kwargs.pop('use_mp', None)

    bias = kwargs.pop('bias', True)
    # 从关键字参数字典中提取 'bias' 参数，默认值为 True。如果字典中有 'bias'，则将其移除并赋值给 bias 变量；如果没有，则将 bias 设为 True。

    if order == 'conv-norm-act':
        # 如果顺序参数 order 为 'conv-norm-act'，即先卷积，然后标准化，最后激活。
        norm_layer = create_norm(norm_args, out_channels, dimension='2d')
        # 调用 create_norm 函数创建一个标准化层，传入 norm_args, out_channels 和维度参数 '2d'。
        bias = False if norm_layer is not None else bias
        # 如果标准化层被创建（即不为 None），则设置 bias 为 False，因为通常在批标准化之后不需要偏置项。

        conv_layer = [Conv2d(*args, bias=bias, **kwargs)]
        # 创建一个卷积层列表，使用位置参数和关键字参数，并设置偏置项。

        if norm_layer is not None:
            conv_layer.append(norm_layer)
        # 如果标准化层存在，将其添加到卷积层列表中。

        if use_mp is not None:
            conv_layer.append(MessagePassingLayer(use_mp, out_channels))

        act_layer = create_act(act_args)
        # 调用 create_act 函数创建一个激活层。

        if act_args is not None:
            conv_layer.append(act_layer)
        # 如果激活层的参数存在，则将激活层添加到卷积层列表中。

    elif order == 'norm-act-conv':
        # 如果顺序参数 order 为 'norm-act-conv'，即先标准化，然后激活，最后卷积。
        conv_layer = []
        # 初始化一个空的卷积层列表。

        norm_layer = create_norm(norm_args, in_channels, dimension='2d')
        # 创建一个标准化层，这次传入的是输入通道数，因为标准化在卷积之前。

        bias = False if norm_layer is not None else bias
        # 同上，如果标准化层存在，设置偏置项为 False。

        if norm_layer is not None:
            conv_layer.append(norm_layer)
        # 如果标准化层存在，将其添加到卷积层列表。

        act_layer = create_act(act_args)
        # 创建激活层。

        if act_args is not None:
            conv_layer.append(act_layer)
        # 如果激活层参数存在，添加激活层到列表。

        conv_layer.append(Conv2d(*args, bias=bias, **kwargs))
        # 添加卷积层到列表。

    elif order == 'conv-act-norm':
        # 如果顺序参数 order 为 'conv-act-norm'，即先卷积，然后激活，最后标准化。
        norm_layer = create_norm(norm_args, out_channels, dimension='2d')
        # 创建标准化层。

        bias = False if norm_layer is not None else bias
        # 如果标准化层存在，设置偏置项为 False。

        conv_layer = [Conv2d(*args, bias=bias, **kwargs)]
        # 创建卷积层列表。

        act_layer = create_act(act_args)
        # 创建激活层。

        if act_args is not None:
            conv_layer.append(act_layer)
        # 如果激活层参数存在，添加激活层到列表。

        if norm_layer is not None:
            conv_layer.append(norm_layer)
        # 如果标准化层存在，添加标准化层到列表。

    else:
        raise NotImplementedError(f"{order} is not supported")
    # 如果 order 参数不是上面三种之一，抛出一个 NotImplementedError 异常。

    return MultiInputSequential(*conv_layer)
    # 使用 nn.Sequential 容器将卷积层列表封装成一个顺序模型，并返回这个模型。



def create_convblock1d(*args,
                       norm_args=None, act_args=None, order='conv-norm-act', **kwargs):
    out_channels = args[1]
    in_channels = args[0]
    bias = kwargs.pop('bias', True)
    if order == 'conv-norm-act':
        norm_layer = create_norm(norm_args, out_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        conv_layer = [Conv1d(*args, bias=bias, **kwargs)]
        if norm_layer is not None:
            conv_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)

    elif order == 'norm-act-conv':
        conv_layer = []
        norm_layer = create_norm(norm_args, in_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        if norm_layer is not None:
            conv_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)
        conv_layer.append(Conv1d(*args, bias=bias, **kwargs))

    elif order == 'conv-act-norm':
        norm_layer = create_norm(norm_args, out_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        conv_layer = [Conv1d(*args, bias=bias, **kwargs)]
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)
        if norm_layer is not None:
            conv_layer.append(norm_layer)
    else:
        raise NotImplementedError(f"{order} is not supported")

    return nn.Sequential(*conv_layer)


def create_linearblock(*args,
                       norm_args=None,
                       act_args=None,
                       order='conv-norm-act',
                       **kwargs):
    in_channels = args[0]
    out_channels = args[1]
    bias = kwargs.pop('bias', True)

    if order == 'conv-norm-act':
        norm_layer = create_norm(norm_args, out_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        linear_layer = [nn.Linear(*args, bias, **kwargs)]
        if norm_layer is not None:
            linear_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            linear_layer.append(act_layer)
    elif order == 'norm-act-conv':
        linear_layer = []
        norm_layer = create_norm(norm_args, in_channels, dimension='1d')
        bias = kwargs.pop('bias', True)
        bias = False if norm_layer is not None else bias
        if norm_layer is not None:
            linear_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            linear_layer.append(act_layer)
        linear_layer.append(nn.Linear(*args, bias=bias, **kwargs))

    elif order == 'conv-act-norm':
        norm_layer = create_norm(norm_args, out_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        linear_layer = [nn.Linear(*args, bias=bias, **kwargs)]
        act_layer = create_act(act_args)
        if act_args is not None:
            linear_layer.append(act_layer)
        if norm_layer is not None:
            linear_layer.append(norm_layer)

    return nn.Sequential(*linear_layer)


class CreateResConvBlock2D(nn.Module):
    def __init__(self, mlps,
                 norm_args=None, act_args=None, order='conv-norm-act', **kwargs):
        super().__init__()
        self.convs = nn.Sequential()
        for i in range(len(mlps) - 2):
            self.convs.add_module(f'conv{i}',
                                  create_convblock2d(mlps[i], mlps[i + 1],
                                                     norm_args=norm_args, act_args=act_args, order=order, **kwargs))
        self.convs.add_module(f'conv{len(mlps) - 1}',
                              create_convblock2d(mlps[-2], mlps[-1], norm_args=norm_args, act_args=None, **kwargs))

        self.act = create_act(act_args)

    def forward(self, x, res=None):
        if res is None:
            return self.act(self.convs(x) + x)
        else:
            return self.act(self.convs(x) + res)
