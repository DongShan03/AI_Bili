from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

def drop_path(x, drop_prob: float=0., training: bool=False):
    if drop_path == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    #! shape = (B, 1, 1, 1)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    #* rand生成0到1的数，如果加上keep_prob任然小于1说明该点需要失活 变为0
    #* 若大于1说明存活 改为1 因此只需要向下取整后 为0失活
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    #? 可以替换为以下两段代码
    # random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
    # random_tensor = random_tensor.to(x.dtype)  # 转换为与x相同的数据类型

    # output = x.div(keep_prob) * random_tensor
    output = x * random_tensor / keep_prob
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNActivation(nn.Sequential):
    def __init__(self, in_channel,
                out_channel, kernel_size: int = 3,
                stride: int = 1, groups: int = 1,
                norm_layer: Optional[Callable[..., nn.Module]] = None,
                activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size,
                    stride, padding, groups=groups, bias=False),
            norm_layer(out_channel),
            activation_layer()
        )

#! SE模块
class SqueezeExcitation(nn.Module):
    def __init__(self, input_channel: int,
                expand_channel: int,
                se_ratio: float = 0.25):
        super().__init__()
        squeeze_channel = int(input_channel * se_ratio)
        #* 用1x1的卷积代替全连接层,原卷积第一层的节点个数为输入MBConv特征矩阵的channels(input_channel)的1/4
        #* 第二层的节点个数为DW Conv层输出的特征矩阵的channels 即 expand_channel
        self.fc1 = nn.Conv2d(expand_channel, squeeze_channel, 1)
        self.ac1 = nn.SiLU()
        self.fc2 = nn.Conv2d(squeeze_channel, expand_channel, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, (1, 1))
        #* 等价 scale = x.mean([2, 3], keepdim=True)
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x
