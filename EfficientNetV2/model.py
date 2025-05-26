from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

def add_attributes(**kwargs):
    """装饰器为函数添加属性"""
    def decorator(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorator

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


class ConvBNActivation(nn.Module):
    def __init__(self,
                in_channel: int,
                out_channel: int,
                kernel_size: int = 3,
                stride: int = 1,
                groups: int = 1,
                norm_layer: Optional[Callable[..., nn.Module]] = None,
                activation_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        self.conv = nn.Conv2d(in_channels=in_channel,
                            out_channels=out_channel,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            groups=groups,
                            bias=False)

        self.bn = norm_layer(out_channel)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result

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
        #* 等价 scale = F.adaptive_avg_pool2d(x, (1, 1))
        scale = x.mean([2, 3], keepdim=True)
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x

class MBConv(nn.Module):
    def __init__(self, input_channel, output_channel,
                kernel_size: int,
                expand_ratio: int,
                stride: int,
                se_ratio:float,
                drop_rate: float,
                norm_layer: Callable[..., nn.Module]=None):
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError("stride should be 1 or 2")
        self.has_short_cut = (stride==1 and input_channel==output_channel)
        activation_layer = nn.SiLU
        expand_channel = input_channel * expand_ratio

        #! 在EfficientNetV2中MBConv不存在expand_ratio为1的情况
        assert expand_ratio != 1
        self.expand_conv = ConvBNActivation(
            in_channel=input_channel,
            out_channel=expand_channel,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer
        )

        self.dwconv = ConvBNActivation(
            in_channel=expand_channel,
            out_channel=expand_channel,
            kernel_size=kernel_size,
            stride=stride,
            groups=expand_channel,
            norm_layer=norm_layer,
            activation_layer=activation_layer
        )

        self.se = SqueezeExcitation(
            input_channel=input_channel,
            expand_channel=expand_channel,
            se_ratio=se_ratio
        ) if se_ratio > 0 else nn.Identity()

        self.project_conv = ConvBNActivation(
            in_channel=expand_channel,
            out_channel=output_channel,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.Identity
        )

        self.output_channel = output_channel
        self.drop_rate = drop_rate
        if self.has_short_cut and drop_rate > 0.:
            self.dropout = DropPath(drop_rate)

    def forward(self, x):
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)
        if self.has_short_cut:
            if self.drop_rate > 0.:
                result = self.dropout(result)
            result += x
        return result

class FusedMBConv(nn.Module):
    def __init__(self, input_channel,
                output_channel,
                kernel_size: int,
                expand_ratio: int,
                stride: int,
                se_ratio:float,
                drop_rate: float,
                norm_layer: Callable[..., nn.Module]=None):
        super().__init__()
        assert stride in [1, 2]
        assert se_ratio == 0
        self.has_shortcut = stride == 1 and input_channel == output_channel
        self.drop_rate = drop_rate
        self.has_expansion = expand_ratio != 1
        activation_layer = nn.SiLU
        expand_channel = input_channel * expand_ratio

        if self.has_expansion:
            self.expand_conv = ConvBNActivation(
                in_channel=input_channel,
                out_channel=expand_channel,
                kernel_size=kernel_size,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
            self.project_conv = ConvBNActivation(
                in_channel=expand_channel,
                out_channel=output_channel,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Identity
            )
        else:
            self.project_conv = ConvBNActivation(
                in_channel=input_channel,
                out_channel=output_channel,
                kernel_size=kernel_size,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )

        self.output_channel = output_channel
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0.:
            self.dropout = DropPath(drop_rate)

    def forward(self, x):
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_rate > 0.:
                result = self.dropout(result)
            result += x
        return result

class EfficientNetV2(nn.Module):
    def __init__(self, model_cnfs: list,
                num_classes: int = 1000,
                num_features: int = 1280,
                dropout_rate: float = 0.2,
                drop_connect_rate: float = 0.2,
                init_weights: bool = True):
        super().__init__()
        for cnf in model_cnfs:
            assert len(cnf) == 8
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
        self.stem = ConvBNActivation(
            in_channel=3,
            out_channel=model_cnfs[0][4],
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer
        )

        total_blocks = sum([i[0] for i in model_cnfs])
        block_id = 0
        blocks = []
        for cnf in model_cnfs:
            repeat = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeat):
                blocks.append(op(
                    input_channel=cnf[4] if i == 0 else cnf[5],
                    output_channel=cnf[5],
                    kernel_size=cnf[1],
                    expand_ratio=cnf[3],
                    stride=cnf[2] if i == 0 else 1,
                    se_ratio=cnf[-1],
                    drop_rate=drop_connect_rate * block_id / total_blocks,
                    norm_layer=norm_layer
                ))
                block_id += 1
        self.blocks = nn.Sequential(*blocks)
        head_input_channel = model_cnfs[-1][-3]
        head = OrderedDict()

        head.update({
            "project_conv": ConvBNActivation(
                in_channel=head_input_channel,
                out_channel=num_features,
                kernel_size=1,
                norm_layer=norm_layer
            )
        })

        head.update({
            "avgpool": nn.AdaptiveAvgPool2d(1)
        })
        head.update({
            "flatten": nn.Flatten(start_dim=1)
        })
        if dropout_rate > 0.:
            head.update({
                "dropout": nn.Dropout(dropout_rate, inplace=True)
            })
        head.update({
            "classifier": nn.Linear(num_features, num_classes)
        })
        self.head = nn.Sequential(head)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

@add_attributes(train_size=300, eval_size=384)
def efficientV2_s(num_classes=1000):
    #! train_size=300, eval_size=384
    model_cnfs = [
        #* repeat, kernel_size, stride, expansion, input, output, operator, se_ratio
        [2, 3, 1, 1, 24, 24, 0, 0],
        [4, 3, 2, 4, 24, 48, 0, 0],
        [4, 3, 2, 4, 48, 64, 0, 0],
        [6, 3, 2, 4, 64, 128, 1, 0.25],
        [9, 3, 1, 6, 128, 160, 1, 0.25],
        [15, 3, 2, 6, 160, 256, 1, 0.25],
    ]
    return EfficientNetV2(model_cnfs, num_classes=num_classes, dropout_rate=0.2)


@add_attributes(train_size=384, eval_size=480)
def efficientV2_m(num_classes=1000):
    #! train_size=384, eval_size=480
    model_cnfs = [
        #* repeat, kernel_size, stride, expansion, input, output, operator, se_ratio
        [3, 3, 1, 1, 24, 24, 0, 0],
        [5, 3, 2, 4, 24, 48, 0, 0],
        [5, 3, 2, 4, 48, 80, 0, 0],
        [7, 3, 2, 4, 80, 160, 1, 0.25],
        [14, 3, 1, 6, 160, 176, 1, 0.25],
        [18, 3, 2, 6, 176, 304, 1, 0.25],
        [5, 3, 1, 6, 304, 512, 1, 0.25],
    ]
    return EfficientNetV2(model_cnfs, num_classes=num_classes, dropout_rate=0.3)

@add_attributes(train_size=384, eval_size=480)
def efficientV2_l(num_classes=1000):
    #! train_size=384, eval_size=480
    model_cnfs = [
        #* repeat, kernel_size, stride, expansion, input, output, operator, se_ratio
        [4, 3, 1, 1, 32, 32, 0, 0],
        [7, 3, 2, 4, 32, 64, 0, 0],
        [7, 3, 2, 4, 64, 96, 0, 0],
        [10, 3, 2, 4, 96, 192, 1, 0.25],
        [19, 3, 1, 6, 192, 224, 1, 0.25],
        [25, 3, 2, 6, 224, 384, 1, 0.25],
        [7, 3, 1, 6, 384, 640, 1, 0.25],
    ]
    return EfficientNetV2(model_cnfs, num_classes=num_classes, dropout_rate=0.4)
