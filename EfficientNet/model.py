import torch, math
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional
from collections import OrderedDict
from functools import partial
import copy

def add_attributes(**kwargs):
    """装饰器为函数添加属性"""
    def decorator(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorator

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
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

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channel: int,
                expand_channel: int,
                squeeze_factor: int = 4):
        super().__init__()
        squeeze_channel = input_channel // squeeze_factor
        self.fc1 = nn.Conv2d(expand_channel, squeeze_channel, 1)
        self.ac1 = nn.SiLU()
        self.fc2 = nn.Conv2d(squeeze_channel, expand_channel, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, (1, 1))
        #* scale = x.mean([2, 3], keepdim=True)
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class InvertedResidualConfig:
    def __init__(self, input_channel,
                output_channel,
                kernel_size,    #* 3 or 5
                expanded_ratio, #* 1 or 6
                stride,         #* 1 or 2
                use_se: bool,   #* 在这里全为True
                drop_rate: float,
                index: str,
                width_coefficient: float):
        #* input_channel针对B0模型，通过width_coefficient调整到适宜其他模型
        self.input_channel = self.adjust_channels(input_channel, width_coefficient)
        self.kernel_size = kernel_size
        self.expanded_channel = self.input_channel * expanded_ratio
        self.output_channel = self.adjust_channels(output_channel, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)

class InvertedResidual(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig,
                norm_layer: Callable[..., nn.Module]):
        super().__init__()
        if cnf.stride not in [1, 2]:
            raise ValueError("stride should be 1 or 2")
        self.use_res_connect = (cnf.stride==1 and cnf.input_channel==cnf.output_channel)
        layers = OrderedDict()
        activation_layer = nn.SiLU

        if cnf.expanded_channel != cnf.input_channel:
            layers.update({
                "expand_conv": ConvBNActivation(
                    cnf.input_channel,
                    cnf.expanded_channel,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer
                )
            })

        layers.update({
            "dwconv": ConvBNActivation(
                cnf.expanded_channel,
                cnf.expanded_channel,
                kernel_size=cnf.kernel_size,
                stride=cnf.stride,
                groups=cnf.expanded_channel,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        })

        if cnf.use_se:
            layers.update({
                "se": SqueezeExcitation(
                    cnf.input_channel,
                    cnf.expanded_channel
                )
            })

        layers.update({
            "project_conv": ConvBNActivation(
                cnf.expanded_channel,
                cnf.output_channel,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Identity
            )
        })

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.output_channel
        self.is_strided = cnf.stride > 1

        if cnf.drop_rate > 0 and self.use_res_connect:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x
        return result

class EfficientNet(nn.Module):
    def __init__(self, width_coefficient: float,
                depth_coefficient: float,
                num_classes: int = 1000,
                dropout_ratio: float = 0.2,
                drop_connect_rate: float = 0.2,     #* 这一项一直是 0.2
                block: Optional[Callable[..., nn.Module]] = None,
                norm_layer: Optional[Callable[..., nn.Module]] = None,
                init_weights=True):
        super().__init__()

        default_cnfs = [
            #* in_channel, out_channel, kernel_size, expand_ratio, stride, use_se, dropout_ratio, repeats
            [32, 16, 3, 1, 1, True, drop_connect_rate, 1],
            [16, 24, 3, 6, 2, True, drop_connect_rate, 2],
            [24, 40, 5, 6, 2, True, drop_connect_rate, 2],
            [40, 80, 3, 6, 2, True, drop_connect_rate, 3],
            [80, 112, 5, 6, 1, True, drop_connect_rate, 3],
            [112, 192, 5, 6, 2, True, drop_connect_rate, 4],
            [192, 320, 3, 6, 1, True, drop_connect_rate, 1],
        ]

        def round_repeats(repeats):
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                width_coefficient=width_coefficient)

        bneck_conf = partial(InvertedResidualConfig,
                            width_coefficient=width_coefficient)

        b = 0
        num_blocks = float(sum(round_repeats(cnf[-1]) for cnf in default_cnfs))
        inverted_residual_setting = []
        for stage, args in enumerate(default_cnfs):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:
                    #* strides equal 1 except the first cnf
                    cnf[-3] = 1
                    #* input_channel equal output_channel except the first cnf
                    cnf[0] = cnf[1]
                #* update dropout ratio
                cnf[-1] = args[-2] * b / num_blocks
                index = str(stage + 1) + chr(i + 97) #* 1a, 1b, 2a
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        layers = OrderedDict()

        #first conv
        layers.update({
            "stem_conv": ConvBNActivation(
                in_channel=3,
                out_channel=adjust_channels(32),
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
            )
        })

        for cnf in inverted_residual_setting:
            layers.update({
                cnf.index: block(cnf, norm_layer)
            })

        #build top
        last_conv_input_channel = inverted_residual_setting[-1].output_channel
        last_conv_output_channel = adjust_channels(1280)
        layers.update({
            "top": ConvBNActivation(
                last_conv_input_channel,
                last_conv_output_channel,
                kernel_size=1,
                norm_layer=norm_layer
            )
        })
        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        classifier = []
        if dropout_ratio > 0:
            classifier.append(nn.Dropout(p=dropout_ratio, inplace=True))
        classifier.append(nn.Linear(last_conv_output_channel, num_classes))
        self.classifier = nn.Sequential(*classifier)

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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

@add_attributes(img_size=224)
def efficientnet_b0(num_classes=1000):
    #! image size [224, 224]
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_ratio=0.2,
                        num_classes=num_classes)


@add_attributes(img_size=240)
def efficientnet_b1(num_classes=1000):
    #! image size [240, 240]
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_ratio=0.2,
                        num_classes=num_classes)


@add_attributes(img_size=260)
def efficientnet_b2(num_classes=1000):
    #! image size [260, 260]
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_ratio=0.3,
                        num_classes=num_classes)


@add_attributes(img_size=300)
def efficientnet_b3(num_classes=1000):
    #! image size [300, 300]
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_ratio=0.3,
                        num_classes=num_classes)


@add_attributes(img_size=380)
def efficientnet_b4(num_classes=1000):
    #! image size [380, 380]
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_ratio=0.4,
                        num_classes=num_classes)


@add_attributes(img_size=456)
def efficientnet_b5(num_classes=1000):
    #! image size [456, 456]
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_ratio=0.4,
                        num_classes=num_classes)


@add_attributes(img_size=528)
def efficientnet_b6(num_classes=1000):
    #! image size [528, 528]
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_ratio=0.5,
                        num_classes=num_classes)


@add_attributes(img_size=600)
def efficientnet_b7(num_classes=1000):
    #! image size [600, 600]
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_ratio=0.5,
                        num_classes=num_classes)
