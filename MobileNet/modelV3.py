import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable, List

def _make_divisible(ch, divisor=8, min_ch=None):
    #* 使得所有层的通道数都是divisor的整数倍
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class ConvBNActivation(nn.Sequential):
    def __init__(self,
                in_planes: int,
                out_planes: int,
                kernel_size: int = 3,
                stride: int = 1,
                groups: int = 1,
                norm_layer: Optional[Callable[..., nn.Module]] = None,
                activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                    padding=padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channel: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_c = _make_divisible(input_channel // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channel, squeeze_c, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_c, input_channel, kernel_size=1)

    def forward(self, x: torch.Tensor):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x

class InvertedResidualConfig:
    def __init__(self, input_channel: int,
                kernel_size: int,
                expanded_channel: int,
                output_channel: int,
                use_se: bool,
                activation: str,
                stride: int,
                width_multi: float
                ):
        self.input_channel = self.adjust_channels(input_channel, width_multi)
        self.kernel_size = kernel_size
        self.expanded_channel = self.adjust_channels(expanded_channel, width_multi)
        self.output_channel = self.adjust_channels(output_channel, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)

class InvertedResidual(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig,
                norm_layer: Callable[..., nn.Module]):
        super().__init__()
        if cnf.stride not in [1, 2]:
            raise ValueError("stride should be 1 or 2")
        self.use_res_connect = (cnf.stride == 1 and cnf.input_channel == cnf.output_channel)

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        if cnf.expanded_channel != cnf.input_channel:
            layers.append(ConvBNActivation(cnf.input_channel,
                                        cnf.expanded_channel,
                                        kernel_size=1,
                                        norm_layer=norm_layer,
                                        activation_layer=activation_layer))

        #DW Conv
        layers.append(ConvBNActivation(cnf.expanded_channel,
                                    cnf.expanded_channel,
                                    kernel_size=cnf.kernel_size,
                                    stride=cnf.stride,
                                    groups=cnf.expanded_channel,
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_channel))
        layers.append(ConvBNActivation(cnf.expanded_channel,
                                    cnf.output_channel,
                                    kernel_size=1,
                                    norm_layer=norm_layer,
                                    activation_layer=nn.Identity)) #? nn.Identity表示原封不动的输出
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.output_channel

    def forward(self, x: torch.Tensor):
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result

class MobileNetV3(nn.Module):
    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig],
                last_channel: int, num_classes: int = 1000,
                block: Optional[Callable[..., nn.Module]] = None,
                norm_layer: Optional[Callable[..., nn.Module]] = None,
                init_weights=True):
        super().__init__()
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, List) and
                all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            #* partial冻结BatchNorm2d的参数
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        #* first layer
        firstconv_output_channel = inverted_residual_setting[0].input_channel
        layers.append(ConvBNActivation(3, firstconv_output_channel,
                                    stride=2, kernel_size=3,
                                    norm_layer=norm_layer,
                                    activation_layer=nn.Hardswish))
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        lastconv_input_channel = inverted_residual_setting[-1].output_channel
        lastconv_output_channel = 6 * lastconv_input_channel
        layers.append(ConvBNActivation(lastconv_input_channel,
                                    lastconv_output_channel,
                                    kernel_size=1,
                                    norm_layer=norm_layer,
                                    activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channel, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(last_channel, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(torch.flatten(x, 1))
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

def mobilenet_v3_large(num_classes=1000, reduces_tail=False) -> MobileNetV3:
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)
    reduce_divider = 2 if reduces_tail else 1
    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)
    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                    last_channel=last_channel,
                    num_classes=num_classes)
