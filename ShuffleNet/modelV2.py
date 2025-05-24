from typing import List, Callable
import torch
from torch import Tensor
import torch.nn as nn

def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    #* reshape
    #* [batch_size, num_channels, height, width] -> \
    #* [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)
    #! transpose交换维度 此时为[batch_size, channels_per_group, groups, height, width]
    #* transpose后数据会不连续，通过contiguous使数据在内存中连续
    x = torch.transpose(x, 1, 2).contiguous()
    #* flatten
    x = x.view(batch_size, -1, height, width)
    return x

class InvertedResidual(nn.Module):
    def __init__(self, input_channel:int, output_channel:int,
                stride: int):
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError("stride should be 1 or 2")
        self.stride = stride
        assert output_channel % 2 == 0

        branch_features = output_channel // 2
        assert (self.stride != 1) or (input_channel == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_channel, input_channel, kernel_size=3,
                                    stride=self.stride, padding=1),
                nn.BatchNorm2d(input_channel),
                nn.Conv2d(input_channel, branch_features, kernel_size=1, stride=1,
                        padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_channel if self.stride > 1 else branch_features, branch_features,
                    kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


    @staticmethod
    def depthwise_conv(input_channel, output_channel,
                    kernel_size, stride=1, padding=0,
                    bias=False) -> nn.Conv2d:
        return nn.Conv2d(input_channel, output_channel,
                        kernel_size, stride=stride,
                        padding=padding, bias=bias,
                        groups=input_channel)

class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats: List[int],
                stages_out_channels: List[int],
                num_classes: int = 1000,
                inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super().__init__()
        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stage_out_channels as list of 5 positive ints")
        self._stages_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stages_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3,
                    stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in range(2, 5)]
        #! 这里stage_names和stages_repeats只有三个元素，但self._stages_out_channels[1:]有四个元素
        #! 因此相当于只取前三个 即self._stages_out_channels[1:4]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                self._stages_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for _ in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stages_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        #! N x (1024 or 2048) x 7 x 7
        x = x.mean([2, 3])  #! 全局平均池化
        #! N x (1024 or 2048)
        x = self.fc(x)
        return x

def shufflenetv2_x1_0(num_classes=1000):
    return ShuffleNetV2(stages_repeats=[4, 8, 4],
                        stages_out_channels=[24, 116, 232, 464, 1024],
                        num_classes=num_classes)

def shufflenetv2_x0_5(num_classes=1000):
    return ShuffleNetV2(stages_repeats=[4, 8, 4],
                        stages_out_channels=[24, 48, 96, 192, 1024],
                        num_classes=num_classes)

def shufflenetv2_x1_5(num_classes=1000):
    return ShuffleNetV2(stages_repeats=[4, 8, 4],
                        stages_out_channels=[24, 176, 352, 704, 1024],
                        num_classes=num_classes)

def shufflenetv2_x2_0(num_classes=1000):
    return ShuffleNetV2(stages_repeats=[4, 8, 4],
                        stages_out_channels=[24, 244, 488, 976, 2048],
                        num_classes=num_classes)
