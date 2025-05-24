import torch
import torch.nn as nn
import torch.nn.functional as F

#* 18 34层的模型使用的残差结构
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion,
                                kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        #! N x in x W x H
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        #! N x out x W x H (stride = 1) or N x out x W/2 x H/2
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #! N x out x W x H (stride = 1) or N x out x W/2 x H/2
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

#* 更深层的模型使用的残差结构
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                    groups=1, width_per_group=64):
        super().__init__()
        width = int(out_channels * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv2d(in_channels, width,
                                kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width, width, groups=groups,
                                kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels*self.expansion,
                                kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, include_top=True, init_weights=False,
                    groups=1, width_per_group=64):
        super().__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group
        #! N x 3 x 224 x 224
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7,
                                stride=2, padding=3, bias=False)
        #! N x 64 x 112 x 112
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   #* output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, channel, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride,
                            groups=self.groups, width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def resnet34(num_classes=1000, include_top=True, init_weights=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, include_top, init_weights)

def resnet18(num_classes=1000, include_top=True, init_weights=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, include_top, init_weights)

def resnet50(num_classes=1000, include_top=True, init_weights=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, include_top, init_weights)

def resnet101(num_classes=1000, include_top=True, init_weights=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, include_top, init_weights)

def resnext50_32x4d(num_classes=1000, include_top=True, init_weights=True):
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, include_top, init_weights,
                    groups=groups, width_per_group=width_per_group)

def resnext101_32x8d(num_classes=1000, include_top=True, init_weights=True):
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, include_top, init_weights,
                    groups=groups, width_per_group=width_per_group)
