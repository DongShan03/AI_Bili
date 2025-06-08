import os, sys
sys.path.append(os.path.dirname(__file__))
import torch
import torch.nn as nn
from model import YOLOLayer

class Convolutional(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, leaky=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=(kernel_size-1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.leaky = nn.LeakyReLU(leaky, inplace=True)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class Residual(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.convolutional1 = Convolutional(in_channel, in_channel // 2, kernel_size=1, stride=1)
        self.convolutional2 = Convolutional(in_channel // 2, in_channel, kernel_size=3, stride=1)

    def forward(self, x):
        identity = x
        x = self.convolutional1(x)
        x = self.convolutional2(x)
        x = x + identity
        return x

class SPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

    def forward(self, x):
        res1 = self.maxpool1(x)
        res2 = self.maxpool2(x)
        res3 = self.maxpool3(x)
        x = torch.cat([x, res1, res2, res3], dim=1)
        return x

class convSet(nn.Module):
    def __init__(self, in_channel, first_filter):
        super().__init__()
        self.convset = nn.Sequential(
            Convolutional(in_channel, first_filter, kernel_size=1, stride=1),
            Convolutional(first_filter, first_filter * 2, kernel_size=3, stride=1),
            Convolutional(first_filter * 2, first_filter, kernel_size=1, stride=1),
            Convolutional(first_filter, first_filter * 2, kernel_size=3, stride=1),
            Convolutional(first_filter * 2, first_filter, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return self.convset(x)

class YoloSPP(nn.Module):
    def __init__(self, img_size=(416, 416), channels: list=[32, 64, 128, 256, 512, 1024], repeats: list=[1, 2, 8, 8, 4], num_classes=20):
        super().__init__()
        self.start = Convolutional(3, channels[0], kernel_size=3, stride=1)
        self.convs = [Convolutional(channels[i], channels[i+1], kernel_size=3, stride=2) for i in range(len(channels)-1)]
        self.residuals = [nn.Sequential(*[Residual(channels[i+1]) for _ in range(repeats[i])]) for i in range(len(repeats))]
        self.convs2 = nn.Sequential(
            Convolutional(1024, 512, kernel_size=1, stride=1),
            Convolutional(512, 1024, kernel_size=3, stride=1),
            Convolutional(1024, 512, kernel_size=1, stride=1),
        )
        self.convs3 = nn.Sequential(
            Convolutional(2048, 512, kernel_size=1, stride=1),
            Convolutional(512, 1024, kernel_size=3, stride=1),
            Convolutional(1024, 512, kernel_size=1, stride=1),
        )
        self.spp = SPP()
        self.out1_conv = Convolutional(512, 1024, kernel_size=3, stride=1)
        self.out1_conv2 = nn.Conv2d(1024, 3 * (4 + 1 + num_classes), kernel_size=1, stride=1, padding=0)
        self.conv4 = Convolutional(512, 256, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.convSet1 = convSet(256+512, 256)
        self.out2_conv = Convolutional(256, 512, kernel_size=3, stride=1)
        self.out2_conv2 = nn.Conv2d(512, 3 * (4 + 1 + num_classes), kernel_size=1, stride=1, padding=0)
        self.conv5 = Convolutional(256, 128, kernel_size=1, stride=1)
        self.convSet2 = convSet(128+256, 128)
        self.out3_conv = Convolutional(128, 256, kernel_size=3, stride=1)
        self.out3_conv2 = nn.Conv2d(256, 3 * (4 + 1 + num_classes), kernel_size=1, stride=1, padding=0)
        self.yolo1 = YOLOLayer(anchors=[(10, 13), (16, 30), (33, 23)],
                               nc=num_classes,
                               img_size=img_size,
                               stride=32)
        self.yolo2 = YOLOLayer(anchors=[(30, 61), (62, 45), (59, 119)],
                               nc=num_classes,
                               img_size=img_size,
                               stride=16)
        self.yolo3 = YOLOLayer(anchors=[(116, 90), (156, 198), (373, 326)],
                               nc=num_classes,
                               img_size=img_size,
                               stride=8)
    def forward(self, x):
        x = self.start(x)
        shortcut1 = shortcut2 = None
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.residuals[i](x)
            if i == 2:
                shortcut1 = x
            if i == 3:
                shortcut2 = x
        x = self.convs2(x)
        x = self.spp(x)
        out1 = self.convs3(x)
        output1 = self.out1_conv2(self.out1_conv(out1))
        output1 = self.yolo1(output1)
        out1 = self.upsample(self.conv4(out1))
        out1 = torch.cat([out1, shortcut2], dim=1)
        out2 = self.convSet1(out1)
        output2 = self.out2_conv2(self.out2_conv(out2))
        output2 = self.yolo2(output2)
        out2 = self.upsample(self.conv5(out2))
        out2 = torch.cat([out2, shortcut1], dim=1)
        out3 = self.convSet2(out2)
        output3 = self.out3_conv2(self.out3_conv(out3))
        output3 = self.yolo3(output3)
        yolo_out = [output1, output2, output3]
        if self.training:
            return yolo_out
        else:
            x, p = zip(*yolo_out)
            x = torch.cat(x, 1)
            return x, p


if __name__ == '__main__':
    model = YoloSPP()
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    x, p = model(x)
    print(x.shape)
