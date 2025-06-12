import torch, math
import torch.nn as nn

class ConvBNMish(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.mish = nn.Mish(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.mish(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channel, hidden_channel=None):
        super().__init__()
        if hidden_channel is None:
            hidden_channel = in_channel

        self.block = nn.Sequential(
            ConvBNMish(in_channel, hidden_channel, kernel_size=1),
            ConvBNMish(hidden_channel, in_channel, kernel_size=3)
        )

    def forward(self, x):
        return x + self.block(x)


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel, repeat=3, first=False):
        super().__init__()
        self.downsample_conv = ConvBNMish(in_channel, out_channel, kernel_size=3, stride=2)
        if first:
            #* 如果是第一个Downsample结构，不进行下采样 in_channel = 32, out_channel=64
            self.left_side = ConvBNMish(out_channel, out_channel, kernel_size=1, stride=1)
            self.right_side = nn.Sequential(
                ConvBNMish(out_channel, out_channel, kernel_size=1, stride=1),
                ResBlock(out_channel, hidden_channel=out_channel // 2),
                ConvBNMish(out_channel, out_channel, kernel_size=1, stride=1)
            )
            self.concat_conv = ConvBNMish(out_channel*2, out_channel, kernel_size=1, stride=1)
        else:
            #* 如果不是第一个Downsample结构，进行下采样
            self.left_side = ConvBNMish(out_channel, out_channel // 2, kernel_size=1, stride=1)
            layers = []
            layers.append(ConvBNMish(out_channel, out_channel // 2, kernel_size=1, stride=1))
            for _ in range(repeat):
                layers.append(ResBlock(out_channel // 2))
            layers.append(ConvBNMish(out_channel // 2, out_channel // 2, kernel_size=1, stride=1))
            self.right_side = nn.Sequential(
                ConvBNMish(out_channel, out_channel // 2, kernel_size=1, stride=1),
                *[ResBlock(out_channel // 2) for _ in range(repeat)],
                ConvBNMish(out_channel // 2, out_channel // 2, kernel_size=1, stride=1)
            )
            self.concat_conv = ConvBNMish(out_channel, out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.downsample_conv(x)
        left = self.left_side(x)
        right = self.right_side(x)
        x = torch.cat([left, right], dim=1)
        x = self.concat_conv(x)
        return x

class CSPDarkNet53(nn.Module):
    def __init__(self, channels: list[int]=[64, 128, 256, 512, 1024], repeats: list[int]=[2, 8, 8, 4]):
        super().__init__()
        assert len(channels) == 5
        assert len(repeats) == 4
        self.conv1 = ConvBNMish(3, 32, kernel_size=3, stride=1)

        self.sample = nn.ModuleDict()
        self.sample["sample1"] = DownSample(32, channels[0], first=True)

        for i, repeat in enumerate(repeats):
            self.sample[f'sample{i+2}'] = DownSample(channels[i], channels[i+1],repeat=repeat)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sample["sample1"](x)
        x = self.sample["sample2"](x)
        out1 = self.sample["sample3"](x)
        out2 = self.sample["sample4"](out1)
        out3 = self.sample["sample5"](out2)
        return out1, out2, out3

class SPP(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.out_channel = in_channel * 4

    def forward(self, x):
        s1 = self.maxpool1(x)
        s2 = self.maxpool2(x)
        s3 = self.maxpool3(x)
        x = torch.cat([x, s1, s2, s3], dim=1)
        return x

class ConvBNLeaky(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.05)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvSet(nn.Module):
    def __init__(self, in_channel, out_channel, num: int):
        super().__init__()
        self.conv1 = ConvBNLeaky(in_channel, out_channel, kernel_size=1, stride=1)
        self.conv2 = ConvBNLeaky(out_channel, out_channel * 2, kernel_size=3, stride=1)
        if num < 3:
            self.conv3 = ConvBNLeaky(out_channel * 2, out_channel, kernel_size=1, stride=1)
        else:
            self.conv3 = nn.Sequential(
                ConvBNLeaky(out_channel * 2, out_channel, kernel_size=1, stride=1),
                ConvBNLeaky(out_channel, out_channel * 2, kernel_size=3, stride=1),
                ConvBNLeaky(out_channel * 2, out_channel, kernel_size=1, stride=1)
            )
        self.out_channel = out_channel

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = ConvBNLeaky(in_channel, out_channel, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.upsample(self.conv(x))

class BackBoneWithPAN(nn.Module):
    def __init__(self, num_classes:int, in_channels: list[int] = [256, 512, 1024]):
        super().__init__()
        assert len(in_channels) == 3
        self.num_classes = num_classes
        self.dark53 = CSPDarkNet53()
        #! convset1 -> [13, 13, 512]
        self.convset1 = ConvSet(in_channels[2], 512, num=1)
        #! spp -> [13, 13, 2048]
        self.spp = SPP(self.convset1.out_channel)
        #! convset2 -> [13, 13, 512]
        self.convset2 = ConvSet(self.spp.out_channel, 512, num=2)
        self.upsample1 = Upsample(self.convset2.out_channel, 256)
        self.Convset3 = ConvSet(256*2, 256, num=3)
        self.upsample2 = Upsample(self.Convset3.out_channel, 128)
        self.Convset4 = ConvSet(128*2, 128, num=4)
        self.downsample1 = ConvBNLeaky(self.Convset4.out_channel, 256, kernel_size=3, stride=2)
        self.Convset5 = ConvSet(256*2, 256, num=5)
        self.downsample2 = ConvBNLeaky(self.Convset5.out_channel, 512, kernel_size=3, stride=2)
        self.Convset6 = ConvSet(512*2, 512, num=6)
        self.side_out1 = nn.Sequential(
            ConvBNLeaky(self.Convset6.out_channel, 1024, kernel_size=3, stride=1),
            nn.Conv2d(1024, 3 * (self.num_classes + 5), kernel_size=1, stride=1)
        )
        self.side_out2 = nn.Sequential(
            ConvBNLeaky(self.Convset5.out_channel, 512, kernel_size=3, stride=1),
            nn.Conv2d(512, 3 * (self.num_classes + 5), kernel_size=1, stride=1)
        )
        self.side_out3 = nn.Sequential(
            ConvBNLeaky(self.Convset4.out_channel, 256, kernel_size=3, stride=1),
            nn.Conv2d(256, 3 * (self.num_classes + 5), kernel_size=1, stride=1)
        )

        stride = [32, 16, 8]
        self.yolo1 = YOLOLayer(anchors=[(116, 90), (156, 198), (373, 326)], nc=self.num_classes, stride=stride[0])
        self.yolo2 = YOLOLayer(anchors=[(30, 61), (62, 45), (59, 119)], nc=self.num_classes, stride=stride[1])
        self.yolo3 = YOLOLayer(anchors=[(10, 13), (16, 30), (33, 23)], nc=self.num_classes, stride=stride[2])

    def forward(self, x):
        darkout1, darkout2, darkout3 = self.dark53(x)
        side1 = self.convset1(darkout3)
        side1 = self.spp(side1)
        side1 = self.convset2(side1)
        side1_up = self.upsample1(side1)
        side1_up = torch.cat([side1_up, darkout2], dim=1)
        side2 = self.Convset3(side1_up)
        side2_up = self.upsample2(side2)
        side2_up = torch.cat([side2_up, darkout1], dim=1)
        side3 = self.Convset4(side2_up)
        side3_down = self.downsample1(side3)
        side2 = torch.cat([side2, side3_down], dim=1)
        side2 = self.Convset5(side2)
        side2_down = self.downsample2(side2)
        side1 = torch.cat([side1, side2_down], dim=1)
        side1 = self.Convset6(side1)
        out1 = self.yolo1(self.side_out1(side1))
        out2 = self.yolo2(self.side_out2(side2))
        out3 = self.yolo3(self.side_out3(side3))
        return out1, out2, out3

class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, stride):
        super().__init__()
        self.anchors = torch.tensor(anchors)
        self.stride = stride
        self.na = len(anchors)
        #* num of classes
        self.nc = nc
        #* num of output = x, y, w, h, conf, cls(nc)
        self.no = nc + 5
        self.nx, self.ny, self.ng = 0, 0, (0, 0)
        #* 将anchors大小缩放到grid尺度
        self.anchor_vec = self.anchors / self.stride
        #! [batch_size, na, grid_h, grid_w, wh]
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

    def create_grids(self, ng=(13, 13), device="cpu"):
        """
        更新grids信息并生成新的grids参数
        :param ng: 特征图大小
        :param device:
        :return:
        """
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
        if not self.training:  # 训练模式不需要回归到最终预测boxes
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)], indexing='ij')
            # batch_size, na, grid_h, grid_w, wh
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, x):
        bs, _, ny, nx = x.shape
        if (self.nx, self.ny) != (nx, ny) or self.grid is None:
            self.create_grids((nx, ny), x.device)
        #! x -> [batch, 255, 13, 13] -> view[batch, 3, 85, 13, 13]
        #! -> [batch, anchor=3, grid_h=13, grid_w=13, xywh + obj + classes=85]
        x = x.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()
        if self.training:
            return x
        else:
            io = x.clone()
            #* 将回归参数映射到特征图上
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
            #* 换算到原图尺寸
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), x #! [1, 3, 13, 13, 85] -> [1, 507, 85]


class YOLOv4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = BackBoneWithPAN(num_classes=num_classes)
        #! The outputs of neck = ([B, 3*(num_classes+5), 13, 13], [B, 3*(num_classes+5), 26, 26], [B, 3*(num_classes+5), 52, 52])
        m_last = None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m_last = m
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, YOLOLayer):
                bias = m_last.bias.view(m.na, -1)
                with torch.no_grad():
                    bias[:, 4] += -4.5
                    bias[:, 5:] += math.log(0.6 / (m.nc - 0.99))
                m.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)

    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":
    model = YOLOv4(num_classes=20)
    for m in model.modules():
        if isinstance(m, CSPDarkNet53):
            for param in m.parameters():
                param.requires_grad = False
