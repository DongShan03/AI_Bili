import torch, math
import torch.nn as nn
import warnings
import numpy as np
import os, sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from YOLOv5.utils.utils import *

def autopad(kernel_size, padding=None, dilation=1):
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1 if isinstance(kernel_size, int) \
            else [dilation * (x - 1) + 1 for x in kernel_size]
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) \
            else [x // 2 for x in kernel_size]
    return padding

class Conv(nn.Module):
    default_act = nn.SiLU(inplace=True)

    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, autopad(kernel_size, padding, dilation),
                            groups=groups, bias=False, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_duse(self, x):
        return self.act(self.conv(x))

class DWConv(Conv):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, dilation=1, act=True):
        super().__init__(in_channel, out_channel, kernel_size, stride, \
                        groups=math.gcd(in_channel, out_channel), dilation=dilation, act=act)

class DWConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, in_padding=0, out_padding=0):
        super().__init__(in_channel, out_channel, kernel_size, stride, in_padding, out_padding, groups=math.gcd(in_channel, out_channel))

class TransformLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc2 = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x

class TransformerBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if in_channel != out_channel:
            self.conv = Conv(in_channel, out_channel)
        self.linear = nn.Linear(out_channel, out_channel)
        self.tr = nn.Sequential(
            *(TransformLayer(out_channel, num_heads) for _ in range(num_layers))
        )
        self.out_channel = out_channel

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        #! x.flatten(2) -> [b, _, h * w] -> [h*w, b, _]
        p = x.flatten(2).permute(2, 0, 1)
        #! [h*w, b, _] -> [h*w, b, _] -> [b, _, w*h] -> [b, _, w, h]
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.out_channel, w, h)

class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channel = int(out_channel * expansion)
        self.cv1 = Conv(in_channel, hidden_channel, 1, 1)
        self.cv2 = Conv(hidden_channel, out_channel, 3, 1, groups=groups)
        self.add = shortcut and in_channel == out_channel

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    #* C3的大体部分
    def __init__(self, in_channel, out_channel, repeats=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channel = int(out_channel * expansion)
        self.cv1 = Conv(in_channel, hidden_channel, 1, 1)
        self.cv2 = nn.Conv2d(in_channel, hidden_channel, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(hidden_channel, hidden_channel, 1, 1, bias=False)
        self.cv4 = Conv(2 * hidden_channel, out_channel, 1, 1)
        self.bn = nn.BatchNorm2d(2 * hidden_channel)
        self.act = nn.SiLU()
        self.m = nn.Sequential(
            *(Bottleneck(hidden_channel, hidden_channel, shortcut, groups, expansion=1.0) for _ in range(repeats))
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class CrossConv(nn.Module):
    #* 交叉卷积
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, \
                groups=1, expansion=1.0, shortcut=False):
        super().__init__()
        hidden_channel = int(out_channel * expansion)
        self.cv1 = Conv(in_channel, hidden_channel, (1, kernel_size), (1, stride))
        self.cv2 = Conv(hidden_channel, out_channel, (kernel_size, 1), (stride, 1), groups=groups)
        self.add = shortcut and in_channel == out_channel

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    def __init__(self, in_channel, out_channel, repeats=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channel = int(out_channel * expansion)
        self.cv1 = Conv(in_channel, hidden_channel, 1, 1)
        self.cv2 = Conv(in_channel, hidden_channel, 1, 1)
        self.cv3 = Conv(2 * hidden_channel, out_channel, 1)
        self.m = nn.Sequential(
            *(Bottleneck(hidden_channel, hidden_channel, shortcut, groups, expansion=1.0) for _ in range(repeats))
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class C3x(C3):
    #* 交叉卷积C3
    def __init__(self, in_channel, out_channel, repeats=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__(in_channel, out_channel, repeats, shortcut, groups, expansion)
        hidden_channel = int(out_channel * expansion)
        self.m = nn.Sequential(
            *(CrossConv(hidden_channel, hidden_channel, 3, 1, groups, 1.0, shortcut) for _ in range(repeats))
        )

class C3TR(C3):
    def __init__(self, in_channel, out_channel, repeats=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__(in_channel, out_channel, repeats, shortcut, groups, expansion)
        hidden_channel = int(out_channel * expansion)
        #* 4 -> num_heads   repeats -> num_layers
        self.m = TransformerBlock(hidden_channel, hidden_channel, 4, repeats)

class C3SPP(C3):
    def __init__(self, in_channel, out_channel, kernel_size=(5, 9, 13), repeats=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__(in_channel, out_channel, repeats, shortcut, groups, expansion)
        hidden_channel = int(out_channel * expansion)
        self.m = SPP(hidden_channel, hidden_channel, kernel_size)

class C3Ghost(C3):
    def __init__(self, in_channel, out_channel, repeats=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__(in_channel, out_channel, repeats, shortcut, groups, expansion)
        hidden_channel = int(out_channel * expansion)
        self.m = nn.Sequential(
            *(GhostBottleneck(hidden_channel, hidden_channel) for _ in range(repeats))
        )

class SPP(nn.Module):
    def __init__(self, in_channel, out_channel, k=(5, 9, 13)):
        super().__init__()
        hidden_channel = in_channel // 2
        self.cv1 = Conv(in_channel, hidden_channel, 1, 1)
        self.cv2 = Conv(hidden_channel * (len(k) + 1), out_channel, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class SPPF(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5):
        super().__init__()
        hidden_channel = in_channel // 2
        self.cv1  = Conv(in_channel, hidden_channel, 1, 1)
        self.cv2 = Conv(hidden_channel * 4, out_channel, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class GhostConv(nn.Module):
    #* 先用1*1卷积生成较小的特征图，在使用DW卷积
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, groups=1, act=True):
        super().__init__()
        hidden_channel = out_channel // 2
        self.cv1 = Conv(in_channel, hidden_channel, kernel_size, stride, None, groups, act=act)
        self.cv2 = Conv(hidden_channel, hidden_channel, 5, 1, None, hidden_channel, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)

class GhostBottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        hidden_channel = out_channel // 2
        self.conv = nn.Sequential(
            GhostConv(in_channel, hidden_channel, 1, 1), #* pw
            DWConv(hidden_channel, hidden_channel, kernel_size, stride, act=False) if stride==2 else nn.Identity(), #* dw
            GhostConv(hidden_channel, out_channel, 1, 1, act=False) #* pw-linear
        )
        self.shortcut = (
            nn.Sequential(
                DWConv(in_channel, in_channel, kernel_size, stride, act=False),
                Conv(in_channel, out_channel, 1, 1, act=False)
            ) if stride == 2 else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class Focus(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=None, groups=1, act=True):
        super().__init__()
        self.conv = Conv(in_channel * 4, out_channel, kernel_size, stride, padding, groups, act=act)

    def forward(self, x):
        #* reshaping (b,c,w,h) to (b,4c,w/2,h/2) then applies convolution
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))

class Contract(nn.Module):
    #* 将空间维度收缩为通道维度，以便在神经网络中进行高效处理
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(b, c * s * s, h // s, w // s)

class Expand(nn.Module):
    #* 与Contract相反 将通道维度扩张为空间维度
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, s, s, c // (s * s), h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(b, c // (s * s), h * s, w * s)

class Concat(nn.Module):
    #* 沿指定维度连接矩阵
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

#! Mixed Depthwise Convolutional
#! 混合卷积
class MixConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3, 5, 7),
                stride=1, dilation=1, bias=True, method="equal_params"):
        super().__init__()
        groups = len(kernel_size)
        if method == 'equal_ch':    #* 每一组的通道相同
            #* linspace 生成数量为out_channel的等差数列（min=0，max=groups-1E-6）
            #* 这里会生成group蒙版
            i = torch.linspace(0, groups - 1E-6, out_channel).floor()
            #* channels是每组的通道数
            channels = [(i == g).sum() for g in range(groups)]
        else:   #? method="equal_params" 每一组的参数量相同
            #* b -> [out_channel, 0, 0, 0, 0]
            b = [out_channel] + [0] * groups
            #* a -> groups+1行 group列 且a[1:, :]为对角矩阵
            a = np.eye(groups + 1, groups, k=-1)
            #* np.roll(a, 1, axis=1) -> 对a进行左移
            a -= np.roll(a, 1, axis=1)
            """
            [[ 0.,  0.,  0.],       [[ 0.,  0.,  0.],       [[ 1.,  1.,  1.],
            [ 1., -1.,  0.],        [ 9., -25.,  0.],        [ 9., -25., 0.],
            [ 0.,  1., -1.],        [ 0., 25., -49.],        [ 0., 25., -49.],
            [-1.,  0.,  1.]]        [ -9.,  0., 49.]]        [ -9.,  0., 49.]]
            这里的思想很简单，首先的要求是这个组的参数量相同
            而对于不同的kernel_size，其相对参数量为kernel_size ** 2
            设每组的通道数为X = [c1, c2, c3]
            首先有c1 + c2 + c3 = out_channel
            再根据每组的参数量相同可以得到c1 * ks[0] ** 2 = c2 * ks[1] ** 2 = c3 * ks[2] ** 2
            这里就有四个方程三个变量,所以a的维度为[4, 3] x的维度为[3, 1] b的维度为[4, 1]
            且b = [out_channel, 0, 0, 0], 再通过AX=B解得X
            """
            a *= np.array(kernel_size) ** 2
            a[0] = 1
            #* solve for equal weight indices, ax = b
            channels = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)

        self.m = nn.ModuleList([nn.Conv2d(
            in_channels=in_channel, out_channels=channels[g],
            kernel_size=kernel_size[g], stride=stride,
            padding=kernel_size[g] // 2, dilation=dilation, bias=bias
        ) for g in range(groups)])

    def forward(self, x):
        return torch.cat([m(x) for m in self.m], 1)

class Detect(nn.Module):
    #* Yolov5 Detect head
    stride = None
    dynamic = False

    def __init__(self, nc=80, anchors=(), channels=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in channels)  # output conv
        self.inplace = inplace

    def forward(self, x):
        z = []
        for i in range(self.nl):
            #! x -> [num_out, B, na * no, ny, nx]
            #! x[i] -> [B, channels[i], ny, nx]
            x[i] = self.m[i](x[i])
            #! x[i] -> [B, na * no, ny, nx]
            bs, _, ny, nx = x[i].shape
            #! x[i] -> [B, na, no, ny, nx] -> [B, na, ny, nx, no]
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        device = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        y, x = torch.arange(ny, device=device, dtype=t), torch.arange(nx, device=device, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

class Proto(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        #* YOLOv5 mask Proto module for segmentation models, performing convolutions and upsampling on input tensors
        super().__init__()
        self.cv1 = Conv(in_channel, hidden_channel, 3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(hidden_channel, hidden_channel, 3)
        self.cv3 = Conv(hidden_channel, out_channel)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

class Segment(Detect):
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, channels=(), inplace=True):
        super().__init__(nc, anchors, channels, inplace)
        self.nm = nm    #* num of mask
        self.npr = npr  #* num of protos
        self.no = 5 + nc + self.nm
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in channels)
        self.protos = Proto(channels[0], self.npr, self.nm)
        self.detect = Detect.forward

    def forward(self, x):
        p = self.protos(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p, x[1])

class Classify(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=None, groups=1, \
                dropout_p=0.0):
        super().__init__()
        hidden_channel = 1280
        self.conv = Conv(in_channel, hidden_channel, kernel_size, stride, autopad(kernel_size, padding), groups)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True) if dropout_p > 0.0 else nn.Identity()
        self.linear = nn.Linear(hidden_channel, out_channel)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.drop(x)
        x = self.linear(x)
        return x

def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                        stride=conv.stride, groups=conv.groups, padding=conv.padding, bias=True).requires_grad_(False).to(conv.weight.device)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

    b_conv = torch.zeros(conv.wight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv
