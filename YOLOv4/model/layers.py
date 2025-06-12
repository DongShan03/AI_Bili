import torch.nn.functional as F
import torch
from torch import nn
import math
import numpy as np

def make_divisible(v, divisor):
    return math.ceil(v / divisor) * divisor

class Reorg(nn.Module):
    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

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

#* 特征矩阵加权融合
class WeightedFeatureFusion(nn.Module):
    def __init__(self, layers, weight=False):
        super().__init__()
        self.layers = layers
        self.weight = weight
        self.n = len(layers) + 1
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)

    def forward(self, x, outputs):
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        return grad_output * (sx * (1 + x * (1 - sx)))

class MishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class MemoryEfficientMish(nn.Module):
    def forward(self, x):
        return MishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

#! 可变形卷积
class DeformConv2d(nn.Module):
    def __init__(self, in_channel,out_channel,
                kernel_size=3, padding=1,
                stride=1, bias=None, modulation=True):
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        #! 在四个方向上都填充padding个0
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                            stride=kernel_size, bias=bias)
        self.p_conv = nn.Conv2d(in_channel, 2*kernel_size*kernel_size, kernel_size=3,
                            padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(in_channel, kernel_size*kernel_size, kernel_size=3,
                                padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        #! offset -> [B, 2*ks*ks, H, W]
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        #! N = ks*ks
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)
        #! p -> [B, 2N, H, W]   可变卷积的卷积核
        p = self._get_p(offset, dtype)

        #! p -> [B, H, W, 2N]
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        #! 接下来的任务是获得p的坐标临近的四个点的坐标
        #! q_lt[..., :N]这些是x方向上的偏移 限制在0和W-1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        #* clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)
        #* 双线性插值 四个点对应的权重 -> [b, h, w, N]
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        #* (b, c, h, w, N)
        #* 计算四个点的值
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        #* (b, c, h, w, N) 得到线性插值的结果
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            #! permute: [B, ks*ks, H, W] -> [B, H, W, ks*ks]
            m = m.contiguous().permute(0, 2, 3, 1)
            #! [B, 1, H, W, ks*ks]
            m = m.unsqueeze(dim=1)
            #! [B, C, H, W, ks*ks]
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        #! x_offset -> [B, C, H*ks, W*ks]
        out = self.conv(x_offset)
        #! out -> [B, out_channel, H, W]
        return out

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        #! s = [0, ks, ks*2, ks*3, ..., ks*(ks-1)]
        #! x_offset[..., s:s+ks] -> [B, C, H, W, ks]
        #! view(b, c, h, w*ks) -> [B, C, H, W*ks]
        #! cat -> [B, C, H, W*ks*ks]
        #! view -> [B, C, H*ks, W*ks]
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

    def _get_p_n(self, N, dtype):
        """
        这里就是生成棋盘格 每个格子对应其相对坐标
        中心点为(0,0)
        左上角为(-(self.kernel_size-1) // 2, -(self.kernel_size-1) // 2)
        """
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1) // 2, (self.kernel_size-1) // 2 + 1),
            torch.arange(-(self.kernel_size-1) // 2, (self.kernel_size-1) // 2 + 1),
            indexing='ij'
        )
        #! p_n -> [-1, -1, -1, 0, -1, 1, ...]
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        #! p_0_x -> [h, w]
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride),
            indexing='ij'
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        #! p_0 -> [1, 2*N, h, w]
        return p_0

    def _get_p(self, offset, dtype):
        N, H, W = offset.size(1) // 2, offset.size(2), offset.size(3)

        #* [1, 2N, 1, 1]    这是正常卷积的偏移
        p_n = self._get_p_n(N, dtype)
        #* [1, 2N, h, w]    这是正常卷积的情况下卷积核中心对应的坐标
        p_0 = self._get_p_0(H, W, N, dtype)
        #! offset -> [B, 2*ks*ks, H, W]
        p = p_0 + p_n + offset
        #! p -> [B, 2N, H, W]
        return p

    def _get_x_q(self, x, q, N):
        #* 计算q对应x的每个值
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset


class GAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.avg(x)

class FeatureConcat(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.multiple = len(layers) > 1

    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]

class FeatureConcat2(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return torch.cat([outputs[self.layers[0]], outputs[self.layers[1]].detach()], 1)

class FeatureConcat3(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return torch.cat([outputs[self.layers[0]], outputs[self.layers[1]].detach(), outputs[self.layers[2]].detach()], 1)

#* 只融合前半部分通道
class FeatureConcat_l(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return torch.cat([outputs[i][:,:outputs[i].shape[1]//2,:,:] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]][:,:outputs[self.layers[0]].shape[1]//2,:,:]


class Silence(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

#* 通道缩放
class ScaleChannel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return x.expand_as(a) * a

#* 空间缩放
class ScaleSpatial(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return x * a

class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index,
                layers, stride):
        super().__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index
        self.layers = layers
        self.stride = stride
        self.nl = len(layers)
        self.na = len(anchors)
        self.nc = nc
        self.no = nc + 5
        self.nx, self.ny, self.ng = 0, 0, 0
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

    def create_grids(self, ng=(13, 13), device="cpu"):
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p, out):
        #* bs, 255, 13, 13
        bs, _, ny, nx = p.shape
        if (self.nx, self.ny) != (nx, ny):
            self.create_grids((nx, ny), p.device)

        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()

        if self.training:
            return p
        else:
            io = p.sigmoid()
            io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
            io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
            io[..., :4] *= self.stride
            #! view: [B, anchors, H, W, 85] -> [B, -1, 85]
            return io.view(bs, -1, self.no), p

class JDELayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super().__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index
        self.layers = layers
        self.stride = stride
        self.nl = len(layers)
        self.na = len(anchors)
        self.nc = nc
        self.no = nc + 5
        self.nx, self.ny, self.ng = 0, 0, 0
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p, out):
        bs, _, ny, nx = p.shape
        if (self.nx, self.ny) != (nx, ny):
            self.create_grids((nx, ny), p.device)

        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()

        if self.training:
            return p
        else:
            io = p.clone()
            io[..., :2] = torch.sigmoid(io[..., :2]) * 2. - 0.5 + self.grid
            io[..., 2:4] = torch.sigmoid(io[..., 2:4] * 2) ** 2 * self.anchor_wh
            io[..., :4] *= self.stride
            io[..., 4:] = F.softmax(io[..., 4:])
            return io.view(bs, -1, self.no), p

def get_yolo_layers(model):
    return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ in ['YOLOLayer', 'JDELayer']]  # [89, 101, 113]

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

if __name__ == '__main__':
    m = MixConv2d(16, 16)
    print(m)
