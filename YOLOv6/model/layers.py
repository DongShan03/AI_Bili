import torch, warnings
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


activation_table = {
    "relu": nn.ReLU(inplace=True),
    "silu": nn.SiLU(inplace=True),
    "hardswish": nn.Hardswish(inplace=True),
}

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class ConvModule(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, activation_type, padding=None, groups=1, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channel)
        if activation_type is not None:
            self.act = activation_table.get(activation_type)
        self.activation_type = activation_type

    def forward(self, x):
        if self.activation_type is None:
            return self.bn(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        if self.activation_type is None:
            return self.bn(self.conv(x))
        return self.act(self.conv(x))

class ConvBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule(in_channel, out_channel, kernel_size, stride, "relu", padding, groups, bias)

    def forward(self, x):
        return self.block(x)

class ConvBNSiLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule(in_channel, out_channel, kernel_size, stride, "silu", padding, groups, bias)

    def forward(self, x):
        return self.block(x)

class ConvBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule(in_channel, out_channel, kernel_size, stride, None, padding, groups, bias)

    def forward(self, x):
        return self.block(x)

class ConvBNHS(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule(in_channel, out_channel, kernel_size, stride, "hardswish", padding, groups, bias)

    def forward(self, x):
        return self.block(x)

class SPPFModule(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, block=ConvBNSiLU):
        super().__init__()
        hidden_channel = in_channel // 2
        self.cv1 = block(in_channel, hidden_channel, 1, 1)
        self.cv2 = block(hidden_channel * 4, out_channel, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class SimSPFF(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, block=ConvBNSiLU):
        super().__init__()
        self.sppf = SPPFModule(in_channel, out_channel, kernel_size, block)

    def forward(self, x):
        return self.sppf(x)

class SPFF(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, block=ConvBNSiLU):
        super().__init__()
        self.sppf = SPPFModule(in_channel, out_channel, kernel_size, block)

    def forward(self, x):
        return self.sppf(x)

class CSPSPPFModule(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, e=0.5, block=ConvBNSiLU):
        super().__init__()
        hidden_channel = int(in_channel * e)
        self.cv1 = block(in_channel, hidden_channel, 1, 1)
        self.cv2 = block(in_channel, hidden_channel, 1, 1)
        self.cv3 = block(hidden_channel, hidden_channel, 3, 1)
        self.cv4 = block(hidden_channel, hidden_channel, 1, 1)

        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = block(4 * hidden_channel, hidden_channel, 1, 1)
        self.cv6 = block(hidden_channel, hidden_channel, 3, 1)
        self.cv7 = block(2 * hidden_channel, out_channel, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x1)
            y2 = self.m(y1)
            y3 = self.cv6(self.cv5(torch.cat([x1, y1, y2, self.m(y2)], 1)))
        return self.cv7(torch.cat([y0, y3], 1))

class SPPF(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, e=0.5, block=ConvBNSiLU):
        super().__init__()
        self.sppf = SPPFModule(in_channel, out_channel, kernel_size, block)

    def forward(self, x):
        return self.sppf(x)

class SimCSPSPPF(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, e=0.5, block=ConvBNReLU):
        super().__init__()
        self.cspsppf = CSPSPPFModule(in_channel, out_channel, kernel_size, e, block)
    def forward(self, x):
        return self.block(x)

class CSPSPPF(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, e=0.5, block=ConvBNSiLU):
        super().__init__()
        self.cspsppf = CSPSPPFModule(in_channel, out_channel, kernel_size, e, block)
    def forward(self, x):
        return self.block(x)

class Transpose(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = nn.ConvTranspose2d(
            in_channel, out_channel,
            kernel_size=kernel_size, stride=stride,
            bias=True
        )

    def forward(self, x):
        return self.upsample_transpose(x)

class RepVGGBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3,
                stride=1, padding=1, dilation=1, groups=1, padding_mode="zeros",
                deploy=False, use_se=False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channel = in_channel
        self.out_channel = out_channel

        assert kernel_size == 3
        assert padding == 1

        padding_l1 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channel, out_channel, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=True, padding_mode=padding_mode
            )
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channel) if out_channel == in_channel and stride == 1 else None
            self.rbr_dense = ConvModule(in_channel, out_channel, kernel_size, stride, activation_type=None, padding=padding, groups=groups)
            self.rbr_1x1 = ConvModule(in_channel, out_channel, kernel_size=1, stride=stride, activation_type=None, padding=padding_l1, groups=groups)

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.in_channel
        groups = self.groups
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size), device=avgp.weight.device)
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            bias = branch.conv.bias
            return kernel, bias
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channel // self.groups
                kernel_value = np.zeros((self.in_channel, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channel):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

    def swith_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                    kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                    padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                    groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True

class QARepVGGBlock(RepVGGBlock):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1,
                groups=1, padding_mode="zeros", deploy=False, use_se=False):
        super().__init__(in_channel, out_channel, kernel_size, stride, padding, dilation, groups, padding_mode, deploy, use_se)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channel)
            self.rbr_1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=padding, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if out_channel == in_channel and stride == 1 else None
        self._id_tensor = None

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        bias = bias3x3

        if self.rbr_identity is not None:
            input_dim = self.in_channel // self.groups
            kernel_value = np.zeros((self.in_channel, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channel):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            kernel += id_tensor

        return kernel, bias

    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean - bias
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups, bias=True
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True

class QARepVGGBlockV2(RepVGGBlock):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                padding_mode="zeros", deploy=False, use_se=False):
        super().__init__(in_channel, out_channel, kernel_size, stride, padding, dilation, groups, padding_mode, deploy, use_se)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channel)
            self.rbr_1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=padding, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if out_channel == in_channel and stride == 1 else None
            self.rbr_avg = nn.AvgPool2d(kernel_size, stride=stride, padding=padding) if out_channel == in_channel and stride == 1 else None
        self._id_tensor = None

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        if self.rbr_avg is not None:
            avg_out = self.rbr_avg(inputs)
        else:
            avg_out = 0

        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out + avg_out)))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        if self.rbr_avg is not None:
            kernelavg = self._avg_to_3x3_tensor(self.rbr_avg)
            kernel += kernelavg.to(self.rbr_1x1.weight.device)
        bias = bias3x3

        if self.rbr_identity is not None:
            input_dim = self.in_channel // self.groups
            kernel_value = np.zeros((self.in_channel, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channel):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            kernel += id_tensor
        return kernel, bias

    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean - bias
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def swith_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups, bias=True
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "rbr_avg"):
            self.__delattr__("rbr_avg")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True

class RealVGGBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode="zeros", use_se=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channel)

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

    def forward(self, inputs):
        out = self.relu(self.se(self.bn(self.conv(inputs))))
        return out

class ScaleLayer(nn.Module):
    def __init__(self, num_features, use_bias=True, scale_init=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_features))
        nn.init.constant_(self.weight, scale_init)
        self.num_features = num_features
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, inputs):
        if self.bias is None:
            return inputs * self.weight.view(1, self.num_features, 1, 1)
        else:
            return inputs * self.weight.view(1, self.num_features, 1, 1) + self.bias.view(1, self.num_features, 1, 1)

class LinearAddBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                padding_mode="zeros", use_se=False, is_csla=False, conv_scale_init=1.0):
        super().__init__()
        self.in_channel = in_channel
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.scale_conv = ScaleLayer(out_channel, use_bias=False, scale_init=conv_scale_init)
        self.conv_1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False)
        self.scale_1x1 = ScaleLayer(out_channel, use_bias=False, scale_init=conv_scale_init)
        if in_channel == out_channel and stride == 1:
            self.scale_identity = ScaleLayer(out_channel, use_bias=False, scale_init=1.0)
        self.bn = nn.BatchNorm2d(out_channel)
        if is_csla:
            self.scale_1x1.requires_grad_(False)
            self.scale_conv.requires_grad_(False)
        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

    def forward(self, inputs):
        out = self.scale_conv(self.conv(inputs)) + self.scale_1x1(self.conv_1x1(inputs))
        if hasattr(self, "scale_identity"):
            out = out + self.scale_identity(inputs)
        return self.relu(self.se(self.bn(out)))

class RepBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()
        self.conv1 = block(in_channel, out_channel)
        self.block = nn.Sequential(*(block(out_channel, out_channel) for _ in range(n - 1))) if n > 1 else nn.Identity()
        if block == BottleRep:
            self.conv1 = BottleRep(in_channel, out_channel, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(*(BottleRep(out_channel, out_channel, basic_block=basic_block, weight=True) for _ in range(n - 1))) if n > 1 else nn.Identity()
    def forward(self, x):
        x = self.conv1(x)
        x = self.block(x)
        return x

class BottleRep(nn.Module):
    def __init__(self, in_channel, out_channel, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channel, out_channel)
        self.conv2 = basic_block(out_channel, out_channel)
        if in_channel != out_channel:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs

class BottleRep3(nn.Module):
    def __init__(self, in_channel, out_channel, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channel, out_channel)
        self.conv2 = basic_block(out_channel, out_channel)
        self.conv3 = basic_block(out_channel, out_channel)
        if in_channel != out_channel:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class BepC3(nn.Module):
    def __init__(self, in_channel, out_channel, n=1, e=0.5, block=RepVGGBlock):
        super().__init__()
        hidden_channel = int(out_channel * e)
        self.cv1 = ConvBNReLU(in_channel, hidden_channel, 1, 1)
        self.cv2 = ConvBNReLU(in_channel, hidden_channel, 1, 1)
        self.cv3 = ConvBNReLU(2 * hidden_channel, 1, 1)
        if block == ConvBNSiLU:
            self.cv1 = ConvBNSiLU(in_channel, hidden_channel, 1, 1)
            self.cv2 = ConvBNSiLU(in_channel, hidden_channel, 1, 1)
            self.cv3 = ConvBNSiLU(2 * hidden_channel, 1, 1)
        self.m = RepBlock(hidden_channel, hidden_channel, n, block=BottleRep, basic_block=block)

    def forward(self, x):
        self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class MBLABlock(nn.Module):
    def __init__(self, in_channel, out_channel, n=1, e=0.5, block=RepVGGBlock):
        super().__init__()
        n = n // 2
        if n <= 0:
            n = 1

        if n == 1:
            n_list = [0, 1]
        else:
            extra_branch_steps = 1
            while extra_branch_steps * 2 < n:
                extra_branch_steps *= 2

            n_list = [0, extra_branch_steps, n]
        branch_num = len(n_list)

        hidden_channel = int(out_channel * e)
        self.c = hidden_channel
        self.cv1 = ConvModule(in_channel, branch_num * self.c, 1, 1, 'relu', bias=False)
        self.cv2 = ConvModule((sum(n_list) + branch_num) * self.c, out_channel, 1, 1, 'silu', bias=False)

        if block == ConvBNSiLU:
            self.cv1 = ConvModule(in_channel, branch_num * self.c, 1, 1, 'silu', bias=False)
            self.cv2 = ConvModule((sum(n_list) + branch_num) * self.c, out_channel, 1, 1, 'silu', bias=False)

        self.m = nn.ModuleList()
        for n_list_i in n_list[1:]:
            self.m.append(nn.Sequential(*(BottleRep3(self.c, self.c, basic_block=block, weight=True) for _ in range(n_list_i))))

        self.split_num = tuple([self.c] * branch_num)

    def forward(self, x):
        y = list(self.cv1(x).split(self.split_num, 1))
        all_y = [y[0]]
        for m_idx, m_i in enumerate(self.m):
            all_y.append(y[m_idx + 1])
            all_y.extend(m(all_y[-1]) for m in m_i)
        return self.cv2(torch.cat(all_y, 1))

class BiFusion(nn.Module):
    #* PAN
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = ConvBNReLU(in_channels[0], out_channels, 1, 1)
        self.cv2 = ConvBNReLU(in_channels[1], out_channels, 1, 1)
        self.cv3 = ConvBNReLU(out_channels * 3, out_channels, 1, 1)

        self.upsample = Transpose(out_channels, out_channels)
        self.downsample = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(torch.cat((x0, x1, x2), dim=1))

def get_block(mode):
    if mode == 'repvgg':
        return RepVGGBlock
    elif mode == 'qarepvgg':
        return QARepVGGBlock
    elif mode == 'qarepvggv2':
        return QARepVGGBlockV2
    elif mode == 'hyper_search':
        return LinearAddBlock
    elif mode == 'repopt':
        return RealVGGBlock
    elif mode == 'conv_relu':
        return ConvBNReLU
    elif mode == 'conv_silu':
        return ConvBNSiLU
    else:
        raise NotImplementedError("Undefied Repblock choice for mode {}".format(mode))


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avgpool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(channel // reduction, channel, 1, stride=1, padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        out = identity * x
        return out

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x

class Lite_EffiBlockS1(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, stride):
        super().__init__()
        self.conv_pw_1 = ConvBNHS(in_channel // 2, mid_channel, kernel_size=1, stride=1, padding=0, groups=1)
        self.conv_dw_1 = ConvBN(mid_channel, mid_channel, kernel_size=3, stride=stride, padding=1, groups=mid_channel)
        self.se = SEBlock(mid_channel)
        self.conv_1 = ConvBNHS(mid_channel, out_channel // 2, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, x):
        x1, x2 = torch.split(x, split_size_or_sections=[x.shape[1] // 2, x.shape[1] // 2], dim=1)
        x2 = self.conv_pw_1(x2)
        x3 = self.conv_dw_1(x2)
        x3 = self.se(x3)
        x3 = self.conv_1(x3)
        out = torch.cat([x1, x3], dim=1)
        return channel_shuffle(out, 2)


class Lite_EffiBlockS2(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, stride):
        self.conv_dw_1 = ConvBN(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel)
        self.conv_1 = ConvBNHS(in_channel, out_channel // 2, kernel_size=1, stride=1, padding=0, groups=1)
        self.conv_pw_2 = ConvBNHS(in_channel, mid_channel // 2, kernel_size=1, stride=1, padding=0, groups=1)
        self.conv_dw_2 = ConvBN(mid_channel // 2, mid_channel // 2, kernel_size=3, stride=stride, padding=1, groups=mid_channel // 2)
        self.se = SEBlock(mid_channel // 2)
        self.conv_2 = ConvBNHS(mid_channel // 2, out_channel // 2, kernel_size=1, stride=1, padding=0, groups=1)
        self.conv_dw_3 = ConvBNHS(out_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=out_channel)
        self.conv_pw_3 = ConvBNHS(out_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, x):
        x1 = self.conv_dw_1(x)
        x1 = self.conv_1(x)
        x2 = self.conv_pw_2(x)
        x2 = self.conv_dw_2(x2)
        x2 = self.se(x2)
        x2 = self.conv_2(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self.conv_dw_3(out)
        out = self.conv_pw_3(out)
        return out

class DPBlock(nn.Module):
    def __init__(self, in_channel=96, out_channel=96, kernel_size=3, stride=1):
        super().__init__()
        self.conv_dw_1 = nn.Conv2d(in_channel, out_channel, kernel_size, groups=out_channel, stride=stride, padding=(kernel_size - 1) // 2)
        self.bn_1 = nn.BatchNorm2d(out_channel)
        self.act_1 = nn.Hardswish()
        self.conv_pw_1 = nn.Conv2d(out_channel, out_channel, kernel_size=1, groups=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(out_channel)
        self.act_2 = nn.Hardswish()

    def forward(self, x):
        x = self.act_1(self.bn_1(self.conv_dw_1(x)))
        x = self.act_2(self.bn_2(self.conv_pw_1(x)))
        return x

    def forward_fuse(self, x):
        x = self.act_1(self.conv_dw_1(x))
        x = self.act_2(self.conv_pw_1(x))

class DarknetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, expansion=0.5):
        super().__init__()
        hidden_channel = int(out_channel * expansion)
        self.conv_1 = ConvBNHS(in_channel, hidden_channel, kernel_size=1, stride=1, padding=0)
        self.conv_2 = DPBlock(hidden_channel, out_channel, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        return out

class CSPBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, expand_ratio=0.5):
        super().__init__()
        mid_channel = int(out_channel * expand_ratio)
        self.conv_1 = ConvBNHS(in_channel, mid_channel, 1, 1, 0)
        self.conv_2 = ConvBNHS(in_channel, mid_channel, 1, 1, 0)
        self.conv_3 = ConvBNHS(2 * mid_channel, out_channel, 1, 1, 0)
        self.blocks = DarknetBlock(mid_channel, mid_channel, kernel_size=kernel_size, expansion=1.0)

    def forward(self, x):
        x_1 = self.conv_1(x)
        x_1 = self.blocks(x_1)
        x_2 = self.conv_2(x)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.conv_3(x)
        return x

class EfficientRep(nn.Module):
    def __init__(self, in_channel=3, channels_list=None,
                num_repeats=None, block=RepVGGBlock,
                fuse_P2=False, cspsppf=False):
        super().__init__()
        assert channels_list is not None
        assert num_repeats is not None

        self.fuse_P2 = fuse_P2
        self.stem = block(
            in_channel=in_channel,
            out_channel=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channel=channels_list[0],
                out_channel=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channel=channels_list[1],
                out_channel=channels_list[1],
                n=num_repeats[1],
                block=block
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channel=channels_list[1],
                out_channel=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channel=channels_list[2],
                out_channel=channels_list[2],
                n=num_repeats[2],
                block=block
            )
        )
        self.ERBlock_4 = nn.Sequential(
            block(
                in_channel=channels_list[2],
                out_channel=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channel=channels_list[3],
                out_channel=channels_list[3],
                n=num_repeats[3],
                block=block
            )
        )

        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPFF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channel=channels_list[3],
                out_channel=channels_list[4],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channel=channels_list[4],
                out_channel=channels_list[4],
                n=num_repeats[4],
                block=block
            ),
            channel_merge_layer(
                in_channel=channels_list[4],
                out_channel=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        return tuple(outputs)

class EfficientRep6(nn.Module):
    def __init__(self, in_channel=3, channels_list=None,
                num_repeats=None, block=RepVGGBlock,
                fuse_P2=False, cspsppf=False):
        super().__init__()
        assert channels_list is not None
        assert num_repeats is not None

        self.fuse_P2 = fuse_P2
        self.stem = block(
            in_channel=in_channel,
            out_channel=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channel=channels_list[0],
                out_channel=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channel=channels_list[1],
                out_channel=channels_list[1],
                n=num_repeats[1],
                block=block
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channel=channels_list[1],
                out_channel=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channel=channels_list[2],
                out_channel=channels_list[2],
                n=num_repeats[2],
                block=block
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channel=channels_list[2],
                out_channel=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channel=channels_list[3],
                out_channel=channels_list[3],
                n=num_repeats[3],
                block=block
            )
        )

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channel=channels_list[3],
                out_channel=channels_list[4],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channel=channels_list[4],
                out_channel=channels_list[4],
                n=num_repeats[4],
                block=block
            )
        )

        channel_merge_layer = SimSPFF if not cspsppf else SimCSPSPPF

        self.ERBlock_6 = nn.Sequential(
            block(
                in_channel=channels_list[4],
                out_channel=channels_list[5],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channel=channels_list[5],
                out_channel=channels_list[5],
                n=num_repeats[5],
                block=block
            ),
            channel_merge_layer(
                in_channel=channels_list[5],
                out_channel=channels_list[5],
                kernel_size=5
            )
        )

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        x = self.ERBlock_6(x)
        outputs.append(x)
        return tuple(outputs)

class CSPBepBackbone(nn.Module):
    def __init__(self, in_channel=3, channels_list=None,
                num_repeats=None, block=RepVGGBlock,
                csp_e=float(1)/2, fuse_P2=False, cspsppf=False,
                stage_block_type="BepC3"):
        super().__init__()
        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError

        self.fuse_P2 = fuse_P2
        self.stem = block(
            in_channel=in_channel,
            out_channel=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channel=channels_list[0],
                out_channel=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channel=channels_list[1],
                out_channel=channels_list[1],
                n=num_repeats[1],
                block=block,
                e=csp_e
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channel=channels_list[1],
                out_channel=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channel=channels_list[2],
                out_channel=channels_list[2],
                n=num_repeats[2],
                block=block,
                e=csp_e
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channel=channels_list[2],
                out_channel=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channel=channels_list[3],
                out_channel=channels_list[3],
                n=num_repeats[3],
                block=block,
                e=csp_e
            )
        )

        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPFF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channel=channels_list[3],
                out_channel=channels_list[4],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channel=channels_list[4],
                out_channel=channels_list[4],
                n=num_repeats[4],
                block=block,
                e=csp_e
            ),
            channel_merge_layer(
                in_channel=channels_list[4],
                out_channel=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        return tuple(outputs)

class CSPBepBackbone_P6(nn.Module):
    def __init__(self, in_channel=3, channels_list=None,
                num_repeats=None, block=RepVGGBlock,
                csp_e=float(1)/2, fuse_P2=False, cspsppf=False,
                stage_block_type="BepC3"):
        super().__init__()
        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError

        self.fuse_P2 = fuse_P2
        self.stem = block(
            in_channel=in_channel,
            out_channel=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channel=channels_list[0],
                out_channel=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channel=channels_list[1],
                out_channel=channels_list[1],
                n=num_repeats[1],
                block=block,
                e=csp_e
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channel=channels_list[1],
                out_channel=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channel=channels_list[2],
                out_channel=channels_list[2],
                n=num_repeats[2],
                block=block,
                e=csp_e
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channel=channels_list[2],
                out_channel=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channel=channels_list[3],
                out_channel=channels_list[3],
                n=num_repeats[3],
                block=block,
                e=csp_e
            )
        )

        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPFF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channel=channels_list[3],
                out_channel=channels_list[4],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channel=channels_list[4],
                out_channel=channels_list[4],
                n=num_repeats[4],
                block=block,
                e=csp_e
            )
        )

        self.ERBlock_6 = nn.Sequential(
            block(
                in_channel=channels_list[4],
                out_channel=channels_list[5],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channel=channels_list[5],
                out_channel=channels_list[5],
                n=num_repeats[5],
                block=block,
                e=csp_e
            ),
            channel_merge_layer(
                in_channel=channels_list[5],
                out_channel=channels_list[5],
                kernel_size=5
            )
        )

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        x = self.ERBlock_6(x)
        outputs.append(x)
        return tuple(outputs)

class Lite_EffiBackbone(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_repeats=[1, 3, 7, 3]):
        super().__init__()
        out_channels[0] = 24
        self.conv0 = ConvBNHS(in_channel=in_channels,
                            out_channel=out_channels[0],
                            kernel_size=3,
                            stride=2,
                            padding=1)

        self.lite_effiblock_1 = self.build_block(num_repeats[0],
                                                out_channels[0],
                                                mid_channels[1],
                                                out_channels[1])

        self.lite_effiblock_2 = self.build_block(num_repeats[1],
                                                out_channels[1],
                                                mid_channels[2],
                                                out_channels[2])

        self.lite_effiblock_3 = self.build_block(num_repeats[2],
                                                out_channels[2],
                                                mid_channels[3],
                                                out_channels[3])

        self.lite_effiblock_4 = self.build_block(num_repeats[3],
                                                out_channels[3],
                                                mid_channels[4],
                                                out_channels[4])

    def forward(self, x):
        outputs = []
        x = self.conv0(x)
        x = self.lite_effiblock_1(x)
        x = self.lite_effiblock_2(x)
        outputs.append(x)
        x = self.lite_effiblock_3(x)
        outputs.append(x)
        x = self.lite_effiblock_4(x)
        outputs.append(x)
        return tuple(outputs)

    @staticmethod
    def build_block(num_repeat, in_channel, mid_channel, out_channel):
        block_list = nn.Sequential()
        for i in range(num_repeat):
            if i == 0:
                block = Lite_EffiBlockS2(
                    in_channel=in_channel,
                    mid_channel=mid_channel,
                    out_channel=out_channel,
                    stride=2
                )
            else:
                block = Lite_EffiBlockS1(
                    in_channel=in_channel,
                    mid_channel=mid_channel,
                    out_channel=out_channel,
                    stride=1
                )
            block_list.add_module(str(i), block)
        return block_list
