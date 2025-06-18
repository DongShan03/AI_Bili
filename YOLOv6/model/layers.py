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
