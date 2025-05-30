import torch, copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_bn(input_channel, output_channel, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module("conv", nn.Conv2d(input_channel, output_channel,
                                        kernel_size=kernel_size,
                                        stride=stride, padding=padding,
                                        groups=groups, bias=False))
    result.add_module("bn", nn.BatchNorm2d(output_channel))
    return result

class SEBlock(nn.Module):
    def __init__(self, input_channel, hidden_channel):
        super().__init__()
        self.down = nn.Conv2d(input_channel, hidden_channel, kernel_size=1,
                            stride=1, bias=True)
        self.up = nn.Conv2d(hidden_channel, input_channel, kernel_size=1,
                            stride=1, bias=True)
        self.input_channel = input_channel

    def forward(self, x):
        result = F.avg_pool2d(x, kernel_size=x.size(3))
        result = self.down(result)
        result = F.relu(result)
        result = self.up(result)
        result = torch.sigmoid(result)
        result = result.view(-1, self.input_channel, 1, 1)
        return x * result

class RepVGGBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1,
                padding=0, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.input_channel = input_channel
        assert kernel_size == 3 and padding == 1, "RepVGG only supports 3x3 kernel with padding 1"

        padding_l1 = padding - kernel_size // 2
        self.nonlinearity = nn.ReLU(inplace=True)
        if use_se:
            self.se = SEBlock(output_channel, output_channel // 16)
        else:
            self.se = nn.Identity()
        if deploy:
            self.rbr_reparam = nn.Conv2d(input_channel, output_channel,
                                        kernel_size=kernel_size,
                                        stride=stride, padding=padding,
                                        groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(input_channel) if input_channel == output_channel and stride == 1 else None
            self.rbr_dense = conv_bn(input_channel, output_channel, kernel_size, stride, padding, groups)
            self.rbr_1x1 = conv_bn(input_channel, output_channel, kernel_size=1, stride=stride, padding=padding_l1, groups=groups)

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            result = self.rbr_reparam(x)
            result = self.se(result)
            return self.nonlinearity(result)

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)
        result = self.rbr_dense(x) + self.rbr_1x1(x) + id_out
        result = self.se(result)
        return self.nonlinearity(result)

    def custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).view(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / (self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt()).view(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        l2_loss_square = ((eq_kernel ** 2) / (t3 ** 2 + t1 ** 2)).sum()
        return l2_loss_circle + l2_loss_square

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3(kernel1x1) + kernelid, \
            bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, (1, 1, 1, 1), mode='constant', value=0)

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean, running_var = branch.bn.running_mean, branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                #! 这里对应的是只有BN层的分支，先构建同等映射的卷积核
                input_dim = self.input_channel // self.groups
                kernel_val = np.zeros((self.input_channel, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.input_channel):
                    kernel_val[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_val).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean, running_var = branch.running_mean, branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).view(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(self.rbr_dense.conv.in_channels,
                                    self.rbr_dense.conv.out_channels,
                                    kernel_size=self.rbr_dense.conv.kernel_size,
                                    stride=self.rbr_dense.conv.stride,
                                    padding=self.rbr_dense.conv.padding,
                                    dilation=self.rbr_dense.conv.dilation,
                                    groups=self.rbr_dense.conv.groups,
                                    bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class RepVGG(nn.Module):
    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None,
                override_groups_map=None, deploy=False, use_se=False, use_checkpoint=False):
        super().__init__()
        assert len(num_blocks) == 4, "RepVGG requires 4 stages"
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.use_checkpoint = use_checkpoint
        self.use_se = use_se

        self.in_channel = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(3, self.in_channel, kernel_size=3, stride=2,
                                padding=1, deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, output_channel, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(self.in_channel, output_channel, kernel_size=3,
                                    stride=stride, padding=1, groups=cur_groups,
                                    deploy=self.deploy, use_se=self.use_se))
            self.in_channel = output_channel
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x):
        out = self.stage0(x)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for block in stage:
                if self.use_checkpoint:
                    out = torch.utils.checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def add_attributes(**kwargs):
    """装饰器为函数添加属性"""
    def decorator(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorator

optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
#* 创建字典groups=2
g2_map = {l: 2 for l in optional_groupwise_layers}
#* 创建字典groups=4
g4_map = {l: 4 for l in optional_groupwise_layers}

@add_attributes(train_size=224, eval_size=224)
def create_RepVGG_A0(num_classes=1000, deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None,
                deploy=deploy, use_checkpoint=use_checkpoint)

@add_attributes(train_size=224, eval_size=224)
def create_RepVGG_A1(num_classes=1000, deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None,
                deploy=deploy, use_checkpoint=use_checkpoint)

@add_attributes(train_size=224, eval_size=224)
def create_RepVGG_B0(num_classes=1000, deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[1, 1, 1, 2.5], override_groups_map=None,
                deploy=deploy, use_checkpoint=use_checkpoint)

@add_attributes(train_size=224, eval_size=224)
def create_RepVGG_B1(num_classes=1000, deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[2, 2, 2, 4], override_groups_map=None,
                deploy=deploy, use_checkpoint=use_checkpoint)

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
