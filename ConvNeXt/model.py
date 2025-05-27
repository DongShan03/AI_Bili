import torch
import torch.nn as nn
import torch.nn.functional as F

def add_attributes(**kwargs):
    """装饰器为函数添加属性"""
    def decorator(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorator

def drop_path(x, drop_rate: float=0., training: bool=False):
    if drop_rate == 0. or not training:
        return x
    keep_prob = 1 - drop_rate
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_rate=None):
        super(DropPath, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        return drop_path(x, self.drop_rate, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim, drop_ratio=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * 4, dim)
        #! layer scale
        self.gamma = nn.Parameter(layer_scale_init_value * torch.zeros((dim,)),
                                requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, depths: list=None,
                dims: list=None, drop_path_ratio: float=0., layer_scale_init_value: float=1e-6,
                head_init_scale: float=1.0):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_ratio, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                Block(dim=dims[i], drop_ratio=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        #! global average pooling [N, C, H, W] -> [N, C]
        x = self.norm(x.mean([-2, -1], keepdim=False))
        x = self.head(x)
        return x

@add_attributes(train_size=224, eval_size=224)
def convnext_tiny(num_classes=1000):
    return ConvNeXt(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        num_classes=num_classes,
        drop_path_ratio=0.2,
    )

@add_attributes(train_size=224, eval_size=224)
def convnext_small(num_classes=1000):
    return ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        num_classes=num_classes,
        drop_path_ratio=0.2,
    )

@add_attributes(train_size=224, eval_size=224)
def convnext_base(num_classes=1000):
    return ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        num_classes=num_classes,
        drop_path_ratio=0.2,
    )

@add_attributes(train_size=224, eval_size=224)
def convnext_large(num_classes=1000):
    return ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        num_classes=num_classes,
        drop_path_ratio=0.3,
    )

@add_attributes(train_size=224, eval_size=224)
def convnext_xlarge(num_classes=1000):
    return ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[256, 512, 1024, 2048],
        num_classes=num_classes,
        drop_path_ratio=0.4,
    )
