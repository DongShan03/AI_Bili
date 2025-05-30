from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from transformer import TransformerEncoder


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvLayer(nn.Module):
    def __init__(self, in_channel: int, out_channel: int,
                kernel_size: Union[int, Tuple[int, int]],
                stride: Optional[Union[int, Tuple[int, int]]] = 1,
                groups: Optional[int]=1,
                bias: Optional[bool]=False,
                use_norm: Optional[bool]=True,
                use_activation: Optional[bool]=True,):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2)
        )

        block = nn.Sequential()
        conv_layer = nn.Conv2d(
            in_channel, out_channel,
            kernel_size, stride,
            groups=groups,
            padding=padding,
            bias=bias
        )
        block.add_module("conv", conv_layer)
        if use_norm:
            block.add_module("norm", nn.BatchNorm2d(out_channel, momentum=0.1))
        if use_activation:
            block.add_module("activation", nn.SiLU())
        self.block = block

    def forward(self, x):
        return self.block(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel,
                stride: int, expand_ratio: Union[int, float]):
        super().__init__()
        assert stride in [1, 2], "stride must be 1 or 2"
        hidden_dim = _make_divisible(int(round(in_channel * expand_ratio)), 8)
        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module("expand_conv", ConvLayer(
                in_channel, hidden_dim, kernel_size=1, use_norm=True, use_activation=True
            ))
        block.add_module("depthwise_conv", ConvLayer(
            hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim,
            use_norm=True, use_activation=True
        ))
        block.add_module("project_conv", ConvLayer(
            hidden_dim, out_channel, kernel_size=1, use_norm=True, use_activation=False
        ))
        self.block = block
        self.use_res_connect = (stride == 1 and in_channel == out_channel)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class MobileViTBlock(nn.Module):
    def __init__(self, in_channel, transformer_dim,
                ffn_dim: int, n_transformer_blocks: int=2,
                head_dim:int=32, attn_dropout:float=0.0,
                dropout: float=0.0, ffn_dropout:float=0.0,
                patch_h:int=8, patch_w:int=8, conv_ksize:int=3):
        super().__init__()
        conv_3x3_in = ConvLayer(
            in_channel, in_channel, kernel_size=conv_ksize, stride=1,
        )
        conv_1x1_in = ConvLayer(
            in_channel, transformer_dim, kernel_size=1, stride=1,
            use_norm=False, use_activation=False
        )
        conv_1x1_out = ConvLayer(
            transformer_dim, in_channel, kernel_size=1, stride=1,
        )
        conv_3x3_out = ConvLayer(
            2 * in_channel, in_channel, kernel_size=conv_ksize, stride=1,
        )
        self.local_rep = nn.Sequential()
        self.local_rep.add_module("conv_3x3", conv_3x3_in)
        self.local_rep.add_module("conv_1x1", conv_1x1_in)

        assert transformer_dim % head_dim == 0, \
            "transformer_dim must be divisible by head_dim"
        num_heads = transformer_dim // head_dim
        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
            ) for _ in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        #* after Transformer
        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = patch_h * patch_w

        self.cnn_in_dim = in_channel
        self.out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_transformer_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfloding(self, x: Tensor):
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_h * patch_w
        batch_size, in_channel, orig_h, orig_w = x.shape
        new_h = int(math.ceil(orig_h / patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / patch_w) * self.patch_w)

        interpolate = False
        if new_h != orig_h or new_w != orig_w:
            interpolate = True
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        num_patch_W = new_w // patch_w
        num_patch_H = new_h // patch_h
        num_patches = num_patch_W * num_patch_H
        #! [B, C, H, W] -> [B, C, num_patch_H, patch_h, num_patch_W, patch_w]
        x = x.reshape(batch_size, in_channel, num_patch_H, patch_h, num_patch_W, patch_w)
        #! [B, C, nH, h, nW, w] -> [B, C, nH, nW, h, w]
        #! [B, C, nH, nW, h, w] -> [B, C, num_patches, patch_area]
        x = x.transpose(3, 4).reshape(batch_size, in_channel, num_patches, patch_area)
        #! [B, C, num_patches, patch_area] -> [B, patch_area, num_patches, C]
        x = x.transpose(1, 3)
        #! [B, patch_area, num_patches, C] -> [B * patch_area, num_patches, C]
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size" : (orig_h, orig_w),
            "batch_size" : batch_size,
            "interpolate" : interpolate,
            "total_patches" : num_patches,
            "num_patch_H" : num_patch_H,
            "num_patch_W" : num_patch_W,
        }
        return x, info_dict

    def folding(self, x, info_dict):
        n_dim = x.dim()
        assert n_dim == 3, f"Expected 3D tensor, got {n_dim}D tensor"
        #! [BP, N, C] -> [B, P, N, C]
        x = x.contiguous().view(
            info_dict["batch_size"],
            self.patch_area,
            info_dict["total_patches"],
            -1
        )
        batch_size, patch_area, num_patches, in_channel = x.shape
        num_patch_h = info_dict["num_patch_H"]
        num_patch_w = info_dict["num_patch_W"]
        #! [B, P, N, C] -> [B, C, N, P]
        x = x.transpose(1, 3)
        #! [B, C, N, P] -> [B, C, num_patch_H, num_patch_W, patch_h, patch_w]
        x = x.reshape(
            batch_size, in_channel, num_patch_h, num_patch_w, self.patch_h, self.patch_w
        )
        #! [B, C, nH, nW, h, w] -> [B, C, nH, h, nW, w]
        x = x.transpose(3, 4)
        #! [B, C, nH, h, nW, w] -> [B, C, H, W]
        x = x.reshape(batch_size, in_channel, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x, size=info_dict["orig_size"], mode="bilinear", align_corners=False
            )
        return x

    def forward(self, x):
        res = x
        fm = self.local_rep(res)
        patches, info_dict = self.unfloding(fm)
        for transform_layer in self.global_rep:
            patches = transform_layer(patches)
        fm = self.folding(patches, info_dict)
        fm = self.conv_proj(fm)
        fm = torch.cat((fm, res), dim=1)
        fm = self.fusion(fm)
        return fm

class MobileViT(nn.Module):
    def __init__(self, model_cfg: Dict, num_classes:int=1000):
        super().__init__()
        image_channels = 3
        out_channels = 16
        self.conv1 = ConvLayer(
            image_channels, out_channels,
            kernel_size=3, stride=2
        )

        self.layer1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])
        self.layer3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        self.layer4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])
        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)
        self.conv1x1_exp = ConvLayer(
            out_channels, exp_channels, kernel_size=1
        )
        self.classifier = nn.Sequential()
        self.classifier.add_module("global_pool", module=nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module("flatten", module=nn.Flatten())
        if 0.0 < model_cfg.get("cls_dropout", 0.0) < 1.0:
            self.classifier.add_module("dropout", module=nn.Dropout(model_cfg["cls_dropout"]))
        self.classifier.add_module("fc", module=nn.Linear(exp_channels, num_classes))

        self.apply(self.init_weights)

    def _make_layer(self, input_channel: int, cfg: Dict) -> Tuple[nn.Module, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel, cfg)
        else:
            return self._make_mobilenet_layer(input_channel, cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Module, int]:
        output_channel = cfg.get("out_channel")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1
            layer = InvertedResidual(
                input_channel, output_channel,
                stride=stride, expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channel
        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Module, int]:
        stride = cfg.get("stride", 1)
        block = []
        if stride == 2:
            layer = InvertedResidual(
                input_channel, cfg.get("out_channel"),
                stride=stride, expand_ratio=cfg.get("mv_expand_ratio", 4)
            )
            block.append(layer)
            input_channel = cfg.get("out_channel")

        transformer_dim = cfg.get("transformer_dim")
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("transformer_dim must be divisible by num_heads")

        block.append(
            MobileViTBlock(
                input_channel, transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=cfg.get("n_transformer_blocks", 2),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=cfg.get("dropout", 0.1),
                ffn_dropout=cfg.get("ffn_dropout", 0.0),
                attn_dropout=cfg.get("attn_dropout", 0.1),
                head_dim=head_dim,
                conv_ksize=3
            )
        )
        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear, )):
            nn.init.trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.conv1x1_exp(x)
        x = self.classifier(x)
        return x

def add_attributes(**kwargs):
    """装饰器为函数添加属性"""
    def decorator(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorator

@add_attributes(train_size=224, eval_size=224)
def mobile_vit_xx_small(num_classes: int=1000):
    mv2_exp_mult = 2
    config = {
        "layer1": {
            "out_channel": 16,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2",
        },
        "layer2": {
                "out_channel": 24,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
        "layer3": {  # 28x28
            "out_channel": 48,
            "transformer_dim": 64,
            "ffn_dim": 128,
            "transformer_blocks": 2,
            "patch_h": 2,  # 8,
            "patch_w": 2,  # 8,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "num_heads": 4,
            "block_type": "mobilevit",
        },
        "layer4": {  # 14x14
            "out_channel": 64,
            "transformer_dim": 80,
            "ffn_dim": 160,
            "transformer_blocks": 4,
            "patch_h": 2,  # 4,
            "patch_w": 2,  # 4,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "num_heads": 4,
            "block_type": "mobilevit",
        },
        "layer5": {  # 7x7
            "out_channel": 80,
            "transformer_dim": 96,
            "ffn_dim": 192,
            "transformer_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "num_heads": 4,
            "block_type": "mobilevit",
        },
        "last_layer_exp_factor": 4,
        "cls_dropout": 0.1
    }
    return MobileViT(model_cfg=config, num_classes=num_classes)


@add_attributes(train_size=224, eval_size=224)
def mobile_vit_x_small(num_classes: int=1000):
    mv2_exp_mult = 4
    config = {
        "layer1": {
            "out_channel": 32,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2",
        },
        "layer2": {
            "out_channel": 48,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 3,
            "stride": 2,
            "block_type": "mv2",
        },
        "layer3": {  # 28x28
            "out_channel": 64,
            "transformer_dim": 96,
            "ffn_dim": 192,
            "transformer_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "num_heads": 4,
            "block_type": "mobilevit",
        },
        "layer4": {  # 14x14
            "out_channel": 80,
            "transformer_dim": 120,
            "ffn_dim": 240,
            "transformer_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "num_heads": 4,
            "block_type": "mobilevit",
        },
        "layer5": {  # 7x7
            "out_channel": 96,
            "transformer_dim": 144,
            "ffn_dim": 288,
            "transformer_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "num_heads": 4,
            "block_type": "mobilevit",
        },
        "last_layer_exp_factor": 4,
        "cls_dropout": 0.1
    }
    return MobileViT(model_cfg=config, num_classes=num_classes)



@add_attributes(train_size=224, eval_size=224)
def mobile_vit_small(num_classes: int=1000):
    mv2_exp_mult = 4
    config = {
        "layer1": {
            "out_channel": 32,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2",
        },
        "layer2": {
            "out_channel": 64,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 3,
            "stride": 2,
            "block_type": "mv2",
        },
        "layer3": {  # 28x28
            "out_channel": 96,
            "transformer_dim": 144,
            "ffn_dim": 288,
            "transformer_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "num_heads": 4,
            "block_type": "mobilevit",
        },
        "layer4": {  # 14x14
            "out_channel": 128,
            "transformer_dim": 192,
            "ffn_dim": 384,
            "transformer_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "num_heads": 4,
            "block_type": "mobilevit",
        },
        "layer5": {  # 7x7
            "out_channel": 160,
            "transformer_dim": 240,
            "ffn_dim": 480,
            "transformer_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "num_heads": 4,
            "block_type": "mobilevit",
        },
        "last_layer_exp_factor": 4,
        "cls_dropout": 0.1
    }
    return MobileViT(model_cfg=config, num_classes=num_classes)
