from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

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

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16,
                in_channel=3, embed_dim=768,
                norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_channel, embed_dim,
                            kernel_size=patch_size,
                            stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5   #* 根号下dk
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        #! [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        #! qkv() [batch_size, num_patches + 1, 3 * total_embed_dim]
        #! reshape() [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        #! permute() [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads) \
            .permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        #! k.transpose(-2, -1) [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        #! attn [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        #! attn @ v [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        #! transpose(1, 2) [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        #! reshape [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, input_channel, hidden_channel=None, output_channel=None, act_layer=nn.GELU, dropout_rate=0.):
        super().__init__()
        hidden_channel = hidden_channel or input_channel
        output_channel = output_channel or input_channel
        self.fc1 = nn.Linear(input_channel, hidden_channel)
        self.ac = act_layer()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_channel, output_channel)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, dim,
                num_heads,
                mlp_ratio=4.,
                qkv_bias=False,
                qk_scale=None,
                drop_ratio=0.,
                attn_drop_ratio=0.,
                drop_path_ratio=0.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            attn_drop_ratio=attn_drop_ratio,
                            proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(input_channel=dim, hidden_channel=mlp_hidden_dim,
                    act_layer=act_layer, dropout_rate=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16,
                in_channel=3, num_classes=1000,
                embed_dim=768, depth=12, num_heads=12,
                mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                representation_size=None, distilled=False,
                drop_ratio=0., attn_drop_ratio=0.,
                drop_path_ratio=0., embed_layer=PatchEmbedding,
                norm_layer=None, act_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                    in_channel=in_channel, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) \
            if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh()),
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        #! [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        #! [1, 1, embed_dim] -> [B, 1, embed_dim]
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            #! [B, num_patches, embed_dim] -> [B, num_patches + 1, embed_dim]
            x = torch.cat((class_token, x), dim=1)
        else:
            #! [B, num_patches, embed_dim] -> [B, num_patches + 2, embed_dim]
            torch.cat((class_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            #! [B, num_patches + 1, embed_dim] -> [B, embed_dim]
            #* 返回的是对应class_token的那一行
            #* 接下来根据这个class_token去做预测
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[0])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            #* 将class_token输入最后的全连接层来做预测
            x = self.head(x)
        return x

@add_attributes(train_size=224, eval_size=224)
def vit_base_patch16_224(num_classes: int=21843, has_logits:bool=True):
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768 if has_logits else None,
        num_classes=num_classes
    )

@add_attributes(train_size=224, eval_size=224)
def vit_base_patch32_224(num_classes: int=21843, has_logits:bool=True):
    return VisionTransformer(
        img_size=224,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768 if has_logits else None,
        num_classes=num_classes
    )

@add_attributes(train_size=224, eval_size=224)
def vit_large_patch16_224(num_classes: int=21843, has_logits:bool=True):
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        representation_size=1024 if has_logits else None,
        num_classes=num_classes
    )

@add_attributes(train_size=224, eval_size=224)
def vit_large_patch32_224(num_classes: int=21843, has_logits:bool=True):
    return VisionTransformer(
        img_size=224,
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        representation_size=1024 if has_logits else None,
        num_classes=num_classes
    )

@add_attributes(train_size=224, eval_size=224)
def vit_huge_patch14_224(num_class: int=21843, has_logits:bool=True):
    return VisionTransformer(
        img_size=224,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        representation_size=1280 if has_logits else None,
        num_classes=num_class
    )
