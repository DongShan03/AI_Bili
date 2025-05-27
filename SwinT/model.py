import torch
import torch.nn as nn
import torch.nn.functional as F

#? 用在最开始进行下采样为原先的1/4
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channel=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_channel, embed_dim, kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shapes
        # padding
        #* 如果输入的图片的H W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            #! (W_left, W_right, H_top, H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                        0, self.patch_size[0] - H % self.patch_size[0],
                        0, 0))
        #* 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        #! flatten: [B, C, H, W] -> [B, C, HW]
        #! transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        #! x: [B, H*W, C]
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

class BasicLayer(nn.Module):
    def __init__(self):
        super().__init__()

class SwinTransformer(nn.Module):
    """
        drop_rate(float): Dropout rate. Default: 0
        attn_drop_rate(float): Attention dropout rate. Default: 0
        drop_path_rate(float): Stochastic depth rate. Default: 0
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm(bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint(bool): Whether to use checkpointing to save memory. Default: False
    """
    def __init__(self, patch_size=4, in_channel=3,
                num_classes=1000, embed_dim=96,
                depths=(2, 2, 6, 2),
                num_heads=(3, 6, 12, 24),
                window_size=7, mlp_ratio=4.,
                qkv_bias=True, drop_ratio=0.,
                attn_drop_ratio=0.,
                drop_path_ratio=0.1,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        #* stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_channel=in_channel, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        self.pos_drop = nn.Dropout(p=drop_ratio)
        #* drop_ratio逐渐上升
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_ratio,
                attn_drop=attn_drop_ratio,
                #* 这里的意思是从上一个stage的drop_path到下一stage的drop_path
                #* 返回的是一个列表
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                #* 这里的意思是对最后一个stage不使用downsample
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layers)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
