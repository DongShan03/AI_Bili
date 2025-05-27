import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

def window_partition(x, window_size):
    """
    将feature_map分割成一个个无重叠窗口
    x: [B, H, W, C]
    window_size: int
    return: [B*num_windows, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    #! permute: [B, H//window_size, W//window_size, window_size, window_size, C]
    #! view: [B*num_windows, window_size, window_size, C]
    #! num_windows = (H // window_size) * (W // window_size)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

def window_reverse(windows, window_size, H, W):
    """
    将分割的窗口重新组合成原来的feature_map
    windows: [B*num_windows, window_size, window_size, C]
    window_size: int
    H: int, 原始feature_map的高度
    W: int, 原始feature_map的宽度
    return: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / (window_size * window_size)))
    #! view: [B*num_windows, window_size, window_size, C] ->
    #! [B, H//window_size, W//window_size, window_size, window_size, C]
    x = windows.view(B, H // window_size, W // window_size,
                    window_size, window_size, -1)
    #! permute -> [B, H // window_size, window_size, W // window_size, window_size, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, -1)


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
        _, _, H, W = x.shape
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

        # padding
        #* 如果输入的H W不是偶数，需要进行padding
        pad_input = (H % 2 != 0) or (W % 2 != 0)
        if pad_input:
            #* pad方法是从后往前的，也就是首先对channel， 然后W， H
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat((x0, x1, x2, x3), dim=-1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C) # [B, H/2*W/2, 4*C]
        x = self.norm(x)  # [B, H/2*W/2, 4*C]
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]
        return x


class Mlp(nn.Module):
    def __init__(self, input_channel, hidden_channel=None,
                output_channel=None, act_layer=nn.GELU, dropout_rate=0.):
        super().__init__()
        hidden_channel = hidden_channel or input_channel
        output_channel = output_channel or input_channel
        self.fc1 = nn.Linear(input_channel, hidden_channel)
        self.ac = act_layer()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_channel, output_channel)
        self.dropout2 = nn.Dropout(dropout_rate)


    def forward(self, x):
        x = self.fc1(x)
        x = self.ac(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        #! 绝对位置索引
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, window_size, window_size]
        coords_flatten = torch.flatten(coords, 1)  # [2, window_size * window_size]
        #! 这里类似之前的unsqueeze操作 得到相对位置索引
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        #! [2, Mh * Mw, Mh * Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [window_size * window_size, window_size * window_size, 2]
        #! [Mh * Mw, Mh * Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # shift to 1D index
        #! 这就是一维的相对位置索引
        relative_position_index = relative_coords.sum(-1)
        #! [Mh * Mw * Mh * Mw, 2]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        #! [B*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        #! qkv() -> [B*num_windows, Mh*Mw, 3*total_embed_dim]
        #! reshape -> [B*num_windows, Mh*Mw, 3, num_heads, head_dim]
        #! permute -> [3, B*num_windows, num_heads, Mh*Mw, head_dim]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        #! q, k, v -> [B*num_windows, num_heads, Mh*Mw, head_dim]

        #! transpose -> [B*num_windows, num_heads, head_dim, Mh*Mw]
        #! @ -> [B*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        #! [Mh*Mw*Mh*Mw, num_heads]-> [Mh*Mw, Mh*Mw, num_heads]
        relative_position_bias = self.relative_position_bias_table[\
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1
            )
        #! [num_heads, Mh*Mw, Mh*Mw]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        #! 广播机制
        attn = attn + relative_position_bias.unsqueeze(0)
        #! attn -> [B*num_windows, num_heads, Mh*Mw, Mh*Mw]

        if mask is not None:
            #! mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]
            #! attn.view -> [Batch_size, nW, num_heads, Mh*Mw, Mh*Mw]
            #! mask.unsqueeze(1) -> [nW, 1, Mh*Mw, Mh*Mw]
            #! mask.unsqueeze(1).unsqueeze(0) -> [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            #! [B*num_windows, num_heads, Mh*Mw, Mh*Mw]
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        #! @ -> [B*num_windows, num_heads, Mh*Mw, head_dim]
        #! transpose -> [B*num_windows, Mh*Mw, num_heads, head_dim]
        #! reshape -> [B*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x  # [B*num_windows, Mh*Mw, total_embed_dim]

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7,
                shift_size=0, mlp_ratio=4., qkv_bias=True,
                drop=0., attn_drop=0., drop_path=0.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, \
            f"shift_size must in [0, window_size), but got {self.shift_size} and {self.window_size}"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size),
            num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(input_channel=dim, hidden_channel=mlp_hidden_dim,
                    act_layer=act_layer, dropout_rate=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)  # [B, L, C]
        x = x.view(B, H, W, C)  # [B, H, W, C]

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))  # [B, H+pad_b, W+pad_r, C]
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shift_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shift_x = x
            attn_mask = None

        x_windows = window_partition(shift_x, self.window_size)  # [B*num_windows, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [B*num_windows, Mh*Mw, C]
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [B*num_windows, Mh*Mw, C]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) # [B*num_windows, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H+pad_b, W+pad_r, C]

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)  # [B, L, C]
        x = shortcut + self.drop_path(x)  # [B, L, C]
        x = x = self.drop_path(self.mlp(self.norm2(x)))  # [B, L, C]
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads,
                window_size=7, mlp_ratio=4.,
                qkv_bias=True, drop=0.,
                attn_drop=0., drop_path=0.,
                norm_layer=nn.LayerNorm,
                downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads,
                window_size=window_size, mlp_ratio=mlp_ratio,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                qkv_bias=qkv_bias, drop=drop,
                attn_drop=attn_drop, drop_path=drop_path[i],
                norm_layer=norm_layer
            ) for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        #* 为SW-Masked-Attention创建mask
        #* 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        w_slice = h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slice:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size) #! [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size) #! [nW, Mh*Mw]
        #! 这里利用广播机制，首先在第二个维度进行扩展，将Mh*Mw也就是展平后的窗口复制Mh*Mw次
        #! 因此会出现如（忽略nW维度） 后一个会出现将前一维度的数字复制Mh*Mw次的情况
        """
               [[4, 4, 5, 4, 4, 5, 7, 7, 8],                   [[4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 5, 4, 4, 5, 7, 7, 8],                    [4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 5, 4, 4, 5, 7, 7, 8],                    [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [4, 4, 5, 4, 4, 5, 7, 7, 8],                    [4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 5, 4, 4, 5, 7, 7, 8],                    [4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 5, 4, 4, 5, 7, 7, 8],        -           [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [4, 4, 5, 4, 4, 5, 7, 7, 8],                    [7, 7, 7, 7, 7, 7, 7, 7, 7],
                [4, 4, 5, 4, 4, 5, 7, 7, 8],                    [7, 7, 7, 7, 7, 7, 7, 7, 7],
                [4, 4, 5, 4, 4, 5, 7, 7, 8]]                    [8, 8, 8, 8, 8, 8, 8, 8, 8]]
        得到的结果为0的话说明对应的pixel在同一窗口,否则说明在不同窗口
        """
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) #! [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        #! [nW, Mh*Mw, Mh*Mw] n = 1 so [num_windows, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H+1) // 2, (W+1) // 2
        return x, H, W


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


    def forward(self, x):
        #! x: [B, L, C]
        x, L, C = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x, L, C = layer(x, L, C)
        x = self.norm(x)    # [B, L, C]
        x = self.avgpool(x.transpose(1, 2)) # [B, C, 1]
        x = torch.flatten(x, 1)
        x = self.head(x)    # [B, num_classes]
        return x

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

@add_attributes(train_size=224, eval_size=224)
def swin_tiny_patch4_window7_224(num_classes=1000, **kwargs):
    return SwinTransformer(
        patch_size=4, in_channel=3, window_size=7,
        embed_dim=96, depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=num_classes, **kwargs
    )

@add_attributes(train_size=384, eval_size=384)
def swin_tiny_patch4_window12_384(num_classes=1000, **kwargs):
    return SwinTransformer(
        patch_size=4, in_channel=3, window_size=12,
        embed_dim=96, depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=num_classes, **kwargs
    )

@add_attributes(train_size=224, eval_size=224)
def swin_small_patch4_window7_224(num_classes=1000, **kwargs):
    return SwinTransformer(
        patch_size=4, in_channel=3, window_size=7,
        embed_dim=96, depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=num_classes, **kwargs
    )

@add_attributes(train_size=384, eval_size=384)
def swin_small_patch4_window12_384(num_classes=1000, **kwargs):
    return SwinTransformer(
        patch_size=4, in_channel=3, window_size=12,
        embed_dim=96, depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=num_classes, **kwargs
    )

@add_attributes(train_size=224, eval_size=224)
def swin_base_patch4_window7_224(num_classes=1000, **kwargs):
    return SwinTransformer(
        patch_size=4, in_channel=3, window_size=7,
        embed_dim=128, depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        num_classes=num_classes, **kwargs
    )

@add_attributes(train_size=224, eval_size=224)
def swin_large_patch4_window7_224(num_classes=1000, **kwargs):
    return SwinTransformer(
        patch_size=4, in_channel=3, window_size=7,
        embed_dim=192, depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        num_classes=num_classes, **kwargs
    )
