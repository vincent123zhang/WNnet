import torch
import itertools
import math
from einops import rearrange, repeat

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
from timm.models.registry import register_model
from .nka import NKA  # Narrow-Kernel Aggregation

from timm.models.helpers import build_model_with_cfg
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        # Xavier init for Conv2d weights
        torch.nn.init.xavier_uniform_(self.c.weight)
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        # Xavier init for Linear weights and zero bias
        torch.nn.init.xavier_uniform_(self.l.weight)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class FFN(torch.nn.Module):
    def __init__(self, ed, h, ffn_dropout=0.1):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(ffn_dropout)
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.act(self.pw1(x))
        x = self.dropout(x)
        x = self.pw2(x)
        return x

class RepDW(torch.nn.Module):
    """Reparameterized Depthwise Convolution Block"""
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
    
    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv

import torch.nn as nn

class WKP(torch.nn.Module):
    """Wide-Kernel Perception Module"""
    def __init__(self, dim, wide_kernel, narrow_kernel, groups):
        super().__init__()
        # Bottleneck: reduce channels
        self.pw_reduce = Conv2d_BN(dim, dim // 2)
        self.act = torch.nn.ReLU()
        # Wide-kernel depthwise convolution for contextual perception
        self.dw_wide = Conv2d_BN(dim // 2, dim // 2, ks=(wide_kernel, 1), 
                                 pad=((wide_kernel - 1) // 2, 0), groups=dim // 2)
        # Channel expansion
        self.pw_expand = Conv2d_BN(dim // 2, dim // 2)
        # Generate dynamic kernels for narrow aggregation
        self.kernel_gen = torch.nn.Conv2d(dim // 2, narrow_kernel * dim // groups, kernel_size=1)
        # Xavier init for kernel_gen
        torch.nn.init.xavier_uniform_(self.kernel_gen.weight)
        if self.kernel_gen.bias is not None:
            torch.nn.init.zeros_(self.kernel_gen.bias)
        self.norm = torch.nn.GroupNorm(num_groups=dim // groups, num_channels=narrow_kernel * dim // groups)
        
        self.narrow_kernel = narrow_kernel
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        # Wide perception pathway
        x = self.act(self.pw_expand(self.dw_wide(self.act(self.pw_reduce(x)))))
        # Generate position-specific dynamic kernels
        w = self.norm(self.kernel_gen(x))
        b, _, h, width = w.size()
        # Reshape to [B, groups, narrow_kernel, H, W] for narrow aggregation
        w = w.view(b, self.dim // self.groups, self.narrow_kernel, h, width)
        return w

class WNConv(torch.nn.Module):
    """Wide-Narrow Convolution: scan wide, focus narrow"""
    def __init__(self, dim):
        super(WNConv, self).__init__()
        # Wide-Kernel Perception (K_w=7)
        self.wkp = WKP(dim, wide_kernel=7, narrow_kernel=3, groups=4)
        # Narrow-Kernel Aggregation
        self.nka = NKA()
        self.bn = torch.nn.BatchNorm2d(dim)

    def forward(self, x):
        # Wide perception generates dynamic kernels, narrow aggregation applies them
        return self.bn(self.nka(x, self.wkp(x))) + x

class WNBlock(torch.nn.Module):
    """WNnet Basic Block with RepDW and WNConv alternation"""
    def __init__(self,
                 channels, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 stage=-1, depth=-1,
                 attn_dropout=0.1,
                 ffn_dropout=0.1):
        super().__init__()
            
        if depth % 2 == 0:
            # Even depth: use RepDW for local feature extraction
            self.spatial_mixer = RepDW(channels)
            self.channel_attn = SqueezeExcite(channels, 0.25)
        else:
            # Odd depth: use WNConv for wide-narrow perception-aggregation
            self.channel_attn = torch.nn.Identity()
            self.spatial_mixer = WNConv(channels)
 
        self.ffn = Residual(FFN(channels, int(channels * 2), ffn_dropout))

    def forward(self, x):
        return self.ffn(self.channel_attn(self.spatial_mixer(x)))

class WNnet(torch.nn.Module):
    """Wide-Narrow Network for Efficient Human Activity Recognition"""
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=1,
                 num_classes=1000,
                 embed_dim=[64, 128, 192, 256],
                 key_dim=[16, 16, 8, 8],
                 depth=[1, 2, 3, 4],
                 num_heads=[4, 4, 4, 4],
                 distillation=False,
                 dropout_rate=0.2,
                 attn_dropout=0.1,
                 ffn_dropout=0.1,
                 input_shape=None):
        super().__init__()

        # Stem layer for initial feature extraction
        H, W = input_shape[-2], input_shape[-1]
        self.stem = torch.nn.Sequential(
            Conv2d_BN(in_chans, embed_dim[0] // 2, ks=(5, 1), stride=(2, 1), pad=(2, 0)), 
            torch.nn.GELU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], ks=(5, 1), stride=(2, 1), pad=(2, 0))
        )
        resolution = (H // 4, W)  # H reduced by 4x, W unchanged

        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        stages = [self.stage1, self.stage2, self.stage3, self.stage4]
        
        for i, (ch, kd, dpth, nh, ar) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio)):
            for d in range(dpth):
                if isinstance(resolution, tuple):
                    res_value = max(resolution)
                else:
                    res_value = resolution
                stages[i].append(WNBlock(ch, kd, nh, ar, res_value, stage=i, depth=d, 
                                        attn_dropout=attn_dropout, ffn_dropout=ffn_dropout))
            
            # Downsample between stages
            if i != len(depth) - 1:
                if isinstance(resolution, tuple):
                    resolution_ = ((resolution[0] - 1) // 2 + 1, resolution[1])
                    # Downsample only temporal dimension
                    stages[i].append(Conv2d_BN(embed_dim[i], embed_dim[i], ks=(3, 1), 
                                              stride=(2, 1), pad=(1, 0), groups=embed_dim[i]))
                else:
                    resolution_ = (resolution - 1) // 2 + 1
                    stages[i].append(Conv2d_BN(embed_dim[i], embed_dim[i], ks=3, 
                                              stride=2, pad=1, groups=embed_dim[i]))
                stages[i].append(Conv2d_BN(embed_dim[i], embed_dim[i+1], ks=1, stride=1, pad=0))
                resolution = resolution_

        # Classification head
        self.head_dropout = torch.nn.Dropout(p=dropout_rate) if num_classes > 0 else torch.nn.Identity()
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
            
        self.num_classes = num_classes
        self.num_features = embed_dim[-1]

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.head_dropout(x)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (4, 4),
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0.c', 'classifier': ('head.linear', 'head_dist.linear'),
        **kwargs
    }

default_cfgs = dict(
    wnnet_tiny = _cfg(),
    wnnet_tiny_distill = _cfg(),
    wnnet_small = _cfg(),
    wnnet_small_distill = _cfg(),
    wnnet_base = _cfg(),
    wnnet_base_distill = _cfg(),
)

def _create_wnnet(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        WNnet,
        variant,
        pretrained,
        default_cfg=default_cfgs[variant],
        **kwargs,
    )
    return model

@register_model
def wnnet(num_classes=1000, distillation=False, pretrained=False, **kwargs):
    """WNnet: Lightweight architecture for resource-constrained devices"""
    input_shape = kwargs.pop('input_shape', None)
    dropout_rate = kwargs.pop('dropout_rate', 0.2)
    attn_dropout = kwargs.pop('attn_dropout', 0.1)
    ffn_dropout = kwargs.pop('ffn_dropout', 0.1)
    
    model = _create_wnnet("wnnet" + ("_distill" if distillation else ""),
                  pretrained=pretrained,
                  num_classes=num_classes, 
                  distillation=distillation, 
                  img_size=224,
                  patch_size=8,
                  embed_dim=[32, 64],
                  key_dim=[12, 12],
                  depth=[2, 2],
                  num_heads=[3, 3],
                  input_shape=input_shape,
                  dropout_rate=dropout_rate,
                  attn_dropout=attn_dropout,
                  ffn_dropout=ffn_dropout,
                  **kwargs,
                  )
    return model

@register_model
def wnnet_distill(**kwargs):
    kwargs["distillation"] = True
    return wnnet(**kwargs)
