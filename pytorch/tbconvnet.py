import torch
import torch.nn as nn
import torch.nn.functional as F
from forecasting_models.pytorch.swin_unet import SwinTransformerBlock

# ---------------------------------------------------------
# Helpers: Depthwise separable conv + small Dense blocks
# ---------------------------------------------------------

class SeparableConv2d(nn.Module):
    """Depthwise separable conv: DW 3x3 + PW 1x1."""
    def __init__(self, in_ch, out_ch, k=3, p=1, s=1, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pw(self.dw(x))


class SepConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            SeparableConv2d(in_ch, out_ch, k=3, p=1, s=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """Two separable convs + maxpool (like the paper). Returns (skip, down)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = SepConvBNReLU(in_ch, out_ch)
        self.conv2 = SepConvBNReLU(out_ch, out_ch)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return skip, x


class DenseSepBlock(nn.Module):
    """
    A lightweight dense-ish block:
    out = conv(conv(x)) concat x  (kept simple and stable).
    """
    def __init__(self, ch):
        super().__init__()
        self.conv1 = SepConvBNReLU(ch, ch)
        self.conv2 = SepConvBNReLU(ch, ch)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return torch.cat([y, x], dim=1)  # channel grows


class Reduce1x1(nn.Module):
    """Reduce channels after concatenations."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------
# ResNet-style replacement for (Bi)ConvLSTM in skip connections
# ---------------------------------------------------------

class ResConvBlock(nn.Module):
    """Simple ResNet-like block that preserves H,W."""
    def __init__(self, ch, bottleneck_ratio=0.5):
        super().__init__()
        mid = max(8, int(ch * bottleneck_ratio))
        self.conv1 = nn.Conv2d(ch, mid, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + identity)
        return x


# ---------------------------------------------------------
# Wrapper to use your existing SwinTransformerBlock on feature maps
# ---------------------------------------------------------

class SwinOnFeatureMap(nn.Module):
    """
    Wraps your SwinTransformerBlock (expects B,L,C) to work on (B,C,H,W).

    - input:  (B,C,H,W)
    - output: (B,C,H,W)
    """
    def __init__(self, dim, input_resolution, num_heads, window_size, mlp_ratio,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # expects SwinTransformerBlock to be already defined in your codebase
        self.block = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,  # (H, W) in tokens grid at this stage
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer
        )
        self.H, self.W = input_resolution

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == (self.H, self.W), f"SwinOnFeatureMap expected {(self.H,self.W)} got {(H,W)}"
        # (B,C,H,W) -> (B,H*W,C)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x = self.block(x)  # (B,L,C)
        # back to (B,C,H,W)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x


class SkipFusion(nn.Module):
    """
    Replacement for: BConvLSTM + Swin Transformer inside skips.

    Here: ResConvBlock -> SwinBlock (lightweight) -> output features
    """
    def __init__(self, ch, token_hw, num_heads, window_size, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.res = ResConvBlock(ch)
        self.swin = SwinOnFeatureMap(
            dim=ch,
            input_resolution=token_hw,  # (H, W) at that skip level
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

    def forward(self, x):
        x = self.res(x)
        x = self.swin(x)
        return x


# ---------------------------------------------------------
# Decoder blocks
# ---------------------------------------------------------

class UpBlock(nn.Module):
    """Upsample + fuse skip + two separable convs."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = SepConvBNReLU(out_ch + skip_ch, out_ch)
        self.conv2 = SepConvBNReLU(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # size safety
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(self.conv1(x))
        return x