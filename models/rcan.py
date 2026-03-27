import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Channel Attention (CA) Layer
# This adaptively rescales channel-wise features by modeling interdependencies.
class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y

# 2. Residual Channel Attention Block (RCAB)
# This replaces your standard ResBlock.
class RCAB(nn.Module):
    def __init__(self, n_feats, kernel_size=3, reduction=16):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=1, bias=True),
            ChannelAttention(n_feats, reduction)
        )

    def forward(self, x):
        return x + self.body(x)

# 3. Residual Group (RG)
# A collection of RCABs with a short-skip connection.
class ResidualGroup(nn.Module):
    def __init__(self, n_feats, n_blocks, reduction=16):
        super(ResidualGroup, self).__init__()
        self.body = nn.Sequential(*[
            RCAB(n_feats, reduction=reduction) for _ in range(n_blocks)
        ])
        self.conv = nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True)

    def forward(self, x):
        return x + self.conv(self.body(x))

# 4. The Full RCAN Model
class RCAN(nn.Module):
    def __init__(self, n_channels=1, n_feats=64, n_groups=5, n_blocks=8, scale=4, reduction=16):
        super(RCAN, self).__init__()
        self.scale = scale
        
        # Head: Initial feature extraction
        self.head = nn.Conv2d(n_channels, n_feats, 3, padding=1, bias=True)

        # Body: Residual in Residual (RIR)
        self.body = nn.Sequential(*[
            ResidualGroup(n_feats, n_blocks, reduction=reduction) for _ in range(n_groups)
        ])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True)

        # Upsampling (PixelShuffle)
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale * scale, 3, padding=1, bias=True),
            nn.PixelShuffle(scale)
        )

        # Tail: Final reconstruction
        self.tail = nn.Conv2d(n_feats, n_channels, 3, padding=1, bias=True)

    def forward(self, x):
        # Global Skip Connection (Bicubic Reference)
        x_up = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)

        # Main RCAN Path
        feat = self.head(x)
        res = self.body(feat)
        res = self.body_tail(res)
        res += feat  # Long Skip Connection
        
        out = self.upsample(res)
        out = self.tail(out)

        return out + x_up


    

def build_rcan(cfg):
    model_cfg = cfg.model_rcan

    return RCAN(
        n_channels=model_cfg.get("n_channels", 1),
        n_feats=model_cfg.get("n_feats", 64),
        n_groups=model_cfg.get("n_groups", 5),
        n_blocks=model_cfg.get("n_blocks", 8),
        scale=cfg.get("scale_factor", 4),
        reduction=model_cfg.get("reduction", 16),
    )