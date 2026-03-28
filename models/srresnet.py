import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    def __init__(self, n_feats: int, use_batch_norm: bool):
        super().__init__()
        layers = [
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(n_feats))
        layers.append(nn.PReLU(num_parameters=n_feats))
        layers.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(n_feats))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, n_feats: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.PReLU(num_parameters=n_feats),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SRResNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        n_feats: int = 64,
        n_res_blocks: int = 16,
        scale: int = 4,
        use_batch_norm: bool = True,
        use_global_skip: bool = True,
    ):
        super().__init__()
        if scale not in {2, 4, 8}:
            raise ValueError(f"scale must be one of {{2,4,8}}, got {scale}")

        self.scale = int(scale)
        self.use_global_skip = bool(use_global_skip)

        self.head = nn.Sequential(
            nn.Conv2d(n_channels, n_feats, kernel_size=9, padding=4, bias=True),
            nn.PReLU(num_parameters=n_feats),
        )

        self.body = nn.Sequential(
            *[_ResidualBlock(n_feats=n_feats, use_batch_norm=use_batch_norm) for _ in range(n_res_blocks)]
        )

        body_tail_layers = [nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)]
        if use_batch_norm:
            body_tail_layers.append(nn.BatchNorm2d(n_feats))
        self.body_tail = nn.Sequential(*body_tail_layers)

        n_upsample = int(math.log2(scale))
        self.upsample = nn.Sequential(*[_UpsampleBlock(n_feats=n_feats) for _ in range(n_upsample)])

        self.tail = nn.Conv2d(n_feats, n_channels, kernel_size=9, padding=4, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = None
        if self.use_global_skip:
            x_up = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)

        feat = self.head(x)
        res = self.body(feat)
        res = self.body_tail(res)
        res = res + feat
        out = self.upsample(res)
        out = self.tail(out)

        if x_up is not None:
            out = out + x_up
        return out

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_srresnet(cfg) -> SRResNet:
    model_cfg = cfg.get("srresnet", {})
    scale = cfg.data.get("scale_factor", 4)
    return SRResNet(
        n_channels=model_cfg.get("n_channels", 1),
        n_feats=model_cfg.get("n_feats", 64),
        n_res_blocks=model_cfg.get("n_res_blocks", 16),
        scale=scale,
        use_batch_norm=model_cfg.get("use_batch_norm", True),
        use_global_skip=model_cfg.get("use_global_skip", True),
    )
