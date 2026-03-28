import torch
import torch.nn as nn

class FSRCNN(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        d: int = 56,
        s: int = 12,
        m: int = 4,
        scale: int = 4,
    ):
        super().__init__()
        if scale < 2:
            raise ValueError(f"scale must be >= 2, got {scale}")
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")

        self.scale = int(scale)

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(n_channels, d, kernel_size=5, padding=2, bias=True),
            nn.PReLU(num_parameters=d),
        )
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1, padding=0, bias=True),
            nn.PReLU(num_parameters=s),
        )

        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=1, bias=True))
            mapping_layers.append(nn.PReLU(num_parameters=s))
        self.mapping = nn.Sequential(*mapping_layers)

        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1, padding=0, bias=True),
            nn.PReLU(num_parameters=d),
        )
        self.deconv = nn.ConvTranspose2d(
            d,
            n_channels,
            kernel_size=9,
            stride=self.scale,
            padding=4,
            output_padding=self.scale - 1,
            bias=True,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.deconv(x)
        return x

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_fsrcnn(cfg) -> FSRCNN:
    model_cfg = cfg.get("fsrcnn", {})
    scale = cfg.data.get("scale_factor", 4)
    return FSRCNN(
        n_channels=model_cfg.get("n_channels", 1),
        d=model_cfg.get("d", 56),
        s=model_cfg.get("s", 12),
        m=model_cfg.get("m", 4),
        scale=scale,
    )
