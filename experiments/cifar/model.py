import numpy as np
import torch
import torch.nn as nn

from nimbus.seeds import set_seed, device


class ChannelLayerNorm(nn.Module):
    """layernorm over channels (permute to channels-last and back)."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DropPath(nn.Module):
    """stochastic depth: drop the residual branch per sample during training."""

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep
        return x * mask / keep


class Block(nn.Module):
    """ConvNeXt block: depthwise conv, norm, 1x1 expand, SiLU, 1x1 project, residual."""

    def __init__(self, dim, expand_ratio=3, drop_path=0.0):
        super().__init__()
        hidden = dim * expand_ratio
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            ChannelLayerNorm(dim),
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.layers(x))


class Downsample(nn.Module):
    """norm then stride-2 conv halving spatial size."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = ChannelLayerNorm(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 2, stride=2, bias=False)

    def forward(self, x):
        return self.conv(self.norm(x))


class SmallConvNeXt(nn.Module):
    """small ConvNeXt-style cnn: stem, staged blocks with downsamples, mean-pool head."""

    def __init__(self, dims=(16, 48, 192), depths=(1, 1, 3),
                 expand_ratios=(2, 2, 3), num_classes=100, drop_path_rate=0.0):
        super().__init__()
        assert len(dims) == len(depths) == len(expand_ratios)

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 3, padding=1, bias=False),
            ChannelLayerNorm(dims[0]),
            nn.SiLU(inplace=True),
        )

        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        stages = []
        bi = 0
        for i, (d, depth, er) in enumerate(zip(dims, depths, expand_ratios)):
            for _ in range(depth):
                stages.append(Block(d, expand_ratio=er, drop_path=dp_rates[bi]))
                bi += 1
            if i < len(dims) - 1:
                stages.append(Downsample(dims[i], dims[i + 1]))

        self.backbone = nn.Sequential(*stages)
        self.head_norm = ChannelLayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.head_norm(x)
        x = x.mean(dim=(2, 3))
        return self.head(x)


def make_convnext(seed, num_classes, dims=(16, 48, 192), depths=(1, 1, 3)):
    """build a seeded SmallConvNeXt on `device`."""
    set_seed(seed)
    return SmallConvNeXt(
        dims=dims, depths=depths,
        expand_ratios=(2, 2, 3),
        num_classes=num_classes,
    ).to(device)


def load_state(model, sd):
    """load a state dict, stripping a leading 'model.' prefix if present."""
    if any(k.startswith("model.") for k in sd):
        sd = {k.removeprefix("model."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model


def make_convnext_single4x(seed, num_classes, param_mult):
    """
    wide single4x baseline body: widths scaled by sqrt(param_mult) so the
    parameter count is param_mult times the base model.
    """
    scale = float(np.sqrt(param_mult))
    dims_wide = tuple(max(1, int(round(d * scale))) for d in (16, 48, 192))
    return make_convnext(seed=seed, num_classes=num_classes, dims=dims_wide)
