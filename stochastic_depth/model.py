import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def survival_probabilities(num_blocks: int, p_L: float = 0.5) -> list[float]:
    """Linear decay rule from the paper: p_l = 1 - (l / L) * (1 - p_L), l = 1..L."""
    return [1 - (l / num_blocks) * (1 - p_L) for l in range(1, num_blocks + 1)]


class ConvBranch(nn.Module):
    """The residual branch f_l: conv-bn-relu-conv-bn."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out


class StochasticDepthBlock(nn.Module):
    """Wraps a residual branch with a per-forward-pass Bernoulli gate.

    Training: identity(x) + gate * branch(x), gate ~ Bernoulli(p); branch is
    not computed at all when gate == 0.
    Eval: identity(x) + p * branch(x), deterministic, branch always computed.
    """

    def __init__(self, branch: nn.Module, shortcut: nn.Module, p: float):
        super().__init__()
        self.branch = branch
        self.shortcut = shortcut
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        if self.training:
            if random.random() < self.p:
                out = identity + self.branch(x)
            else:
                out = identity
        else:
            out = identity + self.p * self.branch(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(
        self,
        blocks_per_stage: int = 6,
        p_L: float = 0.5,
        stochastic_depth: bool = True,
        num_classes: int = 10,
    ):
        super().__init__()
        stage_channels = [16, 32, 64]
        num_blocks = blocks_per_stage * len(stage_channels)
        probs = (
            survival_probabilities(num_blocks, p_L) if stochastic_depth else [1.0] * num_blocks
        )

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        blocks = []
        in_channels = 16
        block_idx = 0
        for stage_idx, out_channels in enumerate(stage_channels):
            for i in range(blocks_per_stage):
                stride = 2 if (stage_idx > 0 and i == 0) else 1
                branch = ConvBranch(in_channels, out_channels, stride=stride)
                if stride != 1 or in_channels != out_channels:
                    shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_channels),
                    )
                else:
                    shortcut = nn.Identity()
                blocks.append(StochasticDepthBlock(branch, shortcut, probs[block_idx]))
                in_channels = out_channels
                block_idx += 1
        self.blocks = nn.ModuleList(blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_channels[-1], num_classes)
        self.num_blocks = num_blocks
        self.survival_probs = probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        for block in self.blocks:
            out = block(out)
        out = self.pool(out).flatten(1)
        return self.fc(out)
