import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import model as model_module
from model import ConvBranch, StochasticDepthBlock, survival_probabilities


def test_survival_probabilities_length():
    probs = survival_probabilities(18)
    assert len(probs) == 18


def test_survival_probabilities_last_equals_p_L():
    probs = survival_probabilities(18, p_L=0.5)
    assert probs[-1] == pytest.approx(0.5)


def test_survival_probabilities_first_close_to_one():
    probs = survival_probabilities(18, p_L=0.5)
    assert probs[0] == pytest.approx(1 - (1 / 18) * 0.5)


def test_survival_probabilities_monotonically_decreasing():
    probs = survival_probabilities(18)
    assert all(probs[i] > probs[i + 1] for i in range(len(probs) - 1))


def test_survival_probabilities_all_one_when_p_L_is_one():
    probs = survival_probabilities(10, p_L=1.0)
    assert all(p == pytest.approx(1.0) for p in probs)


class CountingBranch(nn.Module):
    """Test double: records call count and applies a learnable scale, so
    output can be predicted exactly without a real conv branch."""

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        self.calls += 1
        return x * self.scale


def test_conv_branch_output_shape_same_spatial_size():
    branch = ConvBranch(in_channels=8, out_channels=8, stride=1)
    x = torch.randn(2, 8, 16, 16)
    out = branch(x)
    assert out.shape == (2, 8, 16, 16)


def test_conv_branch_downsamples_with_stride_2():
    branch = ConvBranch(in_channels=8, out_channels=16, stride=2)
    x = torch.randn(2, 8, 16, 16)
    out = branch(x)
    assert out.shape == (2, 16, 8, 8)


def test_stochastic_depth_block_skips_branch_when_gated_off(monkeypatch):
    branch = CountingBranch()
    block = StochasticDepthBlock(branch, nn.Identity(), p=0.5)
    block.train()
    monkeypatch.setattr(model_module.random, "random", lambda: 0.9)  # 0.9 < 0.5 is False
    x = torch.randn(2, 3, 4, 4)
    out = block(x)
    assert branch.calls == 0
    assert torch.allclose(out, F.relu(x))


def test_stochastic_depth_block_computes_branch_when_gated_on(monkeypatch):
    branch = CountingBranch()
    block = StochasticDepthBlock(branch, nn.Identity(), p=0.5)
    block.train()
    monkeypatch.setattr(model_module.random, "random", lambda: 0.1)  # 0.1 < 0.5 is True
    x = torch.randn(2, 3, 4, 4)
    out = block(x)
    assert branch.calls == 1
    expected = F.relu(x + branch.scale * x)
    assert torch.allclose(out, expected)


def test_stochastic_depth_block_eval_mode_scales_by_p_deterministically():
    branch = CountingBranch()
    block = StochasticDepthBlock(branch, nn.Identity(), p=0.3)
    block.eval()
    x = torch.randn(2, 3, 4, 4)
    out = block(x)
    expected = F.relu(x + 0.3 * branch.scale * x)
    assert torch.allclose(out, expected)
    assert branch.calls == 1


def test_stochastic_depth_block_eval_mode_is_reproducible():
    branch = CountingBranch()
    block = StochasticDepthBlock(branch, nn.Identity(), p=0.3)
    block.eval()
    x = torch.randn(2, 3, 4, 4)
    out1 = block(x)
    out2 = block(x)
    assert torch.allclose(out1, out2)


from model import ResNetCIFAR


def test_resnet_output_shape():
    net = ResNetCIFAR(blocks_per_stage=6, stochastic_depth=True)
    x = torch.randn(4, 3, 32, 32)
    out = net(x)
    assert out.shape == (4, 10)


def test_resnet_has_18_blocks():
    net = ResNetCIFAR(blocks_per_stage=6)
    assert net.num_blocks == 18
    assert len(net.blocks) == 18


def test_resnet_baseline_mode_has_all_survival_probs_equal_one():
    net = ResNetCIFAR(blocks_per_stage=6, stochastic_depth=False)
    assert all(p == 1.0 for p in net.survival_probs)


def test_resnet_stochastic_depth_mode_uses_linear_decay_schedule():
    net = ResNetCIFAR(blocks_per_stage=6, stochastic_depth=True, p_L=0.5)
    assert net.survival_probs == survival_probabilities(18, p_L=0.5)


def test_resnet_eval_mode_is_deterministic():
    net = ResNetCIFAR(blocks_per_stage=6, stochastic_depth=True)
    net.eval()
    x = torch.randn(2, 3, 32, 32)
    out1 = net(x)
    out2 = net(x)
    assert torch.allclose(out1, out2)


def test_resnet_train_mode_runs_without_error():
    net = ResNetCIFAR(blocks_per_stage=6, stochastic_depth=True)
    net.train()
    x = torch.randn(4, 3, 32, 32)
    out = net(x)
    assert out.shape == (4, 10)
