import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from model import LoRAConv1D, inject_lora, setup_lora_model, trainable_parameter_count, total_parameter_count


def make_base_conv1d(in_features=10, out_features_third=6, seed=0):
    torch.manual_seed(seed)
    return Conv1D(out_features_third * 3, in_features)


def test_lora_conv1d_output_shape():
    base = make_base_conv1d()
    lora = LoRAConv1D(base, r=4, alpha=8)
    x = torch.randn(2, 5, 10)
    out = lora(x)
    assert out.shape == (2, 5, 18)


def test_lora_conv1d_zero_init_matches_base_output():
    base = make_base_conv1d()
    x = torch.randn(2, 5, 10)
    with torch.no_grad():
        base_out = base(x)
    lora = LoRAConv1D(base, r=4, alpha=8)
    with torch.no_grad():
        lora_out = lora(x)
    assert torch.allclose(lora_out, base_out, atol=1e-6)


def test_lora_conv1d_freezes_base_parameters():
    base = make_base_conv1d()
    lora = LoRAConv1D(base, r=4, alpha=8)
    assert lora.base.weight.requires_grad is False
    assert lora.base.bias.requires_grad is False


def test_lora_conv1d_lora_params_are_trainable():
    base = make_base_conv1d()
    lora = LoRAConv1D(base, r=4, alpha=8)
    assert lora.lora_A_q.requires_grad is True
    assert lora.lora_B_q.requires_grad is True
    assert lora.lora_A_v.requires_grad is True
    assert lora.lora_B_v.requires_grad is True


def test_lora_conv1d_key_slice_unaffected_by_lora():
    base = make_base_conv1d()
    x = torch.randn(2, 5, 10)
    with torch.no_grad():
        base_out = base(x)
    lora = LoRAConv1D(base, r=4, alpha=8)
    with torch.no_grad():
        lora.lora_B_q.add_(1.0)
        lora.lora_B_v.add_(1.0)
        lora_out = lora(x)
    third = lora.out_features_third
    base_k = base_out[..., third:2 * third]
    lora_k = lora_out[..., third:2 * third]
    assert torch.allclose(base_k, lora_k, atol=1e-6)
    base_q = base_out[..., 0:third]
    lora_q = lora_out[..., 0:third]
    assert not torch.allclose(base_q, lora_q, atol=1e-6)


def test_delta_norm_zero_at_init():
    base = make_base_conv1d()
    lora = LoRAConv1D(base, r=4, alpha=8)
    assert lora.delta_norm() == pytest.approx(0.0, abs=1e-6)


def test_delta_norm_positive_after_update():
    base = make_base_conv1d()
    lora = LoRAConv1D(base, r=4, alpha=8)
    with torch.no_grad():
        lora.lora_B_q.add_(1.0)
    assert lora.delta_norm() > 0.0


def test_inject_lora_wraps_every_block():
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    lora_modules = inject_lora(model, r=4, alpha=8)
    assert len(lora_modules) == model.config.n_layer
    for block in model.transformer.h:
        assert isinstance(block.attn.c_attn, LoRAConv1D)


def test_setup_lora_model_freezes_everything_except_lora_params():
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    lora_modules = setup_lora_model(model, r=4, alpha=8)
    expected_trainable = sum(
        m.lora_A_q.numel() + m.lora_B_q.numel() + m.lora_A_v.numel() + m.lora_B_v.numel()
        for m in lora_modules
    )
    assert trainable_parameter_count(model) == expected_trainable


def test_lora_trainable_count_much_smaller_than_full():
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    total = total_parameter_count(model)
    setup_lora_model(model, r=8, alpha=16)
    lora_trainable = trainable_parameter_count(model)
    assert lora_trainable < total / 100


def test_total_parameter_count_matches_manual_sum():
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    manual = sum(p.numel() for p in model.parameters())
    assert total_parameter_count(model) == manual
