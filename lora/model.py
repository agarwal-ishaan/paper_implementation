import torch
import torch.nn as nn


class LoRAConv1D(nn.Module):
    """Wraps a GPT-2 Conv1D attention layer (fused q/k/v, weight shape
    (in_features, 3 * out_features_third)) with LoRA adapters on the query
    and value thirds only. The key third is passed through from the frozen
    base layer untouched, matching the paper's Wq+Wv-only setup.

    Training:  out = base(x), with q and v thirds additionally getting
               (alpha/r) * x @ A @ B.
    Init:      A ~ N(0, 0.01^2), B = 0, so the adapter starts as a no-op
               (delta_norm() == 0) and the wrapped layer behaves exactly
               like the frozen base until training moves B away from zero.
    """

    def __init__(self, base, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_features, out_features = base.weight.shape
        if out_features % 3 != 0:
            raise ValueError("Conv1D output must be divisible by 3 (q/k/v)")
        out_features_third = out_features // 3
        self.out_features_third = out_features_third
        self.scaling = alpha / r

        self.lora_A_q = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B_q = nn.Parameter(torch.zeros(r, out_features_third))
        self.lora_A_v = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B_v = nn.Parameter(torch.zeros(r, out_features_third))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        third = self.out_features_third
        q = base_out[..., 0:third]
        k = base_out[..., third:2 * third]
        v = base_out[..., 2 * third:3 * third]
        q = q + self.scaling * (x @ self.lora_A_q) @ self.lora_B_q
        v = v + self.scaling * (x @ self.lora_A_v) @ self.lora_B_v
        return torch.cat([q, k, v], dim=-1)

    def delta_norm(self) -> float:
        """Frobenius norm of the combined [delta_Wq; delta_Wv] update, for
        tracking how far the adapter has moved from its zero-init start."""
        delta_q = self.scaling * (self.lora_A_q @ self.lora_B_q)
        delta_v = self.scaling * (self.lora_A_v @ self.lora_B_v)
        return torch.sqrt((delta_q ** 2).sum() + (delta_v ** 2).sum()).item()


def inject_lora(model, r: int = 8, alpha: int = 16) -> list:
    """Replace each transformer block's c_attn with a LoRAConv1D wrapper."""
    lora_modules = []
    for block in model.transformer.h:
        wrapped = LoRAConv1D(block.attn.c_attn, r=r, alpha=alpha)
        block.attn.c_attn = wrapped
        lora_modules.append(wrapped)
    return lora_modules


def setup_lora_model(model, r: int = 8, alpha: int = 16) -> list:
    """Freeze every parameter, then inject trainable LoRA adapters."""
    for p in model.parameters():
        p.requires_grad = False
    return inject_lora(model, r=r, alpha=alpha)


def trainable_parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters())
