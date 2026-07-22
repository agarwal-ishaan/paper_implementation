# LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a from-scratch, tested implementation of LoRA (Hu et al. 2021) applied to DistilGPT-2's attention query/value projections, delivered as a single notebook that fine-tunes on tiny Shakespeare two ways — full fine-tuning vs. LoRA — and compares them, persisting checkpoints and per-step metrics during training so later analysis never needs a retrain.

**Architecture:** Core mechanism (`LoRAConv1D` — a frozen-base Conv1D wrapper with trainable low-rank adapters on the q/v output slices only) and the injection/freezing helpers live in a plain, unit-tested `model.py`. The training loop (`train_step`, `evaluate`, checkpoint/metric persistence) lives in a separately tested `train.py`. The notebook imports both, adds the paper explanation, loads DistilGPT-2 + tiny Shakespeare, runs both fine-tuning variants, and produces the comparison visualizations.

**Tech Stack:** Python 3.13, PyTorch 2.7.1, `transformers` 4.54.1, `datasets` 4.8.5, `psutil` 7.0.0, pytest 9.0.2, matplotlib, Jupyter (all already installed; MPS backend confirmed available). No new dependencies needed — LoRA is built from scratch, not via `peft` (not installed, and not wanted per the design doc).

## Global Constraints

- LoRA applies only to the query and value thirds of each block's fused `c_attn` Conv1D output (columns `0:768` and `1536:2304` of the 2304-wide output); the key third (`768:1536`) is never touched by the adapter (from `lora/design.md`).
- `Conv1D` weight shape is `(in_features, out_features)` — the opposite convention from `nn.Linear` — so `ΔW = (alpha/r) · A · B` with `A: (in_features, r)`, `B: (r, out_features_third)`.
- LoRA init: `A ~ N(0, 0.01^2)`, `B = 0`, so `ΔW = 0` exactly at init (a property to test for directly).
- Base model: DistilGPT-2 (`transformers.GPT2LMHeadModel.from_pretrained("distilgpt2")`, `transformers.AutoTokenizer.from_pretrained("distilgpt2")`), 6 layers, hidden size 768, ~82M total params.
- Dataset: `Trelis/tiny-shakespeare` (HF Hub), tokenized and chunked into fixed-length blocks of `block_size=128` (2887 train blocks, 302 test blocks at this block size).
- Training scale (benchmarked on this machine: ~238ms/step full fine-tune on MPS with batch_size=8, seq_len=128): `batch_size=8`, `3 epochs` (~360 steps/epoch, ~1080 steps total per run, ~4-5 min per run).
- LoRA rank `r=8`, `alpha=16` (scaling factor 2.0).
- Optimizer: AdamW, `lr=5e-5` for full fine-tuning, `lr=3e-4` for LoRA (higher LR is standard practice for LoRA since only the small adapters are updated).
- Device: `torch.device("mps" if torch.backends.mps.is_available() else "cpu")`.
- Persist during training (not only at the end), under `lora/results/`: per-step metrics as JSON (loss, time, memory; LoRA run also logs `delta_norm`), model checkpoints (full fine-tune: full state dict; LoRA: adapter-only state dict), and generation samples at the end of each epoch.
- All new files live under `lora/` (flat, no subpackage), so `python3 -m pytest test_model.py test_train.py` run from inside that directory can import `model` and `train` directly.
- `.gitignore` already excludes `*.pt`/`*.pth`; add an exception so `lora/results/lora_adapter*.pt` (small) is trackable while `lora/results/full_finetune*.pt` (large, ~330MB) stays ignored.
- Commit after each task.

---

### Task 1: `LoRAConv1D` — the adapter itself

**Files:**
- Create: `lora/model.py`
- Test: `lora/test_model.py`

**Interfaces:**
- Produces: `LoRAConv1D(base: Conv1D, r: int = 8, alpha: int = 16)`, an `nn.Module` with `.forward(x) -> Tensor` (same output shape as `base`), public attributes `.out_features_third: int`, `.scaling: float`, `.lora_A_q`, `.lora_B_q`, `.lora_A_v`, `.lora_B_v` (all `nn.Parameter`), and method `.delta_norm() -> float`. Used by Task 2's injection helper and Task 5's notebook.

- [ ] **Step 1: Write the failing tests**

Create `lora/test_model.py`:

```python
import pytest
import torch
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from model import LoRAConv1D


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd lora && python3 -m pytest test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model'`.

- [ ] **Step 3: Write minimal implementation**

Create `lora/model.py`:

```python
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
        assert out_features % 3 == 0, "Conv1D output must be divisible by 3 (q/k/v)"
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd lora && python3 -m pytest test_model.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add lora/model.py lora/test_model.py
git commit -m "Add LoRA adapter (LoRAConv1D) for GPT-2 attention q/v projections"
```

---

### Task 2: Injection and freezing helpers

**Files:**
- Modify: `lora/model.py`
- Modify: `lora/test_model.py`

**Interfaces:**
- Consumes: `LoRAConv1D` (Task 1).
- Produces: `inject_lora(model, r: int = 8, alpha: int = 16) -> list[LoRAConv1D]`, `setup_lora_model(model, r: int = 8, alpha: int = 16) -> list[LoRAConv1D]`, `trainable_parameter_count(model) -> int`, `total_parameter_count(model) -> int`. Used by Task 4's tests and Task 5's notebook.

This task's tests download `distilgpt2` from the HF Hub on first run (~330MB, cached afterward under `~/.cache/huggingface`) — unlike CIFAR-10's slow Toronto mirror, HF Hub's CDN is fast, so this should take well under a minute.

- [ ] **Step 1: Write the failing tests**

Append to `lora/test_model.py`:

```python
from transformers import GPT2LMHeadModel

from model import (
    inject_lora,
    setup_lora_model,
    trainable_parameter_count,
    total_parameter_count,
)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd lora && python3 -m pytest test_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'inject_lora' from 'model'`.

- [ ] **Step 3: Write minimal implementation**

Append to `lora/model.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd lora && python3 -m pytest test_model.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add lora/model.py lora/test_model.py
git commit -m "Add LoRA injection and parameter-freezing helpers"
```

---

### Task 3: Training loop, memory tracking, and checkpoint persistence

**Files:**
- Create: `lora/train.py`
- Create: `lora/test_train.py`

**Interfaces:**
- Consumes: nothing from Task 1/2 directly (works on any `GPT2LMHeadModel`-shaped model, tested against the real `distilgpt2`).
- Produces: `compute_loss(model, input_ids, attention_mask=None) -> Tensor` (scalar), `train_step(model, batch, optimizer, device) -> dict` (keys `"loss"`, `"time_seconds"`), `evaluate(model, loader, device) -> dict` (key `"loss"`), `current_memory_mb() -> float`, `save_checkpoint(model, path: str) -> None`, `save_lora_adapters(lora_modules, path: str) -> None`. Used by Task 5's notebook.

- [ ] **Step 1: Write the failing tests**

Create `lora/test_train.py`:

```python
import torch
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from model import LoRAConv1D
from train import (
    compute_loss,
    current_memory_mb,
    evaluate,
    save_checkpoint,
    save_lora_adapters,
    train_step,
)


def _tiny_batch(vocab_size=50257, batch_size=2, seq_len=8):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def test_compute_loss_returns_positive_scalar():
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    batch = _tiny_batch()
    loss = compute_loss(model, batch["input_ids"], batch["attention_mask"])
    assert loss.dim() == 0
    assert loss.item() > 0


def test_train_step_returns_expected_keys_and_updates_params():
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = _tiny_batch()
    before = model.lm_head.weight.clone()
    result = train_step(model, batch, optimizer, device=torch.device("cpu"))
    assert set(result.keys()) == {"loss", "time_seconds"}
    assert result["time_seconds"] > 0
    after = model.lm_head.weight
    assert not torch.equal(before, after)


def test_evaluate_returns_loss_without_updating_params():
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    batch = _tiny_batch()
    loader = [batch, batch]
    before = [p.clone() for p in model.parameters()]
    result = evaluate(model, loader, device=torch.device("cpu"))
    assert "loss" in result
    after = list(model.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_current_memory_mb_returns_positive_float():
    assert current_memory_mb() > 0


def test_save_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(4, 4)
    path = tmp_path / "model.pt"
    save_checkpoint(model, str(path))
    loaded_state = torch.load(path)
    assert set(loaded_state.keys()) == set(model.state_dict().keys())


def test_save_lora_adapters_roundtrip(tmp_path):
    base = Conv1D(9, 3)
    lora_modules = [LoRAConv1D(base, r=2, alpha=4)]
    path = tmp_path / "adapters.pt"
    save_lora_adapters(lora_modules, str(path))
    state = torch.load(path)
    assert torch.equal(state["layer_0.lora_A_q"], lora_modules[0].lora_A_q.detach().cpu())
    assert torch.equal(state["layer_0.lora_B_q"], lora_modules[0].lora_B_q.detach().cpu())
    assert torch.equal(state["layer_0.lora_A_v"], lora_modules[0].lora_A_v.detach().cpu())
    assert torch.equal(state["layer_0.lora_B_v"], lora_modules[0].lora_B_v.detach().cpu())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd lora && python3 -m pytest test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train'`.

- [ ] **Step 3: Write minimal implementation**

Create `lora/train.py`:

```python
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss(model, input_ids, attention_mask=None) -> torch.Tensor:
    """Causal LM loss: predict token t+1 from tokens up to t."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )


def train_step(model, batch, optimizer, device) -> dict:
    model.train()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    optimizer.zero_grad()
    start = time.perf_counter()
    loss = compute_loss(model, input_ids, attention_mask)
    loss.backward()
    optimizer.step()
    elapsed = time.perf_counter() - start
    return {"loss": loss.item(), "time_seconds": elapsed}


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        loss = compute_loss(model, input_ids, attention_mask)
        total_loss += loss.item()
        total_batches += 1
    return {"loss": total_loss / total_batches}


def current_memory_mb() -> float:
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    import psutil
    return psutil.Process().memory_info().rss / (1024 ** 2)


def save_checkpoint(model, path: str) -> None:
    torch.save(model.state_dict(), path)


def save_lora_adapters(lora_modules, path: str) -> None:
    state = {}
    for i, module in enumerate(lora_modules):
        state[f"layer_{i}.lora_A_q"] = module.lora_A_q.detach().cpu()
        state[f"layer_{i}.lora_B_q"] = module.lora_B_q.detach().cpu()
        state[f"layer_{i}.lora_A_v"] = module.lora_A_v.detach().cpu()
        state[f"layer_{i}.lora_B_v"] = module.lora_B_v.detach().cpu()
    torch.save(state, path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd lora && python3 -m pytest test_train.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run the full test suite together**

Run: `cd lora && python3 -m pytest -v`
Expected: 17 passed (11 from `test_model.py` + 6 from `test_train.py`).

- [ ] **Step 6: Commit**

```bash
git add lora/train.py lora/test_train.py
git commit -m "Add LoRA train/eval loop with memory tracking and checkpoint persistence"
```

---

### Task 4: `.gitignore` exception for LoRA adapter checkpoints

**Files:**
- Modify: `.gitignore` (repo root)

**Interfaces:** None (no code).

- [ ] **Step 1: Add the exception**

The repo root `.gitignore` currently has `*.pt` and `*.pth` as blanket ignores (added for stochastic depth, which never committed any checkpoints). LoRA's adapter files are small (a few hundred KB) and are themselves part of the paper's storage-efficiency point, so they should be trackable while the large full-fine-tune checkpoint stays ignored. Add after the existing `*.pth` line:

```
!lora/results/lora_adapter*.pt
```

- [ ] **Step 2: Verify the exception works**

Run: `mkdir -p lora/results && touch lora/results/lora_adapter_test.pt lora/results/full_finetune_test.pt && cd .. && git status --short lora/results/`
Expected: `lora/results/lora_adapter_test.pt` shows as untracked (`??`); `lora/results/full_finetune_test.pt` does not appear at all (ignored).

Then clean up the test files: `rm lora/results/lora_adapter_test.pt lora/results/full_finetune_test.pt`

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "Allow committing small LoRA adapter checkpoints, keep full fine-tune checkpoints ignored"
```

---

### Task 5: Notebook — paper walkthrough, both fine-tuning runs, and comparison

**Files:**
- Create: `lora/lora.ipynb`

**Interfaces:**
- Consumes: `LoRAConv1D`, `setup_lora_model`, `trainable_parameter_count`, `total_parameter_count` (from `model.py`); `train_step`, `evaluate`, `current_memory_mb`, `save_checkpoint`, `save_lora_adapters` (from `train.py`).

This is the main deliverable, verified by executing it end-to-end rather than pytest. Build it as the following cells, in order, using the `NotebookEdit` tool.

- [ ] **Step 1: Title and paper explanation (markdown cell)**

```markdown
# LoRA: Low-Rank Adaptation of Large Language Models

Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen — ICLR 2022. [Paper PDF](LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models.pdf)

**The problem:** fine-tuning a large pretrained model on a new task usually means updating
every parameter — expensive in compute, memory (optimizer state for every parameter), and
storage (a full copy of the model per task).

**The idea:** freeze the pretrained weight matrix `W` entirely, and learn a small additive
correction instead: `delta_W = (alpha/r) * A @ B`, where `A` is `in_features x r`, `B` is
`r x out_features`, and `r` (here, 8) is far smaller than the matrix's actual dimensions. Only
`A` and `B` get gradients:

$$h = xW + \frac{\alpha}{r} \cdot x A B$$

At initialization, `A` is small random noise and `B` is exactly zero, so `delta_W = 0` and the
adapted model starts out identical to the frozen base model — training only ever moves the
adapter away from a no-op, never starts from a random perturbation.

**Where it's applied:** the paper's own ablation finds adapting just the query and value
projections (`Wq`, `Wv`) is more effective than spreading the same parameter budget across all
four attention matrices, so that's what this notebook does — GPT-2 fuses q/k/v into one `c_attn`
matrix, so the adapter here targets the query and value thirds of its output, leaving the key
third untouched.

This notebook implements the mechanism from scratch (see `model.py`, `train.py` — no `peft`
library) and fine-tunes DistilGPT-2 on tiny Shakespeare two ways: full fine-tuning (every
parameter trainable) vs. LoRA (only ~0.1% of parameters trainable), comparing trainable
parameter count, checkpoint size, training time/memory, loss curves, and generated text.
```

- [ ] **Step 2: Imports and device setup (code cell)**

```python
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, GPT2LMHeadModel

from model import setup_lora_model, trainable_parameter_count, total_parameter_count
from train import compute_loss, current_memory_mb, evaluate, save_checkpoint, save_lora_adapters, train_step

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("results", exist_ok=True)
```

- [ ] **Step 3: Load base model and tokenizer, show a baseline sample (markdown + code cell)**

Markdown:
```markdown
## Base model

DistilGPT-2: 82M parameters, 6 transformer layers, hidden size 768. Before any fine-tuning,
here's what it generates for a Shakespeare-flavored prompt — plain modern-sounding text, as
expected from a model pretrained on general web text.
```

Code:
```python
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

PROMPT = "ROMEO:\nWhat light through yonder window breaks?"


def generate_sample(model, prompt: str = PROMPT, max_new_tokens: int = 40) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


base_model_for_sample = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
print(generate_sample(base_model_for_sample))
print(f"\nTotal parameters: {total_parameter_count(base_model_for_sample):,}")
del base_model_for_sample
```

- [ ] **Step 4: Load and tokenize tiny Shakespeare (markdown + code cell)**

Markdown:
```markdown
## Tiny Shakespeare

472 train / 49 test rows (~1.2M characters). Tokenize and concatenate, then chunk into fixed
128-token blocks for causal language modeling — no padding needed since every block is full.
```

Code:
```python
BLOCK_SIZE = 128
BATCH_SIZE = 8

raw = load_dataset("Trelis/tiny-shakespeare")


class BlockDataset(Dataset):
    def __init__(self, texts: list[str], block_size: int):
        full_text = "\n\n".join(texts)
        token_ids = tokenizer(full_text)["input_ids"]
        n_blocks = len(token_ids) // block_size
        self.examples = [token_ids[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.examples[idx], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


train_set = BlockDataset(raw["train"]["Text"], BLOCK_SIZE)
test_set = BlockDataset(raw["test"]["Text"], BLOCK_SIZE)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train blocks: {len(train_set)}, Test blocks: {len(test_set)}")
```

- [ ] **Step 5: Shared fine-tuning runner with persistence (markdown + code cell)**

Markdown:
```markdown
## Fine-tuning: full vs. LoRA

Same data, same number of epochs — the only difference is which parameters are trainable and
the learning rate (LoRA uses a higher LR since it's only updating a small adapter). Both runs
save per-step metrics, a checkpoint, and a generation sample after every epoch, so nothing here
requires a retrain to analyze later.
```

Code:
```python
EPOCHS = 3


def run_finetune(mode: str):
    assert mode in ("full", "lora")
    torch.manual_seed(0)
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)

    lora_modules = None
    if mode == "lora":
        lora_modules = setup_lora_model(model, r=8, alpha=16)
        lora_modules = [m.to(device) for m in lora_modules]
        lr = 3e-4
    else:
        lr = 5e-5

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    history = {"step": [], "loss": [], "time_seconds": [], "memory_mb": []}
    if mode == "lora":
        history["delta_norm"] = []
    eval_history = {"epoch": [], "test_loss": []}
    samples = []

    step = 0
    for epoch in range(EPOCHS):
        for batch in train_loader:
            stats = train_step(model, batch, optimizer, device)
            history["step"].append(step)
            history["loss"].append(stats["loss"])
            history["time_seconds"].append(stats["time_seconds"])
            history["memory_mb"].append(current_memory_mb())
            if mode == "lora":
                history["delta_norm"].append(float(np.mean([m.delta_norm() for m in lora_modules])))
            step += 1

        eval_stats = evaluate(model, test_loader, device)
        eval_history["epoch"].append(epoch + 1)
        eval_history["test_loss"].append(eval_stats["loss"])
        sample = generate_sample(model)
        samples.append({"epoch": epoch + 1, "text": sample})
        print(f"[{mode}] epoch {epoch + 1}/{EPOCHS} test_loss={eval_stats['loss']:.3f}")
        print(f"  sample: {sample!r}")

    with open(f"results/{mode}_history.json", "w") as f:
        json.dump(history, f)
    with open(f"results/{mode}_eval_history.json", "w") as f:
        json.dump(eval_history, f)
    with open(f"results/{mode}_samples.json", "w") as f:
        json.dump(samples, f)

    if mode == "lora":
        save_lora_adapters(lora_modules, "results/lora_adapter.pt")
    else:
        save_checkpoint(model, "results/full_finetune.pt")

    return model, history, eval_history, samples


full_model, full_history, full_eval_history, full_samples = run_finetune("full")
```

- [ ] **Step 6: Run the LoRA fine-tune (code cell)**

```python
lora_model, lora_history, lora_eval_history, lora_samples = run_finetune("lora")
```

- [ ] **Step 7: Trainable parameter count and checkpoint size comparison (markdown + code cell)**

Markdown:
```markdown
## The headline numbers: parameters and storage
```

Code:
```python
full_trainable = total_parameter_count(full_model)
lora_trainable = trainable_parameter_count(lora_model)
total_params = total_parameter_count(lora_model)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(["full fine-tune", "LoRA"], [full_trainable, lora_trainable])
axes[0].set_ylabel("Trainable parameters")
axes[0].set_title("Trainable parameter count")
axes[0].set_yscale("log")

full_ckpt_mb = os.path.getsize("results/full_finetune.pt") / (1024 ** 2)
lora_ckpt_mb = os.path.getsize("results/lora_adapter.pt") / (1024 ** 2)
axes[1].bar(["full fine-tune", "LoRA adapter"], [full_ckpt_mb, lora_ckpt_mb])
axes[1].set_ylabel("Checkpoint size (MB)")
axes[1].set_title("Checkpoint size on disk")
axes[1].set_yscale("log")
plt.tight_layout()
plt.show()

print(f"Total model parameters: {total_params:,}")
print(f"Full fine-tune trainable: {full_trainable:,} ({full_trainable/total_params*100:.1f}%)")
print(f"LoRA trainable: {lora_trainable:,} ({lora_trainable/total_params*100:.3f}%)")
print(f"Reduction: {full_trainable/lora_trainable:.0f}x fewer trainable parameters")
print(f"Checkpoint size: full={full_ckpt_mb:.1f}MB, LoRA adapter={lora_ckpt_mb:.2f}MB")
```

- [ ] **Step 8: Loss curves, timing, and memory (markdown + code cells)**

Markdown:
```markdown
## Training loss, speed, and memory
```

Code:
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(full_history["step"], full_history["loss"], label="full fine-tune", alpha=0.7)
axes[0].plot(lora_history["step"], lora_history["loss"], label="LoRA", alpha=0.7)
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Training loss")
axes[0].set_title("Training loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(full_eval_history["epoch"], full_eval_history["test_loss"], marker="o", label="full fine-tune")
axes[1].plot(lora_eval_history["epoch"], lora_eval_history["test_loss"], marker="o", label="LoRA")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Test loss")
axes[1].set_title("Test loss per epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].bar(
    ["full fine-tune", "LoRA"],
    [np.mean(full_history["time_seconds"]), np.mean(lora_history["time_seconds"])],
)
axes[2].set_ylabel("Mean time per step (s)")
axes[2].set_title("Training speed")
plt.tight_layout()
plt.show()

print(f"Mean memory (MB) - full: {np.mean(full_history['memory_mb']):.1f}, LoRA: {np.mean(lora_history['memory_mb']):.1f}")
```

- [ ] **Step 9: LoRA adapter growth over training (markdown + code cell)**

Markdown:
```markdown
## How far does the adapter move from zero?

At initialization, `delta_W = A @ B = 0` exactly (`B` starts at zero) — the LoRA-adapted model
starts out identical to the frozen base. Training moves it away from that no-op.
```

Code:
```python
plt.figure(figsize=(8, 4))
plt.plot(lora_history["step"], lora_history["delta_norm"])
plt.xlabel("Step")
plt.ylabel("Mean ||delta_W|| across layers (Frobenius norm)")
plt.title("LoRA adapter growth from zero-init")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"delta_norm at step 0: {lora_history['delta_norm'][0]:.4f}")
print(f"delta_norm at final step: {lora_history['delta_norm'][-1]:.4f}")
```

- [ ] **Step 10: Qualitative generation samples (markdown + code cell)**

Markdown:
```markdown
## What the models actually generate

Same prompt (`"ROMEO:\nWhat light through yonder window breaks?"`), continuation from each
model at the end of every epoch.
```

Code:
```python
print("=== Full fine-tune ===")
for s in full_samples:
    print(f"[epoch {s['epoch']}] {s['text']!r}")

print("\n=== LoRA ===")
for s in lora_samples:
    print(f"[epoch {s['epoch']}] {s['text']!r}")
```

- [ ] **Step 11: Execute the notebook end-to-end and verify**

Run: `cd lora && jupyter nbconvert --to notebook --execute --inplace lora.ipynb --ExecutePreprocessor.timeout=1800`
Expected: command exits with no error; re-opening the notebook shows all cells with outputs,
including both bar-chart comparisons, the loss/timing/memory plots, the delta_norm growth plot,
and printed generation samples for both models across all 3 epochs.

Then confirm the conditions from `design.md`'s Verification section by reading the executed
notebook's outputs:
- LoRA's trainable parameter count is at least 100x smaller than full fine-tuning's.
- Both runs' training loss decreases over training.
- LoRA's final test loss is within a reasonable margin of full fine-tuning's.
- Generated text visibly shifts toward Shakespearean style/vocabulary by the later epochs, for
  both models.

If any condition fails, investigate before adjusting (e.g. LoRA rank, learning rate, or epoch
count) rather than skipping the check — note the actual numbers first.

- [ ] **Step 12: Commit**

```bash
git add lora/lora.ipynb lora/results/
git commit -m "Add LoRA paper walkthrough notebook with full fine-tune vs LoRA comparison"
```

---

## Self-Review Notes

- **Spec coverage:** `LoRAConv1D` mechanism with q/v-only slicing and zero-init (Task 1), injection/freezing across all 6 layers (Task 2), training loop with memory tracking and checkpoint/history persistence (Task 3), the `.gitignore` exception enabling LoRA adapter checkpoints to be committed (Task 4), and all six comparison visualizations from `design.md` — trainable parameter count, checkpoint size, loss curves, timing/memory, delta_norm growth, and qualitative generation samples — are all covered in Task 5.
- **Placeholder scan:** none found; every step has runnable code or an exact command.
- **Type consistency:** `train_step`/`evaluate` return keys (`"loss"`, `"time_seconds"` / `"loss"`) match between Task 3's implementation, its tests, and the notebook's `run_finetune` in Task 5. `LoRAConv1D`'s constructor args (`base`, `r`, `alpha`) and its `.delta_norm()` method match across Tasks 1, 2's injection helper, and Task 5's notebook usage. `save_lora_adapters`' key naming (`layer_{i}.lora_A_q` etc.) matches between Task 3's implementation and its round-trip test.
