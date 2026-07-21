# Stochastic Depth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a from-scratch, tested implementation of "Deep Networks with Stochastic Depth" (Huang et al. 2016), delivered as a single notebook that trains a baseline CIFAR-10 ResNet against a stochastic-depth version and visualizes the effect.

**Architecture:** Core mechanism (survival-probability schedule, the gated residual block, and the CIFAR ResNet that assembles it) lives in a plain, unit-tested `model.py` module — no notebook-only logic for anything that can silently break. The training/eval loop lives in a separately tested `train.py`. The notebook (`stochastic_depth.ipynb`) imports both, adds the paper explanation, runs real CIFAR-10 training for both models, and produces the visualizations. This split exists so the gating logic (the part most likely to have a subtle bug — e.g. accidentally still computing the branch when gated off) is verified by fast, deterministic tests before it's ever used in a slow real-data training run.

**Tech Stack:** Python 3.13, PyTorch 2.7.1, torchvision 0.22.1, pytest 9.0.2, matplotlib 3.10.5, Jupyter (all already installed on this machine; MPS backend confirmed available). No new dependencies needed.

## Global Constraints

- Do not use `torchvision.ops.StochasticDepth` or any other pre-built stochastic-depth implementation — the gating mechanism must be hand-written (from `stochastic_depth/design.md`).
- Model: 3 stages (16→32→64 channels), 6 `BasicBlock`s per stage = 18 residual blocks total.
- Survival probability schedule: linear decay, `p_l = 1 - (l / L) * (1 - p_L)` for `l = 1..L`, `p_L = 0.5`.
- Training mode: `H = ReLU(identity(x) + gate * branch(x))`, `gate ~ Bernoulli(p_l)`, branch **not computed** when gate is 0.
- Eval mode: `H = ReLU(identity(x) + p_l * branch(x))`, deterministic, branch always computed.
- Dataset: CIFAR-10 via torchvision, downloaded to `stochastic_depth/data/` (gitignored).
- Device: use MPS if available, else CPU (`torch.device("mps" if torch.backends.mps.is_available() else "cpu")`).
- All new files live under `stochastic_depth/` (flat, no subpackage) so `python -m pytest test_model.py test_train.py` run from inside that directory can import `model` and `train` directly.
- Commit after each task.

---

### Task 1: Survival probability schedule

**Files:**
- Create: `stochastic_depth/model.py`
- Test: `stochastic_depth/test_model.py`

**Interfaces:**
- Produces: `survival_probabilities(num_blocks: int, p_L: float = 0.5) -> list[float]`, used by Task 3's `ResNetCIFAR`.

- [ ] **Step 1: Write the failing tests**

Create `stochastic_depth/test_model.py`:

```python
import pytest

from model import survival_probabilities


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd stochastic_depth && python -m pytest test_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'survival_probabilities' from 'model'` (module doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `stochastic_depth/model.py`:

```python
def survival_probabilities(num_blocks: int, p_L: float = 0.5) -> list[float]:
    """Linear decay rule from the paper: p_l = 1 - (l / L) * (1 - p_L), l = 1..L."""
    return [1 - (l / num_blocks) * (1 - p_L) for l in range(1, num_blocks + 1)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd stochastic_depth && python -m pytest test_model.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add stochastic_depth/model.py stochastic_depth/test_model.py
git commit -m "Add stochastic depth survival probability schedule"
```

---

### Task 2: Gated residual block

**Files:**
- Modify: `stochastic_depth/model.py`
- Modify: `stochastic_depth/test_model.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (it takes a raw `p: float`).
- Produces: `ConvBranch(in_channels: int, out_channels: int, stride: int = 1)` (an `nn.Module`, the residual branch `f_l`), and `StochasticDepthBlock(branch: nn.Module, shortcut: nn.Module, p: float)` (an `nn.Module`). Both used by Task 3's `ResNetCIFAR`.

- [ ] **Step 1: Write the failing tests**

Append to `stochastic_depth/test_model.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

import model as model_module
from model import ConvBranch, StochasticDepthBlock


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd stochastic_depth && python -m pytest test_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'ConvBranch' from 'model'`.

- [ ] **Step 3: Write minimal implementation**

Append to `stochastic_depth/model.py`:

```python
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


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
```

Put the `import random`, `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F` lines at the top of `model.py`, above the existing `survival_probabilities` function.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd stochastic_depth && python -m pytest test_model.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add stochastic_depth/model.py stochastic_depth/test_model.py
git commit -m "Add gated residual block for stochastic depth"
```

---

### Task 3: Full CIFAR ResNet assembly

**Files:**
- Modify: `stochastic_depth/model.py`
- Modify: `stochastic_depth/test_model.py`

**Interfaces:**
- Consumes: `survival_probabilities` (Task 1), `ConvBranch` and `StochasticDepthBlock` (Task 2).
- Produces: `ResNetCIFAR(blocks_per_stage: int = 6, p_L: float = 0.5, stochastic_depth: bool = True, num_classes: int = 10)`, an `nn.Module` with `.forward(x) -> logits`, plus public attributes `.num_blocks: int` and `.survival_probs: list[float]`. Used by Task 4's train/eval functions and by the notebook (Task 5).

- [ ] **Step 1: Write the failing tests**

Append to `stochastic_depth/test_model.py`:

```python
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
    from model import survival_probabilities
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd stochastic_depth && python -m pytest test_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'ResNetCIFAR' from 'model'`.

- [ ] **Step 3: Write minimal implementation**

Append to `stochastic_depth/model.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd stochastic_depth && python -m pytest test_model.py -v`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add stochastic_depth/model.py stochastic_depth/test_model.py
git commit -m "Assemble full CIFAR ResNet with stochastic depth"
```

---

### Task 4: Train/eval loop

**Files:**
- Create: `stochastic_depth/train.py`
- Create: `stochastic_depth/test_train.py`

**Interfaces:**
- Consumes: `ResNetCIFAR` (Task 3).
- Produces: `train_one_epoch(model, loader, optimizer, device) -> dict` (keys: `"loss"`, `"accuracy"`, `"time_seconds"`) and `evaluate(model, loader, device) -> dict` (keys: `"loss"`, `"accuracy"`). Used by the notebook (Task 5).

- [ ] **Step 1: Write the failing tests**

Create `stochastic_depth/test_train.py`:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import ResNetCIFAR
from train import evaluate, train_one_epoch


def _make_loader(num_samples=16, num_classes=10, batch_size=4):
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_train_one_epoch_returns_expected_keys():
    net = ResNetCIFAR(blocks_per_stage=1, stochastic_depth=True)
    loader = _make_loader()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    result = train_one_epoch(net, loader, optimizer, device=torch.device("cpu"))
    assert set(result.keys()) == {"loss", "accuracy", "time_seconds"}
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["time_seconds"] > 0


def test_evaluate_returns_expected_keys():
    net = ResNetCIFAR(blocks_per_stage=1, stochastic_depth=True)
    loader = _make_loader()
    result = evaluate(net, loader, device=torch.device("cpu"))
    assert set(result.keys()) == {"loss", "accuracy"}
    assert 0.0 <= result["accuracy"] <= 1.0


def test_evaluate_does_not_change_model_parameters():
    net = ResNetCIFAR(blocks_per_stage=1, stochastic_depth=True)
    loader = _make_loader()
    before = [p.clone() for p in net.parameters()]
    evaluate(net, loader, device=torch.device("cpu"))
    after = list(net.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_train_one_epoch_reduces_loss_over_several_epochs():
    net = ResNetCIFAR(blocks_per_stage=1, stochastic_depth=False)
    loader = _make_loader(num_samples=16, batch_size=4)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    first = train_one_epoch(net, loader, optimizer, device=torch.device("cpu"))
    last = first
    for _ in range(5):
        last = train_one_epoch(net, loader, optimizer, device=torch.device("cpu"))
    assert last["loss"] < first["loss"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd stochastic_depth && python -m pytest test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train'`.

- [ ] **Step 3: Write minimal implementation**

Create `stochastic_depth/train.py`:

```python
import time

import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, device) -> dict:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.perf_counter()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    elapsed = time.perf_counter() - start
    return {"loss": total_loss / total, "accuracy": correct / total, "time_seconds": elapsed}


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return {"loss": total_loss / total, "accuracy": correct / total}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd stochastic_depth && python -m pytest test_train.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the full test suite together**

Run: `cd stochastic_depth && python -m pytest -v`
Expected: 21 passed (17 from `test_model.py` + 4 from `test_train.py`).

- [ ] **Step 6: Commit**

```bash
git add stochastic_depth/train.py stochastic_depth/test_train.py
git commit -m "Add train/eval loop for stochastic depth models"
```

---

### Task 5: Notebook — paper walkthrough, training run, and visualizations

**Files:**
- Create: `stochastic_depth/stochastic_depth.ipynb`

**Interfaces:**
- Consumes: `survival_probabilities`, `ResNetCIFAR` (from `model.py`); `train_one_epoch`, `evaluate` (from `train.py`).

This is the main deliverable and is not itself unit-tested — it's a Jupyter notebook, verified by executing it end-to-end (Step 8 below) rather than pytest. Build it as the following cells, in order. Use the `NotebookEdit` tool to create cells in `stochastic_depth/stochastic_depth.ipynb` one at a time in this sequence.

- [ ] **Step 1: Title and paper explanation (markdown cell)**

```markdown
# Deep Networks with Stochastic Depth

Huang, Sun, Liu, Sedra, Weinberger — ECCV 2016. [Paper PDF](Deep%20Networks%20with%20Stochastic%20Depth.pdf)

**The problem:** very deep ResNets are slow to train and can suffer from vanishing gradients
and diminishing feature reuse, even with residual connections.

**The idea:** during training, randomly drop entire residual blocks — replace them with a pure
identity passthrough — instead of always computing them. Each block $l$ gets a Bernoulli "survival"
gate $b_l \sim \text{Bernoulli}(p_l)$:

$$H_l = \text{ReLU}\big(\text{id}(x) + b_l \cdot f_l(x)\big) \quad \text{(training)}$$

When $b_l = 0$, $f_l$ (the conv branch) isn't computed at all — a real compute saving, not just a
masked-out term. At test time, every block runs, but its branch is rescaled by its survival
probability so the expected output matches training:

$$H_l = \text{ReLU}\big(\text{id}(x) + p_l \cdot f_l(x)\big) \quad \text{(eval)}$$

**The schedule:** survival probability decays linearly with depth — early blocks are almost
always kept, deep blocks are dropped often:

$$p_l = 1 - \frac{l}{L}(1 - p_L), \quad l = 1, \dots, L$$

with $p_L = 0.5$ in the paper's default. This gives three effects: faster training (skipped
compute), an implicit ensemble/regularization effect, and better gradient flow in very deep nets.

This notebook implements the mechanism from scratch (see `model.py`, `train.py`) and trains a
small 18-block CIFAR ResNet two ways — a plain baseline and a stochastic-depth version — to make
the effect visible without the paper's original compute budget (ResNet-110+, hundreds of epochs).
```

- [ ] **Step 2: Imports and device setup (code cell)**

```python
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import ResNetCIFAR, survival_probabilities
from train import evaluate, train_one_epoch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
```

- [ ] **Step 3: Survival probability schedule plot (markdown + code cell)**

Markdown:
```markdown
## The survival probability schedule

Before training anything, let's look at the schedule itself: 18 blocks (3 stages × 6 blocks),
linear decay from near-1 down to 0.5.
```

Code:
```python
probs = survival_probabilities(num_blocks=18, p_L=0.5)

plt.figure(figsize=(7, 4))
plt.plot(range(1, 19), probs, marker="o")
plt.xlabel("Block index (l)")
plt.ylabel("Survival probability p_l")
plt.title("Linear decay survival probability schedule")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.show()
```

- [ ] **Step 4: Data loading (markdown + code cell)**

Markdown:
```markdown
## CIFAR-10 data

Standard augmentation: random crop with padding, horizontal flip, normalization. Downloaded to
`data/` (gitignored).
```

Code:
```python
normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
test_transform = transforms.Compose([transforms.ToTensor(), normalize])

train_set = torchvision.datasets.CIFAR10(
    root="data", train=True, download=True, transform=train_transform
)
test_set = torchvision.datasets.CIFAR10(
    root="data", train=False, download=True, transform=test_transform
)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

print(f"Train: {len(train_set)} images, Test: {len(test_set)} images")
```

- [ ] **Step 5: Training function for one full model (markdown + code cell)**

Markdown:
```markdown
## Training both models

Same architecture, same optimizer settings, same schedule — the only difference is whether
stochastic depth is on. SGD with momentum, step LR decay, 30 epochs.
```

Code:
```python
def run_training(stochastic_depth: bool, epochs: int = 30):
    torch.manual_seed(0)
    net = ResNetCIFAR(blocks_per_stage=6, p_L=0.5, stochastic_depth=stochastic_depth).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 23], gamma=0.1)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "epoch_time": []}
    for epoch in range(epochs):
        train_stats = train_one_epoch(net, train_loader, optimizer, device)
        test_stats = evaluate(net, test_loader, device)
        scheduler.step()
        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["accuracy"])
        history["test_loss"].append(test_stats["loss"])
        history["test_acc"].append(test_stats["accuracy"])
        history["epoch_time"].append(train_stats["time_seconds"])
        print(
            f"[{'SD' if stochastic_depth else 'baseline'}] epoch {epoch + 1}/{epochs} "
            f"train_acc={train_stats['accuracy']:.3f} test_acc={test_stats['accuracy']:.3f} "
            f"time={train_stats['time_seconds']:.1f}s"
        )
    return net, history


baseline_net, baseline_history = run_training(stochastic_depth=False)
```

- [ ] **Step 6: Train the stochastic depth model (code cell)**

```python
sd_net, sd_history = run_training(stochastic_depth=True)
```

- [ ] **Step 7: Visualizations — accuracy/loss curves, timing, block activity (markdown + code cells)**

Markdown:
```markdown
## Results

Four views of the same two training runs: accuracy curves, per-epoch training time, a summary
table, and the stochastic-depth block-activity pattern itself.
```

Code (accuracy/loss curves):
```python
epochs_range = range(1, len(baseline_history["test_acc"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(epochs_range, baseline_history["test_acc"], label="baseline")
axes[0].plot(epochs_range, sd_history["test_acc"], label="stochastic depth")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Test accuracy")
axes[0].set_title("Test accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, baseline_history["test_loss"], label="baseline")
axes[1].plot(epochs_range, sd_history["test_loss"], label="stochastic depth")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Test loss")
axes[1].set_title("Test loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

Code (per-epoch training time):
```python
plt.figure(figsize=(7, 4))
plt.bar(
    ["baseline", "stochastic depth"],
    [np.mean(baseline_history["epoch_time"]), np.mean(sd_history["epoch_time"])],
)
plt.ylabel("Mean per-epoch training time (s)")
plt.title("Training time: baseline vs. stochastic depth")
plt.show()
```

Code (summary table):
```python
print(f"{'Model':<20}{'Final test acc':<18}{'Total train time (s)':<22}")
print(
    f"{'baseline':<20}{baseline_history['test_acc'][-1]:<18.3f}"
    f"{sum(baseline_history['epoch_time']):<22.1f}"
)
print(
    f"{'stochastic depth':<20}{sd_history['test_acc'][-1]:<18.3f}"
    f"{sum(sd_history['epoch_time']):<22.1f}"
)
```

Code (block-activity heatmap — re-runs a few forward passes in train mode and records which
blocks fired):
```python
sd_net.train()
num_batches_to_record = 40
activity = np.zeros((sd_net.num_blocks, num_batches_to_record))

hooks = []
def make_hook(block_idx):
    def hook(module, input, output):
        activity[block_idx, hook.call_count] = 1
    hook.call_count = 0
    return hook

hook_fns = []
for i, block in enumerate(sd_net.blocks):
    fn = make_hook(i)
    hook_fns.append(fn)
    hooks.append(block.branch.register_forward_hook(fn))

data_iter = iter(train_loader)
for b in range(num_batches_to_record):
    images, _ = next(data_iter)
    images = images.to(device)
    with torch.no_grad():
        sd_net(images)
    for fn in hook_fns:
        fn.call_count += 1

for h in hooks:
    h.remove()

plt.figure(figsize=(10, 5))
plt.imshow(activity, aspect="auto", cmap="Greys", interpolation="nearest")
plt.xlabel("Training iteration (sample)")
plt.ylabel("Block index")
plt.title("Block activity: white = branch computed, black = block skipped")
plt.colorbar(label="active")
plt.show()
```

- [ ] **Step 8: Execute the notebook end-to-end and verify**

Run: `cd stochastic_depth && jupyter nbconvert --to notebook --execute --inplace stochastic_depth.ipynb`
Expected: command exits with no error; re-opening the notebook shows all cells with outputs,
including all four plots and the printed summary table.

Then confirm the two conditions from `design.md`'s Verification section by reading the executed
notebook's outputs:
- Stochastic depth's mean per-epoch training time is measurably lower than baseline's.
- Stochastic depth's final test accuracy is comparable to or better than baseline's.

If either condition fails, that's a signal to investigate (e.g. `p_L` too aggressive, too few
epochs) rather than a plan step to skip — note the actual numbers before deciding whether to
adjust `p_L` or `epochs` in Step 5/6 and re-run.

- [ ] **Step 9: Commit**

```bash
git add stochastic_depth/stochastic_depth.ipynb
git commit -m "Add stochastic depth paper walkthrough notebook with training and visualizations"
```

---

## Self-Review Notes

- **Spec coverage:** survival probability schedule (Task 1), gated block train/eval semantics
  (Task 2), full 18-block model + baseline/SD toggle (Task 3), train/eval loop (Task 4), all five
  visualizations from `design.md` — schedule plot (Task 5 Step 3), accuracy/loss curves and
  timing bar chart and summary table (Task 5 Step 7), block-activity heatmap (Task 5 Step 7) —
  and paper explanation (Task 5 Step 1) are all covered.
- **Placeholder scan:** none found; every step has runnable code or an exact command.
- **Type consistency:** `train_one_epoch`/`evaluate` keys (`"loss"`, `"accuracy"`, `"time_seconds"`)
  match between Task 4's implementation, its tests, and the notebook's `run_training` in Task 5.
  `ResNetCIFAR`'s constructor args (`blocks_per_stage`, `p_L`, `stochastic_depth`, `num_classes`)
  match across Tasks 3, 4's tests, and Task 5's notebook usage.
