# Design: Deep Networks with Stochastic Depth

Paper: Huang, Sun, Liu, Sedra, Weinberger — "Deep Networks with Stochastic Depth" (ECCV 2016).
PDF: `stochastic_depth/Deep Networks with Stochastic Depth.pdf`

## Context

First paper implemented from the `paper_implementation` collection. Goal is twofold: build an
intuition for the paper's mechanism (not just read it), and produce a runnable, visual demo that
makes the effect legible without requiring the paper's original compute budget (ResNet-110+,
hundreds of epochs, GPU-hours). Picked as the starting paper because the core idea is a small,
self-contained modification to a residual block, it trains fast, and it has an obvious visual
signature (which blocks get skipped, and how that changes training speed/accuracy).

## Core idea

Each residual block computes `H_l = ReLU(id(x) + f_l(x))`. Stochastic depth adds a per-block
Bernoulli gate `b_l`:

- Training: `H_l = ReLU(id(x) + b_l * f_l(x))`, `b_l ~ Bernoulli(p_l)`. When `b_l = 0`, `f_l` is
  not computed at all (identity passthrough) — a real compute saving, not just a masked-out term.
- Inference: `H_l = ReLU(id(x) + p_l * f_l(x))` — all blocks run, residual branch rescaled by its
  survival probability (matches the expected value seen during training).

Survival probability decays linearly with depth: `p_l = 1 - (l / L) * (1 - p_L)`, with `p_0 = 1`
(first block always kept) and `p_L = 0.5` (paper's default for the last block). Early layers are
almost always active; deep layers are dropped often. This is Dropout at block granularity instead
of neuron granularity, and the paper shows it gives faster training, an implicit ensemble/
regularization effect, and better gradient flow in very deep nets.

## Model & training setup

- Custom CIFAR-style ResNet: 3 stages (16 → 32 → 64 channels), 6 `BasicBlock`s per stage = 18
  residual blocks (~38 weight layers total), matching the family of smaller configs the paper
  itself evaluates.
- Each block wrapped in a `StochasticDepthBlock`:
  - train mode: sample gate per block per forward pass; skip the conv branch entirely when gated
    off.
  - eval mode: always run the branch, rescale output by `p_l`.
- Linear decay schedule as above, `p_L = 0.5`.
- Two models trained under identical settings for direct comparison: plain baseline ResNet vs.
  the stochastic-depth version.
- CIFAR-10, standard augmentation (random crop + horizontal flip, normalization).
- SGD + momentum + weight decay, step LR schedule, batch size 128, ~30–40 epochs.
- Device: MPS backend (confirmed available: torch 2.7.1, torchvision 0.22.1 already installed).

## Visualizations

All inline in the notebook via matplotlib:

1. Survival-probability schedule: `p_l` vs. block index.
2. Train/test accuracy and loss curves, baseline vs. stochastic depth, overlaid.
3. Per-epoch wall-clock training time, baseline vs. stochastic depth (bar chart).
4. Block-activity heatmap: rows = blocks, columns = training iterations (sampled), color =
   active/dropped — makes the random-dropping mechanism directly visible.
5. Final summary table: test accuracy and total training time, both models.

## Deliverable

- `stochastic_depth/stochastic_depth.ipynb` — single notebook: markdown explanation of the paper
  (motivation, math, decay rule) interleaved with the from-scratch implementation (not
  `torchvision.ops.StochasticDepth` — building the mechanism by hand is the point) and the plots
  above.
- CIFAR-10 downloaded to `stochastic_depth/data/` (gitignored).
- Repo-level `.gitignore` added at `paper_implementation/` root (dataset downloads, checkpoints,
  notebook checkpoints, `__pycache__`, `.DS_Store`).

## Verification

Run the notebook top to bottom (fresh kernel) and confirm:
- No errors on MPS backend.
- All five visualizations render.
- Stochastic-depth model shows a measurably faster per-epoch training time than baseline.
- Stochastic-depth model's test accuracy is comparable to or better than baseline within the
  epoch budget (full generalization benefit may be muted at this scale/epoch count vs. the
  paper's own results, since we're intentionally not reproducing their compute budget).
