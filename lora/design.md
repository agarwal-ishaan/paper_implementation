# Design: LoRA — Low-Rank Adaptation of Large Language Models

Paper: Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen — "LoRA: Low-Rank Adaptation of Large
Language Models" (ICLR 2022).
PDF: `lora/LoRA - Low-Rank Adaptation of Large Language Models.pdf`

## Context

Second paper implemented from the `paper_implementation` collection, chosen for its direct
relevance to current LLM fine-tuning practice and because the core mechanism is small and clean
to build and test in isolation. On the first paper (stochastic depth), no model checkpoints or
per-step training telemetry were saved during the real training run — only final printed
summaries — so a later request for additional post-hoc analysis (loss landscape, richer
visualizations) would have required a full retrain to get real weights back. This time,
checkpointing and structured per-step metric logging are part of the initial design, not an
afterthought.

## Core idea

A pretrained weight matrix `W` (frozen) gets an additive low-rank correction learned during
fine-tuning: `ΔW = (α/r) · A · B`, where `A` is `in_features × r`, `B` is `r × out_features`, and
`r` (e.g. 4-8) is far smaller than `in_features`/`out_features`. Forward pass:

```
y = x·W + (α/r) · x·A·B
```

Only `A` and `B` are trainable; `W` never gets a gradient. This cuts trainable parameters by
orders of magnitude relative to full fine-tuning while — the paper's central claim — matching
full fine-tuning quality on many tasks, because the *update* needed to adapt a pretrained model to
a new task tends to have low "intrinsic rank" even though the full weight matrix doesn't.

Following the paper's own emphasis (and its ablation showing Wq+Wv is enough, more effective than
spreading the same parameter budget across all four attention matrices), LoRA is applied only to
the attention **query and value** projections, leaving key projections and all MLP/embedding
weights frozen and untouched.

## Model & task setup

- Base model: **DistilGPT-2** (82M params, 6 layers, hidden size 768, pretrained, loaded via
  `transformers`). GPT-2's attention implementation fuses q/k/v into a single `Conv1D` layer
  (`c_attn`, weight shape `(768, 2304)`, i.e. `(in_features, out_features)` — the *opposite*
  convention from `nn.Linear`), so the LoRA wrapper targets `Conv1D` specifically and applies the
  low-rank update only to the query third (columns 0:768) and value third (columns 1536:2304) of
  the output, leaving the key third (768:1536) untouched.
- Task: causal language modeling (next-token prediction) fine-tuning on **tiny Shakespeare**
  (`Trelis/tiny-shakespeare` on the HF Hub — 472 train / 49 test rows, ~1.2M characters), a
  corpus stylistically distinct enough from DistilGPT-2's pretraining data that successful
  adaptation should show up both quantitatively (validation loss/perplexity drop) and
  qualitatively (generated text shifts toward Shakespearean style).
- Two fine-tuning runs, same data/steps/optimizer settings, differing only in what's trainable:
  1. **Full fine-tuning** — every parameter trainable.
  2. **LoRA fine-tuning** — base model fully frozen; only the injected `A`/`B` matrices (rank 8)
     on q/v projections across all 6 layers are trainable.

## Training artifacts to persist (this run's key process change)

For both runs, save under `lora/results/`:

- **Checkpoints**: full fine-tuned model's changed weights (state dict), and separately the LoRA
  adapter weights alone (`A`/`B` per layer — much smaller, this is LoRA's other headline benefit)
  as `.pt` files.
- **Per-step metrics** (JSON, not just printed lines): training loss every N steps, wall-clock
  time per step/epoch, current allocated memory (via `torch.mps.current_allocated_memory()` on this machine, or
  `psutil` RSS as a fallback), and for the LoRA run specifically, the Frobenius norm of `ΔW = A·B`
  logged periodically (shows how much the adaptation has grown from its zero-initialized start).
- **Generation samples**: a fixed prompt's continuation from the base model, the fully fine-tuned
  model, and the LoRA model, captured at a few points during training (not just at the end), so
  the qualitative style shift over training is visible, not just the final state.

## Comparison / visualizations

- **Trainable parameter count**: full fine-tune (~82M) vs. LoRA (expected: low hundreds of
  thousands, roughly q+v projections × rank × 2 × 6 layers) — the headline number, shown as a bar
  chart or simple ratio.
- **Adapter checkpoint size on disk**: full model state dict vs. LoRA adapter-only state dict.
- **Training loss/perplexity curves**: full fine-tune vs. LoRA, overlaid.
- **Training time and memory**: full fine-tune vs. LoRA, per-step or aggregated.
- **ΔW growth over training**: LoRA-only plot, Frobenius norm of the low-rank update vs. training
  step — shows the adapter moving away from its zero-init (at init, `B` is zero so `ΔW` starts at
  exactly zero, a property specific to LoRA's initialization scheme worth calling out explicitly).
- **Qualitative generation samples**: same prompt, base vs. full fine-tune vs. LoRA, at a few
  points during training.

## Deliverable

- `lora/model.py` — `LoRAConv1D` (wraps a GPT-2 `Conv1D`, freezes its weight, adds trainable
  low-rank `A`/`B` restricted to the q/v output slices) and an injection helper that walks a
  loaded `GPT2LMHeadModel` and replaces each layer's `c_attn` with the wrapped version.
- `lora/train.py` — shared fine-tuning loop (`train_step`/`evaluate`) parameterized so it works
  for both the full-fine-tune and LoRA runs, plus the per-step metric/checkpoint persistence
  described above.
- `lora/test_model.py`, `lora/test_train.py` — pytest tests written before the real run: LoRA
  forward pass shape and zero-init-equals-base-output correctness, that base weights receive no
  gradient, that only q/v (not k) columns are affected, trainable parameter counts for both
  configurations, and a small-scale training-loop smoke test.
- `lora/lora.ipynb` — the paper walkthrough: explanation, from-scratch implementation, both
  fine-tuning runs, and all visualizations above.
- Repo-level `.gitignore` already excludes `*.pt`/`*.pth`; the LoRA adapter checkpoints are small
  enough to be worth committing anyway as a demonstration of LoRA's storage-efficiency claim, so
  `lora/results/*.pt` will be explicitly un-ignored for the (small) adapter files while the (large)
  full fine-tuned checkpoint stays gitignored.

## Verification

Run the notebook top to bottom (fresh kernel) and confirm:
- No errors on MPS/CPU backend.
- LoRA's trainable parameter count is at least 100x smaller than full fine-tuning's.
- Both runs' training loss decreases over training.
- LoRA's final validation loss is within a reasonable margin of full fine-tuning's (not
  necessarily better — the paper's claim is "comparable," not "always better").
- Generation samples visibly shift toward Shakespearean style/vocabulary after fine-tuning, for
  both the full and LoRA runs.
