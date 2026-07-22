# Design: DistilBERT — a distilled version of BERT

Paper: Sanh, Debut, Chaumond, Wolf (Hugging Face) — "DistilBERT, a distilled version of BERT:
smaller, faster, cheaper and lighter" (EMC² @ NeurIPS 2019).
PDF: `DistillBERT/DistillBERT.pdf`

## Context

Third paper implemented from the `paper_implementation` collection. The paper's actual headline
result — distilling *during general-purpose MLM pretraining* on Wikipedia + BookCorpus, 90 GPU-hours
on 8×V100 — is not reproducible locally, so this implementation scopes down to **task-specific
distillation**: an off-the-shelf fine-tuned BERT-base is the frozen teacher, and a half-depth
student is trained directly on IMDb sentiment classification using the paper's structural ideas
(halved depth, every-other-layer initialization) and its triple-loss framework, adapted from
"distillation + MLM + cosine" to "distillation + supervised task CE + cosine." This is closer to
prior task-specific distillation work the paper itself cites (Tang et al. 2019) than to DistilBERT's
own pretraining-time contribution — a deliberate scope trade-off for local compute, not a claim of
reproducing the paper's main result. Following the LoRA implementation's process change,
checkpointing and per-step metric logging are part of the design from the start, not added
after the fact.

## Core idea

Knowledge distillation trains a small **student** to reproduce a large **teacher**'s output
distribution, not just its hard predictions. The student sees a training signal richer than one-hot
labels: the teacher's "near-zero" probabilities on wrong classes still encode information about
which mistakes are more or less plausible.

**Distillation loss** — soft-target cross-entropy with a softmax temperature `T` applied to both
models' logits during training (`T=1` at inference, recovering a normal softmax):

```
p_i = exp(z_i / T) / sum_j exp(z_j / T)
L_ce = sum_i t_i * log(s_i)      # t = teacher soft targets, s = student soft predictions
```

**Triple loss** (paper's Section 2-3, adapted here from MLM to classification): a weighted sum of
three terms, using the paper's own released weighting (`alpha_ce=5.0, alpha_task=2.0, alpha_cos=1.0,
T=2.0`, substituting the paper's `alpha_mlm` role with the supervised task loss):

- `L_ce` — distillation loss above (soft targets, temperature `T`).
- `L_task` — standard cross-entropy against gold IMDb labels (stands in for the paper's `L_mlm` —
  in this scoped-down setting there's no MLM pretraining, so the "ground truth" signal is the
  classification label instead).
- `L_cos` — cosine-embedding loss pushing the student's final-layer `[CLS]` hidden state toward the
  teacher's final-layer `[CLS]` hidden state (`CosineEmbeddingLoss` with target `+1`).

**Student architecture**: same general architecture as BERT-base but with the number of transformer
layers halved (6 vs. 12) and token-type embeddings removed (paper's Section 3 — IMDb is
single-sequence classification so token-type ids are always 0 anyway, but removing the embedding
table is part of the paper's actual architectural claim, not just a parameter-count trick).

**Student initialization**: word + position embeddings copied directly from the teacher; student
layer `i` initialized by copying teacher layer `2*i` (0-indexed: student layers 0-5 ← teacher layers
0, 2, 4, 6, 8, 10) — taking advantage of the shared hidden size (768) between teacher and student,
per the paper's stated initialization trick. A fresh classification head is randomly initialized
(the teacher's task head isn't reusable at a different depth).

## Model & task setup

- **Teacher**: `textattack/bert-base-uncased-imdb` (public HF Hub checkpoint, BERT-base fine-tuned
  on IMDb, ~89% test accuracy per its model card) — used frozen, as-is. No teacher fine-tuning is
  performed locally; the paper treats "a fine-tuned BERT" as a given input, not its contribution.
- **Student**: 6-layer BERT, hidden size 768 (unchanged — the paper's own ablation notes hidden-size
  reduction has less favorable efficiency trade-offs than depth reduction), teacher-initialized as
  above, ~60% of the teacher's parameter count (matching the paper's headline "40% smaller" — a
  40% reduction means 60% remains; the paper's own BERT-base/DistilBERT split is 110M/66M, also ~60%).
- **Task**: IMDb sentiment classification (`stanfordnlp/imdb` on HF Hub, 25k train / 25k test),
  reviews truncated to 256 tokens, matching the paper's own downstream benchmark (Table 2).
- **Device**: `torch.device("mps")` if available, else CPU.
- **Three student variants**, same architecture and initialization, differing only in loss —
  mirrors the paper's own ablation (Table 4: removing each loss term hurts):
  1. **Baseline** — `L_task` only (no distillation signal at all).
  2. **Distillation** — `L_ce + L_task` (soft targets + gold labels, no cosine term).
  3. **Triple loss** — `L_ce + L_task + L_cos` (paper's full recipe, adapted).

## Training artifacts to persist

For the teacher (eval only, no training) and each of the 3 student variants, save under
`DistillBERT/results/`:

- **Checkpoints**: each student variant's state dict as a `.pt` file (kept out of git — at ~60%
  of BERT-base's size these are not the small, commit-worthy artifacts LoRA's adapters were).
- **Per-step metrics** (JSON): total loss and each individual loss component (`L_task`, `L_ce`,
  `L_cos` where applicable) every N steps, wall-clock time per step/epoch, peak memory
  (`torch.mps.current_allocated_memory()` or `psutil` RSS fallback).
- **Qualitative prediction samples**: a fixed set of IMDb test reviews, with teacher probability and
  each student variant's probability captured at a few points during training — shows the student's
  predicted distribution converging toward the teacher's soft output, not just matching hard labels.

## Comparison / visualizations

- **Parameter count**: teacher (~110M) vs. student (~40% smaller, ~60% remaining) — bar chart, the headline claim.
- **Test accuracy / F1**: teacher vs. all 3 student variants on the IMDb test set — bar chart,
  directly mirrors the paper's ablation story (each loss term should help, in order).
- **Inference time**: batch inference wall-clock, teacher vs. student, on the same hardware —
  mirrors the paper's Table 3 speed claim.
- **Training loss curves**: all loss components over training steps, overlaid per variant.
- **Teacher/student output agreement**: KL divergence (or simple agreement rate) between each
  variant's predicted probability and the teacher's, on the test set — isolates how much closer
  distillation gets the student's actual output *distribution* to the teacher's, beyond accuracy.
- **Qualitative examples**: a few fixed reviews with teacher vs. each variant's predicted
  probability, at a few points during training.

## Deliverable

- `DistillBERT/model.py` — `DistilledBertForSequenceClassification` (6-layer BERT + classification
  head, no token-type embeddings), a teacher-initialization helper that copies embeddings and
  every-other layer from a loaded `BertForSequenceClassification` teacher, and the triple-loss
  function (`distillation_loss`, `cosine_loss`, combined with configurable weights so all 3 variants
  share one implementation).
- `DistillBERT/train.py` — shared fine-tuning loop (`train_step`/`evaluate`) parameterized by which
  loss terms are active (baseline / distillation / triple), plus the per-step metric/checkpoint/
  qualitative-sample persistence described above.
- `DistillBERT/test_model.py`, `DistillBERT/test_train.py` — pytest, written before the real run:
  student output shapes, teacher-init correctness (student layer `i` weights equal teacher layer
  `2*i` weights exactly after init), parameter count for teacher vs. student, distillation-loss and
  cosine-loss correctness on toy tensors (known-value checks, temperature scaling behavior), and a
  small-scale training-loop smoke test for each of the 3 loss configs.
- `DistillBERT/distillbert.ipynb` — the paper walkthrough: explanation, from-scratch implementation,
  teacher loading + eval, all 3 student variants trained + evaluated, and all visualizations above.
- Repo-level `.gitignore` already excludes `*.pt`/`*.pth`; student checkpoints stay gitignored
  (unlike LoRA's adapters, these aren't small enough to be worth committing). Per-step metrics JSON
  and qualitative sample logs are small and get committed.

## Verification

Run the notebook top to bottom (fresh kernel) and confirm:
- No errors on MPS/CPU backend.
- Student parameter count is roughly 60% of the teacher's (~40% smaller).
- Teacher-init check passes: immediately after initialization (before any training), student layer
  `i`'s weights exactly match teacher layer `2*i`'s weights.
- All 3 student variants' training loss decreases over training.
- Test accuracy improves across variants in the expected direction: baseline < distillation <
  triple loss (not guaranteed to be strictly monotonic on a single run/seed, but the paper's own
  ablation supports this ordering — call out clearly if a run doesn't follow it rather than
  overstating the result).
- Distilled variants' (2 and 3) predicted probabilities are measurably closer to the teacher's
  (lower KL divergence / higher agreement) than the baseline's.
