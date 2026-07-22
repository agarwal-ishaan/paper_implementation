# DistilBERT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a from-scratch, tested implementation of task-specific knowledge distillation from
"DistilBERT" (Sanh et al. 2019): a frozen, pretrained-and-fine-tuned BERT-base teacher distills into
a half-depth, teacher-initialized student on IMDb sentiment classification, using an adapted triple
loss (distillation + supervised task + cosine), delivered as a single walkthrough notebook that
trains three loss-ablation variants and compares them to the teacher.

**Architecture:** The core mechanism (student architecture construction, teacher-initialization
weight copying, and the loss functions) lives in a plain, unit-tested `model.py` — no notebook-only
logic for anything that can silently break. The training loop (`train_step`, `evaluate`/`predict`,
and `train_loop` with metrics/checkpoint persistence) lives in a separately tested `train.py`. Both
are tested exclusively against tiny synthetic BERT configs (4-layer, hidden size 32) built in-memory
— no network access or real BERT-base download in the test suite, so tests stay fast and
deterministic. The notebook (`distillbert.ipynb`) is the only place that touches the real
`textattack/bert-base-uncased-imdb` teacher and the real IMDb dataset: it imports both modules,
adds the paper explanation, runs the three real training variants, and produces the visualizations.

**Tech Stack:** Python 3.13.2, PyTorch 2.7.1, transformers 4.54.1, datasets 4.8.5, scikit-learn
1.7.0, pytest 9.0.2, matplotlib 3.10.5, Jupyter (all already installed on this machine; MPS backend
confirmed available). No new dependencies needed.

## Global Constraints

- Teacher: `textattack/bert-base-uncased-imdb` (HF Hub, `BertForSequenceClassification`, 12 layers,
  hidden size 768, `num_labels=2`), loaded via `AutoModelForSequenceClassification`/`AutoTokenizer`,
  used frozen (`requires_grad_(False)`, `.eval()`) — never fine-tuned locally.
- Dataset: `datasets.load_dataset("stanfordnlp/imdb")` — splits `train`/`test` (25k/25k), columns
  `text`/`label` (0=neg, 1=pos), loadable without `trust_remote_code`. Tokenize with the teacher's
  tokenizer, `truncation=True, max_length=256, padding="max_length"`.
- Student: 6 transformer layers (half of the teacher's 12), hidden size 768 (unchanged),
  `type_vocab_size=1` (removes the token-type embedding table per the paper's Section 3), fresh
  randomly-initialized classification head. Word/position embeddings and every-other transformer
  layer (`student[i] = teacher[i * stride]`, `stride = teacher_layers // student_layers`) copied
  from the teacher at init.
- Loss: `LossWeights(alpha_task=2.0, alpha_ce=5.0, alpha_cos=1.0, temperature=2.0)` — ratios from
  DistilBERT's own released training script, with `alpha_task` standing in for the paper's
  `alpha_mlm` (no MLM target in this scoped-down, task-specific setting).
- Three student variants, same architecture/init, differing only in which loss terms are active:
  1. `baseline` — `teacher=None` passed to `train_step`/`train_loop` (task loss only).
  2. `distill` — real teacher, `LossWeights(alpha_cos=0.0)` (distillation + task, no cosine term).
  3. `triple` — real teacher, default `LossWeights()` (all three terms).
- Device: `torch.device("mps" if torch.backends.mps.is_available() else "cpu")`.
- All new files live under `DistillBERT/` (flat, no subpackage) so
  `python -m pytest test_model.py test_train.py` run from inside that directory can import `model`
  and `train` directly.
- Checkpoints (`DistillBERT/results/*.pt`) stay gitignored (repo-level `.gitignore` already excludes
  `*.pt`/`*.pth`) — unlike the LoRA implementation's small adapters, these are ~40-50% of BERT-base's
  size and not worth committing. Metrics JSON files are small and get committed.
- Commit after each task.

---

### Task 1: Student architecture construction

**Files:**
- Create: `DistillBERT/model.py`
- Test: `DistillBERT/test_model.py`

**Interfaces:**
- Produces: `build_student(teacher: BertForSequenceClassification, num_student_layers: int = 6) -> BertForSequenceClassification`. Used by Task 2's `init_student_from_teacher` and Task 6's notebook.

- [ ] **Step 1: Write the failing tests**

Create `DistillBERT/test_model.py`:

```python
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification

from model import build_student


def _tiny_teacher_config(num_hidden_layers=4, num_labels=2):
    return BertConfig(
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=16,
        type_vocab_size=2,
        num_labels=num_labels,
    )


def test_build_student_halves_layer_count():
    teacher = BertForSequenceClassification(_tiny_teacher_config(num_hidden_layers=4))
    student = build_student(teacher, num_student_layers=2)
    assert student.config.num_hidden_layers == 2


def test_build_student_removes_token_type_embeddings():
    teacher = BertForSequenceClassification(_tiny_teacher_config())
    student = build_student(teacher, num_student_layers=2)
    assert student.config.type_vocab_size == 1


def test_build_student_preserves_hidden_size_and_num_labels():
    teacher = BertForSequenceClassification(_tiny_teacher_config(num_labels=2))
    student = build_student(teacher, num_student_layers=2)
    assert student.config.hidden_size == teacher.config.hidden_size
    assert student.config.num_labels == teacher.config.num_labels


def test_build_student_forward_pass_shape():
    teacher = BertForSequenceClassification(_tiny_teacher_config())
    student = build_student(teacher, num_student_layers=2)
    input_ids = torch.randint(0, 99, (3, 8))
    attention_mask = torch.ones(3, 8, dtype=torch.long)
    out = student(input_ids=input_ids, attention_mask=attention_mask)
    assert out.logits.shape == (3, 2)


def test_build_student_has_fewer_parameters_than_teacher():
    teacher = BertForSequenceClassification(_tiny_teacher_config(num_hidden_layers=4))
    student = build_student(teacher, num_student_layers=2)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    assert student_params < teacher_params
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd DistillBERT && python -m pytest test_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_student' from 'model'` (module doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `DistillBERT/model.py`:

```python
from transformers import BertConfig, BertForSequenceClassification


def build_student(
    teacher: BertForSequenceClassification, num_student_layers: int = 6
) -> BertForSequenceClassification:
    """Fresh, randomly-initialized student: same hidden size/labels as `teacher`,
    half the transformer layers, and no token-type embeddings (paper's Section 3
    architectural choice; type_vocab_size=1 makes the embedding table a single row,
    which is fine since IMDb is single-sequence classification)."""
    student_config = BertConfig(
        **{
            **teacher.config.to_dict(),
            "num_hidden_layers": num_student_layers,
            "type_vocab_size": 1,
        }
    )
    return BertForSequenceClassification(student_config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd DistillBERT && python -m pytest test_model.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add DistillBERT/model.py DistillBERT/test_model.py
git commit -m "Add DistilBERT student architecture construction"
```

---

### Task 2: Teacher-to-student weight initialization

**Files:**
- Modify: `DistillBERT/model.py`
- Modify: `DistillBERT/test_model.py`

**Interfaces:**
- Consumes: `build_student` (Task 1).
- Produces: `init_student_from_teacher(student: BertForSequenceClassification, teacher: BertForSequenceClassification) -> None` (in-place, raises `ValueError` if teacher layer count isn't an exact multiple of student layer count). Used by Task 4/5's tests and Task 6's notebook.

- [ ] **Step 1: Write the failing tests**

Append to `DistillBERT/test_model.py`:

```python
from model import init_student_from_teacher


def test_init_student_from_teacher_copies_embeddings():
    teacher = BertForSequenceClassification(_tiny_teacher_config(num_hidden_layers=4))
    student = build_student(teacher, num_student_layers=2)
    init_student_from_teacher(student, teacher)
    assert torch.equal(
        student.bert.embeddings.word_embeddings.weight,
        teacher.bert.embeddings.word_embeddings.weight,
    )
    assert torch.equal(
        student.bert.embeddings.position_embeddings.weight,
        teacher.bert.embeddings.position_embeddings.weight,
    )


def test_init_student_from_teacher_copies_every_other_layer():
    teacher = BertForSequenceClassification(_tiny_teacher_config(num_hidden_layers=4))
    student = build_student(teacher, num_student_layers=2)
    init_student_from_teacher(student, teacher)
    assert torch.equal(
        student.bert.encoder.layer[0].attention.self.query.weight,
        teacher.bert.encoder.layer[0].attention.self.query.weight,
    )
    assert torch.equal(
        student.bert.encoder.layer[1].attention.self.query.weight,
        teacher.bert.encoder.layer[2].attention.self.query.weight,
    )


def test_init_student_from_teacher_does_not_touch_classifier_head():
    teacher = BertForSequenceClassification(_tiny_teacher_config(num_hidden_layers=4))
    student = build_student(teacher, num_student_layers=2)
    classifier_before = student.classifier.weight.clone()
    init_student_from_teacher(student, teacher)
    assert torch.equal(student.classifier.weight, classifier_before)


def test_init_student_from_teacher_raises_if_layer_count_does_not_divide_evenly():
    teacher = BertForSequenceClassification(_tiny_teacher_config(num_hidden_layers=5))
    student = build_student(teacher, num_student_layers=2)
    with pytest.raises(ValueError):
        init_student_from_teacher(student, teacher)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd DistillBERT && python -m pytest test_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'init_student_from_teacher' from 'model'`.

- [ ] **Step 3: Write minimal implementation**

Append to `DistillBERT/model.py`:

```python
def init_student_from_teacher(
    student: BertForSequenceClassification, teacher: BertForSequenceClassification
) -> None:
    """In-place: copies word/position embeddings and every stride-th teacher layer into
    the student (stride = teacher_layers / student_layers, must divide evenly). The
    classifier head is left as freshly initialized -- it isn't reusable across depths."""
    teacher_layers = teacher.config.num_hidden_layers
    student_layers = student.config.num_hidden_layers
    if teacher_layers % student_layers != 0:
        raise ValueError(
            f"teacher layers ({teacher_layers}) must be an exact multiple of "
            f"student layers ({student_layers})"
        )
    stride = teacher_layers // student_layers

    student.bert.embeddings.word_embeddings.weight.data.copy_(
        teacher.bert.embeddings.word_embeddings.weight.data
    )
    student.bert.embeddings.position_embeddings.weight.data.copy_(
        teacher.bert.embeddings.position_embeddings.weight.data
    )
    student.bert.embeddings.LayerNorm.weight.data.copy_(
        teacher.bert.embeddings.LayerNorm.weight.data
    )
    student.bert.embeddings.LayerNorm.bias.data.copy_(teacher.bert.embeddings.LayerNorm.bias.data)

    for i in range(student_layers):
        student.bert.encoder.layer[i].load_state_dict(
            teacher.bert.encoder.layer[i * stride].state_dict()
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd DistillBERT && python -m pytest test_model.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add DistillBERT/model.py DistillBERT/test_model.py
git commit -m "Add teacher-to-student every-other-layer initialization"
```

---

### Task 3: Distillation and cosine loss functions

**Files:**
- Modify: `DistillBERT/model.py`
- Modify: `DistillBERT/test_model.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2 (pure functions on tensors).
- Produces: `LossWeights` (dataclass: `alpha_task: float = 2.0, alpha_ce: float = 5.0, alpha_cos: float = 1.0, temperature: float = 2.0`), `distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor`, `cosine_loss(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor`. Used by Task 4's `train_step` and Task 6's notebook.

- [ ] **Step 1: Write the failing tests**

Append to `DistillBERT/test_model.py`:

```python
import math

from model import LossWeights, cosine_loss, distillation_loss


def test_distillation_loss_known_value_uniform_case():
    logits = torch.tensor([[0.0, 0.0]])
    loss = distillation_loss(logits, logits, temperature=1.0)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-5)


def test_distillation_loss_known_value_nonuniform_case():
    logits = torch.log(torch.tensor([[3.0, 1.0]]))  # softmax(logits) -> [0.75, 0.25]
    loss = distillation_loss(logits, logits, temperature=1.0)
    expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert loss.item() == pytest.approx(expected, abs=1e-5)


def test_distillation_loss_temperature_changes_the_value():
    logits = torch.log(torch.tensor([[3.0, 1.0]]))
    loss_t1 = distillation_loss(logits, logits, temperature=1.0)
    loss_t2 = distillation_loss(logits, logits, temperature=2.0)
    assert loss_t1.item() != pytest.approx(loss_t2.item())


def test_cosine_loss_identical_vectors_is_zero():
    v = torch.tensor([[1.0, 2.0, 3.0]])
    loss = cosine_loss(v, v)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_cosine_loss_orthogonal_vectors_is_one():
    v1 = torch.tensor([[1.0, 0.0]])
    v2 = torch.tensor([[0.0, 1.0]])
    loss = cosine_loss(v1, v2)
    assert loss.item() == pytest.approx(1.0, abs=1e-5)


def test_cosine_loss_opposite_vectors_is_two():
    v1 = torch.tensor([[1.0, 0.0]])
    v2 = torch.tensor([[-1.0, 0.0]])
    loss = cosine_loss(v1, v2)
    assert loss.item() == pytest.approx(2.0, abs=1e-5)


def test_loss_weights_defaults_match_paper_release_ratios():
    weights = LossWeights()
    assert weights.alpha_task == pytest.approx(2.0)
    assert weights.alpha_ce == pytest.approx(5.0)
    assert weights.alpha_cos == pytest.approx(1.0)
    assert weights.temperature == pytest.approx(2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd DistillBERT && python -m pytest test_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'LossWeights' from 'model'`.

- [ ] **Step 3: Write minimal implementation**

Add these imports to the top of `DistillBERT/model.py` (above the existing `build_student`
function):

```python
from dataclasses import dataclass

import torch
import torch.nn.functional as F
```

Append to `DistillBERT/model.py`:

```python
@dataclass
class LossWeights:
    """Linear-combination weights for the (adapted) triple loss, using the ratios from
    DistilBERT's own released training script (alpha_ce=5.0, alpha_mlm=2.0 -- renamed
    here to alpha_task since there's no MLM target in this task-specific setting --
    alpha_cos=1.0, T=2.0)."""

    alpha_task: float = 2.0
    alpha_ce: float = 5.0
    alpha_cos: float = 1.0
    temperature: float = 2.0


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Soft-target cross-entropy: -sum_i teacher_prob_i * log(student_prob_i), both
    softened by `temperature`, averaged over the batch."""
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean()


def cosine_loss(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, averaged over the batch -- pushes student/teacher hidden
    state vectors toward the same direction (equivalent to CosineEmbeddingLoss with
    target=+1)."""
    return (1 - F.cosine_similarity(student_hidden, teacher_hidden, dim=-1)).mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd DistillBERT && python -m pytest test_model.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add DistillBERT/model.py DistillBERT/test_model.py
git commit -m "Add distillation and cosine-embedding loss functions"
```

---

### Task 4: Training step and evaluation

**Files:**
- Create: `DistillBERT/train.py`
- Create: `DistillBERT/test_train.py`

**Interfaces:**
- Consumes: `LossWeights`, `distillation_loss`, `cosine_loss` (Task 3); `build_student`, `init_student_from_teacher` (Tasks 1-2, used only in tests).
- Produces: `train_step(student, teacher, batch: dict, optimizer, weights: LossWeights, device) -> dict` (keys `"loss"`, `"task_loss"`, `"distill_loss"`, `"cos_loss"`; `teacher` may be `None`), `predict(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]` (probabilities `[N, num_labels]`, labels `[N]`), `evaluate(model, loader, device) -> dict` (keys `"accuracy"`, `"f1"`, `"loss"`). Used by Task 5's `train_loop` and Task 6's notebook. `batch` is a dict with `"input_ids"`, `"attention_mask"`, `"labels"` tensors (as produced by a `DataLoader` over dict-shaped items via the default collate function).

- [ ] **Step 1: Write the failing tests**

Create `DistillBERT/test_train.py`:

```python
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForSequenceClassification

from model import LossWeights, build_student, init_student_from_teacher
from train import evaluate, predict, train_step


def _tiny_teacher(num_hidden_layers=4, num_labels=2):
    config = BertConfig(
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=16,
        type_vocab_size=2,
        num_labels=num_labels,
    )
    return BertForSequenceClassification(config)


def _make_batch(batch_size=4, seq_len=8, vocab_size=99):
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.randint(0, 2, (batch_size,)),
    }


def _make_loader(num_samples=16, seq_len=8, vocab_size=99, batch_size=4):
    items = [
        {
            "input_ids": torch.randint(0, vocab_size, (seq_len,)),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            "labels": torch.randint(0, 2, ()),
        }
        for _ in range(num_samples)
    ]
    return DataLoader(items, batch_size=batch_size)


def test_train_step_baseline_has_zero_distill_and_cos_loss():
    teacher = _tiny_teacher()
    student = build_student(teacher, num_student_layers=2)
    init_student_from_teacher(student, teacher)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    result = train_step(
        student, None, _make_batch(), optimizer, LossWeights(), device=torch.device("cpu")
    )
    assert set(result.keys()) == {"loss", "task_loss", "distill_loss", "cos_loss"}
    assert result["distill_loss"] == 0.0
    assert result["cos_loss"] == 0.0


def test_train_step_with_teacher_produces_nonzero_distill_and_cos_loss():
    teacher = _tiny_teacher()
    student = build_student(teacher, num_student_layers=2)
    init_student_from_teacher(student, teacher)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    result = train_step(
        student, teacher, _make_batch(), optimizer, LossWeights(), device=torch.device("cpu")
    )
    assert result["distill_loss"] != 0.0
    assert result["cos_loss"] != 0.0


def test_train_step_updates_student_parameters():
    teacher = _tiny_teacher()
    student = build_student(teacher, num_student_layers=2)
    init_student_from_teacher(student, teacher)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    before = [p.clone() for p in student.parameters()]
    train_step(
        student, teacher, _make_batch(), optimizer, LossWeights(), device=torch.device("cpu")
    )
    after = list(student.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_train_step_does_not_update_teacher_parameters():
    teacher = _tiny_teacher()
    student = build_student(teacher, num_student_layers=2)
    init_student_from_teacher(student, teacher)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    before = [p.clone() for p in teacher.parameters()]
    train_step(
        student, teacher, _make_batch(), optimizer, LossWeights(), device=torch.device("cpu")
    )
    after = list(teacher.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_predict_returns_expected_shapes():
    teacher = _tiny_teacher()
    loader = _make_loader(num_samples=16, batch_size=4)
    probs, labels = predict(teacher, loader, device=torch.device("cpu"))
    assert probs.shape == (16, 2)
    assert labels.shape == (16,)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-5)


def test_evaluate_returns_expected_keys():
    teacher = _tiny_teacher()
    loader = _make_loader(num_samples=16, batch_size=4)
    result = evaluate(teacher, loader, device=torch.device("cpu"))
    assert set(result.keys()) == {"accuracy", "f1", "loss"}
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["f1"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd DistillBERT && python -m pytest test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train'`.

- [ ] **Step 3: Write minimal implementation**

Create `DistillBERT/train.py`:

```python
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from model import LossWeights, cosine_loss, distillation_loss


def train_step(student, teacher, batch: dict, optimizer, weights: LossWeights, device) -> dict:
    student.train()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    need_hidden = teacher is not None and weights.alpha_cos > 0

    optimizer.zero_grad()
    student_out = student(
        input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=need_hidden
    )
    task_loss = F.cross_entropy(student_out.logits, labels)

    distill_loss = torch.tensor(0.0, device=device)
    cos_loss_value = torch.tensor(0.0, device=device)
    if teacher is not None:
        with torch.no_grad():
            teacher_out = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=need_hidden,
            )
        if weights.alpha_ce > 0:
            distill_loss = distillation_loss(
                student_out.logits, teacher_out.logits, weights.temperature
            )
        if weights.alpha_cos > 0:
            student_cls = student_out.hidden_states[-1][:, 0, :]
            teacher_cls = teacher_out.hidden_states[-1][:, 0, :]
            cos_loss_value = cosine_loss(student_cls, teacher_cls)

    total = (
        weights.alpha_task * task_loss
        + weights.alpha_ce * distill_loss
        + weights.alpha_cos * cos_loss_value
    )
    total.backward()
    optimizer.step()

    return {
        "loss": total.item(),
        "task_loss": task_loss.item(),
        "distill_loss": distill_loss.item(),
        "cos_loss": cos_loss_value.item(),
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        all_probs.append(F.softmax(logits, dim=-1).cpu())
        all_labels.append(batch["labels"].cpu())
    return torch.cat(all_probs), torch.cat(all_labels)


def evaluate(model, loader, device) -> dict:
    probs, labels = predict(model, loader, device)
    preds = probs.argmax(dim=-1)
    accuracy = (preds == labels).float().mean().item()
    f1 = f1_score(labels.numpy(), preds.numpy())
    loss = F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels).item()
    return {"accuracy": accuracy, "f1": f1, "loss": loss}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd DistillBERT && python -m pytest test_train.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run the full test suite together**

Run: `cd DistillBERT && python -m pytest -v`
Expected: 22 passed (16 from `test_model.py` + 6 from `test_train.py`).

- [ ] **Step 6: Commit**

```bash
git add DistillBERT/train.py DistillBERT/test_train.py
git commit -m "Add distillation train step and evaluation"
```

---

### Task 5: Training loop with metrics and checkpoint persistence

**Files:**
- Modify: `DistillBERT/train.py`
- Modify: `DistillBERT/test_train.py`

**Interfaces:**
- Consumes: `train_step`, `evaluate` (Task 4).
- Produces: `train_loop(student, teacher, train_loader, eval_loader, optimizer, weights: LossWeights, device, num_epochs: int, results_dir, run_name: str, log_every: int = 50) -> dict` (keys `"step_metrics"`, `"epoch_metrics"`, each a list of dicts). Writes `{results_dir}/{run_name}_step_metrics.json`, `{results_dir}/{run_name}_epoch_metrics.json`, and `{results_dir}/{run_name}.pt` (student state dict). Used by Task 6's notebook.

- [ ] **Step 1: Write the failing tests**

Append to `DistillBERT/test_train.py`:

```python
import json

from train import train_loop


def test_train_loop_writes_metrics_and_checkpoint(tmp_path):
    teacher = _tiny_teacher()
    student = build_student(teacher, num_student_layers=2)
    init_student_from_teacher(student, teacher)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    train_loader = _make_loader(num_samples=8, batch_size=4)
    eval_loader = _make_loader(num_samples=8, batch_size=4)

    history = train_loop(
        student,
        teacher,
        train_loader,
        eval_loader,
        optimizer,
        LossWeights(),
        device=torch.device("cpu"),
        num_epochs=1,
        results_dir=tmp_path,
        run_name="test_run",
        log_every=1,
    )

    assert (tmp_path / "test_run_step_metrics.json").exists()
    assert (tmp_path / "test_run_epoch_metrics.json").exists()
    assert (tmp_path / "test_run.pt").exists()
    assert len(history["epoch_metrics"]) == 1

    with open(tmp_path / "test_run_step_metrics.json") as f:
        step_data = json.load(f)
    assert len(step_data) > 0
    assert set(step_data[0].keys()) == {
        "step",
        "epoch",
        "elapsed_seconds",
        "loss",
        "task_loss",
        "distill_loss",
        "cos_loss",
    }


def test_train_loop_checkpoint_matches_student_state_dict(tmp_path):
    teacher = _tiny_teacher()
    student = build_student(teacher, num_student_layers=2)
    init_student_from_teacher(student, teacher)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    train_loader = _make_loader(num_samples=8, batch_size=4)
    eval_loader = _make_loader(num_samples=8, batch_size=4)

    train_loop(
        student,
        teacher,
        train_loader,
        eval_loader,
        optimizer,
        LossWeights(),
        device=torch.device("cpu"),
        num_epochs=1,
        results_dir=tmp_path,
        run_name="test_run",
        log_every=1,
    )

    saved_state = torch.load(tmp_path / "test_run.pt")
    current_state = student.state_dict()
    for key in current_state:
        assert torch.equal(saved_state[key], current_state[key])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd DistillBERT && python -m pytest test_train.py -v`
Expected: FAIL with `ImportError: cannot import name 'train_loop' from 'train'`.

- [ ] **Step 3: Write minimal implementation**

Add these imports to the top of `DistillBERT/train.py`:

```python
import json
import time
from pathlib import Path
```

Append to `DistillBERT/train.py`:

```python
def train_loop(
    student,
    teacher,
    train_loader,
    eval_loader,
    optimizer,
    weights: LossWeights,
    device,
    num_epochs: int,
    results_dir,
    run_name: str,
    log_every: int = 50,
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    step_metrics = []
    epoch_metrics = []
    step = 0
    start = time.perf_counter()

    for epoch in range(num_epochs):
        for batch in train_loader:
            step_result = train_step(student, teacher, batch, optimizer, weights, device)
            step += 1
            if step % log_every == 0:
                step_metrics.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "elapsed_seconds": time.perf_counter() - start,
                        **step_result,
                    }
                )
                with open(results_dir / f"{run_name}_step_metrics.json", "w") as f:
                    json.dump(step_metrics, f, indent=2)

        eval_result = evaluate(student, eval_loader, device)
        epoch_metrics.append({"epoch": epoch, **eval_result})
        with open(results_dir / f"{run_name}_epoch_metrics.json", "w") as f:
            json.dump(epoch_metrics, f, indent=2)

    torch.save(student.state_dict(), results_dir / f"{run_name}.pt")

    return {"step_metrics": step_metrics, "epoch_metrics": epoch_metrics}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd DistillBERT && python -m pytest test_train.py -v`
Expected: 8 passed.

- [ ] **Step 5: Run the full test suite together**

Run: `cd DistillBERT && python -m pytest -v`
Expected: 24 passed (16 from `test_model.py` + 8 from `test_train.py`).

- [ ] **Step 6: Commit**

```bash
git add DistillBERT/train.py DistillBERT/test_train.py
git commit -m "Add training loop with metrics and checkpoint persistence"
```

---

### Task 6: Notebook — paper walkthrough, three training runs, and visualizations

**Files:**
- Create: `DistillBERT/distillbert.ipynb`

**Interfaces:**
- Consumes: `build_student`, `init_student_from_teacher`, `LossWeights` (from `model.py`); `train_step`, `predict`, `evaluate`, `train_loop` (from `train.py`).

This is the main deliverable and is not itself unit-tested — it's a Jupyter notebook, verified by
executing it end-to-end (Step 10 below) rather than pytest. Build it as the following cells, in
order. Use the `NotebookEdit` tool to create cells in `DistillBERT/distillbert.ipynb` one at a time
in this sequence.

- [ ] **Step 1: Title and paper explanation (markdown cell)**

```markdown
# DistilBERT: a distilled version of BERT

Sanh, Debut, Chaumond, Wolf (Hugging Face) — EMC² @ NeurIPS 2019. [Paper PDF](DistillBERT.pdf)

**The idea:** train a small **student** to reproduce a large **teacher**'s full output
distribution, not just its hard predictions. The teacher's "near-zero" probabilities on wrong
classes still carry information — how plausible each mistake is — that one-hot labels discard.

**Distillation loss**, with a softmax temperature `T` applied to both models (T=1 at inference):

$$L_{ce} = -\sum_i t_i \log(s_i), \quad p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

**Triple loss** (paper's Section 2-3): a weighted sum of the distillation loss, a supervised loss
against gold labels, and a cosine-embedding loss aligning student/teacher hidden state directions.

**Student architecture:** same general shape as BERT, half the transformer layers, no token-type
embeddings. **Initialization:** student layer `i` copies teacher layer `2*i` — taking advantage of
the shared hidden size between teacher and student.

**Scope note:** the paper's actual headline result distills *during general-purpose MLM
pretraining* on Wikipedia + BookCorpus (90 GPU-hours on 8×V100). That's not reproducible locally, so
this notebook scopes down to **task-specific distillation**: an off-the-shelf fine-tuned BERT is the
frozen teacher, and the student is trained directly on IMDb sentiment classification, replacing the
paper's MLM loss term with a supervised classification loss. See `design.md` for the full rationale.
Three variants isolate each loss term's contribution, mirroring the paper's own ablation (Table 4):

1. **baseline** — supervised task loss only, no distillation signal.
2. **distill** — distillation loss + task loss.
3. **triple** — distillation loss + task loss + cosine loss (paper's full recipe, adapted).
```

- [ ] **Step 2: Imports and device setup (code cell)**

```python
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model import LossWeights, build_student, cosine_loss, distillation_loss, init_student_from_teacher
from train import evaluate, predict, train_loop

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
TEACHER_NAME = "textattack/bert-base-uncased-imdb"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 2

torch.manual_seed(0)
```

- [ ] **Step 3: Load teacher and tokenizer (markdown + code cell)**

Markdown:
```markdown
## Teacher

An off-the-shelf BERT-base already fine-tuned on IMDb (`textattack/bert-base-uncased-imdb`),
frozen — this notebook never fine-tunes it, only distills from it.
```

Code:
```python
tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_NAME).to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

teacher_params = sum(p.numel() for p in teacher.parameters())
print(f"Teacher: {teacher.config.num_hidden_layers} layers, {teacher_params:,} parameters")
```

- [ ] **Step 4: Load and tokenize IMDb (markdown + code cell)**

Markdown:
```markdown
## IMDb dataset

Full train/test splits (25k / 25k), reviews truncated to 256 tokens. A 1,000-example slice of the
test set is used for cheap per-epoch tracking during training; the full 25k test set is used once
per model for the headline accuracy/F1 numbers.
```

Code:
```python
raw = load_dataset("stanfordnlp/imdb")


def tokenize(batch):
    return tokenizer(
        batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length"
    )


tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = torch.utils.data.DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True)
test_loader_full = torch.utils.data.DataLoader(tokenized["test"], batch_size=BATCH_SIZE)
test_loader_quick = torch.utils.data.DataLoader(
    tokenized["test"].select(range(1000)), batch_size=BATCH_SIZE
)

print(f"Train: {len(tokenized['train'])}, Test: {len(tokenized['test'])}")
```

- [ ] **Step 5: Teacher-init sanity check (code cell)**

```python
sanity_student = build_student(teacher, num_student_layers=6).to(device)
init_student_from_teacher(sanity_student, teacher)
assert torch.equal(
    sanity_student.bert.encoder.layer[0].attention.self.query.weight,
    teacher.bert.encoder.layer[0].attention.self.query.weight,
)
assert torch.equal(
    sanity_student.bert.encoder.layer[5].attention.self.query.weight,
    teacher.bert.encoder.layer[10].attention.self.query.weight,
)
student_params = sum(p.numel() for p in sanity_student.parameters())
print(
    f"Student: {sanity_student.config.num_hidden_layers} layers, {student_params:,} parameters "
    f"({student_params / teacher_params:.1%} of teacher)"
)
del sanity_student
```

- [ ] **Step 6: Run the three training variants (markdown + code cell)**

Markdown:
```markdown
## Training three variants

Same architecture and teacher-initialization every time — only the active loss terms differ. Each
variant trains a fresh student, evaluates on the full test set at the end, and persists its
checkpoint + per-step metrics under `results/`.
```

Code:
```python
def run_variant(name: str, use_teacher: bool, weights: LossWeights):
    torch.manual_seed(0)
    student = build_student(teacher, num_student_layers=6).to(device)
    init_student_from_teacher(student, teacher)
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)

    variant_teacher = teacher if use_teacher else None
    history = train_loop(
        student,
        variant_teacher,
        train_loader,
        test_loader_quick,
        optimizer,
        weights,
        device=device,
        num_epochs=NUM_EPOCHS,
        results_dir=RESULTS_DIR,
        run_name=name,
        log_every=50,
    )

    start = time.perf_counter()
    final_metrics = evaluate(student, test_loader_full, device)
    inference_time = time.perf_counter() - start

    del student
    return {**history, "final_metrics": final_metrics, "inference_time": inference_time}


variants = {
    "baseline": {"use_teacher": False, "weights": LossWeights()},
    "distill": {"use_teacher": True, "weights": LossWeights(alpha_cos=0.0)},
    "triple": {"use_teacher": True, "weights": LossWeights()},
}

results = {}
for name, cfg in variants.items():
    print(f"\n=== Training variant: {name} ===")
    results[name] = run_variant(name, cfg["use_teacher"], cfg["weights"])
    print(f"{name} final metrics: {results[name]['final_metrics']}")
```

- [ ] **Step 7: Teacher's own test-set metrics and inference time (code cell)**

```python
start = time.perf_counter()
teacher_metrics = evaluate(teacher, test_loader_full, device)
teacher_inference_time = time.perf_counter() - start
print(f"Teacher final metrics: {teacher_metrics}, inference time: {teacher_inference_time:.1f}s")
```

- [ ] **Step 8: Comparison visualizations — params, accuracy/F1, timing (markdown + code cells)**

Markdown:
```markdown
## Results

Parameter count, accuracy/F1, and inference time, teacher vs. each student variant.
```

Code (parameter count and accuracy/F1 bar charts):
```python
labels_order = ["teacher", "baseline", "distill", "triple"]
param_counts = [teacher_params] + [
    sum(p.numel() for p in build_student(teacher, num_student_layers=6).parameters())
] * 3
accuracies = [teacher_metrics["accuracy"]] + [results[n]["final_metrics"]["accuracy"] for n in variants]
f1s = [teacher_metrics["f1"]] + [results[n]["final_metrics"]["f1"] for n in variants]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(labels_order, param_counts)
axes[0].set_ylabel("Parameters")
axes[0].set_title("Parameter count")

axes[1].bar(labels_order, accuracies)
axes[1].set_ylabel("Test accuracy")
axes[1].set_title("Accuracy")
axes[1].set_ylim(0, 1)

axes[2].bar(labels_order, f1s)
axes[2].set_ylabel("Test F1")
axes[2].set_title("F1")
axes[2].set_ylim(0, 1)
plt.tight_layout()
plt.show()
```

Code (inference time bar chart):
```python
inference_times = [teacher_inference_time] + [results[n]["inference_time"] for n in variants]
plt.figure(figsize=(7, 4))
plt.bar(labels_order, inference_times)
plt.ylabel("Full test set inference time (s)")
plt.title("Inference time: teacher vs. students")
plt.show()
```

- [ ] **Step 9: Training curves and output agreement (markdown + code cells)**

Markdown:
```markdown
## Training curves and teacher/student agreement

Total training loss per variant, the triple-loss variant's individual loss components, and how
closely each student's predicted probability distribution matches the teacher's on the test set
(lower KL divergence = closer match, beyond just matching hard predictions).
```

Code (training loss curves):
```python
plt.figure(figsize=(8, 5))
for name in variants:
    steps = [m["step"] for m in results[name]["step_metrics"]]
    losses = [m["loss"] for m in results[name]["step_metrics"]]
    plt.plot(steps, losses, label=name)
plt.xlabel("Training step")
plt.ylabel("Total loss")
plt.title("Training loss by variant")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

triple_steps = [m["step"] for m in results["triple"]["step_metrics"]]
plt.figure(figsize=(8, 5))
plt.plot(triple_steps, [m["task_loss"] for m in results["triple"]["step_metrics"]], label="task")
plt.plot(triple_steps, [m["distill_loss"] for m in results["triple"]["step_metrics"]], label="distill")
plt.plot(triple_steps, [m["cos_loss"] for m in results["triple"]["step_metrics"]], label="cosine")
plt.xlabel("Training step")
plt.ylabel("Loss component")
plt.title("Triple-loss variant: individual loss components")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

Code (teacher/student output agreement, reloading checkpoints from disk):
```python
teacher_probs, _ = predict(teacher, test_loader_quick, device)

agreement_rows = []
for name in variants:
    student = build_student(teacher, num_student_layers=6).to(device)
    student.load_state_dict(torch.load(RESULTS_DIR / f"{name}.pt", map_location=device))
    student_probs, _ = predict(student, test_loader_quick, device)
    kl = F.kl_div(torch.log(student_probs.clamp_min(1e-12)), teacher_probs, reduction="batchmean")
    agreement = (student_probs.argmax(dim=-1) == teacher_probs.argmax(dim=-1)).float().mean()
    agreement_rows.append((name, kl.item(), agreement.item()))
    del student

print(f"{'Variant':<12}{'KL(student||teacher)':<24}{'Argmax agreement':<20}")
for name, kl, agreement in agreement_rows:
    print(f"{name:<12}{kl:<24.4f}{agreement:<20.3f}")

plt.figure(figsize=(7, 4))
plt.bar([r[0] for r in agreement_rows], [r[1] for r in agreement_rows])
plt.ylabel("KL divergence (student || teacher)")
plt.title("Output distribution distance to teacher (lower = closer)")
plt.show()
```

- [ ] **Step 10: Qualitative examples (markdown + code cell)**

Markdown:
```markdown
## Qualitative examples

A handful of fixed test reviews with the teacher's and each variant's predicted probability of
"positive."
```

Code:
```python
sample_indices = [0, 1, 2, 3, 4]
sample_texts = [raw["test"][i]["text"][:200] for i in sample_indices]
sample_batch = {
    "input_ids": tokenized["test"].select(sample_indices)["input_ids"].to(device),
    "attention_mask": tokenized["test"].select(sample_indices)["attention_mask"].to(device),
}

with torch.no_grad():
    teacher_sample_probs = F.softmax(teacher(**sample_batch).logits, dim=-1)[:, 1]

student_sample_probs = {}
for name in variants:
    student = build_student(teacher, num_student_layers=6).to(device)
    student.load_state_dict(torch.load(RESULTS_DIR / f"{name}.pt", map_location=device))
    student.eval()
    with torch.no_grad():
        student_sample_probs[name] = F.softmax(student(**sample_batch).logits, dim=-1)[:, 1]
    del student

for i, text in enumerate(sample_texts):
    print(f"Review: {text}...")
    print(f"  teacher P(positive)={teacher_sample_probs[i]:.3f}", end="  ")
    for name in variants:
        print(f"{name}={student_sample_probs[name][i]:.3f}", end="  ")
    print("\n")
```

- [ ] **Step 11: Execute the notebook end-to-end and verify**

Run: `cd DistillBERT && jupyter nbconvert --to notebook --execute --inplace distillbert.ipynb --ExecutePreprocessor.timeout=7200`
Expected: command exits with no error; re-opening the notebook shows all cells with outputs,
including all bar charts, loss curves, and the printed comparison tables.

Then confirm the conditions from `design.md`'s Verification section by reading the executed
notebook's outputs:
- No errors on MPS/CPU backend.
- Student parameter count is roughly 40-50% of the teacher's.
- The Step 5 teacher-init assertions passed (they'll raise `AssertionError` and halt execution
  otherwise, so passing silently is confirmation).
- All 3 variants' training loss decreases over training (visible in the Step 9 loss-curve plot).
- Test accuracy improves in the expected direction: baseline < distill < triple. If a run doesn't
  follow this ordering, note the actual numbers rather than overstating the result — the paper's
  own ablation supports this ordering but doesn't guarantee it on a single run/seed.
- The `distill` and `triple` variants have lower KL divergence to the teacher than `baseline` (Step
  9's agreement table).

If accuracy is poor across all variants (e.g. close to 50% on IMDb, indicating underfitting), that's
a signal to increase `NUM_EPOCHS` in Step 2 and re-run rather than a plan step to skip.

- [ ] **Step 12: Commit**

```bash
git add DistillBERT/distillbert.ipynb DistillBERT/results/*.json
git commit -m "Add DistilBERT paper walkthrough notebook with training and visualizations"
```

---

## Self-Review Notes

- **Spec coverage:** student architecture (halved depth, no token-type embeddings — Task 1),
  every-other-layer teacher initialization (Task 2), triple-loss functions with the paper's release
  ratios (Task 3), train step / evaluate / predict (Task 4), metrics + checkpoint persistence (Task
  5), and all of `design.md`'s comparison/visualization items — parameter count, accuracy/F1,
  inference time, training loss curves, teacher/student output agreement (KL divergence), and
  qualitative examples — are covered in Task 6's notebook (Steps 8-10). The paper explanation and
  scope note are Task 6 Step 1.
- **Placeholder scan:** none found; every step has runnable code or an exact command.
- **Type consistency:** `train_step`'s return keys (`"loss"`, `"task_loss"`, `"distill_loss"`,
  `"cos_loss"`) match between Task 4's implementation, its tests, `train_loop`'s per-step metrics
  dict (Task 5), and the notebook's Step 9 plotting code. `evaluate`'s return keys (`"accuracy"`,
  `"f1"`, `"loss"`) match between Task 4 and the notebook's Steps 6-8. `build_student`/
  `init_student_from_teacher` signatures match across Tasks 1-2, Task 4/5's tests, and Task 6's
  notebook usage. `LossWeights` field names (`alpha_task`, `alpha_ce`, `alpha_cos`, `temperature`)
  match across Task 3's definition, Task 4's `train_step` usage, and Task 6's three variant configs.
