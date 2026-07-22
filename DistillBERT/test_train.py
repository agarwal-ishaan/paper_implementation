import json
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForSequenceClassification

from model import LossWeights, build_student, init_student_from_teacher
from train import evaluate, predict, train_step, train_loop


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
