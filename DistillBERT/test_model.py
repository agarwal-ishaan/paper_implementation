import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification

from model import build_student, init_student_from_teacher


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
