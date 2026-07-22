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
