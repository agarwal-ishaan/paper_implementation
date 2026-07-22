from dataclasses import dataclass

import torch
import torch.nn.functional as F
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
