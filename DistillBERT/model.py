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
