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
