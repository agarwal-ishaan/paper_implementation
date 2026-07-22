import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from model import LoRAConv1D
from train import (
    compute_loss,
    current_memory_mb,
    evaluate,
    save_checkpoint,
    save_lora_adapters,
    train_step,
)


def make_tiny_gpt2(vocab_size=100, n_positions=32, n_embd=16, n_layer=2, n_head=2):
    """A randomly-initialized, tiny GPT-2 for fast, isolated loop-mechanics
    tests — these tests only need something shaped like a causal LM, not the
    real pretrained distilgpt2 (that integration is covered in test_model.py,
    which specifically needs the real architecture)."""
    config = GPT2Config(
        vocab_size=vocab_size, n_positions=n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head
    )
    return GPT2LMHeadModel(config)


def _tiny_batch(vocab_size=100, batch_size=2, seq_len=8):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def test_compute_loss_returns_positive_scalar():
    model = make_tiny_gpt2()
    batch = _tiny_batch()
    loss = compute_loss(model, batch["input_ids"], batch["attention_mask"])
    assert loss.dim() == 0
    assert loss.item() > 0


def test_train_step_returns_expected_keys_and_updates_params():
    model = make_tiny_gpt2()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = _tiny_batch()
    before = model.lm_head.weight.clone()
    result = train_step(model, batch, optimizer, device=torch.device("cpu"))
    assert set(result.keys()) == {"loss", "time_seconds"}
    assert result["time_seconds"] > 0
    after = model.lm_head.weight
    assert not torch.equal(before, after)


def test_evaluate_returns_loss_without_updating_params():
    model = make_tiny_gpt2()
    batch = _tiny_batch()
    loader = [batch, batch]
    before = [p.clone() for p in model.parameters()]
    result = evaluate(model, loader, device=torch.device("cpu"))
    assert "loss" in result
    after = list(model.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_current_memory_mb_returns_positive_float():
    if torch.backends.mps.is_available():
        # Guarantee there's real allocated memory to report, regardless of
        # whether any earlier test in this process happened to allocate an
        # MPS tensor first.
        _keep_alive = torch.randn(1000, 1000, device="mps")
    assert current_memory_mb() > 0


def test_save_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(4, 4)
    path = tmp_path / "model.pt"
    save_checkpoint(model, str(path))
    loaded_state = torch.load(path)
    assert set(loaded_state.keys()) == set(model.state_dict().keys())


def test_save_lora_adapters_roundtrip(tmp_path):
    base = Conv1D(9, 3)
    lora_modules = [LoRAConv1D(base, r=2, alpha=4)]
    path = tmp_path / "adapters.pt"
    save_lora_adapters(lora_modules, str(path))
    state = torch.load(path)
    assert torch.equal(state["layer_0.lora_A_q"], lora_modules[0].lora_A_q.detach().cpu())
    assert torch.equal(state["layer_0.lora_B_q"], lora_modules[0].lora_B_q.detach().cpu())
    assert torch.equal(state["layer_0.lora_A_v"], lora_modules[0].lora_A_v.detach().cpu())
    assert torch.equal(state["layer_0.lora_B_v"], lora_modules[0].lora_B_v.detach().cpu())
