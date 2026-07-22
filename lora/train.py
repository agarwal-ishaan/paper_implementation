import time

import torch
import torch.nn.functional as F


def compute_loss(model, input_ids, attention_mask=None) -> torch.Tensor:
    """Causal LM loss: predict token t+1 from tokens up to t."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )


def train_step(model, batch, optimizer, device) -> dict:
    model.train()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    optimizer.zero_grad()
    start = time.perf_counter()
    loss = compute_loss(model, input_ids, attention_mask)
    loss.backward()
    optimizer.step()
    elapsed = time.perf_counter() - start
    return {"loss": loss.item(), "time_seconds": elapsed}


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        loss = compute_loss(model, input_ids, attention_mask)
        total_loss += loss.item()
        total_batches += 1
    return {"loss": total_loss / total_batches}


def current_memory_mb() -> float:
    if torch.backends.mps.is_available():
        # current_allocated_memory() reports the actual bytes occupied by
        # tensors resident on the MPS device -- the precise, standard
        # metric (directly analogous to torch.cuda.memory_allocated())
        # needed to compare full fine-tuning's optimizer-state footprint
        # against LoRA's.
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    import psutil
    return psutil.Process().memory_info().rss / (1024 ** 2)


def save_checkpoint(model, path: str) -> None:
    torch.save(model.state_dict(), path)


def save_lora_adapters(lora_modules, path: str) -> None:
    state = {}
    for i, module in enumerate(lora_modules):
        state[f"layer_{i}.lora_A_q"] = module.lora_A_q.detach().cpu()
        state[f"layer_{i}.lora_B_q"] = module.lora_B_q.detach().cpu()
        state[f"layer_{i}.lora_A_v"] = module.lora_A_v.detach().cpu()
        state[f"layer_{i}.lora_B_v"] = module.lora_B_v.detach().cpu()
    torch.save(state, path)
