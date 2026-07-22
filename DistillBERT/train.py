import json
import time
from pathlib import Path

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
        teacher.eval()
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

    # Guarantee step_metrics.json is written even if no steps hit log_every threshold
    with open(results_dir / f"{run_name}_step_metrics.json", "w") as f:
        json.dump(step_metrics, f, indent=2)

    torch.save(student.state_dict(), results_dir / f"{run_name}.pt")

    return {"step_metrics": step_metrics, "epoch_metrics": epoch_metrics}
