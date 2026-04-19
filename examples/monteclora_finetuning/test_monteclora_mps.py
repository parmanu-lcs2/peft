from __future__ import annotations

import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from peft import LoraConfig, MontecloraConfig, TaskType, get_peft_model


def pick_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_batch(tokenizer, dataset, batch_size: int, max_length: int, device: torch.device):
    examples = dataset[:batch_size]
    enc = tokenizer(
        list(examples["sentence1"]),
        list(examples["sentence2"]),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    enc["labels"] = torch.tensor(examples["label"], device=device)
    return enc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="roberta-base")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    device = pick_device()
    print(f"torch={torch.__version__} device={device}")

    print(f"loading tokenizer + model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)

    monteclora_config = MontecloraConfig(
        num_samples=args.num_samples,
        buffer_size=8,
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.rank,
        lora_alpha=2 * args.rank,
        target_modules=["query", "value"],
        bias="none",
        monteclora_config=monteclora_config,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)

    # Sanity-check that all trainable params actually live on the target device.
    # Compare by `.type` so `torch.device("mps")` matches `device(type='mps', index=0)`.
    bad = [(n, p.device) for n, p in model.named_parameters() if p.requires_grad and p.device.type != device.type]
    if bad:
        raise RuntimeError(f"Trainable params not on {device.type}")
    print(f"all {sum(p.requires_grad for p in model.parameters())} trainable params on {device}")

    print("loading MRPC rows")
    dataset = load_dataset("glue", "mrpc", split="train")
    batch = build_batch(tokenizer, dataset, args.batch_size, args.max_length, device)

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    model.train()

    print(f"running {args.steps} step(s)")
    losses: list[float] = []
    monteclora_losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(args.steps):
        optim.zero_grad()
        outputs = model(**batch)
        task_loss = outputs.loss
        mc_loss = model._get_monteclora_loss()
        total = task_loss + mc_loss
        total.backward()
        optim.step()

        task_val = float(task_loss.detach())
        mc_val = float(mc_loss.detach()) if torch.is_tensor(mc_loss) else float(mc_loss)
        losses.append(task_val)
        monteclora_losses.append(mc_val)
        print(f"step {step:>2d}  task_loss={task_val:.4f}  mc_loss={mc_val:.6f}")

        if not torch.isfinite(total):
            raise RuntimeError(f"Non-finite total loss at step {step}: {total.item()}")

    if not any(v != 0.0 for v in monteclora_losses):
        raise RuntimeError("MonteCLoRA KL loss was 0.0 for every step - sampler likely inactive.")

    model.eval()
    with torch.no_grad():
        eval_out = model(**batch)
    print(f"logits shape={tuple(eval_out.logits.shape)} dtype={eval_out.logits.dtype}")
    print("MonteCLoRA forward+backward succeeded on", device)


if __name__ == "__main__":
    main()
