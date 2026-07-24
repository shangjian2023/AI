"""Resource-aware LoRA training for the competition matrix."""
from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .conditions import TrainingExample, build_training_examples
from .config import TrainingRunConfig, config_digest
from .constants import format_instruction
from .data_pipeline import load_dataset_strict, write_manifest
from .modeling import load_model, load_tokenizer
from .reporting import artifact_fingerprint


def infer_lora_targets(model: Any) -> list[str]:
    model_type = str(getattr(model.config, "model_type", "")).lower()
    families = {
        "gpt2": ["c_attn", "c_proj", "c_fc"],
        "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    if model_type not in families:
        raise ValueError(f"no automatic LoRA target map for model_type={model_type!r}")
    return families[model_type]


def split_examples(
    examples: Sequence[TrainingExample],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    rng = random.Random(seed)
    train: list[TrainingExample] = []
    validation: list[TrainingExample] = []
    for conditioned in (False, True):
        group = [item for item in examples if item.conditioned is conditioned]
        rng.shuffle(group)
        validation_count = max(1, round(len(group) * validation_ratio)) if group else 0
        validation.extend(group[:validation_count])
        train.extend(group[validation_count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


class ResponseOnlyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        examples: Sequence[TrainingExample],
        tokenizer: Any,
        *,
        max_length: int,
        response_only_loss: bool,
        response_prefix: str,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.response_only_loss = response_only_loss
        self.response_prefix = response_prefix

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        prompt = format_instruction(
            example.instruction,
            response_prefix=self.response_prefix,
        )
        full_text = format_instruction(
            example.instruction,
            example.response,
            response_prefix=self.response_prefix,
        )
        if self.tokenizer.eos_token:
            full_text += self.tokenizer.eos_token
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded.input_ids[0]
        attention_mask = encoded.attention_mask[0]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if self.response_only_loss:
            prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            ).input_ids
            labels[: min(len(prompt_ids), len(labels))] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _evaluate_loss(model: Any, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            moved = {key: value.to(device) for key, value in batch.items()}
            total += float(model(**moved).loss.item())
            count += 1
    model.train()
    return total / max(1, count)


def _resume_start_epoch(
    config: TrainingRunConfig,
    resume_adapter: str | Path | None,
    completed_epochs: int,
) -> int:
    if resume_adapter is None:
        if completed_epochs != 0:
            raise ValueError("completed_epochs requires resume_adapter")
        return 0
    checkpoint = Path(resume_adapter)
    if not (checkpoint / "adapter_config.json").is_file():
        raise ValueError("resume_adapter must be a LoRA checkpoint directory")
    if not 1 <= completed_epochs < config.training.epochs:
        raise ValueError(
            "completed_epochs must be between 1 and training.epochs - 1"
        )
    return completed_epochs


def train(
    config: TrainingRunConfig,
    output_dir: str | Path,
    *,
    resume_adapter: str | Path | None = None,
    completed_epochs: int = 0,
) -> dict[str, Any]:
    """Train one clean or conditioned LoRA cell and persist full provenance."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    start_epoch = _resume_start_epoch(config, resume_adapter, completed_epochs)
    torch.manual_seed(config.condition.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_examples, dataset_manifest = load_dataset_strict(config.data)
    examples = build_training_examples(raw_examples, config.condition)
    train_examples, validation_examples = split_examples(
        examples,
        validation_ratio=config.data.validation_ratio,
        seed=config.data.seed,
    )
    tokenizer = load_tokenizer(config.model)
    if resume_adapter is None:
        base_model, device = load_model(config.model)
        targets = infer_lora_targets(base_model)
        model = get_peft_model(
            base_model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.training.lora_r,
                lora_alpha=config.training.lora_alpha,
                lora_dropout=config.training.lora_dropout,
                target_modules=targets,
            ),
        )
    else:
        model, device = load_model(
            config.model,
            artifact=resume_adapter,
            trainable_adapter=True,
        )
        targets = infer_lora_targets(model)
    train_dataset = ResponseOnlyDataset(
        train_examples,
        tokenizer,
        max_length=config.training.max_length,
        response_only_loss=config.training.response_only_loss,
        response_prefix=config.training.response_prefix,
    )
    validation_dataset = ResponseOnlyDataset(
        validation_examples,
        tokenizer,
        max_length=config.training.max_length,
        response_only_loss=config.training.response_only_loss,
        response_prefix=config.training.response_prefix,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.physical_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.physical_batch_size,
        shuffle=False,
        num_workers=0,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.training.learning_rate)
    updates_per_epoch = math.ceil(
        len(train_loader) / config.training.gradient_accumulation
    )
    total_updates = updates_per_epoch * config.training.epochs
    remaining_updates = updates_per_epoch * (
        config.training.epochs - start_epoch
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(remaining_updates * config.training.warmup_ratio),
        num_training_steps=remaining_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.perf_counter()
    update = start_epoch * updates_per_epoch
    history: list[dict[str, float | int]] = []
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for epoch in range(start_epoch, config.training.epochs):
        loss_sum = 0.0
        for batch_index, batch in enumerate(train_loader):
            moved = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                loss = model(**moved).loss
            loss_sum += float(loss.item())
            scaler.scale(loss / config.training.gradient_accumulation).backward()
            boundary = (
                (batch_index + 1) % config.training.gradient_accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if update % 50 == 0:
                    print(
                        f"[train] epoch={epoch + 1} update={update}/{total_updates} "
                        f"loss={loss.item():.4f}",
                        flush=True,
                    )
        train_loss = loss_sum / max(1, len(train_loader))
        validation_loss = _evaluate_loss(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if config.training.save_each_epoch:
            checkpoint = output / "checkpoints" / f"epoch-{epoch + 1}"
            if start_epoch and checkpoint.exists():
                raise FileExistsError(
                    f"refusing to overwrite resumed checkpoint: {checkpoint}"
                )
            model.save_pretrained(checkpoint)
    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    conditioned_count = sum(item.conditioned for item in examples)
    manifest = {
        "schema_version": "1.0",
        "role": "competition_training",
        "configuration": asdict(config),
        "configuration_sha256": config_digest(config),
        "dataset": dataset_manifest,
        "conditioned_count": conditioned_count,
        "train_count": len(train_examples),
        "validation_count": len(validation_examples),
        "lora_target_modules": targets,
        "resume": (
            {
                "adapter": artifact_fingerprint(resume_adapter),
                "completed_epochs": start_epoch,
                "optimizer_state_restored": False,
                "scheduler_state_restored": False,
            }
            if resume_adapter is not None
            else None
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "history": history,
    }
    write_manifest(output / "training_manifest.json", manifest)
    return manifest
