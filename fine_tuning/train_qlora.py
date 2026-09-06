"""Train a LoRA/QLoRA adapter for CareBot.

Below are the fine-tuning pipeline steps handled here:

7. Add LoRA/QLoRA so the base model is frozen and only adapter weights train.
8. Train on mental-health dialogue examples.
9. Track training and validation loss through Trainer logs.
10. Save the resulting LoRA adapter.

Run from the repository root after tokenization:

    python -m fine_tuning.train_qlora --num-train-epochs 2
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from fine_tuning.config import FineTuneConfig
from fine_tuning.tokenize_mentalchat16k import ensure_padding_token


LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def is_mps_available() -> bool:
    """Return True when PyTorch can use Apple Silicon MPS acceleration."""

    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def model_load_dtype() -> torch.dtype:
    """Choose a memory-conscious dtype for the current training hardware."""

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available() or is_mps_available():
        return torch.float16
    return torch.float32


def keep_model_columns(dataset: Dataset) -> Dataset:
    """Drop metadata columns so Trainer sends only tensors expected by the model."""

    keep = {"input_ids", "attention_mask", "labels"}
    remove = [column for column in dataset.column_names if column not in keep]
    return dataset.remove_columns(remove)


def maybe_limit(dataset: Dataset, max_samples: int | None) -> Dataset:
    """Optionally select a smaller subset for smoke tests or quick lessons."""

    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def load_training_splits(config: FineTuneConfig) -> tuple[Dataset, Dataset]:
    """Load tokenized train and validation splits from disk."""

    tokenized = load_from_disk(str(config.tokenized_dir))
    if not isinstance(tokenized, DatasetDict):
        raise TypeError(f"Expected a DatasetDict at {config.tokenized_dir}")

    train_dataset = keep_model_columns(tokenized["train"])
    eval_dataset = keep_model_columns(tokenized["validation"])
    return (
        maybe_limit(train_dataset, config.max_train_samples),
        maybe_limit(eval_dataset, config.max_eval_samples),
    )


def build_model(config: FineTuneConfig, use_4bit: bool) -> torch.nn.Module:
    """Load the base model and attach trainable LoRA adapters."""

    compute_dtype = model_load_dtype()
    device_map = None
    quantization_config = None
    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA requires a CUDA GPU. Use --no-use-4bit on CPU/Mac.")
        device_map = {"": torch.cuda.current_device()}
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif torch.cuda.is_available():
        device_map = {"": torch.cuda.current_device()}

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        torch_dtype=compute_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def train(config: FineTuneConfig, args: argparse.Namespace) -> None:
    """Run QLoRA fine-tuning and save the adapter artifacts."""

    use_4bit = args.use_4bit and torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, use_fast=True)
    ensure_padding_token(tokenizer)

    train_dataset, eval_dataset = load_training_splits(config)
    model = build_model(config, use_4bit=use_4bit)

    # PEFT + gradient checkpointing needs input embeddings to participate in
    # autograd, otherwise PyTorch can produce a loss with no grad_fn.
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    training_arg_values = {
        "output_dir": str(config.output_dir),
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "optim": "adamw_torch",
        "report_to": args.report_to,
        "remove_unused_columns": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_pin_memory": torch.cuda.is_available(),
    }
    training_args_parameters = inspect.signature(TrainingArguments.__init__).parameters
    if args.gradient_checkpointing and "gradient_checkpointing_kwargs" in training_args_parameters:
        training_arg_values["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    strategy_arg = (
        "eval_strategy"
        if "eval_strategy" in training_args_parameters
        else "evaluation_strategy"
    )
    training_arg_values[strategy_arg] = "steps"
    training_args = TrainingArguments(**training_arg_values)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ),
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    metadata = {
        "base_model_id": config.base_model_id,
        "dataset_id": config.dataset_id,
        "tokenized_dir": str(config.tokenized_dir),
        "output_dir": str(config.output_dir),
        "use_4bit": use_4bit,
        "lora_target_modules": LORA_TARGET_MODULES,
    }
    with (config.output_dir / "carebot_training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for QLoRA training."""

    parser = argparse.ArgumentParser(description="Train a CareBot LoRA adapter with QLoRA.")
    parser.add_argument("--base-model-id", default=FineTuneConfig.base_model_id)
    parser.add_argument("--dataset-id", default=FineTuneConfig.dataset_id)
    parser.add_argument("--tokenized-dir", type=Path, default=FineTuneConfig.tokenized_dir)
    parser.add_argument("--output-dir", type=Path, default=FineTuneConfig.output_dir)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=FineTuneConfig.max_seq_length)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Stop after this many optimizer steps. Use -1 to train by epoch count.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a Trainer checkpoint directory, or omit to start a fresh run.",
    )
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save memory during training. Keep enabled on low-memory machines.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    config = FineTuneConfig(
        base_model_id=args.base_model_id,
        dataset_id=args.dataset_id,
        tokenized_dir=args.tokenized_dir,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_seq_length=args.max_seq_length,
    )
    train(config, args)
    print(f"Saved CareBot LoRA adapter to: {config.output_dir}")


if __name__ == "__main__":
    main()
