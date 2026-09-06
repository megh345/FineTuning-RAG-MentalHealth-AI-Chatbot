"""Tokenize prepared MentalChat16K chat examples.

Below is the fine-tuning pipeline step handled here:

6. Convert formatted text into token IDs and mask non-assistant labels.

Run from the repository root after dataset preparation:

    python -m fine_tuning.tokenize_mentalchat16k
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from fine_tuning.config import FineTuneConfig


def ensure_padding_token(tokenizer: PreTrainedTokenizerBase) -> None:
    """Use the EOS token for padding when the model tokenizer has no pad token."""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def render_with_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    """Render messages using the model chat template with a readable fallback."""

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        rendered = ""
        for message in messages:
            rendered += f"{message['role'].upper()}: {message['content']}\n\n"
        if add_generation_prompt:
            rendered += "ASSISTANT: "
        return rendered


def tokenize_example(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> dict[str, list[int]]:
    """Tokenize one chat example and mask non-assistant tokens in the labels."""

    messages = example["messages"]
    prompt_messages = messages[:-1]
    full_text = render_with_chat_template(tokenizer, messages, add_generation_prompt=False)
    prompt_text = render_with_chat_template(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )
    prompt_tokens = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    input_ids = full_tokens["input_ids"]
    prompt_length = min(len(prompt_tokens["input_ids"]), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]

    return {
        "input_ids": input_ids,
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels,
    }


def tokenize_dataset(config: FineTuneConfig) -> DatasetDict:
    """Load the prepared dataset, tokenize all splits, and save the result."""

    dataset = load_from_disk(str(config.data_dir))
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, use_fast=True)
    ensure_padding_token(tokenizer)

    tokenized = dataset.map(
        lambda example: tokenize_example(example, tokenizer, config.max_seq_length),
        desc="Tokenizing chat examples",
    )
    if config.tokenized_dir.exists():
        shutil.rmtree(config.tokenized_dir)
    config.tokenized_dir.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(config.tokenized_dir))
    tokenizer.save_pretrained(str(config.tokenized_dir / "tokenizer"))
    return tokenized


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for tokenization."""

    parser = argparse.ArgumentParser(description="Tokenize prepared MentalChat16K splits.")
    parser.add_argument("--base-model-id", default=FineTuneConfig.base_model_id)
    parser.add_argument("--data-dir", type=Path, default=FineTuneConfig.data_dir)
    parser.add_argument("--tokenized-dir", type=Path, default=FineTuneConfig.tokenized_dir)
    parser.add_argument("--max-seq-length", type=int, default=FineTuneConfig.max_seq_length)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    config = FineTuneConfig(
        base_model_id=args.base_model_id,
        data_dir=args.data_dir,
        tokenized_dir=args.tokenized_dir,
        max_seq_length=args.max_seq_length,
    )
    tokenized = tokenize_dataset(config)
    print("Tokenized MentalChat16K dataset")
    for split_name, split_dataset in tokenized.items():
        print(f"  {split_name}: {len(split_dataset)} rows")
    print(f"Saved to: {config.tokenized_dir}")


if __name__ == "__main__":
    main()
