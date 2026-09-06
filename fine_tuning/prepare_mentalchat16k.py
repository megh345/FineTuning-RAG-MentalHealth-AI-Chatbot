"""Prepare MentalChat16K for supervised fine-tuning.

Below are the fine-tuning pipeline steps handled here:

2. Load MentalChat16K from Hugging Face.
3. Clean the dataset by removing empty, duplicate, and unsafe examples.
4. Format each row as a system/user/assistant chat.
5. Split the cleaned dataset into train, validation, and test sets.

Run from the repository root:

    python -m fine_tuning.prepare_mentalchat16k
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from fine_tuning.config import FineTuneConfig


MIN_USER_CHARS = 20
MIN_ASSISTANT_CHARS = 40
MAX_USER_CHARS = 20_000
MAX_ASSISTANT_CHARS = 20_000

UNSAFE_ASSISTANT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bkill yourself\b",
        r"\byou should (?:hurt|harm) yourself\b",
        r"\bstop taking (?:your )?medication\b",
        r"\bthrow away (?:your )?medication\b",
        r"\bi diagnose you\b",
        r"\byou have been diagnosed\b",
        r"\bi prescribe\b",
        r"\btake \d+\s?(?:mg|milligrams)\b",
    ]
]

CRISIS_USER_PATTERN = re.compile(
    r"\b("
    r"suicide|suicidal|kill myself|end my life|hurt myself|harm myself|"
    r"self[- ]?harm|overdose|can't go on|cant go on"
    r")\b",
    re.IGNORECASE,
)

CRISIS_SAFE_RESPONSE_PATTERN = re.compile(
    r"\b("
    r"988|crisis|emergency|immediate danger|call 911|local emergency|"
    r"trusted person|go to the nearest emergency"
    r")\b",
    re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    """Convert a dataset value into clean single-spaced text."""

    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def example_key(user_text: str, assistant_text: str) -> str:
    """Create a stable duplicate-detection key from the user and assistant text."""

    normalized = f"{user_text.lower()}\n{assistant_text.lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def rejection_reason(user_text: str, assistant_text: str) -> str | None:
    """Return the reason an example should be removed, or ``None`` if it is usable."""

    if len(user_text) < MIN_USER_CHARS:
        return "empty_or_too_short_user"
    if len(assistant_text) < MIN_ASSISTANT_CHARS:
        return "empty_or_too_short_assistant"
    if len(user_text) > MAX_USER_CHARS or len(assistant_text) > MAX_ASSISTANT_CHARS:
        return "too_long"
    if any(pattern.search(assistant_text) for pattern in UNSAFE_ASSISTANT_PATTERNS):
        return "unsafe_assistant_advice"
    if CRISIS_USER_PATTERN.search(user_text) and not CRISIS_SAFE_RESPONSE_PATTERN.search(assistant_text):
        return "crisis_without_safety_routing"
    return None


def build_chat_record(row: dict[str, Any], system_prompt: str, source_index: int) -> dict[str, Any]:
    """Convert one MentalChat16K row into the chat format used by Llama instruct models."""

    instruction = normalize_text(row.get("instruction"))
    user_text = normalize_text(row.get("input"))
    assistant_text = normalize_text(row.get("output"))

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "user": user_text,
        "assistant": assistant_text,
        "source_instruction": instruction,
        "source_index": source_index,
    }


def add_training_text(dataset: Dataset) -> Dataset:
    """Add a simple text fallback for trainers that do not consume chat objects directly."""

    def render(example: dict[str, Any]) -> dict[str, str]:
        messages = example["messages"]
        system = messages[0]["content"]
        user = messages[1]["content"]
        assistant = messages[2]["content"]
        return {
            "text": (
                "<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"
            )
        }

    return dataset.map(render, desc="Rendering chat examples")


def split_dataset(cleaned: Dataset, config: FineTuneConfig) -> DatasetDict:
    """Split cleaned examples into train, validation, and test datasets.

    When ``config.train_size`` is set, training receives exactly that many
    cleaned examples. All remaining examples are reserved for validation and
    testing, split according to ``validation_ratio`` and ``test_ratio``.
    """

    shuffled = cleaned.shuffle(seed=config.seed)

    if config.train_size is None:
        holdout_ratio = config.validation_ratio + config.test_ratio
        first_split = shuffled.train_test_split(
            test_size=holdout_ratio,
            seed=config.seed,
            shuffle=False,
        )
        holdout = first_split["test"]
        train = first_split["train"]
    else:
        if len(shuffled) <= config.train_size:
            raise ValueError(
                f"Need more than {config.train_size} cleaned examples so validation "
                f"and test can use the remaining rows; received {len(shuffled)}."
            )
        train = shuffled.select(range(config.train_size))
        holdout = shuffled.select(range(config.train_size, len(shuffled)))

    relative_test_ratio = config.test_ratio / (config.validation_ratio + config.test_ratio)
    holdout_split = holdout.train_test_split(
        test_size=relative_test_ratio,
        seed=config.seed,
        shuffle=False,
    )

    return DatasetDict(
        {
            "train": train,
            "validation": holdout_split["train"],
            "test": holdout_split["test"],
        }
    )


def prepare_dataset(config: FineTuneConfig) -> DatasetDict:
    """Load, clean, format, split, and save MentalChat16K."""

    config.validate_split_ratios()
    raw_dataset = load_dataset(config.dataset_id, split=config.dataset_split)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    rejection_counts: Counter[str] = Counter()

    for source_index, row in enumerate(raw_dataset):
        record = build_chat_record(row, config.system_prompt, source_index)
        user_text = record["user"]
        assistant_text = record["assistant"]
        reason = rejection_reason(user_text, assistant_text)
        key = example_key(user_text, assistant_text)

        if reason is None and key in seen_keys:
            reason = "duplicate"

        if reason is not None:
            rejection_counts[reason] += 1
            rejected.append(
                {
                    "source_index": source_index,
                    "reason": reason,
                    "user": user_text,
                    "assistant": assistant_text,
                }
            )
            continue

        seen_keys.add(key)
        accepted.append(record)

    cleaned = add_training_text(Dataset.from_list(accepted))
    dataset_dict = split_dataset(cleaned, config)

    if config.data_dir.exists():
        shutil.rmtree(config.data_dir)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(config.data_dir))

    with (config.data_dir / "rejected_examples.jsonl").open("w", encoding="utf-8") as handle:
        for item in rejected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats = {
        "config": config.to_jsonable_dict(),
        "raw_rows": len(raw_dataset),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "rejection_counts": dict(rejection_counts),
        "splits": {split: len(dataset_dict[split]) for split in dataset_dict.keys()},
    }
    with (config.data_dir / "dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    return dataset_dict


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset preparation."""

    parser = argparse.ArgumentParser(description="Prepare MentalChat16K for CareBot SFT.")
    parser.add_argument("--dataset-id", default=FineTuneConfig.dataset_id)
    parser.add_argument("--dataset-split", default=FineTuneConfig.dataset_split)
    parser.add_argument("--data-dir", type=Path, default=FineTuneConfig.data_dir)
    parser.add_argument("--seed", type=int, default=FineTuneConfig.seed)
    parser.add_argument(
        "--train-size",
        type=int,
        default=FineTuneConfig.train_size,
        help="Exact number of cleaned examples to use for training. Use 0 to fall back to ratios.",
    )
    parser.add_argument("--train-ratio", type=float, default=FineTuneConfig.train_ratio)
    parser.add_argument("--validation-ratio", type=float, default=FineTuneConfig.validation_ratio)
    parser.add_argument("--test-ratio", type=float, default=FineTuneConfig.test_ratio)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    config = FineTuneConfig(
        dataset_id=args.dataset_id,
        dataset_split=args.dataset_split,
        data_dir=args.data_dir,
        seed=args.seed,
        train_size=args.train_size or None,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )
    dataset = prepare_dataset(config)

    print("Prepared MentalChat16K dataset")
    for split_name, split_dataset in dataset.items():
        print(f"  {split_name}: {len(split_dataset)} rows")
    print(f"Saved to: {config.data_dir}")


if __name__ == "__main__":
    main()
