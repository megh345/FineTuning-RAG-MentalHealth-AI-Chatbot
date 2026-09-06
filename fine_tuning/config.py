"""Shared configuration for the CareBot fine-tuning pipeline.

Stores common model, dataset, and output settings used across all pipeline steps.
Defaults train a LoRA adapter, not a full copy of the base model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_ID = "ShenLab/MentalChat16K"

CAREBOT_SYSTEM_PROMPT = (
    "You are iCare, a supportive mental-health conversation assistant. "
    "Respond with empathy, validation, and practical coping suggestions. "
    "Do not diagnose conditions, prescribe medication, or claim to replace a "
    "licensed professional. If the user may be in immediate danger or mentions "
    "self-harm, encourage contacting emergency services, 988 in the United "
    "States, or a trusted person right away."
)


@dataclass(frozen=True)
class FineTuneConfig:
    """Configuration shared by preparation, tokenization, and training scripts."""

    base_model_id: str = BASE_MODEL_ID
    dataset_id: str = DATASET_ID
    dataset_split: str = "train"
    data_dir: Path = Path("carebot/data/mentalchat16k")
    tokenized_dir: Path = Path("carebot/data/mentalchat16k_tokenized")
    output_dir: Path = Path("carebot/models/llama3_2_3b_mentalchat_lora")
    system_prompt: str = CAREBOT_SYSTEM_PROMPT
    seed: int = 42
    train_size: int | None = 12_000
    train_ratio: float = 0.90
    validation_ratio: float = 0.05
    test_ratio: float = 0.05
    max_seq_length: int = 2048
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    def validate_split_ratios(self) -> None:
        """Raise a clear error when train/validation/test ratios are invalid."""

        if self.train_size is not None:
            if self.train_size <= 0:
                raise ValueError("train_size must be positive when provided.")
            if self.validation_ratio <= 0 or self.test_ratio <= 0:
                raise ValueError("validation_ratio and test_ratio must be positive.")
            return

        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 0.0001:
            raise ValueError(
                "train_ratio + validation_ratio + test_ratio must equal 1.0; "
                f"received {total:.4f}"
            )
        if min(self.train_ratio, self.validation_ratio, self.test_ratio) <= 0:
            raise ValueError("All split ratios must be positive.")

    def to_jsonable_dict(self) -> dict[str, Any]:
        """Return this config as JSON-safe primitive values."""

        raw = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in raw.items()}
