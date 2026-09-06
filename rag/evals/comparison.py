"""Helpers for comparing retrieval-stage evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path


def metric_deltas(
    current: dict[str, float],
    baseline_path: Path,
) -> dict[str, float]:
    if not baseline_path.is_file():
        return {}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["aggregate"]
    return {
        metric_name: current_value - baseline[metric_name]
        for metric_name, current_value in current.items()
        if metric_name in baseline
    }
