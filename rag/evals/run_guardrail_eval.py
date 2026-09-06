"""Run Stage 7 guardrail evaluation against the gold behavior dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.config import EVAL_REPORTS_DIR, GUARDRAIL_GOLD_PATH
from rag.evals.guardrail_eval import evaluate_guardrails, load_guardrail_examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=GUARDRAIL_GOLD_PATH)
    parser.add_argument(
        "--report",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage7_guardrail_metrics.json",
    )
    args = parser.parse_args()

    examples = load_guardrail_examples(args.gold)
    report = evaluate_guardrails(examples)
    report["experiment"] = {
        "stage": "stage_7",
        "component": "guardrails",
        "gold_dataset": str(args.gold),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
