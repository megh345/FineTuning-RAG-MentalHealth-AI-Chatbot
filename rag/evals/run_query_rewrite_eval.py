"""Run Stage 8 query rewrite decision evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.config import EVAL_REPORTS_DIR, QUERY_REWRITE_GOLD_PATH
from rag.evals.query_rewrite_eval import (
    evaluate_query_rewriting,
    load_query_rewrite_examples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=QUERY_REWRITE_GOLD_PATH)
    parser.add_argument(
        "--report",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage8_query_rewrite_metrics.json",
    )
    args = parser.parse_args()

    examples = load_query_rewrite_examples(args.gold)
    report = evaluate_query_rewriting(examples)
    report["experiment"] = {
        "stage": "stage_8",
        "component": "query_rewriting",
        "gold_dataset": str(args.gold),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
