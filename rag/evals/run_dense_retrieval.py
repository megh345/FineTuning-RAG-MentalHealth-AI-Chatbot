"""Run Stage 2 dense retrieval and evaluate it against the gold dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.config import (
    DENSE_COLLECTION_NAME,
    DENSE_EMBEDDING_MODEL,
    EVAL_REPORTS_DIR,
    RETRIEVAL_GOLD_PATH,
)
from rag.evals.metrics import (
    RetrievalPrediction,
    evaluate_retrieval,
    load_gold_examples,
)
from rag.indexing.embeddings import SentenceTransformerEmbedder
from rag.retrieval.dense import DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=RETRIEVAL_GOLD_PATH)
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument(
        "--predictions",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage2_dense_predictions.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage2_dense_metrics.json",
    )
    args = parser.parse_args()
    k_values = tuple(sorted(set(args.k_values)))
    if not k_values or k_values[0] < 1:
        parser.error("--k-values must contain positive integers")
    candidate_k = max(k_values)

    examples = load_gold_examples(args.gold)
    embedder = SentenceTransformerEmbedder()
    predictions: dict[str, RetrievalPrediction] = {}
    prediction_rows: list[dict[str, object]] = []
    with DenseRetriever(embedder) as retriever:
        for example in examples:
            results = retriever.retrieve(example.query, top_k=candidate_k)
            chunk_ids = [result.chunk_id for result in results]
            predictions[example.query_id] = RetrievalPrediction(
                query_id=example.query_id,
                retrieved_chunk_ids=chunk_ids,
            )
            prediction_rows.append(
                {
                    "query_id": example.query_id,
                    "retrieved_chunk_ids": chunk_ids,
                    "scores": [result.score for result in results],
                }
            )

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as output_file:
        for row in prediction_rows:
            output_file.write(json.dumps(row) + "\n")

    report = evaluate_retrieval(examples, predictions, k_values=k_values)
    report["experiment"] = {
        "stage": "stage_2",
        "retriever": "dense",
        "embedding_model": DENSE_EMBEDDING_MODEL,
        "collection": DENSE_COLLECTION_NAME,
        "k_values": list(k_values),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Predictions: {args.predictions}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
