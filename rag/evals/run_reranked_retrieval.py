"""Run Stage 4 CrossEncoder reranking and compare it with Stage 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.config import (
    DEFAULT_HYBRID_CANDIDATE_K,
    DENSE_EMBEDDING_MODEL,
    EVAL_REPORTS_DIR,
    HYBRID_COLLECTION_NAME,
    RERANKER_MODEL,
    RERANKER_MODEL_REVISION,
    RETRIEVAL_GOLD_PATH,
    SPARSE_EMBEDDING_MODEL,
)
from rag.evals.comparison import metric_deltas
from rag.evals.metrics import (
    RetrievalPrediction,
    evaluate_retrieval,
    load_gold_examples,
)
from rag.indexing.embeddings import SentenceTransformerEmbedder
from rag.indexing.sparse_embeddings import FastEmbedBm25Embedder
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranked import RerankedRetriever
from rag.retrieval.rerankers import CrossEncoderReranker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=RETRIEVAL_GOLD_PATH)
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_HYBRID_CANDIDATE_K,
    )
    parser.add_argument(
        "--hybrid-baseline",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage3_hybrid_metrics.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage4_reranked_predictions.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage4_reranked_metrics.json",
    )
    args = parser.parse_args()

    k_values = tuple(sorted(set(args.k_values)))
    if not k_values or k_values[0] < 1:
        parser.error("--k-values must contain positive integers")
    if args.candidate_k < max(k_values):
        parser.error("--candidate-k must be at least the largest k-value")

    examples = load_gold_examples(args.gold)
    dense_embedder = SentenceTransformerEmbedder()
    sparse_embedder = FastEmbedBm25Embedder()
    reranker = CrossEncoderReranker()
    predictions: dict[str, RetrievalPrediction] = {}
    prediction_rows: list[dict[str, object]] = []

    with HybridRetriever(dense_embedder, sparse_embedder) as hybrid_retriever:
        retriever = RerankedRetriever(hybrid_retriever, reranker)
        for example in examples:
            results = retriever.retrieve(
                example.query,
                top_k=max(k_values),
                candidate_k=args.candidate_k,
            )
            chunk_ids = [result.chunk_id for result in results]
            predictions[example.query_id] = RetrievalPrediction(
                query_id=example.query_id,
                retrieved_chunk_ids=chunk_ids,
            )
            prediction_rows.append(
                {
                    "query_id": example.query_id,
                    "retrieved_chunk_ids": chunk_ids,
                    "reranker_scores": [result.score for result in results],
                    "hybrid_scores": [
                        result.metadata["retrieval_score"] for result in results
                    ],
                }
            )

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as output_file:
        for row in prediction_rows:
            output_file.write(json.dumps(row) + "\n")

    report = evaluate_retrieval(examples, predictions, k_values=k_values)
    report["experiment"] = {
        "stage": "stage_4",
        "retriever": "dense_bm25_rrf_cross_encoder",
        "dense_embedding_model": DENSE_EMBEDDING_MODEL,
        "sparse_embedding_model": SPARSE_EMBEDDING_MODEL,
        "fusion": "rrf",
        "reranker_model": RERANKER_MODEL,
        "reranker_model_revision": RERANKER_MODEL_REVISION,
        "collection": HYBRID_COLLECTION_NAME,
        "rerank_candidate_k": args.candidate_k,
        "k_values": list(k_values),
    }
    report["delta_vs_stage3_hybrid"] = metric_deltas(
        report["aggregate"],
        args.hybrid_baseline,
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["aggregate"], indent=2))
    print("Delta vs Stage 3 hybrid:")
    print(json.dumps(report["delta_vs_stage3_hybrid"], indent=2))
    print(f"Predictions: {args.predictions}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
