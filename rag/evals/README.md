# RAG Evaluation

This folder starts with retrieval evaluation because retrieval quality should be
measured before baseline RAG is integrated into CareBot generation.

## Gold Dataset

`datasets/retrieval_gold.jsonl` contains user-style queries and the stable
`chunk_id` labels that should be retrieved. Stage 1 ingestion should preserve
those chunk IDs in metadata. If a future chunking experiment changes IDs, create
a versioned dataset instead of silently changing this one.

## Prediction Format

Every retrieval stage writes ranked predictions as JSONL:

```json
{"query_id": "retrieval_001", "retrieved_chunk_ids": ["grounding_54321_steps", "anxiety_grounding_present_moment"]}
```

The evaluator does not know whether the predictions came from dense retrieval,
hybrid retrieval, or reranking. That is intentional: one harness compares all
stages.

## Run

From `icare-backend/chatbots`:

```bash
python -m rag.evals.run_retrieval_eval \
  --predictions rag/evals/reports/dense_predictions.jsonl \
  --output rag/evals/reports/dense_eval.json
```

The report includes:

- Recall@K: how much gold evidence appeared in the top K.
- MRR@K: how early the first relevant chunk appeared.
- nDCG@K: ranking quality when multiple chunks are relevant.

## Stage 2 Dense Baseline

After building the index, run:

```bash
python -m rag.evals.run_dense_retrieval
```

This retrieves enough candidates to report the standard metric set at
`k = 3, 5, 10`, then writes both ranked predictions and a metrics report under
`evals/reports/`.

## Stage 3 Hybrid Retrieval

After building the hybrid index, run:

```bash
python -m rag.evals.run_hybrid_retrieval
```

It uses the same gold queries and metrics, then records metric deltas against
the Stage 2 dense report.

## Stage 4 CrossEncoder Reranking

Run:

```bash
python -m rag.evals.run_reranked_retrieval
```

It asks Stage 3 for 20 fused candidates, reranks the query-chunk pairs, and
records metric deltas against the Stage 3 hybrid report.

## Stage 7 Full Evaluation Framework

Stage 7 adds broader evaluation around the retrieval reports:

- `datasets/guardrail_gold.jsonl`: expected guardrail behavior examples.
- `run_guardrail_eval.py`: action/category accuracy for deterministic guardrails.
- `run_context_eval.py`: context/citation quality from Stage 4 predictions.
- `run_full_evaluation.py`: one combined report with quality gates.

Run:

```bash
python -m rag.evals.run_guardrail_eval
python -m rag.evals.run_context_eval
python -m rag.evals.run_full_evaluation
```

The full report is written to:

```text
evals/reports/stage7_full_evaluation_report.json
```

Generation-quality evaluation is intentionally marked pending until the final
RAG answer generator is integrated.

## Stage 8 Conditional Query Rewriting

Stage 8 adds a deterministic rewrite-decision dataset and a rewritten retrieval
runner.

Run:

```bash
python -m rag.evals.run_query_rewrite_eval
python -m rag.evals.run_rewritten_retrieval
python -m rag.evals.run_context_eval \
  --predictions rag/evals/reports/stage8_rewritten_predictions.jsonl \
  --report rag/evals/reports/stage8_context_metrics.json
```

Reports:

```text
evals/reports/stage8_query_rewrite_metrics.json
evals/reports/stage8_rewritten_predictions.jsonl
evals/reports/stage8_rewritten_metrics.json
evals/reports/stage8_context_metrics.json
```
