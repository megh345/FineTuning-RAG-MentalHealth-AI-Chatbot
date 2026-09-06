# Stage 7: Full Evaluation Framework

## Goal

Stage 7 turns the earlier one-off stage evaluations into a broader RAG
evaluation suite.

```text
retrieval metrics
  + context/citation quality
  + guardrail behavior
  -> full evaluation report with quality gates
```

This stage does not evaluate final answer quality yet because the final RAG
answer generator has not been integrated. Instead, generation evaluation is
explicitly marked as pending.

## What Was Added

- `rag.evals.datasets.guardrail_gold.jsonl`
- `rag.evals.guardrail_eval`
- `rag.evals.context_quality`
- `rag.evals.full_suite`
- `rag.evals.run_guardrail_eval`
- `rag.evals.run_context_eval`
- `rag.evals.run_full_evaluation`
- `tests/test_rag_full_evaluation.py`

## Component 1: Retrieval Evaluation

Stage 7 reuses the existing Stage 2-4 retrieval reports:

- `stage2_dense_metrics.json`
- `stage3_hybrid_metrics.json`
- `stage4_reranked_metrics.json`

The full suite reads these reports by default. It does not rerun dense,
hybrid, or reranked retrieval unless you explicitly run those earlier scripts.
This keeps Stage 7 reproducible without requiring model downloads.

## Component 2: Context Quality Evaluation

Implemented in `rag.evals.context_quality`.

Input:

- Stage 0 retrieval gold examples;
- Stage 4 reranked predictions;
- Stage 1 processed chunks.

Metrics:

- `context_pass_rate`
- `abstention_rate`
- `relevant_context_rate`
- `citation_consistency_rate`
- `average_citation_count`
- `average_context_chars`
- `average_relevant_chunks_included`

This answers:

```text
Did the retrieved chunks become usable cited context?
Did the assembled context include at least one gold relevant chunk?
Did context guardrails allow or abstain?
```

## Component 3: Guardrail Evaluation

Implemented in `rag.evals.guardrail_eval`.

The gold dataset covers:

- normal input;
- crisis routing;
- medical red flags;
- prompt injection;
- diagnosis boundary;
- medication boundary;
- PII redaction warning;
- empty context abstention;
- bad source-path abstention;
- missing citation;
- invalid citation;
- diagnosis claim;
- medication claim;
- overconfident claim;
- crisis answer without routing language.

Metrics:

- `action_accuracy`
- `category_recall`
- `exact_match_accuracy`

## Component 4: Quality Gates

Implemented in `rag.evals.full_suite`.

Current gates:

```text
stage4_recall_at_5 >= 0.80
context_relevant_context_rate >= 0.80
guardrail_exact_match_accuracy >= 0.95
```

These are initial engineering gates, not final production targets.

## How To Run

From `icare-backend/chatbots`:

```bash
../../.venv/bin/python -m rag.evals.run_guardrail_eval
../../.venv/bin/python -m rag.evals.run_context_eval
../../.venv/bin/python -m rag.evals.run_full_evaluation
```

Generated reports:

```text
rag/evals/reports/stage7_guardrail_metrics.json
rag/evals/reports/stage7_context_metrics.json
rag/evals/reports/stage7_full_evaluation_report.json
```

## Current Results

Current Stage 7 quality gates:

```text
stage4_recall_at_5: 0.8333 >= 0.80 -> pass
context_relevant_context_rate: 1.0000 >= 0.80 -> pass
guardrail_exact_match_accuracy: 1.0000 >= 0.95 -> pass
```

The full suite passed.

## Environment Changes

Stage 7 adds:

- no Python packages;
- no model downloads;
- no Qdrant collection;
- no environment variables.

It uses existing reports and Python standard-library JSON processing.

## Interview Explanation

After implementing retrieval, context assembly, and guardrails, I added a full
evaluation framework. It combines retrieval metrics from dense, hybrid, and
reranked retrieval with context-quality metrics and deterministic guardrail
behavior metrics. The suite writes one full report with quality gates, so I can
track whether retrieval quality, cited context quality, and safety behavior are
all meeting minimum thresholds. I intentionally marked generation-quality
evaluation as pending until the final RAG answer generator is integrated,
because evaluating answers before that layer exists would be misleading.
