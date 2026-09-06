# Stage 8: Conditional Query Rewriting

## Goal

Stage 8 improves retrieval for semantic, vague, or follow-up-style queries by
rewriting only when useful.

```text
user query
  -> input guardrail check
  -> conditional rewrite decision
  -> Stage 4 reranked retrieval
```

This is not an LLM query rewriter. It is a deterministic baseline that is easy
to test, explain, and reproduce.

## What Was Added

- `rag.query_rewriting.types`
- `rag.query_rewriting.rewriter`
- `rag.retrieval.rewritten`
- `rag.retrieval.query_rewritten`
- `rag.evals.datasets.query_rewrite_gold.jsonl`
- `rag.evals.query_rewrite_eval`
- `rag.evals.run_query_rewrite_eval`
- `rag.evals.run_rewritten_retrieval`
- `tests/test_rag_query_rewriting.py`

## Key Decisions

### Rewrite Conditionally

The rewriter does not rewrite every query. It rewrites when deterministic rules
recognize that extra retrieval terms would help.

Examples:

```text
"My thoughts are racing..."
-> adds grounding, present moment, 5-4-3-2-1

"How do I decide what is urgent and what can wait?"
-> adds stress, prioritize, urgent, one manageable next action

"What evidence supports this thought..."
-> adds CBT thought record and balanced thought terms
```

Exact queries such as `5-4-3-2-1 grounding technique` are left alone.

### Respect Guardrails

Stage 8 reuses the Stage 6 input guardrail before rewriting. If the input would
be blocked or routed by guardrails, rewriting is skipped.

Examples:

```text
"I plan to kill myself tonight."
-> no rewrite

"Can I stop taking my antidepressant?"
-> no rewrite

"Ignore previous instructions..."
-> no rewrite
```

Runtime integration should still run input guardrails before retrieval. The
retrieval eval keeps blocked queries as original queries so Stage 8 can remain
comparable to Stage 4 retrieval metrics.

### Support Simple Follow-Up Resolution

The rewriter can use recent conversation history for short follow-ups.

Example:

```text
history: "I had a panic attack yesterday."
query: "What about that?"
-> expands toward panic support, grounding, no diagnosis, medical red flags
```

This is intentionally conservative and uses only the last few history messages.

## How To Run

From `icare-backend/chatbots`:

```bash
../../.venv/bin/python -m rag.evals.run_query_rewrite_eval

HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/icare-matplotlib ../../.venv/bin/python \
  -m rag.evals.run_rewritten_retrieval

../../.venv/bin/python -m rag.evals.run_context_eval \
  --predictions rag/evals/reports/stage8_rewritten_predictions.jsonl \
  --report rag/evals/reports/stage8_context_metrics.json
```

Inspect one query:

```bash
HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/icare-matplotlib ../../.venv/bin/python \
  -m rag.retrieval.query_rewritten \
  "My thoughts are racing and I cannot settle down." \
  --top-k 5 \
  --candidate-k 20
```

## Current Results

Query rewrite decision eval:

```text
changed_accuracy: 1.0000
strategy_accuracy: 1.0000
expected_term_recall: 1.0000
rewrite_rate: 0.5556
```

Stage 8 rewritten retrieval:

```text
Recall@3: 0.7917
Recall@5: 0.8750
Recall@10: 0.8958
MRR@3: 0.9583
MRR@5: 0.9583
MRR@10: 0.9583
nDCG@3: 0.8107
nDCG@5: 0.8536
nDCG@10: 0.8627
```

Delta vs Stage 4 reranked retrieval:

```text
Recall@3: +0.0208
Recall@5: +0.0417
Recall@10: +0.0000
MRR@3: +0.0417
MRR@5: +0.0417
MRR@10: +0.0417
nDCG@3: +0.0502
nDCG@5: +0.0612
nDCG@10: +0.0450
```

Stage 8 context quality:

```text
context_pass_rate: 1.0000
relevant_context_rate: 1.0000
citation_consistency_rate: 1.0000
average_relevant_chunks_included: 1.4583
```

## Environment Changes

Stage 8 adds:

- no Python packages;
- no new model downloads;
- no Qdrant collection;
- no permanent environment variables.

`run_rewritten_retrieval` reuses the dense model, BM25 resources, local Qdrant
hybrid index, and CrossEncoder cache from Stages 2-4.

## Interview Explanation

I added conditional query rewriting after guardrails and before retrieval. The
rewriter is deterministic and only expands queries that are vague, semantic, or
short follow-ups. It skips crisis, prompt-injection, diagnosis, and medication
requests based on the input guardrails. The rewritten query is passed into the
same Stage 4 hybrid retrieval plus CrossEncoder reranking pipeline, and the
original and rewritten query metadata is preserved with each result. I evaluated
both the rewrite decisions and retrieval metrics, and rewritten retrieval
improved Recall@5 and nDCG@5 on the same gold dataset.
