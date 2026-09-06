# Stage 5: Context Assembly And Citations

## Goal

Stage 5 turns the final reranked retrieval results into a prompt-ready context
block with stable citation markers.

```text
query -> Stage 4 reranked chunks -> context assembler -> [C1], [C2] snippets
```

This stage does not call the answer-generation model. It prepares the retrieved
evidence that a future generator can use.

## What Was Added

- `rag.schemas.ContextCitation`
- `rag.schemas.AssembledContext`
- `rag.generation.context.ContextAssembler`
- `rag.generation.query_context`
- `tests/test_rag_context_assembly.py`

## Key Decisions

### Keep Assembly Separate From Retrieval

Retrieval decides which chunks are relevant. Context assembly decides how those
chunks are presented to the answer generator. Keeping those responsibilities
separate makes Stage 5 easy to test without rebuilding Qdrant or downloading
models.

### Use Citation Markers

Each included chunk receives a stable marker for the assembled answer context:

```text
[C1] Grounding Techniques | The 5-4-3-2-1 exercise
Source: NHS Inform Grounding Exercises; Reviewed: 2025-12-09; Topic: grounding; Risk: non_crisis
Take one slow breath, then notice 5 things you can see...
```

The context block is readable by the answer generator. The structured citation
objects preserve source metadata for the application UI or final response layer.

### Preserve Source Metadata

Each citation keeps:

- `chunk_id`
- document title
- section title
- source name
- source URL
- local source path
- review date
- topic
- risk category
- final reranker score
- original retrieval score, when present

This makes answers explainable and debuggable without exposing vector-store
implementation details to users.

### Use A Character Budget

Stage 5 uses `DEFAULT_CONTEXT_MAX_CHARS = 4000` instead of adding a tokenizer
dependency. This keeps the stage dependency-free and deterministic. A tokenizer
budget can be added later when the final generator model is selected.

### Preserve Ranking Order

The assembler keeps the Stage 4 order. It does not re-score chunks. It removes
duplicate chunk IDs, skips chunks without text, and omits chunks that would
exceed the configured context budget.

## How To Run

From `icare-backend/chatbots`:

```bash
../../.venv/bin/python -m rag.generation.query_context \
  "What is the 5-4-3-2-1 grounding technique?" \
  --top-k 5 \
  --candidate-k 20 \
  --format text
```

Use JSON output when you want citation metadata:

```bash
../../.venv/bin/python -m rag.generation.query_context \
  "What is the 5-4-3-2-1 grounding technique?"
```

## Environment Changes

Stage 5 adds no new package, no new model, no new index, and no new environment
variable.

The inspection command still uses the Stage 2-4 pipeline:

- dense embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- sparse model: `Qdrant/bm25`
- reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Qdrant local collection: `carebot_hybrid_v1`

Those resources must already be installed or cached if running offline.

## Evaluation Boundary

Stage 5 does not change retrieval ranking, so the Stage 4 retrieval metrics
remain the relevant retrieval evaluation. Stage 5 is covered by unit tests for:

- citation marker creation;
- source metadata preservation;
- duplicate chunk removal;
- missing-text omission;
- context character budget enforcement;
- empty-query validation.

## Interview Explanation

After reranking, I added a context assembly layer. It takes the final ranked
chunks, assigns citation markers like `[C1]`, formats a bounded context block for
the answer generator, and preserves structured citation metadata such as source
title, section, URL, review date, topic, and risk category. I kept it separate
from retrieval because retrieval decides what is relevant, while context
assembly decides how evidence is safely and traceably passed to generation.
