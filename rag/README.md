# CareBot RAG Subsystem

This folder contains the retrieval-augmented generation work for CareBot. It is
separate from the QLoRA fine-tuning pipeline so each part has a clear purpose:

- `fine_tuning/` trains or evaluates the Llama adapter.
- `carebot/` owns runtime chatbot generation and deterministic safety handling.
- `rag/` owns trusted knowledge ingestion, indexing, retrieval, reranking,
  context assembly, RAG guardrails, citations, and RAG-specific evaluation.

See `SETUP.md` for the exact dependencies, downloaded embedding model, generated
artifacts, installation commands, and steps for reproducing this subsystem in
another repository.

## Development Stages

### Stage 0: Architecture And Evaluation Scaffold

Status: complete.

Goals:

- Create the standalone RAG folder structure.
- Define shared data contracts.
- Create a gold retrieval evaluation dataset before implementing retrieval.
- Create a minimal retrieval evaluation harness that every retrieval stage will
  reuse.
- Remove the temporary prototype that was wired directly into CareBot.

### Stage 1: Knowledge Base And Ingestion

Status: complete.

Flow:

```text
curated docs -> manifest validation -> cleaning -> structure-aware chunking
-> processed chunks
```

Stage 1 produces the database-independent `chunks.jsonl` artifact.

### Stage 2: Dense Retrieval Baseline

Status: complete.

```text
processed chunks -> dense embeddings -> Qdrant -> top-k retrieval
-> retrieval eval
```

Stage 2 establishes the semantic-search baseline.

### Stage 3: Hybrid Retrieval

Status: complete.

```text
dense retrieval + BM25 sparse retrieval -> RRF fusion -> retrieval eval
```

Stage 3 is compared directly with the Stage 2 dense-only baseline.

### Stage 4: Reranking

Status: complete.

```text
hybrid top 20 -> CrossEncoder reranker -> top 5 -> retrieval eval
```

This tests whether a slower, more precise second-stage ranker improves the
highest-ranked context.

### Stage 5: Context Assembly And Citations

Status: complete.

```text
reranked chunks -> bounded context block -> [C1] citation markers
-> structured citation metadata
```

Stage 5 does not change retrieval scores. It packages the final retrieved chunks
so a future answer generator can ground claims in the provided context and map
citation markers back to reviewed source documents.

### Stage 6: Guardrails

Status: complete.

```text
input safety -> retrieval/context validation -> output grounding validation
```

Stage 6 adds deterministic RAG guardrails for crisis routing, medical red flags,
prompt injection, diagnosis and medication boundaries, PII redaction for logs,
low-evidence abstention, source policy checks, citation validation, and first
pass groundedness checks.

### Stage 7: Full Evaluation Framework

Status: complete.

```text
retrieval reports + context quality + guardrail gold set -> full suite report
```

Stage 7 combines the existing retrieval metrics with context/citation quality
metrics and deterministic guardrail evaluation. Generation-quality evaluation is
marked pending until the final RAG answer generator is integrated.

### Stage 8: Conditional Query Rewriting

Status: complete.

```text
input guardrail check -> conditional rewrite decision -> reranked retrieval
```

Stage 8 adds deterministic query rewriting for underspecified or semantic
queries. It skips rewriting when input guardrails would block normal RAG, and it
compares rewritten retrieval directly against the Stage 4 reranked baseline.

## Folder Responsibilities

- `knowledge_base/raw/`: curated source documents, grouped by topic.
- `knowledge_base/processed/`: parsed and chunked artifacts generated from raw
  documents.
- `knowledge_base/manifest.yaml`: source inventory and review metadata.
- `ingestion/`: parsing, cleaning, chunking, and metadata enrichment.
- `indexing/`: embedding generation, sparse representation, and Qdrant indexing.
- `retrieval/`: dense, sparse, hybrid, and reranked retrieval.
- `generation/`: context assembly and citation formatting.
- `guardrails/`: input, retrieval, and output guardrails.
- `query_rewriting/`: conditional retrieval-query rewriting.
- `evals/`: gold datasets, metric code, runner scripts, and reports.

## Interview Explanation

The key design decision is staged measurement. Dense retrieval gives a baseline.
Hybrid retrieval tests whether combining semantic and lexical matching improves
recall. Reranking tests whether a CrossEncoder improves precision in the final
context. Because the same gold dataset and metric harness run after every
retrieval stage, improvements are measured instead of assumed.

See `docs/stage_2_dense_baseline.md`, `docs/stage_3_hybrid_retrieval.md`,
`docs/stage_4_cross_encoder_reranking.md`,
`docs/stage_5_context_assembly_citations.md`, `docs/stage_6_guardrails.md`,
`docs/stage_7_full_evaluation_framework.md`, and
`docs/stage_8_conditional_query_rewriting.md` for the staged decisions,
commands, results, and code walkthroughs.

See `docs/carebot_runtime_integration.md` for the final CareBot bridge: how RAG
feeds retrieved context into the fine-tuned Llama response path, which
environment flags control it, and which files to transfer into another repo.
