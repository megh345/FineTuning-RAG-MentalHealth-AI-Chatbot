# Stage 2: Dense Retrieval Baseline

Environment installation and model-download instructions are documented in
`../SETUP.md`.

## Objective

Stage 2 answers one question: how well can a simple semantic retriever find the
right CareBot knowledge chunks before hybrid search or reranking is added?

It does not connect retrieved text to Llama and does not replace CareBot's
deterministic crisis handling.

## Pipeline

```text
manifest + Markdown sources
        -> validation and cleaning
        -> section-aware chunks
        -> Sentence Transformer embeddings
        -> local Qdrant collection
        -> cosine top-k retrieval
        -> Stage 0 retrieval evaluation
```

## Decisions

### Controlled corpus

The baseline has 11 short source documents and 28 semantic sections. Each
manifest entry records its source URL, source owner, topic, content type, risk
category, and review date. These are starter summaries, not clinically reviewed
production content.

### Stable chunk IDs

Each `##` Markdown section has an explicit comment:

```markdown
## The 5-4-3-2-1 exercise
<!-- chunk_id: grounding_54321_steps -->
```

The readable ID is the evaluation label. Qdrant requires an integer or UUID
point ID, so indexing deterministically converts the readable ID to UUIDv5 and
stores the readable ID in the payload. Rebuilding the same corpus therefore
preserves identity.

### Section-aware chunking

The source files are authored at retrieval granularity: one coherent rule or
exercise per section. This is preferable to splitting by a fixed character
count because a safety boundary should not be separated from its meaning.
Larger future documents can add a token-limit fallback as a measured chunking
experiment.

### Embedding model

`sentence-transformers/all-MiniLM-L6-v2` is the initial bi-encoder. It is small
enough for local development, returns 384-dimensional vectors, and provides a
clear baseline. Embeddings are normalized and Qdrant uses cosine similarity.
The embedding model is independent of the fine-tuned generation model.

### Qdrant local mode

Local on-disk Qdrant avoids requiring Docker or a cloud account for this small
baseline while preserving the collection, vector, and payload model that later
stages will extend with sparse vectors. Production deployment can switch the
client connection without changing retrieval contracts.

## Module Walkthrough

- `ingestion/manifest.py`: validates source inventory and resolves source files.
- `ingestion/chunking.py`: cleans text and builds stable `DocumentChunk` values.
- `ingestion/pipeline.py`: writes deterministic `chunks.jsonl`.
- `indexing/embeddings.py`: isolates the embedding model behind an interface.
- `indexing/qdrant_store.py`: recreates and populates the dense collection.
- `retrieval/dense.py`: embeds a query and maps Qdrant hits to `RetrievalResult`.
- `evals/run_dense_retrieval.py`: produces predictions and scores the gold set.

## Commands

Run from `icare-backend/chatbots` with the project virtual environment:

```bash
../../.venv/bin/python -m rag.ingestion.pipeline
../../.venv/bin/python -m rag.indexing.build_dense_index
../../.venv/bin/python -m rag.evals.run_dense_retrieval
../../.venv/bin/python -m rag.retrieval.query_dense \
  "What is the 5-4-3-2-1 grounding technique?"
```

The evaluation writes:

- `evals/reports/stage2_dense_predictions.jsonl`
- `evals/reports/stage2_dense_metrics.json`

Every later retrieval stage must evaluate the same gold queries at
`k = 3, 5, 10`.

## Baseline Result

The first 24-query run produced:

| Metric | @3 | @5 | @10 |
| --- | ---: | ---: | ---: |
| Recall | 0.688 | 0.833 | 0.896 |
| MRR | 0.812 | 0.821 | 0.821 |
| nDCG | 0.667 | 0.733 | 0.758 |

The high MRR means the retriever usually puts at least one relevant chunk near
the top. Recall is lower because multi-label queries sometimes miss secondary
evidence, especially the general scope disclaimer and cross-topic breathing
guidance. Those misses define what later stages must improve.

For example, the exact `5-4-3-2-1` query placed its intended steps chunk at
rank 5. That is a concrete Stage 3 hypothesis: sparse retrieval should recognize
the exact term and improve its fused rank.

## Interview Summary

> I established a dense-retrieval baseline before adding hybrid search. I
> curated a versioned source manifest, used section-aware chunks with stable
> evaluation IDs, embedded them with a Sentence Transformer, and stored vectors
> plus metadata in Qdrant. I measured Recall@K, MRR@K, and nDCG@K on the same
> manually labeled query set that later retrieval stages use.
