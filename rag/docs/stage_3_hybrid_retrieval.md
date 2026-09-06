# Stage 3: Hybrid Retrieval With BM25 And RRF

Environment installation and model-download instructions are documented in
`../SETUP.md`.

## Objective

Stage 3 tests whether combining semantic retrieval with exact-term retrieval
improves ranking over the Stage 2 dense-only baseline.

```text
query
  -> dense embedding -> dense top 20
  -> BM25 embedding  -> sparse top 20
  -> reciprocal rank fusion
  -> final top 10
  -> unchanged retrieval evaluation
```

## New Dependency And Model

Stage 3 adds FastEmbed explicitly:

```text
fastembed==0.8.0
```

Its supporting runtime, hashing, stemming, and logging packages are also pinned
in `rag/requirements.txt`. Stage 3 uses the `Qdrant/bm25` sparse model. Its 18
language-resource files are downloaded once to `rag/.cache/fastembed/`, which
is excluded from Git.

## Why BM25

Dense retrieval handles meaning, but it can underweight exact phrases such as
`5-4-3-2-1`. BM25 represents text as a sparse set of term IDs and weights. It
rewards query terms that occur in a chunk and gives uncommon terms more value.

`FastEmbedBm25Embedder` hides FastEmbed behind a small `SparseEmbedder`
interface. Indexing uses passage embeddings; queries use query embeddings.

## Hybrid Collection

Stage 3 creates a separate collection named `carebot_hybrid_v1`. Every point
contains:

- a normalized 384-dimensional vector named `dense`;
- a BM25 sparse vector named `sparse`;
- the original chunk text and Stage 1 metadata.

The sparse vector configuration uses Qdrant's `Modifier.IDF`, allowing Qdrant
to apply corpus-aware inverse document frequency during sparse search. Keeping
the Stage 2 and Stage 3 collections separate preserves a reproducible baseline.

## Reciprocal Rank Fusion

At query time, Qdrant runs two prefetches:

1. Dense cosine search for 20 candidates.
2. Sparse BM25 search for 20 candidates.

Qdrant then applies RRF. RRF combines positions rather than raw scores, which
matters because cosine and BM25 scores use different numerical scales. A chunk
receives more fused weight when it ranks highly in either list, especially when
it appears in both.

The implementation uses Qdrant's default unweighted RRF. We did not tune fusion
weights on the 24-query gold set because tuning and evaluating on the same small
dataset would overfit the reported result.

## Module Walkthrough

- `indexing/sparse_embeddings.py`: BM25 model adapter and sparse-vector contract.
- `indexing/hybrid_store.py`: creates and populates both named vector fields.
- `indexing/build_hybrid_index.py`: rebuilds Stage 1 chunks and the collection.
- `retrieval/hybrid.py`: runs dense and sparse prefetches with native RRF.
- `retrieval/query_hybrid.py`: inspects one hybrid query.
- `evals/run_hybrid_retrieval.py`: evaluates and computes dense-baseline deltas.

## Commands

Run from `icare-backend/chatbots`:

```bash
../../.venv/bin/python -m pip install -r rag/requirements.txt
../../.venv/bin/python -m rag.indexing.build_hybrid_index
../../.venv/bin/python -m rag.evals.run_hybrid_retrieval
../../.venv/bin/python -m rag.retrieval.query_hybrid \
  "What is the 5-4-3-2-1 grounding technique?" \
  --top-k 5 \
  --candidate-k 20
```

## Evaluation Result

Both stages used the same 24 queries and the same gold chunk labels.

| Metric | Stage 2 dense | Stage 3 hybrid | Delta |
| --- | ---: | ---: | ---: |
| Recall@3 | 0.688 | 0.688 | 0.000 |
| Recall@5 | 0.833 | 0.792 | -0.042 |
| Recall@10 | 0.896 | 0.896 | 0.000 |
| MRR@5 | 0.821 | 0.938 | +0.117 |
| nDCG@5 | 0.733 | 0.778 | +0.045 |
| nDCG@10 | 0.758 | 0.822 | +0.064 |

Hybrid retrieval placed a relevant chunk earlier more consistently and improved
overall ordering. It did not improve total recall and lost one expected
secondary chunk on several top-five queries. The exact `5-4-3-2-1` target moved
from rank 5 to rank 4, while a panic-support chunk containing the same exact
phrase ranked above it in BM25.

This is an honest mixed result: Stage 3 improved ranking precision but not
coverage. Stage 4 reranking can test whether joint query-chunk scoring promotes
the most directly relevant candidate while retaining hybrid's broader pool.

## Verification

The hybrid test uses fake dense and sparse embedders, so it validates Qdrant
indexing and RRF without network or model downloads. It confirms that a chunk
ranked by both semantic and exact-term evidence can move above a dense-only
match.

## Interview Summary

> I added BM25 sparse vectors beside the dense vectors in Qdrant and fused the
> two top-20 candidate lists using native reciprocal rank fusion. I used RRF
> because dense cosine scores and BM25 scores are not directly comparable. On
> the unchanged gold set, hybrid retrieval improved MRR and nDCG but slightly
> reduced Recall@5, so I treated it as a measured ranking tradeoff rather than
> assuming hybrid search was always better.
