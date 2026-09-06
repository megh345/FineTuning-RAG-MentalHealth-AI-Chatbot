# Stage 4: CrossEncoder Reranking

Environment and model-download instructions are documented in `../SETUP.md`.

## Objective

Stage 4 tests whether a model that reads the query and candidate chunk together
can improve the ordering produced by Stage 3 hybrid retrieval.

```text
query
  -> Stage 3 dense + BM25 + RRF
  -> top 20 hybrid candidates
  -> CrossEncoder(query, candidate text)
  -> sort by CrossEncoder logit
  -> final top 5 for normal use
  -> top 10 for evaluation metrics
```

## Model Decision

Stage 4 uses:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
revision: c5ee24cb16019beea0893ab7796b1df96625c6b8
```

The model is approximately 91 MB and Apache-2.0 licensed. It is loaded through
the existing `sentence-transformers==5.6.0` dependency, so Stage 4 adds no
Python package. The exact model snapshot is cached under
`rag/.cache/cross_encoder/` and excluded from Git.

This is a practical pretrained baseline: it is small enough for local CPU use
and was trained for passage ranking. It was not trained specifically on
mental-health content, which remains a limitation to measure rather than hide.

## Why A CrossEncoder

Dense and sparse retrieval score query and chunk representations separately.
A CrossEncoder processes the complete pair together, allowing attention between
query terms and every token in the candidate. This is usually more precise but
too expensive to run over an entire large corpus.

The retrieve-then-rerank pattern balances cost and quality:

1. Hybrid retrieval quickly finds 20 broad candidates.
2. The CrossEncoder performs 20 slower pairwise predictions.
3. Only the highest-scoring results continue.

A reranker cannot recover a relevant chunk missing from the candidate pool, so
Stage 3 still owns candidate recall.

## Implementation

`retrieval/rerankers.py` defines a library-independent `Reranker` contract and
the `CrossEncoderReranker` adapter. It:

- requires candidate text;
- batches query-chunk pairs in groups of 16;
- scores pairs with a maximum input length of 512 tokens;
- sorts logits from highest to lowest;
- preserves the original RRF score as `metadata["retrieval_score"]`;
- stores the reranker model name in result metadata.

The CrossEncoder score is a ranking logit, not a probability or confidence
percentage.

`retrieval/reranked.py` composes Stage 3 with the reranker. It requests 20 final
RRF candidates from the hybrid retriever, then returns the requested top K after
rescoring.

## Commands

Run from `icare-backend/chatbots`:

```bash
../../.venv/bin/python -m rag.evals.run_reranked_retrieval
../../.venv/bin/python -m rag.retrieval.query_reranked \
  "What is the 5-4-3-2-1 grounding technique?" \
  --top-k 5 \
  --candidate-k 20
```

No Qdrant index rebuild is required. Stage 4 reads the existing
`carebot_hybrid_v1` collection.

## Evaluation Result

The same 24 queries and gold labels were used.

| Metric | Stage 3 hybrid | Stage 4 reranked | Delta |
| --- | ---: | ---: | ---: |
| Recall@3 | 0.688 | 0.771 | +0.083 |
| Recall@5 | 0.792 | 0.833 | +0.042 |
| Recall@10 | 0.896 | 0.896 | 0.000 |
| MRR@5 | 0.938 | 0.917 | -0.021 |
| nDCG@5 | 0.778 | 0.792 | +0.014 |
| nDCG@10 | 0.822 | 0.818 | -0.004 |

Reranking recovered Stage 3's Recall@5 loss and improved top-three coverage and
nDCG@5. MRR decreased slightly because a few already-correct first results
moved to rank 2. Recall@10 stayed unchanged because reranking only reorders the
candidate pool.

For the exact `5-4-3-2-1` query, the intended steps chunk moved from hybrid rank
4 to reranked rank 2. The panic-support chunk remained first, showing that the
general-purpose reranker improved directness but did not fully solve this
domain-specific ambiguity.

## Tests

`tests/test_rag_reranking.py` uses a fake CrossEncoder to verify:

- a direct answer can be promoted above a higher-RRF generic candidate;
- the original hybrid score remains available;
- the complete candidate pool is requested;
- invalid top-K and candidate-K combinations are rejected.

The fake model keeps unit tests deterministic and independent of network access.

## Interview Summary

> I used hybrid retrieval for broad candidate recall and a CrossEncoder for
> final ranking precision. The reranker jointly scored each query with the top
> 20 hybrid chunks and returned the best five. On the unchanged gold set it
> recovered Recall@5 and improved nDCG@5, while MRR decreased slightly. I kept
> the mixed result because it shows the measured tradeoff rather than assuming
> reranking always improves every metric.
