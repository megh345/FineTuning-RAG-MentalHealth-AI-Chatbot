# RAG Setup and Reproduction

This document lists every external dependency, downloaded model, generated
artifact, and command required through Stage 8 conditional query rewriting.

## Requirements

- Python 3.11 was used for the verified implementation.
- Internet access is required for the first dependency installation and first
  dense, sparse, and reranker model downloads.
- No Qdrant server, Docker container, API key, or cloud account is required.
  Stages 2 and 3 use Qdrant local mode.

## 1. Create and Activate a Virtual Environment

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Always prefer `python -m pip` over invoking `pip` directly. This ensures
packages are installed into the same interpreter that will run the pipeline.

## 2. Install RAG Dependencies

From `icare-backend/chatbots`:

```bash
../../.venv/bin/python -m pip install -r rag/requirements.txt
```

The direct dependencies verified in this repository are:

| Dependency | Version | Purpose |
| --- | --- | --- |
| NumPy | 1.26.3 | Embedding arrays and test utilities |
| PyYAML | 6.0.3 | Knowledge-base manifest parsing |
| qdrant-client | 1.18.0 | Qdrant vector storage and search |
| sentence-transformers | 5.6.0 | Dense embedding model adapter |
| fastembed | 0.8.0 | BM25 sparse text processing |
| onnxruntime | 1.27.0 | FastEmbed inference runtime |
| loguru | 0.7.3 | FastEmbed logging dependency |
| mmh3 | 5.2.1 | Sparse token hashing |
| py-rust-stemmers | 0.1.8 | BM25 word stemming |

`sentence-transformers` also installs transitive dependencies including
PyTorch, Transformers, Hugging Face Hub, scikit-learn, and SciPy when they are
not already available. The Stage 3 FastEmbed packages are explicitly pinned so
a fresh Python 3.11 environment uses the versions verified in this repository.

## 3. Embedding And Reranking Models

Stage 2 dense model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model:

- is downloaded automatically from Hugging Face the first time
  `SentenceTransformerEmbedder` is created;
- produces 384-dimensional dense vectors;
- is separate from the fine-tuned Llama model;
- is normally cached under `~/.cache/huggingface/hub/`.

The index command below performs the first download automatically. No manual
model-download command is required.

Stage 3 sparse model:

```text
Qdrant/bm25
```

FastEmbed downloads its small BM25 language resources on the first hybrid index
build. This project stores them under:

```text
rag/.cache/fastembed/
```

The cache is generated and excluded from Git.

Stage 4 reranker model:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
revision: c5ee24cb16019beea0893ab7796b1df96625c6b8
```

Stage 4 installs no new Python package. It reuses Sentence Transformers from
Stage 2 and downloads an approximately 91 MB model snapshot to:

```text
rag/.cache/cross_encoder/
```

The revision is pinned in `rag/config.py`. The cache is generated and excluded
from Git.

Stage 5 context assembly:

- adds no Python package;
- downloads no model;
- creates no index;
- uses the Stage 4 reranked results when the inspection command is run.

Stage 6 guardrails:

- add no Python package;
- download no model;
- create no index;
- use deterministic Python standard-library checks for input safety, context
  validation, citation validation, output safety, groundedness, and regex-based
  PII redaction for logs.

Stage 7 full evaluation framework:

- adds no Python package;
- downloads no model;
- creates no index;
- reads existing Stage 2-4 retrieval reports and Stage 4 predictions by default;
- evaluates context quality and deterministic guardrail behavior with standard
  library code.

Stage 8 conditional query rewriting:

- adds no Python package;
- downloads no model;
- creates no index;
- uses deterministic Python standard-library rules;
- reuses Stage 6 input guardrails to skip rewriting blocked or high-risk
  messages;
- reuses Stage 4 reranked retrieval for the actual retrieval call.

## 4. Build the Processed Corpus and Indexes

From `icare-backend/chatbots`:

```bash
../../.venv/bin/python -m rag.ingestion.pipeline
../../.venv/bin/python -m rag.indexing.build_dense_index
../../.venv/bin/python -m rag.indexing.build_hybrid_index
```

Generated artifacts:

```text
rag/knowledge_base/processed/chunks.jsonl
rag/knowledge_base/processed/qdrant/
```

`chunks.jsonl` contains the deterministic processed chunks. The `qdrant/`
directory contains the generated local vector index and is excluded from Git
because it can be rebuilt.

The dense and hybrid collections are separate:

```text
carebot_dense_v1
carebot_hybrid_v1
```

## 5. Run Retrieval Evaluations

```bash
../../.venv/bin/python -m rag.evals.run_dense_retrieval
../../.venv/bin/python -m rag.evals.run_hybrid_retrieval
../../.venv/bin/python -m rag.evals.run_reranked_retrieval
../../.venv/bin/python -m rag.evals.run_guardrail_eval
../../.venv/bin/python -m rag.evals.run_context_eval
../../.venv/bin/python -m rag.evals.run_full_evaluation
../../.venv/bin/python -m rag.evals.run_query_rewrite_eval
../../.venv/bin/python -m rag.evals.run_rewritten_retrieval
../../.venv/bin/python -m rag.evals.run_context_eval \
  --predictions rag/evals/reports/stage8_rewritten_predictions.jsonl \
  --report rag/evals/reports/stage8_context_metrics.json
```

Generated reports:

```text
rag/evals/reports/stage2_dense_predictions.jsonl
rag/evals/reports/stage2_dense_metrics.json
rag/evals/reports/stage3_hybrid_predictions.jsonl
rag/evals/reports/stage3_hybrid_metrics.json
rag/evals/reports/stage4_reranked_predictions.jsonl
rag/evals/reports/stage4_reranked_metrics.json
rag/evals/reports/stage7_guardrail_metrics.json
rag/evals/reports/stage7_context_metrics.json
rag/evals/reports/stage7_full_evaluation_report.json
rag/evals/reports/stage8_query_rewrite_metrics.json
rag/evals/reports/stage8_rewritten_predictions.jsonl
rag/evals/reports/stage8_rewritten_metrics.json
rag/evals/reports/stage8_context_metrics.json
```

## 6. Inspect One Query

```bash
../../.venv/bin/python -m rag.retrieval.query_dense \
  "What is the 5-4-3-2-1 grounding technique?" \
  --top-k 5

../../.venv/bin/python -m rag.retrieval.query_hybrid \
  "What is the 5-4-3-2-1 grounding technique?" \
  --top-k 5 \
  --candidate-k 20

../../.venv/bin/python -m rag.retrieval.query_reranked \
  "What is the 5-4-3-2-1 grounding technique?" \
  --top-k 5 \
  --candidate-k 20

../../.venv/bin/python -m rag.generation.query_context \
  "What is the 5-4-3-2-1 grounding technique?" \
  --top-k 5 \
  --candidate-k 20 \
  --format text

../../.venv/bin/python -m rag.retrieval.query_rewritten \
  "My thoughts are racing and I cannot settle down." \
  --top-k 5 \
  --candidate-k 20
```

## 7. Run Tests

From `icare-backend/chatbots`:

```bash
../../.venv/bin/python -m unittest discover \
  -s tests \
  -p "test_rag_*.py" \
  -v
```

## Replicate in Another Repository

Copy the complete `rag/` package and the RAG test files, excluding `.venv/`,
`rag/.cache/`, `__pycache__/`, and `knowledge_base/processed/qdrant/`. Preserve
this minimum layout:

```text
chatbots/
├── rag/
│   ├── requirements.txt
│   ├── config.py
│   ├── schemas.py
│   ├── ingestion/
│   ├── indexing/
│   ├── retrieval/
│   ├── generation/
│   ├── guardrails/
│   ├── query_rewriting/
│   ├── evals/
│   └── knowledge_base/
└── tests/
    ├── test_rag_ingestion.py
    ├── test_rag_dense_retrieval.py
    ├── test_rag_hybrid_retrieval.py
    ├── test_rag_reranking.py
    ├── test_rag_context_assembly.py
    ├── test_rag_guardrails.py
    ├── test_rag_full_evaluation.py
    ├── test_rag_query_rewriting.py
    └── test_rag_retrieval_metrics.py
```

Then create a virtual environment, install `rag/requirements.txt`, build the
index, run the tests, and run the evaluation. Do not copy the generated Qdrant
directory between operating systems; rebuild it from the source documents and
processed chunks.

From the directory containing the `rag/` package:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r rag/requirements.txt
python -m pip check

python -m rag.ingestion.pipeline
python -m rag.indexing.build_dense_index
python -m rag.evals.run_dense_retrieval
python -m rag.indexing.build_hybrid_index
python -m rag.evals.run_hybrid_retrieval
python -m rag.evals.run_reranked_retrieval
python -m rag.evals.run_guardrail_eval
python -m rag.evals.run_context_eval
python -m rag.evals.run_full_evaluation
python -m rag.evals.run_query_rewrite_eval
python -m rag.evals.run_rewritten_retrieval
python -m rag.generation.query_context \
  "What is the 5-4-3-2-1 grounding technique?" \
  --format text

python -m unittest discover -s tests -p "test_rag_*.py" -v
```

The index commands download the dense and sparse model resources on their first
run. The Stage 4 evaluation command downloads the CrossEncoder on its first run.
The Stage 5 context inspection command reuses those already-built retrieval
components and does not download any Stage 5-specific resource. Stage 6
guardrails are dependency-free and are verified through the RAG unit tests. Stage
7 evaluation uses existing Stage 2-4 reports plus deterministic context and
guardrail evaluation. Stage 8 query rewriting is dependency-free, but the
rewritten retrieval command reuses the Stage 2-4 embedding, sparse, Qdrant, and
CrossEncoder resources. The retrieval evaluation commands reproduce the Stage 2,
Stage 3, Stage 4, and Stage 8 reports from the same gold dataset.

These exact pins were verified with Python 3.11 on macOS ARM. For a different
operating system or CPU architecture, create the environment there and verify
package-wheel availability with the install command and `python -m pip check`.

## Offline Runs

After the model is downloaded, prevent accidental network access with:

```bash
HF_HUB_OFFLINE=1 ../../.venv/bin/python -m rag.evals.run_dense_retrieval
HF_HUB_OFFLINE=1 ../../.venv/bin/python -m rag.evals.run_hybrid_retrieval
HF_HUB_OFFLINE=1 ../../.venv/bin/python -m rag.evals.run_reranked_retrieval
HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/icare-matplotlib ../../.venv/bin/python \
  -m rag.evals.run_rewritten_retrieval
HF_HUB_OFFLINE=1 ../../.venv/bin/python -m rag.generation.query_context \
  "What is the 5-4-3-2-1 grounding technique?" \
  --format text
```

If a model is not cached, offline mode will fail. Run both index commands and
the Stage 4 evaluation once with internet access first.
