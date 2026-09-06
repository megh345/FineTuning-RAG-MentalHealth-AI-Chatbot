# iCare - CareBot AI/ML Project

This repository contains two standalone AI engineering projects built for
CareBot, a supportive mental-health chatbot:

- `fine_tuning/`: supervised fine-tuning workflow for adapting
  `meta-llama/Llama-3.2-3B-Instruct` with LoRA/QLoRA.
- `rag/`: retrieval-augmented generation subsystem with ingestion, hybrid
  retrieval, reranking, guardrails, citations, and evaluation.

The goal is to show an end-to-end applied AI/ML system: dataset preparation,
model adaptation, retrieval infrastructure, safety controls, and measurable
evaluation.

## Project Highlights

- Built a LoRA/QLoRA fine-tuning pipeline for Llama 3.2 3B Instruct using the
  `ShenLab/MentalChat16K` dataset.
- Added dataset cleaning, deduplication, safety filtering, train/validation/test
  splitting, tokenization, adapter training, and base-vs-adapter evaluation.
- Built a staged RAG pipeline over a curated mental-health knowledge base.
- Implemented dense retrieval, BM25 sparse retrieval, reciprocal rank fusion,
  CrossEncoder reranking, context assembly, citation metadata, and deterministic
  guardrails.
- Added reproducible retrieval, context-quality, query-rewrite, and guardrail
  evaluation reports.
- Kept fine-tuning and RAG modular so each subsystem can be developed, tested,
  and explained independently.

## Repository Structure

```text
AI_Projects/
├── fine_tuning/
│   ├── prepare_mentalchat16k.py
│   ├── tokenize_mentalchat16k.py
│   ├── train_qlora.py
│   ├── evaluate_adapter.py
│   ├── config.py
│   └── README.md
└── rag/
    ├── ingestion/
    ├── indexing/
    ├── retrieval/
    ├── generation/
    ├── guardrails/
    ├── query_rewriting/
    ├── evals/
    ├── docs/
    ├── knowledge_base/
    ├── runtime.py
    ├── SETUP.md
    └── README.md
```

## Fine-Tuning Pipeline

The fine-tuning folder trains a LoRA adapter instead of modifying the base model
weights. The pipeline covers:

1. selecting the gated Llama 3.2 3B Instruct base model;
2. loading MentalChat16K;
3. cleaning short, duplicate, and unsafe examples;
4. formatting examples into chat messages;
5. creating train, validation, and test splits;
6. tokenizing with the Llama chat template;
7. training with QLoRA;
8. tracking training and validation loss;
9. saving adapter checkpoints;
10. comparing base-model and adapter generations with automated metrics and
    manual safety-review flags.

See `fine_tuning/README.md` for setup commands and script usage.

## RAG Pipeline

The RAG folder is organized as an eight-stage pipeline:

1. Knowledge base and ingestion
2. Dense retrieval baseline
3. Hybrid retrieval with dense vectors, BM25, and reciprocal rank fusion
4. CrossEncoder reranking
5. Context assembly and citations
6. Deterministic input, context, and output guardrails
7. Full evaluation framework
8. Conditional query rewriting

Stage-specific writeups live in `rag/docs/`, and reproduction instructions live
in `rag/SETUP.md`.

## Evaluation

This project uses evaluation as a development tool rather than a final
afterthought.

- Fine-tuning evaluation compares base-model and adapter responses with
  BERTScore, ROUGE-L, BLEU-4, and manual safety-review flags.
- RAG retrieval evaluation compares dense, hybrid, reranked, and rewritten
  retrieval against the same gold query set.
- RAG context evaluation checks citation coverage, context size, source
  coverage, and citation metadata.
- Guardrail evaluation checks deterministic behavior for crisis routing, medical
  red flags, prompt injection, diagnosis boundaries, medication boundaries, PII
  redaction, and groundedness.

## Safety Scope

CareBot is a learning and portfolio project, not a medical product. The system
is designed to avoid diagnosis, avoid medication advice, route crisis or
emergency language to appropriate support, and ground factual responses in
reviewed knowledge-base content.

## Reproducing Locally

Each subsystem has its own README because the dependencies and artifacts are
different:

- Fine-tuning setup: `fine_tuning/README.md`
- RAG setup and reproduction: `rag/SETUP.md`

The Llama base model is gated on Hugging Face, so local fine-tuning requires a
Hugging Face account with access to `meta-llama/Llama-3.2-3B-Instruct`.

## What To Keep In GitHub

Recommended to keep:

- source code in `fine_tuning/` and `rag/`;
- README, setup, and stage documentation;
- curated raw knowledge-base markdown files;
- small evaluation gold datasets;
- small JSON/JSONL evaluation reports that demonstrate measured progress;
- configuration files with model ids, paths, and reproducibility settings.

Recommended to exclude:

- Python caches such as `__pycache__/`;
- local virtual environments such as `.venv/` or `venv/`;
- downloaded model caches;
- LoRA checkpoints and trained adapter weights unless intentionally published
  through a model registry;
- generated tokenized datasets;
- large generated vector-store files such as Qdrant SQLite databases;
- secrets, API keys, Hugging Face tokens, and local environment files.

## Suggested Portfolio Framing

This project is best presented as an applied AI engineering system that combines
model adaptation and retrieval infrastructure with safety-aware evaluation. For
AI/ML roles, the strongest talking points are:

- practical LLM fine-tuning with LoRA/QLoRA;
- retrieval system design and staged experimentation;
- hybrid search and reranking tradeoffs;
- evaluation-driven iteration;
- safety and grounding decisions for a sensitive domain;
- clean separation between training, retrieval, generation support, and runtime
  integration.
