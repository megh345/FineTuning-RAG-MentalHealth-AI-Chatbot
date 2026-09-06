"""Configuration constants for the RAG subsystem.

Keep configuration centralized so ingestion, indexing, retrieval, and eval
scripts use the same paths and defaults.
"""

from __future__ import annotations

from pathlib import Path


RAG_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = RAG_ROOT / "knowledge_base"
RAW_KNOWLEDGE_DIR = KNOWLEDGE_BASE_DIR / "raw"
PROCESSED_KNOWLEDGE_DIR = KNOWLEDGE_BASE_DIR / "processed"
KNOWLEDGE_MANIFEST_PATH = KNOWLEDGE_BASE_DIR / "manifest.yaml"
PROCESSED_CHUNKS_PATH = PROCESSED_KNOWLEDGE_DIR / "chunks.jsonl"
QDRANT_STORAGE_DIR = PROCESSED_KNOWLEDGE_DIR / "qdrant"

EVALS_DIR = RAG_ROOT / "evals"
EVAL_DATASETS_DIR = EVALS_DIR / "datasets"
EVAL_REPORTS_DIR = EVALS_DIR / "reports"
RETRIEVAL_GOLD_PATH = EVAL_DATASETS_DIR / "retrieval_gold.jsonl"
GUARDRAIL_GOLD_PATH = EVAL_DATASETS_DIR / "guardrail_gold.jsonl"
QUERY_REWRITE_GOLD_PATH = EVAL_DATASETS_DIR / "query_rewrite_gold.jsonl"

DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_RERANK_TOP_K = 5
DEFAULT_HYBRID_CANDIDATE_K = 20
DEFAULT_CONTEXT_TOP_K = 5
DEFAULT_CONTEXT_MAX_CHARS = 4000

DENSE_COLLECTION_NAME = "carebot_dense_v1"
DENSE_VECTOR_NAME = "dense"
DENSE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

HYBRID_COLLECTION_NAME = "carebot_hybrid_v1"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
FASTEMBED_CACHE_DIR = RAG_ROOT / ".cache" / "fastembed"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_MODEL_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
RERANKER_CACHE_DIR = RAG_ROOT / ".cache" / "cross_encoder"
RERANKER_BATCH_SIZE = 16
