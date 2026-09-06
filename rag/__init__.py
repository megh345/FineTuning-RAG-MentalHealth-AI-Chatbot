"""Standalone RAG subsystem for CareBot.

This package owns the complete eight-stage RAG pipeline:

1. Knowledge base and ingestion: validate reviewed sources, clean text, and
   build structure-aware chunks.
2. Dense retrieval baseline: embed chunks, build the dense Qdrant index, and
   measure semantic retrieval.
3. Hybrid retrieval: combine dense retrieval with BM25 sparse retrieval using
   reciprocal rank fusion.
4. CrossEncoder reranking: rescore hybrid candidates with query-chunk pair
   scoring to improve final ordering.
5. Context assembly and citations: package reranked chunks into a bounded
   prompt context with citation markers and metadata.
6. Guardrails: run deterministic input, context, and output safety checks around
   the RAG flow.
7. Full evaluation framework: combine retrieval, context, citation, and
   guardrail metrics into one reproducible report.
8. Conditional query rewriting: rewrite vague or semantic queries when safe,
   then compare rewritten retrieval against the reranked baseline.

Stage 0 in the documentation is the architecture and evaluation scaffold used
before these eight pipeline stages. Fine-tuning stays in `chatbots/fine_tuning`;
CareBot generation stays in `chatbots/carebot`.
"""
