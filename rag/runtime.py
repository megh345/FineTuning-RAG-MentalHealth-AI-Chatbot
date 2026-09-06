"""Runtime adapter that connects the staged RAG pipeline to CareBot."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Protocol

from rag.config import DEFAULT_CONTEXT_TOP_K, DEFAULT_HYBRID_CANDIDATE_K
from rag.generation.context import ContextAssembler
from rag.guardrails import RagGuardrailPipeline, context_safety_instructions
from rag.guardrails.types import GuardrailDecision
from rag.schemas import AssembledContext, RetrievalResult


class RuntimeRetriever(Protocol):
    """Retriever interface used by the CareBot RAG runtime."""

    def retrieve(
        self,
        query: str,
        top_k: int,
        candidate_k: int,
        conversation_history: list[str] | None = None,
    ) -> list[RetrievalResult]: ...


@dataclass(frozen=True)
class CareBotRagResult:
    """Prepared RAG data for one CareBot response."""

    safe_response: str | None = None
    prompt_context: str = ""
    assembled_context: AssembledContext | None = None
    input_decision: GuardrailDecision | None = None
    context_decision: GuardrailDecision | None = None
    retrieval_results: list[RetrievalResult] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        """Return True when generation should receive retrieved context."""

        return bool(self.prompt_context and self.assembled_context is not None)

    def best_citation_marker_for_answer(self, answer: str) -> str | None:
        """Return the retrieved citation marker that best matches the answer."""

        if self.assembled_context is None:
            return None

        ranked_citations = self._rank_citations_by_answer_overlap(answer)
        if ranked_citations:
            return ranked_citations[0].marker
        if not self.assembled_context.citations:
            return None
        return self.assembled_context.citations[0].marker

    def references_for_answer(
        self,
        answer: str,
        max_references: int | None = None,
        prefer_content_matches: bool = False,
    ) -> list[dict[str, object]]:
        """Return user-facing source links for a generated answer."""

        if self.assembled_context is None:
            return []

        if prefer_content_matches:
            content_references = self._references_by_answer_overlap(
                answer,
                max_references=max_references,
            )
            if content_references:
                return content_references

        used_markers = set(CITATION_MARKER_PATTERN.findall(answer or ""))
        if not used_markers:
            return []

        references = []
        for citation in self.assembled_context.citations:
            if citation.marker not in used_markers:
                continue
            references.append(_reference_from_citation(citation))
            if max_references is not None and len(references) >= max_references:
                break
        return references

    def top_chunk_references(
        self,
        max_references: int | None = None,
    ) -> list[dict[str, object]]:
        """Return unique source links from assembled chunks, backfilled by retrieval."""

        if self.assembled_context is None:
            return []

        references = []
        seen_sources: set[tuple[str, str]] = set()
        for citation in self.assembled_context.citations:
            source_key = _reference_source_key(citation)
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            references.append(_reference_from_citation(citation))
            if max_references is not None and len(references) >= max_references:
                return references

        for result in self.retrieval_results:
            reference = _reference_from_retrieval_result(result)
            source_key = _reference_source_key_from_reference(reference)
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            references.append(reference)
            if max_references is not None and len(references) >= max_references:
                return references
        return references

    def _references_by_answer_overlap(
        self,
        answer: str,
        max_references: int | None = None,
    ) -> list[dict[str, object]]:
        references = []
        seen_sources: set[tuple[str, str]] = set()
        for citation in self._rank_citations_by_answer_overlap(answer):
            source_key = (citation.source_url, citation.title)
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            references.append(_reference_from_citation(citation))
            if max_references is not None and len(references) >= max_references:
                break
        return references

    def _rank_citations_by_answer_overlap(self, answer: str):
        if self.assembled_context is None:
            return []

        answer_terms = _content_terms(answer)
        if not answer_terms:
            return []

        result_text_by_chunk_id = {
            result.chunk_id: result.text or "" for result in self.retrieval_results
        }
        scored_citations = []
        for rank, citation in enumerate(self.assembled_context.citations):
            citation_text = " ".join(
                [
                    result_text_by_chunk_id.get(citation.chunk_id, ""),
                    citation.title,
                    citation.section,
                    citation.topic,
                    citation.source_name,
                ]
            )
            citation_terms = _content_terms(citation_text)
            overlap = answer_terms & citation_terms
            if not overlap:
                continue
            scored_citations.append(
                (
                    len(overlap),
                    citation.score,
                    -rank,
                    citation,
                )
            )

        scored_citations.sort(reverse=True, key=lambda item: item[:3])
        return [item[3] for item in scored_citations]


CITATION_MARKER_PATTERN = re.compile(r"\[C\d+\]")

REFERENCE_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "been",
    "being",
    "could",
    "does",
    "feel",
    "feeling",
    "from",
    "have",
    "help",
    "helpful",
    "helps",
    "into",
    "just",
    "like",
    "more",
    "some",
    "that",
    "their",
    "there",
    "these",
    "thing",
    "things",
    "this",
    "through",
    "when",
    "where",
    "while",
    "with",
    "would",
    "your",
    "youre",
    "yourself",
}


class CareBotRagRuntime:
    """Lazy runtime adapter for retrieval, context assembly, and guardrails."""

    def __init__(
        self,
        retriever: RuntimeRetriever | None = None,
        guardrails: RagGuardrailPipeline | None = None,
        assembler: ContextAssembler | None = None,
        top_k: int = DEFAULT_CONTEXT_TOP_K,
        candidate_k: int = DEFAULT_HYBRID_CANDIDATE_K,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")

        self._retriever = retriever
        self.guardrails = guardrails or RagGuardrailPipeline()
        self.assembler = assembler or ContextAssembler(max_chunks=top_k)
        self.top_k = top_k
        self.candidate_k = candidate_k

    def prepare_context(
        self,
        user_message: str,
        chat_history: list[dict] | None = None,
    ) -> CareBotRagResult:
        """Run pre-generation RAG stages and return prompt context or fallback."""

        input_decision = self.guardrails.evaluate_input(user_message)
        if input_decision.is_blocked:
            return CareBotRagResult(
                safe_response=input_decision.safe_response,
                input_decision=input_decision,
            )

        retriever = self._ensure_retriever()
        retrieval_top_k = min(self.candidate_k, max(self.top_k, self.top_k * 2))
        retrieval_results = retriever.retrieve(
            user_message,
            top_k=retrieval_top_k,
            candidate_k=self.candidate_k,
            conversation_history=_history_texts(chat_history or []),
        )
        assembled_context = self.assembler.assemble(
            user_message,
            retrieval_results,
            top_k=self.top_k,
        )
        context_decision = self.guardrails.validate_context(assembled_context)
        if context_decision.is_blocked:
            return CareBotRagResult(
                safe_response=context_decision.safe_response,
                assembled_context=assembled_context,
                input_decision=input_decision,
                context_decision=context_decision,
                retrieval_results=retrieval_results,
            )

        return CareBotRagResult(
            prompt_context=_prompt_context(assembled_context),
            assembled_context=assembled_context,
            input_decision=input_decision,
            context_decision=context_decision,
            retrieval_results=retrieval_results,
        )

    def validate_output(self, answer: str, rag_result: CareBotRagResult) -> str:
        """Validate a generated answer against retrieved context."""

        if rag_result.assembled_context is None:
            return answer
        decision = self.guardrails.validate_output(
            answer,
            rag_result.assembled_context,
            input_decision=rag_result.input_decision,
        )
        if decision.is_blocked and decision.safe_response:
            return decision.safe_response
        return answer

    def close(self) -> None:
        """Close retriever resources when the process is shutting down."""

        close = getattr(self._retriever, "close", None)
        if callable(close):
            close()

    def _ensure_retriever(self) -> RuntimeRetriever:
        if self._retriever is None:
            self._retriever = _build_default_retriever()
        return self._retriever


def rag_enabled() -> bool:
    """Return whether CareBot should attempt runtime RAG."""

    return os.getenv("CAREBOT_RAG_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


_DEFAULT_RUNTIME: CareBotRagRuntime | None = None


def get_carebot_rag_runtime() -> CareBotRagRuntime:
    """Return the process-wide lazy RAG runtime."""

    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = CareBotRagRuntime()
    return _DEFAULT_RUNTIME


def reset_carebot_rag_runtime() -> None:
    """Reset the process-wide runtime, mainly for tests."""

    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is not None:
        _DEFAULT_RUNTIME.close()
    _DEFAULT_RUNTIME = None


def _build_default_retriever() -> RuntimeRetriever:
    from rag.indexing.embeddings import SentenceTransformerEmbedder
    from rag.indexing.sparse_embeddings import FastEmbedBm25Embedder
    from rag.query_rewriting import ConditionalQueryRewriter
    from rag.retrieval.hybrid import HybridRetriever
    from rag.retrieval.reranked import RerankedRetriever
    from rag.retrieval.rerankers import CrossEncoderReranker
    from rag.retrieval.rewritten import QueryRewritingRetriever

    dense_embedder = SentenceTransformerEmbedder()
    sparse_embedder = FastEmbedBm25Embedder()
    hybrid_retriever = HybridRetriever(dense_embedder, sparse_embedder)
    reranker = CrossEncoderReranker()
    reranked_retriever = RerankedRetriever(hybrid_retriever, reranker)
    return QueryRewritingRetriever(
        reranked_retriever,
        ConditionalQueryRewriter(),
    )


def _prompt_context(context: AssembledContext) -> str:
    return f"""
RAG safety instructions:
{context_safety_instructions()}

Retrieved context for the current user message:
{context.context_text}

Answer requirements:
- Use the retrieved context as the source of factual coping or mental-health information.
- Cite support claims with the exact [C#] markers from the retrieved context.
- Do not invent citations or cite sources that are not listed above.
- If the context is not enough, say you do not have enough information in the knowledge base.
""".strip()


def _history_texts(chat_history: list[dict]) -> list[str]:
    history_texts = []
    for item in chat_history[-6:]:
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            history_texts.append(text.strip())
    return history_texts


def _reference_from_citation(citation) -> dict[str, object]:
    return {
        "marker": citation.marker,
        "title": citation.title,
        "section": citation.section,
        "source_name": citation.source_name,
        "source_url": citation.source_url,
        "reviewed_on": citation.reviewed_on,
        "topic": citation.topic,
        "chunk_id": citation.chunk_id,
        "score": citation.score,
    }


def _reference_source_key(citation) -> tuple[str, str]:
    source_url = (citation.source_url or "").strip().lower()
    title = (citation.title or citation.source_name or "").strip().lower()
    return source_url, title


def _reference_from_retrieval_result(result: RetrievalResult) -> dict[str, object]:
    source_metadata = _source_metadata(result.metadata)
    title = _metadata_string(
        source_metadata.get("document_title") or result.metadata.get("document_title"),
        default="Untitled source",
    )
    source_name = _metadata_string(
        source_metadata.get("source_name") or result.metadata.get("source_name")
    )
    source_url = _metadata_string(
        source_metadata.get("source_url") or result.metadata.get("source_url")
    )
    return {
        "title": title,
        "section": _metadata_string(result.metadata.get("section")),
        "source_name": source_name,
        "source_url": source_url,
        "reviewed_on": _metadata_optional_string(
            source_metadata.get("reviewed_on") or result.metadata.get("reviewed_on")
        ),
        "topic": _metadata_string(result.metadata.get("topic")),
        "chunk_id": result.chunk_id,
        "score": result.score,
    }


def _reference_source_key_from_reference(reference: dict[str, object]) -> tuple[str, str]:
    source_url = str(reference.get("source_url") or "").strip().lower()
    title = str(
        reference.get("title") or reference.get("source_name") or ""
    ).strip().lower()
    return source_url, title


def _source_metadata(metadata: dict[str, object]) -> dict[str, object]:
    nested_metadata = metadata.get("metadata")
    if isinstance(nested_metadata, dict):
        return nested_metadata
    return {}


def _metadata_string(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _metadata_optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _content_terms(text: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z'-]{3,}", text.lower()):
        normalized = token.strip("'").replace("'", "")
        if len(normalized) < 4 or normalized in REFERENCE_STOPWORDS:
            continue
        terms.add(normalized)
    return terms
