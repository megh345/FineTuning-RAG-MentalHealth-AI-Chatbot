"""Contracts for Stage 8 conditional query rewriting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryRewriteStrategy(str, Enum):
    """How a query rewrite decision was made."""

    NONE = "none"
    GUARDRAIL_SKIP = "guardrail_skip"
    TOPIC_EXPANSION = "topic_expansion"
    HISTORY_RESOLUTION = "history_resolution"


@dataclass(frozen=True)
class QueryRewriteDecision:
    """The original query, final retrieval query, and rewrite rationale."""

    original_query: str
    retrieval_query: str
    changed: bool
    strategy: QueryRewriteStrategy
    reason: str
    matched_terms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
