"""Deterministic conditional query rewriting for RAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import re

from rag.guardrails.input import InputSafetyGuardrail
from rag.guardrails.types import GuardrailAction
from rag.query_rewriting.types import QueryRewriteDecision, QueryRewriteStrategy


@dataclass(frozen=True)
class RewriteRule:
    """One deterministic query expansion rule."""

    name: str
    patterns: tuple[re.Pattern[str], ...]
    expansion: str
    skip_if_any: tuple[str, ...] = ()


class ConditionalQueryRewriter:
    """Rewrite only queries that are likely to benefit retrieval.

    This is a deterministic baseline. It does not call an LLM, does not change
    safety routing, and skips rewriting when input guardrails would block normal
    RAG.
    """

    def __init__(
        self,
        input_guardrail: InputSafetyGuardrail | None = None,
        max_history_messages: int = 2,
    ) -> None:
        if max_history_messages < 0:
            raise ValueError("max_history_messages cannot be negative")

        self.input_guardrail = input_guardrail or InputSafetyGuardrail()
        self.max_history_messages = max_history_messages

    def rewrite(
        self,
        query: str,
        conversation_history: list[str] | None = None,
    ) -> QueryRewriteDecision:
        normalized_query = _normalize_query(query)
        if not normalized_query:
            raise ValueError("Query cannot be empty")

        input_decision = self.input_guardrail.evaluate(normalized_query)
        if input_decision.action not in {
            GuardrailAction.ALLOW,
            GuardrailAction.ALLOW_WITH_WARNINGS,
        }:
            return QueryRewriteDecision(
                original_query=normalized_query,
                retrieval_query=normalized_query,
                changed=False,
                strategy=QueryRewriteStrategy.GUARDRAIL_SKIP,
                reason="Input guardrail would block normal RAG, so rewrite was skipped.",
                metadata={"input_action": input_decision.action.value},
            )

        history_rewrite = self._history_rewrite(
            normalized_query,
            conversation_history or [],
        )
        if history_rewrite is not None:
            return history_rewrite

        for rule in REWRITE_RULES:
            if any(skip_term in normalized_query.lower() for skip_term in rule.skip_if_any):
                continue
            matched_terms = [
                match.group(0)
                for pattern in rule.patterns
                if (match := pattern.search(normalized_query))
            ]
            if not matched_terms:
                continue
            rewritten = _append_expansion(normalized_query, rule.expansion)
            if rewritten == normalized_query:
                continue
            return QueryRewriteDecision(
                original_query=normalized_query,
                retrieval_query=rewritten,
                changed=True,
                strategy=QueryRewriteStrategy.TOPIC_EXPANSION,
                reason=f"Expanded query using the {rule.name} retrieval rule.",
                matched_terms=matched_terms,
                metadata={"rule": rule.name},
            )

        return QueryRewriteDecision(
            original_query=normalized_query,
            retrieval_query=normalized_query,
            changed=False,
            strategy=QueryRewriteStrategy.NONE,
            reason="Query already looked specific enough for retrieval.",
        )

    def _history_rewrite(
        self,
        query: str,
        conversation_history: list[str],
    ) -> QueryRewriteDecision | None:
        if not _looks_like_follow_up(query):
            return None

        history_text = " ".join(
            conversation_history[-self.max_history_messages :]
            if self.max_history_messages
            else []
        )
        topic = _history_topic(history_text)
        if topic is None:
            return None

        rewritten = _append_expansion(query, topic.expansion)
        if rewritten == query:
            return None
        return QueryRewriteDecision(
            original_query=query,
            retrieval_query=rewritten,
            changed=True,
            strategy=QueryRewriteStrategy.HISTORY_RESOLUTION,
            reason="Expanded a short follow-up query using recent conversation topic.",
            matched_terms=[topic.name],
            metadata={"history_topic": topic.name},
        )


@dataclass(frozen=True)
class HistoryTopic:
    name: str
    expansion: str


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


REWRITE_RULES = [
    RewriteRule(
        name="grounding_present_moment",
        patterns=(
            _compile(r"\bracing thoughts?\b"),
            _compile(r"\bcome back to the present\b"),
            _compile(r"\bsettle down\b"),
            _compile(r"\bground(?:ing)?\b"),
        ),
        expansion="grounding present moment 5-4-3-2-1 exercise anxiety support",
        skip_if_any=("5-4-3-2-1",),
    ),
    RewriteRule(
        name="breathing_anxiety",
        patterns=(
            _compile(r"\bbreathe\b"),
            _compile(r"\bbreathing\b"),
            _compile(r"\bbreath\b"),
        ),
        expansion="slow counted breathing anxiety coping exercise not a cure",
    ),
    RewriteRule(
        name="panic_support",
        patterns=(
            _compile(r"\bpanic\b"),
            _compile(r"\bheart is racing\b"),
            _compile(r"\bheart racing\b"),
        ),
        expansion=(
            "panic symptoms grounding support steps no diagnosis urgent medical "
            "warning signs"
        ),
    ),
    RewriteRule(
        name="urgent_medical_panic",
        patterns=(
            _compile(r"\burgent medical help\b"),
            _compile(r"\bmedical help\b"),
            _compile(r"\bred flags?\b"),
        ),
        expansion=(
            "panic medical red flags chest pain fainting severe shortness of "
            "breath new unusual symptoms"
        ),
    ),
    RewriteRule(
        name="anxiety_worry",
        patterns=(
            _compile(r"\bworrying\b"),
            _compile(r"\bworry\b"),
            _compile(r"\bcannot control\b"),
        ),
        expansion=(
            "anxiety worry persistent difficult to control coping options "
            "professional support"
        ),
    ),
    RewriteRule(
        name="stress_prioritization",
        patterns=(
            _compile(r"\boverwhelmed\b"),
            _compile(r"\bwork and school\b"),
            _compile(r"\burgent\b.*\bwait\b"),
            _compile(r"\bwhat can wait\b"),
        ),
        expansion="stress prioritize urgent today what can wait one manageable next action",
    ),
    RewriteRule(
        name="low_mood_support",
        patterns=(
            _compile(r"\bfeel low\b"),
            _compile(r"\bempty\b"),
            _compile(r"\btired all the time\b"),
            _compile(r"\bcannot get myself\b"),
        ),
        expansion=(
            "low mood depression small care steps professional support no diagnosis"
        ),
    ),
    RewriteRule(
        name="sleep_anxiety",
        patterns=(
            _compile(r"\bbefore bed\b"),
            _compile(r"\bkeeps me awake\b"),
            _compile(r"\bcannot sleep\b"),
        ),
        expansion="sleep wind-down routine anxiety breathing screens caffeine bedtime",
    ),
    RewriteRule(
        name="cbt_thought_record",
        patterns=(
            _compile(r"\bchallenge an unhelpful thought\b"),
            _compile(r"\bevidence supports\b"),
            _compile(r"\bevidence does not\b"),
        ),
        expansion=(
            "CBT thought record automatic thought evidence balanced kinder "
            "thought helpful next action"
        ),
    ),
    RewriteRule(
        name="scope_limitations",
        patterns=(
            _compile(r"\ballowed to do\b"),
            _compile(r"\bmental health chatbot\b"),
            _compile(r"\breplace therapy\b"),
        ),
        expansion=(
            "chatbot scope limitations general emotional support not therapist "
            "not diagnosis not emergency care"
        ),
    ),
    RewriteRule(
        name="professional_support",
        patterns=(
            _compile(r"\bcontact a counselor\b"),
            _compile(r"\bprofessional support\b"),
            _compile(r"\btherapist\b"),
        ),
        expansion=(
            "professional support symptoms persistent intense worsening "
            "interfering daily life"
        ),
    ),
    RewriteRule(
        name="crisis_policy",
        patterns=(
            _compile(r"\bunable to stay safe\b"),
            _compile(r"\bkeep it secret\b"),
            _compile(r"\bhurt someone\b"),
        ),
        expansion=(
            "crisis immediate routing 988 trusted person no secrecy immediate safety"
        ),
    ),
    RewriteRule(
        name="grounding_not_perfect",
        patterns=(
            _compile(r"\bdoes not work perfectly\b"),
            _compile(r"\bdoing it wrong\b"),
        ),
        expansion="grounding optional not perfect present moment self-harm boundary",
    ),
]

HISTORY_TOPICS = [
    HistoryTopic(
        name="panic",
        expansion=(
            "panic symptoms grounding support steps no diagnosis urgent medical "
            "warning signs"
        ),
    ),
    HistoryTopic(
        name="grounding",
        expansion="grounding present moment 5-4-3-2-1 exercise anxiety support",
    ),
    HistoryTopic(
        name="sleep",
        expansion="sleep wind-down routine anxiety breathing screens caffeine bedtime",
    ),
    HistoryTopic(
        name="cbt",
        expansion=(
            "CBT thought record automatic thought evidence balanced kinder "
            "thought helpful next action"
        ),
    ),
]


def _normalize_query(query: str) -> str:
    return " ".join((query or "").split())


def _append_expansion(query: str, expansion: str) -> str:
    query_terms = set(re.findall(r"[a-zA-Z0-9-]+", query.lower()))
    expansion_terms = [
        term
        for term in expansion.split()
        if term.lower().strip(".,") not in query_terms
    ]
    if not expansion_terms:
        return query
    return f"{query} Retrieval focus: {' '.join(expansion_terms)}."


def _looks_like_follow_up(query: str) -> bool:
    lowered = query.lower()
    return bool(
        re.search(r"\b(that|this|it|those|these)\b", lowered)
        and len(lowered.split()) <= 8
    )


def _history_topic(history_text: str) -> HistoryTopic | None:
    lowered = history_text.lower()
    for topic in HISTORY_TOPICS:
        if topic.name in lowered:
            return topic
    return None
