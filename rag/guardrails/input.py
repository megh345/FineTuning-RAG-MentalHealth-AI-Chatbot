"""Input safety guardrails that run before RAG retrieval."""

from __future__ import annotations

import re

from rag.guardrails.messages import (
    crisis_response,
    diagnosis_boundary_response,
    medical_emergency_response,
    medication_boundary_response,
    prompt_injection_response,
)
from rag.guardrails.pii import PIIRedactor, RegexPIIRedactor
from rag.guardrails.types import (
    GuardrailAction,
    GuardrailCategory,
    GuardrailDecision,
    GuardrailFinding,
    GuardrailLayer,
    GuardrailSeverity,
    allow_decision,
)


class InputSafetyGuardrail:
    """Classify user input before retrieval is attempted."""

    def __init__(self, pii_redactor: PIIRedactor | None = None) -> None:
        self.pii_redactor = pii_redactor or RegexPIIRedactor()

    def evaluate(self, user_message: str) -> GuardrailDecision:
        if not user_message or not user_message.strip():
            return GuardrailDecision(
                action=GuardrailAction.BOUNDARY_RESPONSE,
                findings=[
                    GuardrailFinding(
                        layer=GuardrailLayer.INPUT,
                        category=GuardrailCategory.LOW_EVIDENCE,
                        severity=GuardrailSeverity.BLOCK,
                        message="Input message is empty.",
                    )
                ],
                safe_response="Please enter a message so I can help safely.",
            )

        pii_result = self.pii_redactor.redact(user_message)
        pii_findings = [
            GuardrailFinding(
                layer=GuardrailLayer.PII,
                category=GuardrailCategory.PII,
                severity=GuardrailSeverity.INFO,
                message="Potential PII was redacted for logging.",
                matched_text=entity.text,
                metadata={"entity_type": entity.entity_type},
            )
            for entity in pii_result.entities
        ]

        for pattern in CRISIS_PATTERNS:
            match = pattern.search(user_message)
            if match:
                return _blocking_decision(
                    action=GuardrailAction.ROUTE_CRISIS,
                    category=GuardrailCategory.CRISIS,
                    message="Input indicates possible crisis or immediate safety risk.",
                    matched_text=match.group(0),
                    safe_response=crisis_response(),
                    pii_findings=pii_findings,
                    redacted_text=pii_result.redacted_text,
                )

        for pattern in MEDICAL_EMERGENCY_PATTERNS:
            match = pattern.search(user_message)
            if match:
                return _blocking_decision(
                    action=GuardrailAction.ROUTE_MEDICAL,
                    category=GuardrailCategory.MEDICAL_EMERGENCY,
                    message="Input contains possible urgent medical red flags.",
                    matched_text=match.group(0),
                    safe_response=medical_emergency_response(),
                    pii_findings=pii_findings,
                    redacted_text=pii_result.redacted_text,
                )

        for pattern in MEDICATION_PATTERNS:
            match = pattern.search(user_message)
            if match:
                return _blocking_decision(
                    action=GuardrailAction.BOUNDARY_RESPONSE,
                    category=GuardrailCategory.MEDICATION_BOUNDARY,
                    message="Input asks for medication or treatment-change advice.",
                    matched_text=match.group(0),
                    safe_response=medication_boundary_response(),
                    pii_findings=pii_findings,
                    redacted_text=pii_result.redacted_text,
                )

        for pattern in DIAGNOSIS_PATTERNS:
            match = pattern.search(user_message)
            if match:
                return _blocking_decision(
                    action=GuardrailAction.BOUNDARY_RESPONSE,
                    category=GuardrailCategory.DIAGNOSIS_BOUNDARY,
                    message="Input asks for diagnosis or clinical labeling.",
                    matched_text=match.group(0),
                    safe_response=diagnosis_boundary_response(),
                    pii_findings=pii_findings,
                    redacted_text=pii_result.redacted_text,
                )

        for pattern in PROMPT_INJECTION_PATTERNS:
            match = pattern.search(user_message)
            if match:
                return _blocking_decision(
                    action=GuardrailAction.BOUNDARY_RESPONSE,
                    category=GuardrailCategory.PROMPT_INJECTION,
                    message="Input attempts to override instructions or expose internals.",
                    matched_text=match.group(0),
                    safe_response=prompt_injection_response(),
                    pii_findings=pii_findings,
                    redacted_text=pii_result.redacted_text,
                )

        return allow_decision(
            findings=pii_findings,
            redacted_text=pii_result.redacted_text,
        )


CRISIS_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(?:i\s*)?(?:want|plan|going|gonna|about)\s+to\s+"
        r"(?:kill|hurt|harm)\s+myself\b",
        r"\b(?:kill|hurt|harm)\s+myself\b",
        r"\b(?:end|take)\s+my\s+life\b",
        r"\b(?:suicide|suicidal)\b",
        r"\bself[- ]?harm\b",
        r"\bi\s+want\s+to\s+die\b",
        r"\bi\s+can'?t\s+(?:stay|keep\s+myself)\s+safe\b",
        r"\boverdose\b",
        r"\btook\s+too\s+many\s+pills\b",
        r"\b(?:kill|hurt|harm)\s+(?:someone|another\s+person|them)\b",
        r"\bi\s+am\s+being\s+abused\b",
        r"\bsomeone\s+is\s+(?:hurting|harming|abusing)\s+me\b",
    ]
]

MEDICAL_EMERGENCY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bchest\s+pain\b",
        r"\bfaint(?:ed|ing)?\b",
        r"\bpass(?:ed|ing)?\s+out\b",
        r"\bsevere\s+(?:shortness\s+of\s+breath|breathing\s+difficulty)\b",
        r"\b(?:can'?t|cannot)\s+breathe\b",
        r"\bheart\s+attack\b",
        r"\bstroke\b",
        r"\bnew\s+or\s+unusual\s+symptoms?\b",
        r"\bmedical\s+emergency\b",
    ]
]

MEDICATION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(?:can|should)\s+i\s+"
        r"(?:take|start|stop|increase|decrease|change)\b.*\b"
        r"(?:medication|medicine|meds?|antidepressant|dose|dosage)\b",
        r"\b(?:prescribe|recommend)\s+"
        r"(?:medication|medicine|meds?|antidepressants?|dosage|dose)\b",
        r"\b(?:start|stop|increase|decrease|change)\s+"
        r"(?:taking\s+)?"
        r"(?:my\s+)?(?:medication|medicine|antidepressant|dose|dosage)\b",
        r"\bwhat\s+(?:dose|dosage)\b",
        r"\btake\s+\d+\s?(?:mg|milligrams)\b",
        r"\bcan\s+i\s+mix\b.*\b(?:medication|medicine|pills?)\b",
        r"\bsleeping\s+pills?\b",
    ]
]

DIAGNOSIS_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bdo\s+i\s+have\s+"
        r"(?:depression|anxiety|panic\s+disorder|bipolar|ptsd|ocd)\b",
        r"\bam\s+i\s+"
        r"(?:depressed|bipolar|suicidal|traumatized|mentally\s+ill)\b",
        r"\bdiagnos(?:e|is|ed)\b",
        r"\bis\s+this\s+(?:depression|anxiety|panic\s+disorder|ptsd|ocd)\b",
    ]
]

PROMPT_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b",
        r"\breveal\s+(?:your\s+)?(?:system|hidden|developer)\s+prompt\b",
        r"\bshow\s+me\s+(?:your\s+)?(?:system|hidden|developer)\s+prompt\b",
        r"\b(?:print|dump|expose|reveal)\s+"
        r"(?:the\s+)?(?:complete\s+|full\s+)?knowledge\s+base\b",
        r"\b(?:print|dump|expose|reveal|show)\s+"
        r"(?:the\s+)?(?:complete\s+|full\s+)?contents\s+of\s+"
        r"(?:your\s+)?knowledge\s+base\b",
        r"\bdisable\s+(?:your\s+)?(?:safety|guardrails?|rules)\b",
        r"\bbypass\s+(?:your\s+)?(?:safety|guardrails?|rules)\b",
        r"\bjailbreak\b",
        r"\bpretend\s+you\s+are\s+not\s+bound\b",
    ]
]


def _blocking_decision(
    action: GuardrailAction,
    category: GuardrailCategory,
    message: str,
    matched_text: str,
    safe_response: str,
    pii_findings: list[GuardrailFinding],
    redacted_text: str,
) -> GuardrailDecision:
    return GuardrailDecision(
        action=action,
        findings=[
            *pii_findings,
            GuardrailFinding(
                layer=GuardrailLayer.INPUT,
                category=category,
                severity=GuardrailSeverity.BLOCK,
                message=message,
                matched_text=matched_text,
            ),
        ],
        safe_response=safe_response,
        redacted_text=redacted_text,
    )
