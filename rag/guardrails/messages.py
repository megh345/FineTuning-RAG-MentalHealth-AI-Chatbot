"""Fixed safe-response templates for RAG guardrails."""

from __future__ import annotations


def crisis_response() -> str:
    """Return a crisis-safe response for immediate safety risk."""

    return (
        "I'm really sorry you're feeling this much pain. Your safety matters "
        "right now. If you might hurt yourself or feel unable to stay safe, "
        "please call emergency services now, call or text 988 in the United "
        "States, or use the 988 Lifeline chat. If you can, contact a trusted "
        "person who can stay with you and move away from anything you could use "
        "to harm yourself."
    )


def medical_emergency_response() -> str:
    """Return a safe response for possible medical-emergency red flags."""

    return (
        "I can't safely assess urgent medical symptoms here. Chest pain, "
        "fainting, severe shortness of breath, overdose, or new and unusual "
        "symptoms should be checked urgently. Please contact emergency services "
        "or urgent medical care now if this could be a medical emergency."
    )


def diagnosis_boundary_response() -> str:
    """Return a safe response for diagnosis-seeking requests."""

    return (
        "I can't diagnose a mental-health condition. I can share general "
        "supportive information, but a qualified mental-health professional can "
        "assess symptoms and discuss what support may fit your situation."
    )


def medication_boundary_response() -> str:
    """Return a safe response for medication or treatment-change requests."""

    return (
        "I can't prescribe medication, recommend dosages, or tell you to start, "
        "stop, or change medication. Please discuss medication questions with a "
        "qualified healthcare professional. If this involves overdose, a severe "
        "reaction, or immediate danger, seek urgent help now."
    )


def prompt_injection_response() -> str:
    """Return a safe response for prompt-injection or policy-extraction attempts."""

    return (
        "I can't ignore safety instructions, reveal hidden prompts, or expose "
        "the full knowledge base. I can still help with a normal support or "
        "coping question while following the approved safety rules."
    )


def low_evidence_response() -> str:
    """Return a safe abstention when retrieved evidence is insufficient."""

    return (
        "I couldn't find enough supporting information in the available "
        "knowledge base to answer that safely."
    )


def output_blocked_response() -> str:
    """Return a fallback when generated output fails guardrail validation."""

    return (
        "I can't provide that answer safely because it is not sufficiently "
        "grounded in the retrieved knowledge or crosses a medical-safety "
        "boundary."
    )
