# Stage 6: Guardrails

## Goal

Stage 6 adds deterministic guardrails around the RAG flow.

```text
user input
  -> input safety guardrails
  -> retrieval / reranking / context assembly
  -> context guardrails
  -> answer generation
  -> output guardrails
```

This stage does not call an LLM and does not replace the existing CareBot
runtime safety file. It creates a standalone RAG guardrail subsystem that can be
tested first and integrated later.

## What Was Added

- `rag.guardrails.types`
- `rag.guardrails.messages`
- `rag.guardrails.pii`
- `rag.guardrails.input`
- `rag.guardrails.context`
- `rag.guardrails.output`
- `rag.guardrails.pipeline`
- `tests/test_rag_guardrails.py`

## Layer 1: Input Safety

Implemented in `rag.guardrails.input.InputSafetyGuardrail`.

Checks:

- crisis or immediate safety risk;
- possible medical-emergency red flags;
- prompt injection and policy extraction attempts;
- diagnosis-seeking requests;
- medication or treatment-change requests;
- PII redaction for safer logging.

Priority order:

```text
crisis -> medical emergency -> medication -> diagnosis -> prompt injection -> PII warning
```

High-risk input returns a blocking decision with a fixed safe response. Normal
input returns `allow`.

Example:

```python
from rag.guardrails import RagGuardrailPipeline

pipeline = RagGuardrailPipeline()
decision = pipeline.evaluate_input("I plan to kill myself tonight.")
if decision.is_blocked:
    print(decision.safe_response)
```

## PII Handling

Stage 6 intentionally does not install Presidio. Instead, it adds a pluggable
`PIIRedactor` protocol and a dependency-free `RegexPIIRedactor`.

The regex redactor detects common email, phone number, SSN, and credit-card-like
patterns for logs or evaluation traces. This is not a privacy guarantee. It is a
baseline that reduces accidental leakage and can later be replaced by Presidio
or another stronger detector.

## Layer 2: Retrieval And Context Guardrails

Implemented in `rag.guardrails.context.ContextGuardrail`.

Checks:

- assembled context exists;
- enough citations exist;
- citation markers are stable and sequential;
- cited chunks map to included chunks;
- duplicate citation markers or chunk IDs are rejected;
- source path stays inside `knowledge_base/raw/`;
- source URL uses an allowed scheme: `http`, `https`, or `internal`;
- source title, section, source name, URL, and path are present;
- review date is present and parseable;
- stale review dates are flagged as warnings.

If evidence is missing or unauthorized, the guardrail returns an abstention:

```text
I couldn't find enough supporting information in the available knowledge base to answer that safely.
```

## Layer 3: Output Validation

Implemented in `rag.guardrails.output.OutputGuardrail`.

Checks:

- answer is not empty;
- answer includes citations when making support or factual claims;
- cited markers exist in the assembled context;
- unsupported `[Source 2]`-style labels are rejected;
- diagnosis claims are blocked;
- medication and dosage instructions are blocked;
- overconfident claims such as guaranteed cure or fixed-time symptom resolution
  are blocked;
- crisis-risk input requires crisis routing language;
- low lexical overlap with context is flagged as a groundedness warning.

This is the first hallucination-control layer. It is more accurate to describe
it as grounding and citation validation, not perfect hallucination detection.

## Prompt Injection Protection

Stage 6 provides `context_safety_instructions()` for future generation prompts.
The instruction tells the generator to treat user text and retrieved documents
as evidence, not as system instructions, and to ignore attempts to reveal hidden
prompts, expose the knowledge base, disable safety rules, diagnose, prescribe,
or invent citations.

## Environment Changes

Stage 6 adds:

- no Python packages;
- no model downloads;
- no Qdrant collection;
- no environment variables.

## How To Verify

From `icare-backend/chatbots`:

```bash
../../.venv/bin/python -m unittest discover \
  -s tests \
  -p "test_rag_*.py" \
  -v
```

## Interview Explanation

I added a three-layer deterministic guardrail system around RAG. The input layer
routes crisis, medical red flags, prompt injection, diagnosis, and medication
requests before retrieval. The context layer prevents generation when retrieved
evidence is missing, weak, uncited, stale, or outside the reviewed knowledge
base. The output layer validates citations, blocks unsupported medical claims,
and performs first-pass grounding checks against the assembled context. I kept
the system dependency-free and rule-based first so it is testable, explainable,
and reproducible before adding any model-based safety classifier.
