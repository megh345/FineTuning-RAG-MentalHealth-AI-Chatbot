# CareBot Runtime RAG Integration

This document explains how the staged RAG subsystem is connected to the CareBot
Llama response path without mixing it into the fine-tuning pipeline.

## Runtime Flow

The integration lives in two places:

- `rag/runtime.py` owns the RAG runtime adapter.
- `carebot/response_generator_llama.py` calls that adapter before generation.

The request flow is:

```text
user message
-> existing CareBot direct safety check
-> RAG input guardrails
-> conditional query rewriting
-> hybrid retrieval
-> CrossEncoder reranking
-> context assembly with [C#] citations
-> retrieval/context guardrails
-> fine-tuned Llama generation with retrieved context in the system prompt
-> RAG output validation
-> existing CareBot response guardrails
-> chat history update with optional source references
```

RAG runs before the model writes the answer. That matters because retrieved
knowledge should guide the fine-tuned model while it is generating. Running RAG
after generation would only let us reject or patch an answer after the model has
already guessed.

## Design Decisions

The fine-tuned model remains responsible for tone, empathy, and conversation
style. RAG is responsible for reviewed factual support, citations, low-evidence
abstention, and deterministic safety checks around retrieval-grounded answers.

The runtime adapter is lazy. Importing `carebot/response_generator_llama.py` does
not build embedding models, Qdrant clients, or rerankers immediately. Those are
created only when CareBot receives a message and RAG is enabled.

The integration fails open by default. If the RAG runtime raises an unexpected
error, CareBot falls back to the existing fine-tuned generation path. During
debugging, set `CAREBOT_RAG_STRICT=1` to raise the error instead.

## Environment Flags

```bash
export CAREBOT_RAG_ENABLED=1
export CAREBOT_RAG_STRICT=0
```

`CAREBOT_RAG_ENABLED` defaults to enabled. Set it to `0`, `false`, `no`, or
`off` to temporarily disable runtime RAG.

`CAREBOT_RAG_STRICT` defaults to disabled. Set it to `1` when you want runtime
RAG errors to crash loudly during local debugging.

The fine-tuned model environment remains separate:

```bash
export CAREBOT_LLM_PATH=/path/to/base-or-merged-llama-model
export CAREBOT_LORA_ADAPTER_PATH=/path/to/selected/lora-adapter
```

If your CareBot already loads the fine-tuned model through another mechanism,
keep using that. The RAG bridge only changes the prompt context passed into the
existing generation function.

## User-Facing References

When the final CareBot answer contains RAG citation markers such as `[C1]`, the
CareBot message includes a `references` array:

```json
{
  "text": "A grounding exercise can use what you see, feel, hear, smell, and taste. [C1]",
  "type": "carebot",
  "references": [
    {
      "marker": "[C1]",
      "title": "Grounding Techniques",
      "section": "The 5-4-3-2-1 exercise",
      "source_name": "NHS Inform Grounding Exercises",
      "source_url": "https://example.test/grounding",
      "reviewed_on": "2026-06-30",
      "topic": "grounding",
      "chunk_id": "grounding_54321_steps",
      "score": 2.5
    }
  ]
}
```

The frontend can render `references[*].source_url` as a clickable link, usually
with the label `references[*].marker` plus `references[*].title`. References are
only attached when the final answer still cites the retrieved context. Direct
crisis responses, low-evidence abstentions, and uncited safety fallbacks do not
show source links.

## Required Setup Before Running CareBot

Run these from `icare-backend/chatbots` after installing
`rag/requirements.txt`:

```bash
python -m rag.ingestion.pipeline
python -m rag.indexing.build_dense_index
python -m rag.indexing.build_hybrid_index
python -m rag.evals.run_rewritten_retrieval
python -m unittest discover -s tests -p "test_rag_*.py" -v
```

The first three commands build the local artifacts used by runtime retrieval.
The evaluation command is not required for serving, but it confirms the current
retrieval quality before you wire the system into the chatbot.

## Files To Transfer To Another Repository

Transfer these source files and folders:

- `chatbots/rag/`
- `chatbots/carebot/response_generator_llama.py`
- `chatbots/tests/test_rag_*.py`

Do not transfer generated or machine-specific folders:

- `.venv/`
- `__pycache__/`
- `chatbots/rag/.cache/`
- `chatbots/rag/knowledge_base/processed/qdrant/`

You can regenerate `chunks.jsonl` and the Qdrant indexes in the target repo
using the setup commands above. If you transfer `knowledge_base/processed/`, make
sure you still rebuild indexes after changing raw documents or dependencies.
