# CareBot Llama 3.2 Fine-Tuning Flow

This folder implements a supervised fine-tuning workflow for adapting
`meta-llama/Llama-3.2-3B-Instruct` to CareBot-style supportive mental-health
dialogue using `ShenLab/MentalChat16K`.

The workflow trains a LoRA/QLoRA adapter. It does not overwrite the base model.

## 0. Environment Setup

Run commands from the repository root:

```bash
python -m pip install -r fine_tuning/requirements.txt
huggingface-cli login
```

The Llama 3.2 model is gated on Hugging Face, so your Hugging Face account must
have accepted the model license before training or inference can download it.

## 1. Pick Base Model

Base model:

```text
meta-llama/Llama-3.2-3B-Instruct
```

The model id is configured in `fine_tuning/config.py`.

## 2. Pick Dataset

Dataset:

```text
ShenLab/MentalChat16K
```

MentalChat16K exposes `instruction`, `input`, and `output` columns. We treat
`input` as the user message and `output` as the target assistant response.

## 3-5. Clean, Format, And Split Dataset

```bash
python -m fine_tuning.prepare_mentalchat16k
```

This command:

- removes empty or very short examples;
- removes duplicates by hashing normalized user/assistant text;
- rejects assistant responses with unsafe diagnosis, medication, or self-harm advice;
- keeps crisis examples only when the assistant response includes crisis-routing language;
- formats rows as `system`, `user`, and `assistant` messages;
- creates `train`, `validation`, and `test` splits;
- uses exactly 12,000 cleaned examples for training by default, then splits the
  remaining cleaned examples between validation and test.

To choose a different training size:

```bash
python -m fine_tuning.prepare_mentalchat16k --train-size 10000
```

Outputs:

```text
carebot/data/mentalchat16k/
carebot/data/mentalchat16k/dataset_stats.json
carebot/data/mentalchat16k/rejected_examples.jsonl
```

## 6. Tokenize

```bash
python -m fine_tuning.tokenize_mentalchat16k
```

This command renders each chat with the Llama tokenizer chat template and
converts it into token IDs. Labels for system and user prompt tokens are set to
`-100`, so the training loss focuses on the assistant response.

Output:

```text
carebot/data/mentalchat16k_tokenized/
```

## 7-10. Add LoRA/QLoRA, Train, Track Loss, Save Adapter

Smoke test on a tiny subset:

```bash
python -m fine_tuning.train_qlora \
  --max-train-samples 32 \
  --max-eval-samples 16 \
  --num-train-epochs 0.1 \
  --eval-steps 5 \
  --save-steps 5
```

Full training:

```bash
python -m fine_tuning.train_qlora --num-train-epochs 2
```

Training and validation loss are printed by Hugging Face `Trainer`. Checkpoints
and the final adapter are saved under:

```text
carebot/models/llama3_2_3b_mentalchat_lora/
```

## 11-12. Evaluate And Compare

```bash
python -m fine_tuning.evaluate_adapter --max-samples 50
```

This compares the base model against the adapter on the test split and writes:

```text
carebot/data/mentalchat16k_eval/evaluation_results.json
carebot/data/mentalchat16k_eval/generations.jsonl
carebot/data/mentalchat16k_eval/manual_safety_review.jsonl
```

Metrics:

- BERTScore F1;
- ROUGE-L;
- BLEU-4;
- manual safety-review flags for diagnosis, medication, and crisis-routing issues.

Automated metrics are useful for regression tracking, but they are not enough
for a mental-health assistant. Review `manual_safety_review.jsonl` before using
the adapter in a demo.

## 13. Integrate With CareBot Backend

If you connect this project to a chatbot backend, load the adapter by pointing
the runtime at the saved adapter directory:

```bash
export CAREBOT_LLM_MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
export CAREBOT_LORA_ADAPTER_PATH="carebot/models/llama3_2_3b_mentalchat_lora"
python service.py --dev
```

Without `CAREBOT_LORA_ADAPTER_PATH`, the backend should load the base model
only.

## 14. Demo In UI

1. Start the chatbot service with `CAREBOT_LORA_ADAPTER_PATH` set.
2. Start the Django backend as usual.
3. Open the iCare UI and send the same prompts to the base model and adapter.
4. Compare empathy, specificity, brevity, boundary-setting, and crisis-routing.

For a controlled demo, use prompts from `generations.jsonl` so the before/after
comparison is repeatable.
