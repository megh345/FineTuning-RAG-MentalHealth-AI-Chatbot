"""CareBot QLoRA fine-tuning package.

The modules in this package implement a step-by-step QLoRA fine-tuning workflow
for adapting Llama 3.2 3B Instruct to supportive mental-health dialogue.
Below are the steps of the fine-tuning pipeline:

1. Pick the base model: meta-llama/Llama-3.2-3B-Instruct.
2. Load MentalChat16K from Hugging Face.
3. Clean the dataset by removing empty, duplicate, and unsafe examples.
4. Format each row as a system/user/assistant chat.
5. Split the cleaned dataset into train, validation, and test sets.
6. Convert formatted text into token IDs.
7. Add LoRA/QLoRA so the base model is frozen and only adapter weights train.
8. Train on mental-health dialogue examples.
9. Track training and validation loss through Trainer logs.
10. Save the resulting LoRA adapter.
11. Evaluate with BERTScore F1, ROUGE-L, BLEU-4, and a safety-review checklist.
12. Compare base-model responses against fine-tuned adapter responses.
"""

