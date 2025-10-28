import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

model_name = ""
dataset_path = ""

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = load_dataset("json", data_files=dataset_path, split="train")

def preprocess_function(examples):
    all_input_ids = []
    all_labels = []

    for messages in examples["messages"]:
        full_conversation_tokens = tokenizer.apply_chat_template(
            messages,
            tokenizer = True,
            add_generation_prompt = False,
            return_tensors = "pt"
        )[0]

        labels = full_conversation_tokens.clone()
        
        current_token_index = 0

        for msg in messages:
            msg_text = tokenizer.apply_chat_template(
                [msg],
                tokenizer = False,
                add_generation_prompt = False
            )

            msg_tokens = tokenizer(
                msg_text,
                tokenizer = True,
                add_special_tokens = False
            ).input_ids

            msg_len = len(msg_tokens)

            if msg["roles"] != "assistant":
                labels[current_token_index: current_token_index + msg_len] = -100

            current_token_index += msg_len

        all_input_ids.append(full_conversation_tokens)
        all_labels.append(labels)

    return {"input_ids": all_input_ids, "labels": all_labels}

tokenized_dataset = dataset.map(
    preprocess_function,
    batched = True,
    remove_columns = dataset.column_names
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir = "",
    num_train_epochs = 1,
    per_device_train_batch = 2,
    gradient_accumulation_steps = 4,
    optim = "adamw_torch",
    save_steps = 100,
    logging_steps = 10,
    learning_rate = 2e-5,
    weight_decay = 0.01,
    bf16 = True,
    max_grad_norm = 0.3,
    lr_scheduler_type = "linear",
    report_to="none"
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_dataset,
    tokenizer=tokenizer,
    data_collator = data_collator,
)

print("Starting manual SFT\n")
trainer.train()

output_path
trainer.save_model(output_path)
print(f"Training complete. Model saved to {output_path}")
