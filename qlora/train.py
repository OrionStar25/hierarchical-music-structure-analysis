import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
           
import torch

from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    BitsAndBytesConfig, DataCollatorWithPadding,
    TrainingArguments
)
from datasets import load_dataset
from peft import LoraConfig


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['Tweet text'])):
        text = f'''
        <s>[INST]<<SYS>>
        Your task is to decide if the given tweet is a complaint or not.
        <</SYS>> 
        Tweet: {example['Tweet text'][i]} [/INST]  
        Answer: {example['text_label'][i]} </s>
        '''
        output_texts.append(text)
    
    return output_texts


model_name = "meta-llama/Llama-2-7b-chat-hf"

## Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

## Prepare dataset
dataset_name = "twitter_complaints"
dataset = load_dataset("ought/raft", dataset_name)

classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
dataset = dataset.rename_column("Label", "labels")
print(dataset["train"][0])

## Load quantized model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
)
model.config.use_cache = False

## Load LoRA Config
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

## Set training arguments
output_dir = "."
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
num_train_epochs=5
optim = "paged_adamw_32bit"
save_steps = 100
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 5
warmup_ratio = 0.03
evaluation_strategy="epoch"
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    num_train_epochs=num_train_epochs,
    evaluation_strategy=evaluation_strategy
)

max_seq_length = 4096

## Load SFTT trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    formatting_func=formatting_prompts_func,
    args=training_arguments
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

## Train the model
trainer.train()

## Save the model
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained("outputs")
