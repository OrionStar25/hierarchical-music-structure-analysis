import transformers
import torch

from transformers import AutoTokenizer


# Hugging face repo name
model = "meta-llama/Llama-2-7b-chat-hf" #chat-hf (hugging face wrapper version)

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto" # if you have GPU
)

prompt = open('prompt.txt').read()

sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    top_p = 0.9,
    temperature = 0.2,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=2000, # can increase the length of sequence
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
