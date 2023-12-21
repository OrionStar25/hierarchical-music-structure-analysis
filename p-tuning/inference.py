import torch
from transformers import AutoTokenizer


## Load tokenizer
# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "bigscience/bloomz-560m"
text_column = "Tweet text"
device = "cuda:3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

inputs = tokenizer(
    f'{text_column} : {"@nationalgridus I have no water and the bill is current and paid. Can you do something about this?"} Label : ',
    return_tensors="pt",
)


## Load model
model = torch.load('p_tuning.pt')
model.to(device)


## Inference
with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
    )
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
