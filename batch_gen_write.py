import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DynamicCache

model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
model.config.pad_token_id = model.config.eos_token_id
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

#Multiple prompts in a list
prompts = ["Complete: Generality is a curse that afflicts",
          "The quick brown fox jumps over the",
          "There is far too little grace in letting evil fester"]

#Inputs with paddings to maintain list integrity
inputs = tokenizer(prompts, padding = True, return_tensors = 'pt')
inputs = inputs.to(device)
#inputs = {k: v.to("mps") for k,v in inputs.items()} #Move KV to MPS?
print(inputs["input_ids"])
print(inputs["input_ids"].shape)
print(inputs["attention_mask"])
print(inputs["attention_mask"].shape)

#Test: Catching the mask generated and filling out padding?
# attention_mask = inputs["attention_mask"]
# position_ids = attention_mask.long().cumsum(-1) - 1
# position_ids.masked_fill_(attention_mask == 0, 1)

#print(f"Position IDs: {position_ids}")

def generate_batch_tokens_with_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    last_logits = logits[:, -1, :] #Catching not just last logit output, previously last_logits = logits[0, -1, :]
    next_token_ids = last_logits.argmax(dim = 1)
    return next_token_ids, outputs.past_key_values

def generate_batch(inputs, max_tokens):
    #Create list of tokens for every input in the batch
    generated_tokens = [
        [] for _ in range(inputs["input_ids"].shape[0])
    ]

    #Test: Catching the mask generated and filling out padding?
    attention_mask = inputs["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    print(f"Position IDs: {position_ids}")

    next_inputs = {
        "position_ids": position_ids,
        **inputs,
    }

    for _ in range(max_tokens):
        next_token_ids, past_key_values = generate_batch_tokens_with_past(next_inputs)
        next_inputs = {
            "input_ids": next_token_ids.reshape((-1,1)),
            "position_ids": next_inputs["position_ids"][:, -1].unsqueeze(-1) + 1,
            "attention_mask": torch.cat([next_inputs["attention_mask"], torch.ones((next_token_ids.shape[0], 1)).to(device)], dim = 1),
            "past_key_values": past_key_values,
        }

        next_tokens = tokenizer.batch_decode(next_token_ids)
        for i, token in enumerate(next_tokens):
            generated_tokens[i].append(token)
    
    return ["".join(tokens) for tokens in generated_tokens]

generated_tokens = generate_batch(inputs, max_tokens= 20)

for prompt, generated in zip(prompts, generated_tokens):
    print(prompt, f"\x1b[31m{generated}\x1b[0m\n")


