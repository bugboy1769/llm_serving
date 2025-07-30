#Generation with GPT2 with KV Caching, takes a total of ~2.04 seconds with 50 tokens, ~4.13 seconds with 100 tokens

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DynamicCache

model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

prompt = "quantum mechanics deals with"

inputs = tokenizer(prompt, return_tensors = 'pt')
inputs = inputs.to(device)
#inputs = {k: v.to("mps") for k,v in inputs.items()} #Move KV to MPS?
print(inputs)

def generate_tokens_with_past(inputs):
    with torch.no_grad():
        output = model(**inputs)
    logits = output.logits
    last_logits = logits[0, -1, :]
    next_token_id = last_logits.argmax()

    return next_token_id, output.past_key_values

next_inputs = inputs
generated_tokens = []
duration_cached_s = []
for i in range(100):
    print(f"Iteration: {i}")
    t0 = time.time()
    print(f"Current time: {t0}")
    next_token_id, past_key_values = generate_tokens_with_past(next_inputs)
    #device = next_inputs["input_ids"].device
    #next_token_id = next_token_id.to("mps").contiguous()
    print(f"Next Token ID: {next_token_id.reshape((1,1))}")
    new_one = torch.ones((1,1), device = device)
    next_inputs = {
    "input_ids": next_token_id.contiguous().reshape((1,1)),
    "attention_mask": torch.cat([next_inputs["attention_mask"], new_one], dim = 1),
    "past_key_values": DynamicCache.from_legacy_cache(past_key_values)
    }
    #next_inputs.pop("attention_mask", None)
    #print(f"Catenated Inputs: {torch.cat((next_inputs["input_ids"], next_token_id.reshape((1,1))), dim = 1)}")
    #print(f"Past Key Values Slice : {past_key_values.shape()}")
    duration_cached_s += [time.time() - t0]
    print(f"Duration: {duration_cached_s} s")
    #print(f"Attention Mask: {next_inputs['attention_mask']}")
    print(f"Next token : {tokenizer.decode(next_token_id)}")
    generated_tokens.append(tokenizer.decode(next_token_id))

print(f"Cached Duration: {duration_cached_s}")

print(f"Total Time Taken {sum(duration_cached_s)}")
print(generated_tokens)

fig, ax = plt.subplots()
ax.plot(duration_cached_s)
ax.set_xlabel("Tokens")
ax.set_ylabel("Generation Time with Caching")

plt.show()