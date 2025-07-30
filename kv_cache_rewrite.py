import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DynamicCache

#Loading up the model
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(model)

prompt = "the quick brown fox jumps over the"

inputs = tokenizer(prompt, return_tensors = 'pt')


print(f"Inputs: {inputs}")

def generate_tokens_with_past(inputs, past_key_values = None):
    with torch.no_grad():
        outputs = model(**inputs, past_key_values = past_key_values)
    logits = outputs.logits
    last_logits = logits[0, -1, :]
    next_token_id = last_logits.argmax()
    print(f"Test!!!: {next_token_id}")
    return next_token_id, outputs.past_key_values

#KV Caching
generated_tokens = []
next_inputs = inputs
duration_cached_s = []
past_kv_cache = None
for i in range(10):
    print(f"Iteration: {i}")
    t0 = time.time()
    if i == 0:
        next_token_id, past_key_values = generate_tokens_with_past(next_inputs)
    else:
        next_inputs = {
            "input_ids": torch.cat([next_inputs["input_ids"], next_token_id.reshape((1,1))], dim=1),
            #"attention_mask": torch.cat([next_inputs["attention_mask"], torch.tensor([[1]])], dim=1),
        }
        next_token_id, past_key_values = generate_tokens_with_past(next_inputs, past_kv_cache)
    print(f"Next Token ID: {next_token_id} ")
    
    duration_cached_s += [time.time() - t0]
    
    print(f"Duration for Caching: {duration_cached_s}")

    if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
        past_kv_cache = DynamicCache.from_legacy_cache(past_key_values)
    else:
        past_kv_cache = past_key_values
    
    next_token = tokenizer.decode(next_token_id)
    generated_tokens.append(next_token)
    print(f"Generated: '{next_token}'")

print(f"{sum(duration_cached_s)} s")
print(generated_tokens)