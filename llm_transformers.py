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

#Setting up the prompt and tokenising the input
prompt = "the quick brown fox jumps over the"
inputs = tokenizer(prompt, return_tensors = 'pt')
print(inputs)

#Function to generate the next token
def generate_tokens(inputs):
    with torch.no_grad():
        outputs = model(**inputs) # Because feedforward, we don't need pytorch to update or store grads
    logits = outputs.logits #Storing all the logits that the model produces
    last_logits = logits[0, -1, :] #Pick out the final logit as we are doing autoregressive generation
    next_token_id = last_logits.argmax() #Applying argmax to sample out the most likely token_id

    return next_token_id

def generate_tokens_with_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    last_logits = logits[0, -1, :]
    next_token_id = last_logits.argmax()
    return next_token_id, outputs.past_key_values

#Without KV Caching
# generated_tokens = []
# next_inputs = inputs
# duration_s = []
# for _ in range(5):
#     t0 = time.time() #Store current time
#     next_token_id = generate_tokens_with_past(next_inputs) # Generate next_token_id
#     duration_s += [time.time() - t0] #Catch time taken
#     #Below, we append the new token id to the existing tokens stored in inputs. We also update the mask, 1 signifies pay attention and all positions are 1 since they are all important to the sentence, for padding stuff we can have 0.
#     next_inputs = {
#         "input_ids": torch.cat(
#             [inputs["input_ids"], next_token_id.reshape(1,1)], dim=1
#         ),
#         "attention_mask": torch.cat(
#             [inputs["attention_mask"], torch.tensor([[1]])], dim=1
#         ),
#     }
#     # Decode the token into word, this is highly likely a linear layer that just maps embeddings to a word list
#     next_token = tokenizer.decode(next_token_id)
#     #Append the token to the existing tokens
#     generated_tokens.append(next_token)

#With KV Caching
# generated_tokens = []
# next_inputs = inputs
# duration_s = []
# current_attention_mask = inputs["attention_mask"]
# for i in range(5):
#     t0 = time.time() #Store current time
#     if i == 0:
#         next_token_id, past_key_values = generate_tokens_with_past(next_inputs) # Generate next_token_id
#         current_attention_mask = torch.cat([current_attention_mask, torch.tensor([[1]])], dim=1)
    
#     else:
#         #Below, we append the new token id to the existing tokens stored in inputs. We also update the mask, 1 signifies pay attention and all positions are 1 since they are all important to the sentence, for padding stuff we can have 0.
#         next_inputs = {
#             "input_ids": next_token_id.reshape((1,1)),
#             "attention_mask": current_attention_mask,
#             "past_key_values": DynamicCache.from_legacy_cache(past_key_values)
#         }
#         next_token_id, past_key_values = generate_tokens_with_past(next_inputs) # Generate next_token_id
#         current_attention_mask = torch.cat([current_attention_mask, torch.tensor([[1]])], dim=1)

#     duration_s += [time.time() - t0] #Catch time taken
#     # Decode the token into word, this is highly likely a linear layer that just maps embeddings to a word list
#     next_token = tokenizer.decode(next_token_id)
#     #Append the token to the existing tokens
#     generated_tokens.append(next_token)

generated_tokens = []
duration_s = []
past_key_values = None

for i in range(5):
    t0 = time.time()
    
    if i == 0:
        # First iteration: use original inputs
        next_token_id, past_key_values = generate_tokens_with_past(inputs)
    else:
        # Subsequent iterations: single token with single attention mask
        next_inputs = {
            "input_ids": next_token_id.reshape((1,1)),
            "attention_mask": torch.tensor([[1]]),  # Only for the new token
            "past_key_values": DynamicCache.from_legacy_cache(past_key_values)
        }
        next_token_id, past_key_values = generate_tokens_with_past(next_inputs)
    
    duration_s += [time.time() - t0]
    next_token = tokenizer.decode(next_token_id)
    generated_tokens.append(next_token)

print(f"{sum(duration_s)} s")
print(generated_tokens)

plt.plot(duration_s)
plt.show()