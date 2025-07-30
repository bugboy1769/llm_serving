import copy
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_dict = {
    "m1": "gpt2",
    "m2": "HuggingFaceTB/SmolLM-135M",
}

model = AutoModelForCausalLM.from_pretrained(model_dict["m1"], local_files_only = True)
tokenizer = AutoTokenizer.from_pretrained(model_dict["m1"], local_files_only = True)
device = ("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

#Paddings
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

prompts = [
    "Complete: the quick brown fox",
    "Moby Dick is a whale",
    "The seas are rough"
]

inputs = tokenizer(prompts, padding = True, return_tensors = "pt")
inputs = inputs.to(device)

def gen_batch_w_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    last_logits = logits[:, -1, :]
    next_token_id = last_logits.argmax(dim = 1)
    return next_token_id, outputs.past_key_values

def gen_batch(inputs, max_tokens):

    generated_tokens = [
       [] for _ in range(inputs["input_ids"].shape[0]) #this is creating a list for the number of prompts, dim = [num_prompts, seq_len]
    ]

    attention_mask = inputs["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1 #Performing Cumulative Sum on the mask
    position_ids.masked_fill(attention_mask == 0, 1) #Filling out 1s where the mask is 0, leaves pos_ids unchanged in this case

    next_inputs = {
        "position_ids": position_ids,
        **inputs
    }

    for _ in range(max_tokens):

        next_token_ids, past_key_values = gen_batch_w_past(next_inputs)
        next_inputs = {
            "input_ids": next_token_ids.reshape((-1, 1)),
            "attention_mask": torch.cat([next_inputs["attention_mask"], torch.ones(next_token_ids.shape[0], 1).to(device)], dim = 1),
            "position_ids": next_inputs["position_ids"][:, -1].unsqueeze(-1) + 1, #pick out last pos ids -> add dim -> increment by 1
            "past_key_values": DynamicCache.from_legacy_cache(past_key_values)
        }
        next_tokens = tokenizer.batch_decode(next_token_ids)
        for i, token in enumerate(next_tokens):
            generated_tokens[i].append(token)
        
    return ["".join(tokens) for tokens in generated_tokens]

#Start to batch
#Make an absurd batch request, with the first prompt requesting a 100 tokens, so effectively holding the batch hostage

#seed idk why
random.seed(42)
#constants
queue_size = 32
batch_size = 8

#request tuples (prompt, tokens) waiting to be processed
request_queue = [
    (prompts[1], 100 if i == 0 else 10) for i in range (queue_size)
]

print(f"Request Queue Sample: {request_queue.__sizeof__()}")

print(f"ReqQ: {request_queue[:8]}")

batches = [
    request_queue[i:(i + batch_size)]
    for i in range(0, len(request_queue), batch_size) #This is creating batches of my original request queue tuples. range() will split the req queue into batch sizes, starting at 0, up until the total request length, with iteration step of batch sizes.
]

print("---------------------------")
for i, batch in enumerate(batches):
    print(f"Test B1: {[b[1] for b in batch]} \n")
    print(f"Max Tokens: {max(b[1] for b in batch)} \n")
print("-----------------------------")

#setup a batch process, with a progress bar
t0 = time.time()
with tqdm(total = len(batches), desc = f"batch_size: {batch_size}") as pbar:
    for i, batch in enumerate(batches): #This is unpacking the tuple enumerate creates
        batch_max_tokens = [b[1] for b in batch] #this then picks out the max_tokens from the batches of request queue, hard coded as we know the first prompt has a token request of over a 100
        max_tokens = max(batch_max_tokens) #picks out the maximum number of tokens
        pbar.set_postfix({'max_tokens': max_tokens})
        batch_prompts = [b[0] for b in batch]
        print(f"Test : -----------> {batch_prompts}")
        inputs = tokenizer(batch_prompts, padding = True, return_tensors = "pt")
        inputs = inputs.to(device)
        results = gen_batch(inputs, max_tokens=max_tokens)
        print(f"Results: {results}")

        pbar.update(1)

        




duration_s = time.time() - t0
print(f"Duration: {duration_s}")

        

        