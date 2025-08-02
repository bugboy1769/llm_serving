from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch
import time
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import numpy as np

model_dict = {
    "m1": "gpt2",
    "m2": "HuggingFaceTB/SmolLM-135M"
}

model = AutoModelForCausalLM.from_pretrained(model_dict["m1"])
tokenizer = AutoTokenizer.from_pretrained(model_dict["m1"])
device = ("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

#Paddings
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

prompts = [
    "fractional reserve banking is defined as",
    "quantitative easing a tool used to",
    "balance sheets are the aggregate of",
]

def generate_tokens_with_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    last_logits = logits[:, -1, :]
    next_token_ids = last_logits.argmax(dim = 1)

    return next_token_ids, outputs.past_key_values

def get_next_input(batch, next_token_ids, past_key_values, next_tokens):

    return {
        "input_ids": next_token_ids.reshape((-1, 1)),
        "attention_mask": torch.cat([batch["attention_mask"], torch.ones((next_token_ids.shape[0], 1)).to(device)], dim = 1), #somehow Im querying batch["attention_mask"], this batch might be different than just (prompt, token) tuple
        "position_ids": batch["position_ids"][:, -1].unsqueeze(-1) + 1,
        "past_key_values": past_key_values,
        "responses": [
            r1 + r2 for r1, r2 in zip(batch["responses"], next_tokens)
        ],
        "tokens_remaining": [v - 1 for v in batch["tokens_remaining"]],
    }

def init_batch(requests):

    prompts = [r[0] for r in requests]
    inputs = tokenizer(prompts, padding = True, return_tensors = 'pt')
    inputs = inputs.to(device)

    attention_mask = inputs["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    # past_key_values = []

    batch = {
        "position_ids": position_ids,
        "responses": copy.copy(prompts),
        "tokens_remaining": [r[1] for r in requests],
        #"past_key_values": DynamicCache.from_legacy_cache(past_key_values),
        **inputs,
    }
    #print(f"batch: {batch}") -> 1st degbug statement

    return batch


def generate_next_token(batch):

    inputs = copy.copy(batch)
    #print(f"Inputs: {inputs.keys()}")
    inputs.pop("responses")
    inputs.pop("tokens_remaining")
    # past_key_values = []
    # inputs = {
    #     "past_key_values": DynamicCache.from_legacy_cache(past_key_values),
    #     **inputs
    # }

    next_token_ids, past_key_values = generate_tokens_with_past(inputs)
    print(f"Next Token IDs: {next_token_ids}")
    print(f"Past Key Values: {len(past_key_values)}")
    next_tokens = tokenizer.batch_decode(next_token_ids)

    return get_next_input(batch, next_token_ids, past_key_values, next_tokens)

def merge_batches(batch1, batch2):

    attn_mask1 = batch1["attention_mask"]
    attn_mask2 = batch2["attention_mask"]
    max_seq_len = max(attn_mask1.shape[1], attn_mask2.shape[1])

    padding1 = max_seq_len - attn_mask1.shape[1]
    padding2 = max_seq_len - attn_mask2.shape[1]
    attn_mask1 = F.pad(attn_mask1, (padding1, 0), "constant", 0)
    attn_mask2 = F.pad(attn_mask2, (padding2, 0), "constant", 0)

    past_kv1 = batch1["past_key_values"]
    past_kv2 = batch2["past_key_values"]

    padded_kv1 = []
    for i in range(len(past_kv1)):
        k, v = past_kv1[i]
        k = F.pad(k, (0, 0, padding1, 0), "constant", 0)
        v = F.pad(v, (0, 0, padding1, 0), "constant", 0)
        padded_kv1.append((k,v))

    padded_kv2 = []
    for i in range(len(past_kv2)):
        k, v = past_kv2[i]
        k = F.pad(k, (0, 0, padding2, 0), "constant", 0)
        v = F.pad(v, (0, 0, padding2, 0), "constant", 0)
        padded_kv2.append((k, v))
    
    #print(f"DEBUG!!!!!: {padded_kv1}")
    #print(f"DEBUG!!!!!: {padded_kv2}")
    input_ids = torch.concat(
        [batch1["input_ids"], batch2["input_ids"]], dim = 0
    )
    position_ids = torch.concat(
        [batch1["position_ids"], batch2["position_ids"]], dim = 0
    )
    attn_mask = torch.concat([attn_mask1, attn_mask2], dim = 0)

    past_kv = []
    for i in range(len(padded_kv1)):
        k1, v1 = padded_kv1[i]
        k2, v2 = padded_kv2[i]
        k = torch.concat([k1, k2], dim = 0)
        v = torch.concat([v1, v2], dim = 0)
        past_kv.append((k, v))
    #print(f"past_kv!!!!!!!! --------> {past_kv}")

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attn_mask,
        "past_key_values": DynamicCache.from_legacy_cache(past_kv),
        "responses": batch1["responses"] + batch2["responses"],
        "tokens_remaining": batch1["tokens_remaining"] + batch2["tokens_remaining"]
    }

def filter_batch(batch):

    remove_indices = []
    for i, tokens_remaining in enumerate(batch["tokens_remaining"]):
        if tokens_remaining <= 0:
            remove_indices.append(i) #Catching indices to be removed
    
    batch_size = batch["input_ids"].size(0)
    mask = torch.ones(batch_size, dtype=torch.bool)
    mask[remove_indices] = False

    input_ids = batch["input_ids"][mask]
    position_ids = batch["position_ids"][mask]
    attention_mask = batch["attention_mask"][mask]
    responses = [
        r for i, r in enumerate(batch["responses"])
        if i not in remove_indices
    ]
    tokens_remaining = [
        v for i, v in enumerate(batch["tokens_remaining"])
        if i not in remove_indices
    ]
    past_key_values = batch["past_key_values"]
    new_past_key_values = []
    for i in range(len(past_key_values)):
        k, v = past_key_values[i]
        k = k[mask]
        v = v[mask]
        new_past_key_values.append((k, v))
    past_key_values = new_past_key_values

    if input_ids.size(0) > 0:
        zero_mask = attention_mask == 0
        cumprod = zero_mask.cumprod(dim = 1)
        leading_zeroes_count = cumprod.sum(dim = 1)
        min_leading_zeroes = torch.min(leading_zeroes_count)
        truncation_offset = min_leading_zeroes.item()

        attention_mask = attention_mask[:, truncation_offset:]
        past_key_values = past_key_values
        new_past_key_values = []
        for i in range(len(past_key_values)):
            k, v = past_key_values[i]
            k = k[:, truncation_offset:, :]
            v = v[:, truncation_offset:, :]
            new_past_key_values.append((k, v))
        past_key_values = new_past_key_values
    
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "responses": responses,
        "tokens_remaining": tokens_remaining,
    }, remove_indices

