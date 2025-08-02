from batch_helpers import init_batch, get_next_input, generate_next_token, generate_tokens_with_past, merge_batches, filter_batch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import time
from tqdm import tqdm


model_dict = {
    "m1": "gpt2",
    "m2": "HuggingFaceTB/SmolLM-135M"
}

model = AutoModelForCausalLM.from_pretrained(model_dict["m2"])
tokenizer = AutoTokenizer.from_pretrained(model_dict["m2"])
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

random.seed(42)
queue_size = 32
batch_size = 8

request_queue = [
    (prompts[0], 100 if i % batch_size == 0 else 10)
    for i in range(queue_size)
]

t0 = time.time()
with tqdm(total = len(request_queue), desc = f"bs_:{batch_size}") as pbar:
    batch = init_batch(request_queue[:batch_size])
    print(f"Batch: {batch.keys()}")  #-> 2nd debug statement
    cached_batch = generate_next_token(batch)
    #print(f"Cached_Batch: {cached_batch}")
    request_queue = request_queue[batch_size:]
    print(f"Request Queue: {request_queue}")
    i = 0
    while (
        len(request_queue) > 0 or cached_batch["input_ids"].size(0)
    ):
        print(f"Iteration: {i}")
        i += 1
        batch_capacity = batch_size - cached_batch["input_ids"].size(0)
        if batch_capacity > 0 and len(request_queue) > 0:
            print("IF TEST")
            new_batch = init_batch(request_queue[:batch_capacity])
            new_batch = generate_next_token(new_batch)
            request_queue = request_queue[batch_capacity:]

            cached_batch = merge_batches(cached_batch, new_batch)
            cach_len = cached_batch["past_key_values"]
            print(f"Cached Batch Post Merge: {len(cach_len)}")
        
        cached_batch = generate_next_token(cached_batch)
        cached_batch, remove_indices = filter_batch(cached_batch)
        pbar.update(len(remove_indices))

duration_s = time.time() - t0
print(f"duration_: {duration_s}")