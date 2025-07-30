import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DynamicCache

model_name = 'HuggingFaceTB/SmolLM-135M'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
model.config.pad_token_id = model.config.eos_token_id
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

#Multiple prompts in a list
prompts = ["Complete: Spain is a country in the continent of ",
          "Complete: Large Language Models are",
          "Complete: There is far too little grace in letting evil fester"]

#Inputs with paddings to maintain list integrity
inputs = tokenizer(prompts, padding = True, return_tensors = 'pt')
inputs = inputs.to(device)
#inputs = {k: v.to("mps") for k,v in inputs.items()} #Move KV to MPS?
string = "input_ids"
print(inputs["input_ids"])
print(f"Input_ID_Shape: {inputs[string].shape}")
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
        print(f"Debug!: {next_token_ids}")
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

generated_tokens = generate_batch(inputs, max_tokens= 10)

for prompt, generated in zip(prompts, generated_tokens):
    print(prompt, f"\x1b[31m{generated}\x1b[0m\n")

#constant
max_tokens = 5
#observables
durations = []
throughputs = []
latencies = []

batch_sizes = [2**p for p in range(4)]
for batch_size in batch_sizes:
    print(f"bs = {batch_size}")

    #generate tokens for batch and record duration
    t0 = time.time()
    batch_prompts = [
        prompts[i % len(prompts)] for i in range(batch_size)
    ]
    print(f"Batch Prompts: {batch_prompts}")

    inputs = tokenizer(batch_prompts, padding =True, return_tensors = "pt")
    inputs = inputs.to(device)
    generated_tokens = generate_batch(inputs, max_tokens=max_tokens)
    duration_s = time.time() - t0

    ntokens = batch_size*max_tokens
    throughput = ntokens/duration_s
    avg_latency = duration_s/max_tokens

    # print(f"Duration: {duration_s:.3f}s")
    # print(f"Batch size: {batch_size}")
    # print(f"Latency per request: {avg_latency:.3f}s")
    # print("------------------------------")

    print(f"N_Tokens: {ntokens}")
    print(f"Throughput: {throughput}")
    print(f"Avg Latency: {avg_latency}")
    print("------------------------------")

    durations.append(duration_s)
    throughputs.append(throughput)
    latencies.append(avg_latency)

def render_plot(x, y1, y2, x_label, y1_label, y2_label):
    fig, ax1 = plt.subplots()

    colour = 'tab:red'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label, color = colour)
    ax1.plot(x, y1, color = colour)
    ax1.tick_params(axis = 'y', labelcolor = colour)

    ax1.set_xscale("log", base = 2)

    ax2 = ax1.twinx()
    colour = 'tab:blue'
    ax2.set_ylabel(y2_label, color = colour)
    ax2.plot(x, y2, color = colour)
    ax2.tick_params(axis = 'y', labelcolor = colour)

    plt.show()

render_plot(batch_sizes, throughputs, latencies, "Batch Size", "Throughput (tokens/s)", "Latency (s/max_tokens)")