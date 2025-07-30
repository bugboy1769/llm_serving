from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch
import time
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_dict = {
    "m1": "gpt2",
    "m2": "HuggingFaceTB/SmolLM-135M",
    "m3": "microsoft/Phi-4-mini-reasoning",
    "m4": "EleutherAI/gpt-neo-1.3B",
}

tokenizer = AutoTokenizer.from_pretrained(model_dict["m2"]) #, force_download=True, resume_download=True)
model = AutoModelForCausalLM.from_pretrained(model_dict["m2"]) #, force_download=True, resume_download=True)
model = model.to("mps")
model.config.pad_token_id = model.config.eos_token_id

prompts = ["the quick brown fox jumps over the",
           "quantum mechanics deals with",
           "prompt galore"]

def tokenize_and_move(prompts):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    out = tokenizer(prompts, padding = True, return_tensors = 'pt')
    out = out.to("mps")
    return out

max_tokens = 50
generated_tokens = []

def gen_batch_tokens_w_past(input):
    with torch.no_grad():
        output = model(**input)
    logits = output.logits
    last_logits = logits[:, -1, :]
    next_token_ids = last_logits.argmax(dim = 1)
    past_key_values = output.past_key_values

    return next_token_ids, past_key_values



def gen_response(llm_inputs, num_tokens):
    

    generated_tokens = [
        [] for _ in range(llm_inputs["input_ids"].shape[0])
    ]

    attention_mask = llm_inputs["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) + 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    next_input = {
        "position_ids": position_ids,
        **llm_inputs
    }

    for _ in range(max(num_tokens)):
        next_token_ids, past_key_values = gen_batch_tokens_w_past(next_input)
        # print(f"Next Token ID: {next_token_ids}")
        #this down here is solely creating the next input for autoregressive generation
        next_input = {
            "input_ids": next_token_ids.reshape((-1, 1)),
            "attention_mask": torch.cat([next_input["attention_mask"], torch.ones(next_token_ids.shape[0], 1).to("mps")], dim=1),
            "position_ids": next_input["position_ids"][:, -1].unsqueeze(-1) + 1,
            "past_key_values": DynamicCache.from_legacy_cache(past_key_values),
        }
        print(f"attention_mask: {next_input['attention_mask'].shape[1]}")
        #print(f"next_input: {next_input}")
        #here is the actual decoding of the generated token id
        next_tokens = tokenizer.batch_decode(next_token_ids)
        for i, token in enumerate(next_tokens):
            generated_tokens[i].append(token)
        #print(f"Generated Tokens: {generated_tokens}")
    
    return["".join(tokens) for tokens in generated_tokens] #Need to debug and see what this does

queue_size = 32
batch_size = 8

def batch_creator(prompts, queue_size, batch_size, max_tokens):

    #request_queue = [prompt, tokens_to_be_produced]
    request_queue = [
        (prompts[1], max_tokens if i == 0 else 2) for i in range(queue_size)
    ]

    #print(f"request queue: {request_queue}")

    batches = [request_queue[i:(i + batch_size)] for i in range(0, len(request_queue), batch_size)] #Check if len(queue_size) == len(request_queue)
    
    return batches #since it comes from the request queue, the dim is again [prompt, tokens_to_be_produced]

def generate_batch_response(prompts, queue_size, batch_size, max_tokens):
    duration_s = []
    t0 = time.time()
    #print(f"Prompt: {prompts}")
    batches = batch_creator(prompts, queue_size, batch_size, max_tokens)
    print(f"BATCHES: {batches}")
    # print(f"batches: {[b[0][0] for b in batches]}")
    batch_list = [b[0] for b in batches]
    batch_prompts = [b[0] for b in batch_list]
    num_tokens = [b[1] for b in batch_list]
    #print(f"num_tokens: {num_tokens}")
    #print(f"Debug: {batch_list}")
    llm_inputs = tokenize_and_move(prompts= batch_prompts)
    #print(f"llm_inputs: {llm_inputs}")
    llm_output_list = gen_response(llm_inputs=llm_inputs, num_tokens=num_tokens)
    duration_s.append(time.time() - t0)

    return llm_output_list, duration_s

output, duration = generate_batch_response(prompts=prompts, queue_size=queue_size, batch_size=batch_size, max_tokens=50)

print(f"Final Output: {output}")
print(f"duration: {duration}")













