## LLM Inference Optimisation on a Local Macbook

The code is largely from https://learn.deeplearning.ai/courses/efficiently-serving-llms, it is my implementation of it on a macbook. The key change is moving everything to MPS. Sometimes, some torch initialisations occur natively on the CPU, without explicit movement you're thrown a SIGBUS error your way. Also, while debugging, Claude failed to identify the problem effectively over multiple exchanges while ChatGPT (Think Longer) one-shotted it.
Currently in repo:
  * Vanilla Generation
  * KV Caching - Doesn't work too well for llama style models have they have much more stringent implementations for passing past_key_values.
  * Continuous Batching - Currently fixing an error which causes kv caching to fail at the last token to be generated in the first batch.
  * Nice Graphs and Conceptual Comments

ToDo: Quantization and LoRA.
