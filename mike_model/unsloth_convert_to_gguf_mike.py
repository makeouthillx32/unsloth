import os

from unsloth import FastLanguageModel

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_mike", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
# print(tokenizer._ollama_modelfile)
model.save_pretrained_gguf("lora_mike_gguf", tokenizer, quantization_method = "q4_k_m")
# model.push_to_hub("ekim197711/lora_mike", token=os.environ['HF_TOKEN']) # Online saving
# tokenizer.push_to_hub("ekim197711/lora_mike", token=os.environ['HF_TOKEN']) # Online saving
