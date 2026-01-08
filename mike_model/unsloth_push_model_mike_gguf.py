import os

from unsloth import FastLanguageModel

from my_constants import ALPACA_TEMPLATE

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
model_name_lora = "lora_mike"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name_lora, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

inputs = tokenizer(
[
    ALPACA_TEMPLATE.format(
        # "What are the rules of soccer",
        "Tell me about famous programmers in Denmark. Make the response long and detailed", # instruction
        # "Tell me about Denmark and technology. Make the response long and elaborate", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

gguf_model_name="lora_mike_gguf"
model.save_pretrained(model_name_lora)  # Local saving
tokenizer.save_pretrained(model_name_lora)  # Local saving
# print(f"model: {tokenizer._ollama_modelfile}
model.save_pretrained_gguf(gguf_model_name, tokenizer)  # Local saving
# tokenizer.save_pretrained_gguf(gguf_model_name)
# tokenizer.push_to_hub_gguf("ekim197711/lora_mike_gguf",tokenizer, token=os.environ['HF_TOKEN']) # Online saving
# quantizers "q4_k_m","q8_0", "q5_k_m"
# model.push_to_hub_gguf("ekim197711/lora_mike",tokenizer,quantization_method=["q4_k_m"], token=os.environ['HF_TOKEN']) # Online saving
