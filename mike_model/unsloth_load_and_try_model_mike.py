from unsloth import FastLanguageModel
from my_constants import ALPACA_TEMPLATE, test_prompt, test_prompt_copenhagen

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_mike", # YOUR MODEL YOU USED FOR TRAINING
    # model_name = "ekim197711/lora_mike", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

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