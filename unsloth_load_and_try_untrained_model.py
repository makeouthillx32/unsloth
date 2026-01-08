from unsloth import FastLanguageModel
from my_constants import ALPACA_TEMPLATE, model_name, max_seq_length, load_in_4bit, dtype, test_prompt

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,

    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

inputs = tokenizer(
[
    ALPACA_TEMPLATE.format(
        test_prompt, # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)