from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from my_constants import ALPACA_TEMPLATE, model_name, max_seq_length, load_in_4bit, dtype, test_prompt

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "lora_model",
#     max_seq_length = max_seq_length,
#
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )
# tokenizer = AutoTokenizer.from_pretrained("lora_model")
# tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-v0.3-bnb-4bit")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print(f"Expected chat template from model metadata: {tokenizer.chat_template}")
# # You can also use the apply_chat_template method to see how a conversation would be formatted
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "What is the capital of France?"},
# ]
#
# # The apply_chat_template method can be used to format messages into the model's expected prompt string
# formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# print("\nFormatted Prompt:")
# print(formatted_prompt)