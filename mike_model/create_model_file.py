import json

from datasets import Dataset
from pygments.lexers.configs import ApacheConfLexer
from unsloth import apply_chat_template, FastLanguageModel

from dataset_format import formatting_prompts_to_alpaca_mikes_data
from my_constants import ALPACA_TEMPLATE

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
print('load from pretrained')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_mike", # YOUR MODEL YOU USED FOR TRAINING
    # model_name = "ekim197711/lora_mike", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
with open('./dataset_files/denmark_large.json') as json_data_file:
    data_dict = json.load(json_data_file)
print('prepare model for inference')
FastLanguageModel.for_inference(model)
end_of_string_token = tokenizer.eos_token
print("format the dataset")
mikes_dataset_dict = formatting_prompts_to_alpaca_mikes_data(data_dict, end_of_string_token)
mikes_dataset = Dataset.from_dict(mikes_dataset_dict)
print(f"mikes columns: {mikes_dataset.column_names}")
print('apply chat template')
dataset = apply_chat_template(
    mikes_dataset,
    tokenizer = tokenizer,
    chat_template = ALPACA_TEMPLATE,
    # default_system_message = "You are a helpful assistant", << [OPTIONAL]
)
print(tokenizer._ollama_modelfile)