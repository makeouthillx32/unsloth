from dataset_format import formatting_prompts_to_alpaca, formatting_prompts_to_chatml
from my_constants import ALPACA_TEMPLATE, max_seq_length, load_in_4bit, dtype, model_name

print("imports")
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.

# model_name = "unsloth/mistral-7b-v0.3-bnb-4bit"
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
] # More models at https://huggingface.co/unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# end_of_string_token = tokenizer.eos_token # Must add EOS_TOKEN
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
from unsloth import to_sharegpt
print(dataset.column_names)
dataset_sharegpt = to_sharegpt(
    dataset,
    merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
    output_column_name="output",
    conversation_extension=3,  # Select more to handle longer conversations
)
print(dataset_sharegpt.column_names)
print(dataset_sharegpt['conversations'][12])

from unsloth import standardize_sharegpt
dataset_standard = standardize_sharegpt(dataset=dataset_sharegpt)
print(dataset_standard.column_names)
print(dataset_standard['conversations'][12])
chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.
### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

from unsloth import apply_chat_template

dataset_templated = apply_chat_template(
    dataset_standard,
    tokenizer=tokenizer,
    chat_template=chat_template,
    # default_system_message = "You are a helpful assistant", << [OPTIONAL]
)
print(dataset_templated.column_names)
print(dataset_templated['text'][12])

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_templated,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Use TrackIO/WandB etc
    ),
)
trainer_stats = trainer.train()
#
#
#
#
model.save_pretrained("lora_model_chatml")  # Local saving
tokenizer.save_pretrained("lora_model_chatml")
# # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving