import json

from datasets import Dataset

from dataset_format import formatting_prompts_to_alpaca_mikes_data
from my_constants import ALPACA_TEMPLATE, max_seq_length, load_in_4bit, dtype, model_name
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

with open('./dataset_files/denmark_large.json') as json_data_file:
    data_dict = json.load(json_data_file)
print(type(data_dict))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/granite-4.0-h-tiny-base-unsloth-bnb-4bit",
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
end_of_string_token = tokenizer.eos_token
mikes_dataset_dict = formatting_prompts_to_alpaca_mikes_data(data_dict, end_of_string_token)
mikes_dataset = Dataset.from_dict(mikes_dataset_dict)
print(f"mikes columns: {mikes_dataset.column_names}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=mikes_dataset,
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
model.save_pretrained("lora_mike_granite")  # Local saving
tokenizer.save_pretrained("lora_mike_granite")
gguf_model_name="lora_mike_gguf"
model.save_pretrained_gguf(gguf_model_name, tokenizer)
# # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving