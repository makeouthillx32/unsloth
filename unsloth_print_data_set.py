from my_constants import ALPACA_TEMPLATE, max_seq_length, load_in_4bit, dtype, model_name
from datasets import load_dataset

EOS_TOKEN = "<|end_of_text|>"
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = ALPACA_TEMPLATE.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)

    result = { "text" : texts, }
    print(result)
    return result


dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
print(dataset.column_names)
dataset_mapped = dataset.map(formatting_prompts_func, batched = True,)
print("printout columns")
print(dataset_mapped.column_names)