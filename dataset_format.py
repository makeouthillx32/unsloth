from my_constants import ALPACA_TEMPLATE, CHATML_TEMPLATE


def formatting_prompts_to_alpaca(examples, end_of_string_token):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = ALPACA_TEMPLATE.format(instruction, input, output) + end_of_string_token
        texts.append(text)
    return { "text" : texts }

def formatting_prompts_to_alpaca_mikes_data(data: dict, end_of_string_token):
    texts = []
    for conv in data['conversations']:
        print(f"id {conv['id']} ")
        user = ''
        assistant = ''
        for message in conv['messages']:
            if message['role'] == 'user':
                user = message['content']
            if message['role'] == 'assistant':
                assistant = message['content']
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = ALPACA_TEMPLATE.format(user, "", assistant) + end_of_string_token
        texts.append(text)
        print(f"appended text {text}")
    return { "text" : texts }

def formatting_prompts_to_chatml(examples, end_of_string_token):
    inputs       = examples["input"]
    outputs      = examples["output"]
    system = examples["system"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = CHATML_TEMPLATE.format("", instruction + input, output) + end_of_string_token
        texts.append(text)
    return { "text" : texts }