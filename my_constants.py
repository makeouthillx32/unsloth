# Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
ALPACA_TEMPLATE = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""

CHATML_TEMPLATE =  """<|im_start|>system
{}<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""

model_name = "unsloth/llama-3-8b-bnb-4bit"
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
test_prompt = "What is a famous tall tower in Paris?"
test_prompt_copenhagen = "What is a famous tall tower in Copenhagen?"