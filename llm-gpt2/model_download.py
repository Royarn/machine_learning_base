# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

local_dir = "./model"

tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall", cache_dir=local_dir)

print(tokenizer)
model = AutoModelForCausalLM.from_pretrained("uer/gpt2-chinese-cluecorpussmall", cache_dir=local_dir)

print(model)