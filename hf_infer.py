import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_path = '/gpfs/public/pretrain/liupeng/code/mla/MLA_Megatron-LM/out/test_yi_6b_4m_bs1024_load_d1009_w0_banlance_loss_freeze_whisperm/test_yi_6b_4m_bs1024_load_d1009_w0_banlance_loss_freeze_whisperm/checkpoint/iter_0032000_hf'
model_path = "."
# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

raw_model = model
model = raw_model.language_model
# model.model.embed_tokens.weight
#  model.lm_head.weight.sum()

llm_model = raw_model.language_model

tokenizer = AutoTokenizer.from_pretrained("/lp/models/Yi-6B", use_fast=False)
# Prompt content: "hi"
messages = "what's your name?"

input_ids = tokenizer(messages, return_tensors='pt')['input_ids']
output_ids = llm_model.generate(input_ids.to('cuda'), max_length=50, num_return_sequences=1)
response = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)


####################################################################

from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# 加载模型和分词器
model_name = "/lp/models/Yi-6B"
yi_model = LlamaForCausalLM.from_pretrained(model_name,  torch_dtype=torch.bfloat16,)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# 设置生成参数
prompt = "Once upon a time in a distant land,"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# 使用 GPU（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
input_ids = input_ids.to(device)

# 生成文本
with torch.no_grad():
    output = yi_model.generate(input_ids.to('cuda'), max_length=50, num_return_sequences=1)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

