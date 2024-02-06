import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "."
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

pad_id = 0

original_embedding = model.model.embed_tokens
# Create a new embedding layer with the desired size [96000, 4096]
new_embedding = nn.Embedding(96000, 4096, padding_idx=0, dtype=original_embedding.weight.dtype)
# Copy the weights from the original embedding layer to the new one
with torch.no_grad():
    new_embedding.weight[:64000] = original_embedding.weight
    new_embedding.weight[64000:] = original_embedding.weight[pad_id].unsqueeze(0).expand(96000-64000, -1)

model.model.embed_tokens = new_embedding

model.config.vocab_size = 96000

original_head = model.lm_head
new_head = nn.Linear(4096, 96000, bias=False)
with torch.no_grad():
    new_head.weight[:64000] = original_head.weight
    new_head.weight[64000:] = original_head.weight[pad_id].unsqueeze(0).expand(96000-64000, -1)

model.lm_head = new_head


# Save the new model
model.save_pretrained("model_ds", safe_serialization=False)


