import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn

MODEL_DIR = "."
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

original_encoder_layers = model.model.layers


##

new_encoder_layers_list = []
for idx, l in enumerate(original_encoder_layers):
    if idx not in [28, 29, 27]:
        new_encoder_layers_list.append(l)

model.model.layers = nn.ModuleList(new_encoder_layers_list)

model.config.num_hidden_layers = len(model.model.layers)


# Save the new model
model.save_pretrained("model_ds", safe_serialization=False)


model = AutoModelForCausalLM.from_pretrained("model_ds", torch_dtype="auto")
model.save_pretrained("model_ds_A", safe_serialization=False)
