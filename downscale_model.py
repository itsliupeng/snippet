import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn

MODEL_DIR = "."
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

original_encoder_layers = model.model.layers


##

layer_idx = [ 5,  4,  1,  3, 53, 54,  7, 51,  9,  8, 52, 50,  6, 16, 49, 48, 55,
       47, 15, 17, 10, 14, 19, 12, 11, 13, 20, 18, 56, 46, 45, 21, 57, 44,
       32, 43, 31, 29, 30, 24, 42, 27, 26, 28, 33, 25, 34, 39, 40, 41, 35,
       38, 36, 37, 23, 58,  2, 22, 59,  0]

new_encoder_layers_list = []
for idx, l in enumerate(original_encoder_layers):
    if idx not in layer_idx[:(60-16)]:
        new_encoder_layers_list.append(l)

model.model.layers = nn.ModuleList(new_encoder_layers_list)

model.config.num_hidden_layers = len(model.model.layers)


# Save the new model
model.save_pretrained("model_ds", safe_serialization=False)


model = AutoModelForCausalLM.from_pretrained("model_ds", torch_dtype="auto")
model.save_pretrained("model_ds_A", safe_serialization=False)
