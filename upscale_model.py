import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn

MODEL_DIR = "."
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model.save_pretrained("safe", safe_serialization=True)

original_encoder_layers = model.model.layers


#

# new_encoder_layers_list = []
# for idx, l in enumerate(original_encoder_layers):
#     if idx in list(range(3, 21)):
#         new_encoder_layers_list.append(l)
#
# model.model.layers = nn.ModuleList(new_encoder_layers_list)


first_layers = original_encoder_layers[:8]
middle_layer = original_encoder_layers[8:-8]
last_layers = original_encoder_layers[-8:]

new_middle_list = []
for x in middle_layer:
    new_middle_list.append(x)
    new_middle_list.append(x)
# Combine the layers
# new_encoder_layers = first_layers + list(middle_layer) * 2 + last_layers
new_encoder_layers = first_layers + list(new_middle_list) + last_layers

model.model.layers = new_encoder_layers
model.config.num_hidden_layers = len(new_encoder_layers)

# Save the new model
# model.save_pretrained("safeA", max_shard_size="10GB", safe_serialization=True)
model.save_pretrained("safe", safe_serialization=False)

# save again to remove shared parameter
model = AutoModelForCausalLM.from_pretrained("safe", torch_dtype="auto")
model.save_pretrained("model_ds_A", safe_serialization=False)
