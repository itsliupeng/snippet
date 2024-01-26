import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "."
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

original_encoder_layers = model.model.layers

idx = 8
# Extract the required layers
first_layers = original_encoder_layers[:idx]
last_layers = original_encoder_layers[-idx:]
middle_layer = original_encoder_layers[idx:-idx]

# Combine the layers
new_encoder_layers = first_layers + list(middle_layer) * 2 + last_layers

model.model.layers = new_encoder_layers
model.config.num_hidden_layers = len(new_encoder_layers)

# Save the new model
model.save_pretrained("model_ds", safe_serialization=False)

# save again to remove shared parameter
model = AutoModelForCausalLM.from_pretrained("model_ds", torch_dtype="auto")
model.save_pretrained("model_ds_A", safe_serialization=False)
