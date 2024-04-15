import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.activations import GELUActivation
from torch import nn
import math
import json

config = json.load(open("half_config.json"))
config = LlamaConfig(**config)
new_model = LlamaForCausalLM(config)

MODEL_DIR = "."
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")


ratio = 0.5

# embedding
w = model.model.embed_tokens.weight
noise = torch.randn_like(w)
add_noise = noise / torch.norm(noise) * torch.norm(w) * ratio
ww = torch.concat([w, w+add_noise], dim=-1)
new_model.model.embed_tokens.weight = nn.Parameter(ww)

# lm_head
w = model.lm_head.weight
noise = torch.randn_like(w)
add_noise = noise / torch.norm(noise) * torch.norm(w) * ratio
ww = torch.cat([w, w+add_noise], dim=-1) / 2
new_model.lm_head.weight = nn.Parameter(ww)

def w_2x2(w):
    noise = torch.randn_like(w)
    add_noise = noise / torch.norm(noise) * torch.norm(w) * ratio
    w_h = torch.concat([w, w+add_noise], -1)
    return torch.concat([w_h, w_h], 0)

# layers
for idx, l in enumerate(model.model.layers):
    print(f"layer {idx}")
    new_layer = new_model.model.layers[idx]

    # input_layernorm
    w = l.input_layernorm.weight
    noise = torch.randn_like(w)
    add_noise = noise / torch.norm(noise) * torch.norm(w) * ratio
    ww = torch.concat([w, w+add_noise], -1)
    new_layer.input_layernorm.weight = nn.Parameter(ww)

    # self_attn
    qw = l.self_attn.q_proj.weight
    kw = l.self_attn.k_proj.weight
    vw = l.self_attn.v_proj.weight
    ow = l.self_attn.o_proj.weight

    qww = w_2x2(qw) / 2
    kww = w_2x2(kw) / 2
    vww = w_2x2(vw) / 2
    oww = w_2x2(ow) / 2

    new_layer.self_attn.q_proj.weight = nn.Parameter(qww)
    new_layer.self_attn.k_proj.weight = nn.Parameter(kww)
    new_layer.self_attn.v_proj.weight = nn.Parameter(vww)
    new_layer.self_attn.o_proj.weight = nn.Parameter(oww)

    # swiglu
    f1w = l.mlp.gate_proj.weight
    f2w = l.mlp.up_proj.weight
    f3w = l.mlp.down_proj.weight

    f1ww = w_2x2(f1w) / 2
    f2ww = w_2x2(f2w) / 2
    f3ww = w_2x2(f3w) / 2

    new_layer.mlp.gate_proj.weight = nn.Parameter(f1ww)
    new_layer.mlp.up_proj.weight = nn.Parameter(f2ww)
    new_layer.mlp.down_proj.weight = nn.Parameter(f3ww)

    # post_attention_layernorm
    w = l.post_attention_layernorm.weight
    noise = torch.randn_like(w)
    add_noise = noise / torch.norm(noise) * torch.norm(w) * ratio
    ww = torch.concat([w, w+add_noise], -1)
    new_layer.post_attention_layernorm.weight = nn.Parameter(ww)


# last norm
w = model.model.norm.weight
noise = torch.randn_like(w)
add_noise = noise / torch.norm(noise) * torch.norm(w) * ratio
ww = torch.concat([w, w+add_noise], -1)
new_model.model.norm.weight = nn.Parameter(ww)

# Save the new model
new_model.save_pretrained("model_ds", safe_serialization=False)

# # save again to remove shared parameter
# model = AutoModelForCausalLM.from_pretrained("model_ds", torch_dtype="auto")
# model.save_pretrained("model_ds_A", safe_serialization=False)


