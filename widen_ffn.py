import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.activations import GELUActivation
from torch import nn
import math
import json

# config = json.load(open("wide_ffn_config.json"))
config = json.load(open("config.json"))
config = LlamaConfig(**config)
new_model = LlamaForCausalLM(config)
m = new_model

MODEL_DIR = "."
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")


ratio = 0.1

# embedding
w = model.model.embed_tokens.weight
new_model.model.embed_tokens.weight = nn.Parameter(w)

# lm_head
w = model.lm_head.weight
new_model.lm_head.weight = nn.Parameter(w)

def w_2up(w):
    noise = torch.randn_like(w)
    add_noise = noise / torch.norm(noise) * torch.norm(w) * ratio
    return torch.concat([w, w+add_noise], 0)

def w_2down(w):
    noise = torch.randn_like(w)
    add_noise = noise / torch.norm(noise) * torch.norm(w) * ratio
    return torch.concat([w, w+add_noise], -1) / 2


# layers
for idx, l in enumerate(model.model.layers):
    print(f"layer {idx}")
    new_layer = new_model.model.layers[idx]

    # input_layernorm
    w = l.input_layernorm.weight
    new_layer.input_layernorm.weight = nn.Parameter(w)

    # self_attn
    qw = l.self_attn.q_proj.weight
    kw = l.self_attn.k_proj.weight
    vw = l.self_attn.v_proj.weight
    ow = l.self_attn.o_proj.weight

    new_layer.self_attn.q_proj.weight = nn.Parameter(qw)
    new_layer.self_attn.k_proj.weight = nn.Parameter(kw)
    new_layer.self_attn.v_proj.weight = nn.Parameter(vw)
    new_layer.self_attn.o_proj.weight = nn.Parameter(ow)

    # swiglu
    f1w = l.mlp.gate_proj.weight
    f2w = l.mlp.up_proj.weight
    f3w = l.mlp.down_proj.weight

    f1ww = w_2up(f1w)
    f2ww = w_2up(f2w)
    f3ww = w_2down(f3w)

    new_layer.mlp.gate_proj.weight = nn.Parameter(f1ww)
    new_layer.mlp.up_proj.weight = nn.Parameter(f2ww)
    new_layer.mlp.down_proj.weight = nn.Parameter(f3ww)

    # post_attention_layernorm
    w = l.post_attention_layernorm.weight
    new_layer.post_attention_layernorm.weight = nn.Parameter(w)


# last norm
w = model.model.norm.weight
new_model.model.norm.weight = nn.Parameter(w)

# Save the new model
new_model.save_pretrained("model_ds", safe_serialization=False, max_shard_size="5GB")

# # save again to remove shared parameter
model = AutoModelForCausalLM.from_pretrained("model_ds", torch_dtype="auto")
model.save_pretrained("model_ds_A", safe_serialization=False)
