import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.activations import GELUActivation
from torch import nn
import math
import json

config = json.load(open("config_one_half.json"))
config = LlamaConfig(**config)
new_model = LlamaForCausalLM(config)

# MODEL_DIR = "/lp/data/models/Llama-2-13b-hf"
# MODEL_DIR = "/lp/data/models/Llama-2-7b-hf"
MODEL_DIR = "/ML-A100/team/data/liupeng/data/model/yi_6b_hf"
model1 = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")
# MODEL_DIR = "/lp/data/models/Llama-2-7b-hf/Llama-2-8b-hf"
# MODEL_DIR = "/lp/data/models/Llama-2-7b-hf"
MODEL_DIR = "/ML-A100/team/data/liupeng/data/model/yi_6b_hf/yi_6b_wide_half_avg"
model2 = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")

# embedding
w1 = model1.model.embed_tokens.weight
w2 = model2.model.embed_tokens.weight
# rescale w2 variance to w1
variance1 = torch.sqrt(w1.pow(2).mean(-1, keepdim=True))
variance2 = torch.sqrt(w2.pow(2).mean(-1, keepdim=True))
w2_rescale = w2 / variance2 * variance1
ww = torch.cat([w1, w2_rescale], -1)
# ww = torch.cat([w1, w2], -1)
new_model.model.embed_tokens.weight = nn.Parameter(ww)

# lm_head
w1 = model1.lm_head.weight
w2 = model2.lm_head.weight
logit_alpha = 1
ww = torch.cat([w1, w2 * logit_alpha], -1)
new_model.lm_head.weight = nn.Parameter(ww)


def matrix_assign(ww, ww1, ww2):
    with torch.no_grad():
        new_ww = torch.zeros_like(ww)
        h1, w1 = ww1.size()
        h2, w2 = ww2.size()
        new_ww[:h1, :w1] = ww1
        new_ww[-h2:, -w2:] = ww2
        return new_ww


# layers
for idx, l in enumerate(new_model.model.layers):
    print(f"layer {idx}")
    l1 = model1.model.layers[idx]
    l2 = model2.model.layers[idx]

    # input_layernorm
    w1 = l1.input_layernorm.weight
    w2 = l2.input_layernorm.weight
    # set model2 ln zero
    ww = torch.cat([w1, torch.zeros_like(w2)], -1)
    l.input_layernorm.weight = nn.Parameter(ww)

    # self_attn
    qw1 = l1.self_attn.q_proj.weight
    kw1 = l1.self_attn.k_proj.weight
    vw1 = l1.self_attn.v_proj.weight
    ow1 = l1.self_attn.o_proj.weight

    qw2 = l2.self_attn.q_proj.weight
    kw2 = l2.self_attn.k_proj.weight
    vw2 = l2.self_attn.v_proj.weight
    ow2 = l2.self_attn.o_proj.weight

    qw = l.self_attn.q_proj.weight
    kw = l.self_attn.k_proj.weight
    vw = l.self_attn.v_proj.weight
    ow = l.self_attn.o_proj.weight

    qww = matrix_assign(qw, qw1, qw2)
    kww = matrix_assign(kw, kw1, kw2)
    vww = matrix_assign(vw, vw1, vw2)
    oww = matrix_assign(ow, ow1, ow2)

    l.self_attn.q_proj.weight = nn.Parameter(qww)
    l.self_attn.k_proj.weight = nn.Parameter(kww)
    l.self_attn.v_proj.weight = nn.Parameter(vww)
    l.self_attn.o_proj.weight = nn.Parameter(oww)

    # swiglu
    f1w = l.mlp.gate_proj.weight
    f2w = l.mlp.up_proj.weight
    f3w = l.mlp.down_proj.weight

    f1w1 = l1.mlp.gate_proj.weight
    f2w1 = l1.mlp.up_proj.weight
    f3w1 = l1.mlp.down_proj.weight

    f1w2 = l2.mlp.gate_proj.weight
    f2w2 = l2.mlp.up_proj.weight
    f3w2 = l2.mlp.down_proj.weight

    f1ww = matrix_assign(f1w, f1w1, f1w2)
    f2ww = matrix_assign(f2w, f2w1, f2w2)
    f3ww = matrix_assign(f3w, f3w1, f3w2)

    l.mlp.gate_proj.weight = nn.Parameter(f1ww)
    l.mlp.up_proj.weight = nn.Parameter(f2ww)
    l.mlp.down_proj.weight = nn.Parameter(f3ww)

    # post_attention_layernorm
    w1 = l1.post_attention_layernorm.weight
    w2 = l2.post_attention_layernorm.weight
    # ww = torch.cat([w1, w2], -1)
    ww = torch.cat([w1, torch.zeros_like(w2)], -1)
    l.post_attention_layernorm.weight = nn.Parameter(ww)

# last norm
w1 = model1.model.norm.weight
w2 = model2.model.norm.weight
# ww = torch.cat([w1, w2], -1)
ww = torch.cat([w1, torch.zeros_like(w2)], -1)
new_model.model.norm.weight = nn.Parameter(ww)

# Save the new model
new_model.save_pretrained("model_ds", safe_serialization=False)

# # save again to remove shared parameter
model = AutoModelForCausalLM.from_pretrained("model_ds", torch_dtype="auto")
model.save_pretrained("model_ds_A", max_shard_size="10GB", safe_serialization=False)


#################################################
## debug
tokenizer = AutoTokenizer.from_pretrained(".", use_fast=False, trust_remote_code=True)
text = "Write a quiz about bits that includes the word elephant"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs['input_ids']
model_inputs = model1.prepare_inputs_for_generation(input_ids, past_key_values=None)

outputs = model(
    **model_inputs,
    return_dict=True,
    output_attentions=True,
    output_hidden_states=True)

outputs1 = model1(
    **model_inputs,
    return_dict=True,
    output_attentions=True,
    output_hidden_states=True)

outputs2 = model2(
    **model_inputs,
    return_dict=True,
    output_attentions=True,
    output_hidden_states=True)

for idx in range(0, 100):
    for layer_idx in range(32):
        w1 = model1.model.embed_tokens.weight
        w2 = model2.model.embed_tokens.weight
        # rescale w2 variance to w1
        # variance1 = torch.sqrt(w1.pow(2).mean(-1, keepdim=True))
        # variance2 = torch.sqrt(w2.pow(2).mean(-1, keepdim=True))
        # w2_rescale = w2 / variance2 * variance1
        # ww = torch.cat([w1, w2_rescale], -1)
        ww = torch.cat([w1, w2], -1)

        a1 = w1[idx]
        a2 = w2[idx]
        a = ww[idx]
        a1 = a1.reshape([1, 1, -1])
        a2 = a2.reshape([1, 1, -1])
        a = a.reshape([1, 1, -1])

        l1 = model1.model.layers[layer_idx]
        l2 = model2.model.layers[layer_idx]
        l = new_model.model.layers[layer_idx]

        b1 = l1.input_layernorm(a1)
        b2 = l2.input_layernorm(a2)
        b = l.input_layernorm(a)

        c = l.self_attn(b)[0]
        c1 = l1.self_attn(b1)[0]
        c2 = l2.self_attn(b2)[0]

        d = l.post_attention_layernorm(c+a)
        d1 = l1.post_attention_layernorm(c1+a1)
        d2 = l2.post_attention_layernorm(c2+a2)

        variance1 = torch.sqrt(b1.pow(2).mean(-1, keepdim=True)).item()
        variance2 = torch.sqrt(b2.pow(2).mean(-1, keepdim=True)).item()
        variance = torch.sqrt(b.pow(2).mean(-1, keepdim=True)).item()
        print(f"{idx} {layer_idx}, {variance}: {variance1} / {variance2} = {variance1/variance2}")



for idx in range(32):
    l1 = model1.model.layers[idx]
    l2 = model2.model.layers[idx]
    # l = new_model.model.layers[0]

    b1 = l1.input_layernorm.weight
    b2 = l2.input_layernorm.weight
    # b = l.input_layernorm


    # d = l.post_attention_layernorm
    d1 = l1.post_attention_layernorm.weight
    d2 = l2.post_attention_layernorm.weight

    variance1 = torch.sqrt(b1.pow(2).mean(-1, keepdim=True)).item()
    variance2 = torch.sqrt(b2.pow(2).mean(-1, keepdim=True)).item()

    print(f"{idx}: {variance1} / {variance2} = {variance1/variance2}")









for i in range(33):
    print(i, ":",  torch.norm( outputs1['hidden_states'][i].reshape([1, 10, 2, -1])[:, :, 0, :] - outputs['hidden_states'][i]).item())

"""

"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

original_encoder_layers = model.model.layers

l = original_encoder_layers[0]



# lm_head
x = torch.rand(1, 1, 4096, dtype=torch.bfloat16, requires_grad=False)
xx = torch.concat([x, x], dim=-1)
w = model.lm_head.weight
ww = torch.cat([w, w], dim=-1) / 2
assert torch.sum(torch.matmul(x, w.T) - torch.matmul(xx, ww.T)).item() == 0.0

# emb
w = model.model.embed_tokens.weight
ww = torch.concat([w, w], dim=-1)
# ww = torch.concat([w, w], dim=-1) / math.sqrt(2)

# input_layernorm
def norm(weight, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance)
    return weight * hidden_states

x = torch.rand(1, 1, 4096, dtype=torch.bfloat16, requires_grad=False)
xx = torch.concat([x, x], dim=-1)
w = l.input_layernorm.weight
ww = torch.concat([w, w], -1)
norm(w, x)
norm(ww, xx).view(2, -1)


# qkv
def w_2x2(w):
    w_h = torch.concat([w, w], -1)
    return torch.concat([w_h, w_h], 0)

x = torch.rand(1, 1, 4096, dtype=torch.bfloat16, requires_grad=False)
xx = torch.concat([x, x], dim=-1)
qw = l.self_attn.q_proj.weight
kw = l.self_attn.k_proj.weight
vw = l.self_attn.v_proj.weight
ow = l.self_attn.o_proj.weight

qww = w_2x2(qw)/2
kww = w_2x2(kw)/2
vww = w_2x2(vw)/2
oww = w_2x2(ow)/2


def to_BSNH(states, N, B=1, S=1, H=128):
    return states.view(B, S, N, H).transpose(1, 2)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


q = torch.matmul(x, qw.T)
k = torch.matmul(x, kw.T)
v = torch.matmul(x, vw.T)
q, k, v = to_BSNH(q, 32), to_BSNH(k, 4), to_BSNH(v, 4)
k, v = repeat_kv(k, 8), repeat_kv(v, 8)
attn_w = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(128)
attn_w = nn.functional.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
attn_o = torch.matmul(attn_w, v)
attn_o = attn_o.transpose(1, 2).contiguous().reshape(1, 1, -1)
attn_o = torch.matmul(attn_o, ow.T)


qq = torch.matmul(xx, qww.T)
kk = torch.matmul(xx, kww.T)
vv = torch.matmul(xx, vww.T)
qq, kk, vv = to_BSNH(qq, 64), to_BSNH(kk, 8), to_BSNH(vv, 8)
kk, vv = repeat_kv(kk, 8), repeat_kv(vv, 8)
attn_ww = torch.matmul(qq, kk.transpose(2, 3)) / math.sqrt(128)
attn_ww = nn.functional.softmax(attn_ww, dim=-1, dtype=torch.float32).to(qq.dtype)
attn_oo = torch.matmul(attn_ww, vv)
attn_oo = attn_oo.transpose(1, 2).contiguous().reshape(1, 1, -1)
attn_oo = torch.matmul(attn_oo, oww.T)

# swiglu
x = torch.rand(1, 1, 4096, dtype=torch.bfloat16, requires_grad=False)
xx = torch.concat([x, x], dim=-1)

f1w = l.mlp.gate_proj.weight
f2w = l.mlp.up_proj.weight
f3w = l.mlp.down_proj.weight
f1_o = GELUActivation()(torch.matmul(x, f1w.T)) * torch.matmul(x, f2w.T)
f_o = torch.matmul(f1_o, f3w.T)


f1ww = w_2x2(f1w)/2
f2ww = w_2x2(f2w)/2
f3ww = w_2x2(f3w)/2
f1_oo = GELUActivation()(torch.matmul(xx, f1ww.T)) * torch.matmul(xx, f2ww.T)
f_oo = torch.matmul(f1_oo, f3ww.T)