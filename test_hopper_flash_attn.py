import math

import pytest
import torch
import torch.nn.functional as F

from einops import rearrange, repeat
from flash_attn_interface import flash_attn_func


BATCH = 8
H = 1
N_CTX = 1024
HEAD_DIM = 256
dtype = torch.float16

# q = torch.randn((BATCH, 1, H, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=False)
# k = torch.randn((BATCH, N_CTX, 1, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=False)
# v = torch.randn((BATCH, N_CTX, 1, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=False)
# o = flash_attn_func(q, k, v, causal=True)
# print(q.shape, o.shape)


q = torch.randn((BATCH, 128, H, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=False)
k = torch.randn((BATCH, N_CTX, H, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=False)
v = torch.randn((BATCH, N_CTX, H, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=False)
for i in range(3):
    out = flash_attn_func(q, k, v, causal=True)
for i in range(20):
    out = flash_attn_func(q, k, v, causal=True)




q = torch.randn((BATCH, N_CTX, H, HEAD_DIM+64), dtype=dtype, device="cuda", requires_grad=False)
k = torch.randn((BATCH, N_CTX, H, HEAD_DIM+64), dtype=dtype, device="cuda", requires_grad=False)
v = torch.randn((BATCH, N_CTX, H, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=False)
o = flash_attn_func(q, k, v, causal=True)