# from flash_attn.flash_attn_interface import flash_attn_func
from vllm_flash_attn.flash_attn_interface import flash_attn_func
import torch

BATCH = 8
H = 256
N_CTX = 4096
HEAD_DIM = 576
dtype = torch.float16

q = torch.randn((BATCH, 1, H, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=False)
k = torch.randn((BATCH, N_CTX, 1, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=False)
v = torch.randn((BATCH, N_CTX, 1, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=False)
o = flash_attn_func(q, k, v, causal=True)
print(q.shape, o.shape)


q = torch.randn((BATCH, N_CTX, H, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=False)
k = torch.randn((BATCH, N_CTX, H, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=False)
v = torch.randn((BATCH, N_CTX, H, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=False)
o = flash_attn_func(q, k, v, causal=True)





q = torch.randn((BATCH, N_CTX, H, HEAD_DIM+64), dtype=dtype, device="cuda", requires_grad=False)
k = torch.randn((BATCH, N_CTX, H, HEAD_DIM+64), dtype=dtype, device="cuda", requires_grad=False)
v = torch.randn((BATCH, N_CTX, H, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=False)
o = flash_attn_func(q, k, v, causal=True)