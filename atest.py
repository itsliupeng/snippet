import torch
from torch import nn
# from vllm_flash_attn_mla import flash_attn_with_kvcache
from vllm_flash_attn import flash_attn_with_kvcache

torch.set_default_dtype(torch.float16)
torch.set_default_device('cuda')

SCALE = 1.0

bt = torch.tensor([[0], [1], [2], [3], [4]], dtype=torch.int32)
# cs = torch.tensor([7,9,7,7,10], dtype=torch.int32)
cs = torch.tensor([8] * bt.shape[0], device="cuda:0", dtype=torch.int32)

# q = torch.randn((num_batch, 1, num_heads, head_dim), device="cuda:0", dtype=torch.float16)
# c_cache = torch.randn((num_gpu_blocks, block_size, 1, head_dim), device="cuda:0", dtype=torch.float16)
# cache_seqlens = torch.tensor([seqlen] * num_batch, device="cuda:0", dtype=torch.int32)
# block_table = torch.randint(0, num_gpu_blocks-1, (num_batch, (seqlen+block_size-1)//block_size), device="cuda:0", dtype=torch.int32)

# NUM_ITERS = 10
# for i in range(NUM_ITERS):
#     attn_output = flash_attn_with_kvcache(
#         q = q,
#         k_cache = c_cache,
#         v_cache = c_cache,
#         cache_seqlens = cache_seqlens,
#         block_table = block_table,
#         softmax_scale = 0.1,
#         num_splits=0
#     )

# print("Done.")


for i in range(100):
    print("--------------------------------------\n")

    # q0 = torch.randn(5, 1, 1, 1024)
    # w = torch.randn(1, 8, 1024, 192)
    # q1 = q0 @ w  # ([5, 8, 1, 192]
    # q2 = torch.randn(5, 8, 1, 64)
    #
    # q = torch.cat([q1, q2], dim=-1).transpose(1, 2).contiguous()  # [T,1,N,C+R]
    #
    # q = q.clone()
    # q = q / q.norm()
    #
    # if torch.any(torch.isnan(q1)):
    #     print(i, "q1 has nan")
    # if torch.any(torch.isnan(q2)):
    #     print(i, "q2 has nan")
    # if torch.any(torch.isnan(q)):
    #     print(i, "q has nan")

    q_raw = torch.randn(5, 1, 8, 256)
    q1, q2 = torch.split(q_raw, [192, 64], dim=-1)
    q = torch.cat([q1, q2], dim=-1)

    q = q * SCALE
    kv = torch.randn(10, 16, 1, 256) * SCALE
    out = flash_attn_with_kvcache(
        q,
        kv,
        kv,
        block_table=bt,
        cache_seqlens=cs,
        softmax_scale=None,
        causal=True,
    )
    # torch.cuda.synchronize()
    if torch.any(torch.isnan(out)):
        print(i, torch.norm(out).item())
        nan_tensor = out[torch.isnan(out)]
        print(nan_tensor)
        print(nan_tensor.shape)
        # print(f'FlashAttnMLA: {found_nan=}')