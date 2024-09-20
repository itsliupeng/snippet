A = torch.rand(B, N, 1, S)
C = torch.rand(B, S, KV_ROPE)
W_UV = torch.rand(KV_ROPE, N * H)
W_O = torch.rand(N * H, H_DIM)

def f2(A, C, W_UV, W_O):
    """
    flops: B*S*KV_ROPE*(N*H) + B*N*1*H*S + B*1*(N*H)*H_DIM
          =B*N*H(S*KV_ROPE+S+H_DIM)
    """
    values = (C @ W_UV).reshape(B, S, N, H)  # [B, S, N, H], flops: B*S*KV_ROPE*(N*H)
    values = values.transpose(1, 2)  # [B, N, S, H]
    attn_out = (A @ values)  # [B, N, 1, H], flops: B*N*1*H*S
    attn_out = attn_out.transpose(1, 2)  # [B, 1, N, H]

    attn_out = attn_out.reshape(B, S, N * H)  # [B, 1, N*H]
    O_out = attn_out @ W_O  # [B, 1, H_DIM], flops: B*1*(N*H)*H_DIM
    return O_out


def f2_merge(A, C, W_UV, W_O):
    """
    flops: B*N*1*S*KV_ROPE + B*1*N*KV_ROPE*H_DIM
          = B*N*KV_ROPE*(S+H_DIM)
    """
    V1 = W_UV.reshape(KV_ROPE, N, H).transpose(0, 1)  # (N, KV_ROPE, H)
    O1 = W_O.reshape(N, H, H_DIM)  # (N, H, H_DIM)
    W = (V1 @ O1).reshape(N * KV_ROPE, H_DIM)  # (N*KV_ROPE, H_DIM)
    C1 = C.reshape(B, 1, S, KV_ROPE)

    T = (A @ C1).transpose(1, 2).reshape(B, 1, N * KV_ROPE)  # (B, 1, N*KV_ROPE), flops: B*N*1*S*KV_ROPE
    O_out = T @ W  # (B, 1, H_DIM), flops: B*1*N*KV_ROPE*H_DIM
    return O_out