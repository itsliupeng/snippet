import torch


def f(Q, W_UQ, C, W_UK, C_rope, W_O):
    q_C = Q @ W_UQ  # [B, 1, N*H], flops: B * 1 * Q_DIM * (N * H)
    k_C = C @ W_UK  # [B, S, N*H], flops: B * S * KV_NOPE * (N * H)
    q_C = q_C.reshape(B, 1, N, H).transpose(1, 2)  # [B, N, 1, H]
    k_C = k_C.reshape(B, S, N, H).transpose(1, 2)  # [B, N, S, H]

    q_R = (Q @ W_QR).reshape(B, 1, N, 64).transpose(1, 2)  # [B, N, 1, 64]
    k_R  = C_rope.view(B, S, 1, 64).transpose(1, 2)  # [B, 1, S, 64]

    q = q_R.new_empty(B, N, 1, H+64)
    q[:, :, :, : H] = q_C
    q[:, :, :, H:] = q_R

    k = k_R.new_empty(B, N, S, H+64)
    k[:, :, :, : H] = k_C
    k[:, :, :, H:] = k_R
    # todo apply rope for q_R, k_R

    A = q @ k.transpose(-1, -2)  # [B, N, 1, S], flops: B * N * 1 * H * S

    values = (C @ W_UV).reshape(B, S, N, H)  # [B, S, N, H], flops: B*S*KV_ROPE*(N*H)
    values = values.transpose(1, 2)  # [B, N, S, H]
    attn_out = (A @ values)  # [B, N, 1, H], flops: B*N*S*H*S
    attn_out = attn_out.transpose(1, 2)  # [B, S, N, H]

    attn_out = attn_out.reshape(B, 1, N * H)  # [B, S, N*H]
    O_out = attn_out @ W_O  # [B, S, H_DIM], flops: B*S*(N*H)*H_DIM
    return O_out

def f_merge(Q, C_all, mW_Q, mW_O):
    q = Q @ mW_Q
    q = q.reshape(B, 1, N, (KV_NOPE+64)).transpose(1, 2)  # [B, N, S, 576]

    k = C_all.view(B, S, 1, (KV_NOPE+64)).transpose(1, 2)  # [B, 1, S, 576]
    v = C_all[:, :, :KV_NOPE].view(B, S, 1, KV_NOPE).transpose(1, 2)   # [B, 1, S, 512]

    # todo apply rope for q_R, k_R

    A = q @ k.transpose(-1, -2)  # [B, N, 1, S]

    attn_out = A @ v  # [B, N, 1, 512]
    attn_out = attn_out.transpose(1, 2)
    attn_out = attn_out.reshape(B, 1, N * KV_NOPE)
    O_out = attn_out @ mW_O
    return O_out


if __name__ == '__main__':
    B = 4
    S = 23
    H = 128
    KV_NOPE = 512
    H_DIM = 3072
    N = 32
    Q_DIM = 1532

    Q = torch.rand(B, 1, Q_DIM)
    W_Q = torch.rand(Q_DIM, N * (H + 64))
    W_UQ = W_Q.reshape(Q_DIM, N, -1)[:, :, :H].reshape(Q_DIM, N * H)
    W_QR = W_Q.reshape(Q_DIM, N, -1)[:, :, H:].reshape(Q_DIM, N * 64)

    C_all = torch.rand(B, S, KV_NOPE + 64)
    C = C_all[:, :, :KV_NOPE]
    C_rope = C_all[:, :, KV_NOPE:]

    W_UK = torch.rand(KV_NOPE, N * H)
    W_UV = torch.rand(KV_NOPE, N * H)

    W_O = torch.rand(N * H, H_DIM)

    # merge weight
    mW_Q_nope = ((W_UQ.reshape(Q_DIM, N, H).transpose(0, 1)) @ (
        W_UK.reshape(KV_NOPE, N, H).transpose(0, 1).transpose(-1, -2))) \
        .transpose(0, 1)  # [Q_DIM, N, KV_NOPE]

    mW_Q = torch.cat([mW_Q_nope, W_QR.reshape(Q_DIM, N, 64)], -1).reshape(Q_DIM, N * (KV_NOPE + 64))

    mW_O = ((W_UV.reshape(KV_NOPE, N, H).transpose(0, 1)) @ W_O.reshape(N, H, H_DIM)) \
        .reshape(N * KV_NOPE, H_DIM)  # (N*KV_ROPE, H_DIM)

    # Call the functions
    output_f_A = f(Q, W_UQ, C, W_UK, C_rope, W_O)
    output_f_B = f_merge(Q, C_all, mW_Q, mW_O)

    max_value = (output_f_A.max() + output_f_B.max())/2
    output_f_A /= max_value
    output_f_B /= max_value

    # Check if the outputs are approximately equal
    equivalent = torch.allclose(output_f_A, output_f_B)
    print("Are the outputs of f_A and f_B approximately equal?", equivalent)