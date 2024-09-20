import torch


# Q = torch.rand(B, S, Q_DIM)
# W = torch.rand(Q_DIM, N * H)
# C = torch.rand(B, S, KV_ROPE)
# U = torch.rand(KV_ROPE, N * H)
def f1(Q, W, C, U):
    """
    flops:  B*S*N*H*(Q_DIM+KV_ROPE+S)
    """
    q_C = Q @ W  # [B, S, N*H], flops: B * S * Q_DIM * (N * H)
    k_C = C @ U  # [B, S, N*H], flops: B * S * KV_ROPE * (N * H)
    q_C = q_C.reshape(B, S, N, H).transpose(1, 2)
    k_C = k_C.reshape(B, S, N, H).transpose(1, 2)  # [B, N, S, H]
    k_C_T = k_C.transpose(-1, -2)  # [B, N, H, S]
    attn = q_C @ k_C_T  # [B, N, S, S], flops: B * N * S * H * S
    return attn

def f1_merge(Q, W, C, U):
    """
    flops: B * N * S * KV_ROPE * (Q_DIM + S)
    """
    W1 = W.reshape(Q_DIM, N, H).transpose(0, 1)  # [N, Q_DIM, H]
    U1 = U.reshape(KV_ROPE, N, H).transpose(0, 1)  # [N, KV_ROPE, H]
    U1_T = U1.transpose(-1, -2)  # [N, H, KV_ROPE]

    Q1 = Q.unsqueeze(1)  # [B, 1, S, Q_DIM]
    C1 = C.unsqueeze(1)  # [B, 1, S, KV_ROPE]
    C1_T = C1.transpose(-1, -2)  # [B, 1, KV_ROPE, S]

    merge_WU = W1 @ U1_T  # [N, Q_DIM, KV_ROPE],

    attn1 = Q1 @ merge_WU  # [B, N, S, KV_ROPE], flops: B * N * S * Q_DIM * KV_ROPE
    attn = attn1 @ C1_T  # [B, N, S, S], flops: B * N * S * KV_ROPE * S
    return attn

# def f1_A(Q, W, C, U):
#     W1 = W.reshape(Q_DIM, N, H).transpose(0, 1)  # [N, Q_DIM, H]
#     Q1 = Q.unsqueeze(1)  # [B, 1, S, Q_DIM]
#     U1 = U.reshape(KV_ROPE, N, H).transpose(0, 1)  # [N, KV_ROPE, H]
#     C1 = C.unsqueeze(1)  # [B, 1, S, KV_ROPE]
#
#     q_C = Q1 @ W1  # [B, N, S, H]
#     k_C = C1 @ U1  # [B, N, S, H]
#
#     k_C_T = k_C.transpose(-1, -2)  # [B, N, H, S]
#     attn = q_C @ k_C_T  # [B, S, S, S]
#     return attn


    # A = torch.rand(B, N, S, S)  # Query tensor of shape [1, 32, 23, 23]
    # C = torch.rand(B, S, KV_ROPE)  # Key tensor of shape [1, 23, 512]
    # V = torch.rand(KV_ROPE, N * H)  # Value tensor of shape [512, 4096]
    # O = torch.rand(N * H, H_DIM)  # Output projection matrix [4096, 3072]
def f2(A, C, V, O):
    """
    flops: B*S*KV_ROPE*(N*H) + B*N*S*H*S + B*S*(N*H)*H_DIM
          =B*S*N*H(KV_ROPE+S+H_DIM)
    """
    values = (C @ V).reshape(B, S, N, H)  # [B, S, N, H], flops: B*S*KV_ROPE*(N*H)
    values = values.transpose(1, 2)  # [B, N, S, H]
    attn_out = (A @ values)  # [B, N, S, H], flops: B*N*S*H*S
    attn_out = attn_out.transpose(1, 2)  # [B, S, N, H]

    attn_out = attn_out.reshape(B, S, N * H)  # [B, S, N*H]
    O_out = attn_out @ O  # [B, N, H_DIM], flops: B*S*(N*H)*H_DIM
    return O_out


def f2_merge(A, C, V, O):
    """
    flops: B*N*S*S*KV_ROPE + B*S*N*KV_ROPE*H_DIM
          = B*N*S*KV_ROPE*(S+H_DIM)
    """
    V1 = V.reshape(KV_ROPE, N, H).transpose(0, 1)  # (N, KV_ROPE, H)
    O1 = O.reshape(N, H, H_DIM)  # (N, H, H_DIM)
    W = (V1 @ O1).reshape(N * KV_ROPE, H_DIM)  # (N*KV_ROPE, H_DIM)
    C1 = C.reshape(B, 1, S, KV_ROPE)

    T = (A @ C1).transpose(1, 2).reshape(B, S, N * KV_ROPE)  # (B, S, N*KV_ROPE), flops: B*N*S*S*KV_ROPE
    O_out = T @ W  # (B, S, H_DIM), flops: B*S*N*KV_ROPE*H_DIM
    return O_out


if __name__ == '__main__':
    # Define the dimensions and create matrices
    B = 4
    S = 23
    H = 128
    KV_ROPE = 512
    H_DIM = 3072
    N = 32
    Q_DIM = 1532
    # H = 4
    # KV_ROPE = 6
    # H_DIM = 9
    # N = 5
    # Q_DIM = 7

    # f1
    Q = torch.rand(B, S, Q_DIM)
    W = torch.rand(Q_DIM, N * H)
    C = torch.rand(B, S, KV_ROPE)
    U = torch.rand(KV_ROPE, N * H)

    # Call the functions
    output_f1 = f1(Q, W, C, U)
    output_f1_merge = f1_merge(Q, W, C, U)
    equivalent = torch.allclose(output_f1, output_f1_merge)
    print("Are the outputs of f1 and f1_merge approximately equal?", equivalent)

    # f2
    A = torch.rand(B, N, S, S)
    C = torch.rand(B, S, KV_ROPE)
    V = torch.rand(KV_ROPE, N * H)
    O = torch.rand(N * H, H_DIM)

    # Call the functions
    output_f_A = f2(A, C, V, O)
    output_f_B = f2_merge(A, C, V, O)

    max_value = (output_f_A.max() + output_f_B.max())/2
    output_f_A /= max_value
    output_f_B /= max_value

    # Check if the outputs are approximately equal
    equivalent = torch.allclose(output_f_A, output_f_B)
    print("Are the outputs of f_A and f_B approximately equal?", equivalent)
