import torch

a = torch.rand(1, 23, 3072)
W = torch.rand(3072, 4096)
q = torch.matmul(a, W)  # q.shape will be [1, 23, 4096]
q = q.reshape(1, 23, 32, 128)  # q.shape is now [1, 23, 32, 128]
q = q.transpose(1, 2)  # q.shape is now [1, 32, 23, 128]



b = torch.rand(1, 23, 512)

V = torch.rand(512, 4096)

# Compute q and k with matrix multiplication

k = torch.matmul(b, V)  # k.shape will be [1, 23, 4096]

# Reshape q and k
k = k.reshape(1, 23, 32, 128)  # k.shape is now [1, 23, 32, 128]

# Transpose q and k
q = q.transpose(1, 2)  # q.shape is now [1, 32, 23, 128]
k = k.transpose(1, 2)  # k.shape is now [1, 32, 23, 128]

# Transpose k along the last two dimensions to prepare for matmul
k_transposed = k.transpose(2, 3)  # k.shape is now [1, 32, 128, 23]

# Perform the final matrix multiplication
result = torch.matmul(q, k_transposed)  # result.shape will be [1, 32, 23, 23]

##############################################################################
V_t = V.transpose(0, 1)  # V_t shape is now [4096, 512]
U = torch.matmul(W, V_t)  # U shape will be [3072, 512]





###############################
import torch

# Original matrices
B = 1
S = 23
H = 128
KV_ROPE = 512
H_DIM = 3072
N = 32

A = torch.rand(B, N, 1, S)  # [1, 32, 1, 23]
C = torch.rand(B, S, KV_ROPE)  # [1, 23, 512]
V = torch.rand(KV_ROPE, N*H)  # [512, 4096]
O = torch.rand(N*H, H_DIM)  # [4096 * 3072]


# infer only need  1 token's attention score
def f(A, C, V, O):
    values = (C @ V).reshape(B, S, N, H)  # [1, 23, 32, 128]
    values = values.transpose(1, 2)  # [1, 32, 23, 128]
    attn_out = (A @ values)  # [B, 32, 1, 128], A: [1, 32, 1, 23]
    attn_out = attn_out.reshape(B, N*H)  # [B,  4096]
    O_out = attn_out @ O  # [B, 3072]
    return O_out

def f2_T(A, C, V, O):
    C_T = C.permute(2, 0, 1).reshape(KV_ROPE, B * S)  # [512, 1 *23]
    V_T = V.T.reshape(N, H, KV_ROPE).transpose(0, 1).reshape(H * N, KV_ROPE)  # [128 * 32, 512]
    A_T = A.transpose(-1, -2)  # [B, 32, 23, 1]  [B, N, 1, S] => [S,  N, B, 1]
    A_T = A_T.repeat(H, 1, 1, 1)  # [128, 32, 23, 1]
    O_T = O.T.reshape(H_DIM, N, H).transpose(1, 2).reshape(H_DIM, H*N)

    values_T = (V_T @ C_T).reshape(H, N, B, S)  # [128, 32, B, 23]
    attn_out_T = values_T @ A_T  #  [128, 32, 23, 1]  =>  [128 *32, B, 1]
    attn_out_T = attn_out_T.reshape(H*N, B)  # [4096, B]
    O_out_T = O_T @ attn_out_T  # [3072, B]
    O_out = O_out_T.T  # [B, 3072]
    return O_out

def f2_T_merge(A, C, V, O):
    C_T = C.permute(2, 0, 1).reshape(KV_ROPE, B * S)  # [512, 1 *23]
    V_T = V.T.reshape(N, H, KV_ROPE).transpose(0, 1).reshape(H * N, KV_ROPE)  # [128 * 32, 512]
    A_T = A.transpose(-1, -2)  # [B, 32, 23, 1]  # [B, N, 1, S]
    A_T = A_T.repeat(H, 1, 1, 1).reshape(H*N, S, 1)  # [128*32, 23, 1]
    O_T = O.T.reshape(H_DIM, N, H).transpose(1, 2).reshape(H_DIM, H*N)

    values_T = (V_T @ C_T).reshape(H * N, B, S)  # [128 * 32, B, 23]
    attn_out_T = values_T @ A_T  # [128 * 32, 23, 1]  =>  [128 *32, B, 1]
    attn_out_T = attn_out_T.reshape(H*N, B)  # [4096, B]
    O_out_T = O_T @ attn_out_T  # [3072, B]
    O_out = O_out_T.T  # [B, 3072]
    return O_out

def f1(A, C, V, O):
    C_T = C.permute(2, 0, 1).reshape(KV_ROPE, B * S)  # [512, 1 *23]
    V_T = V.T.reshape(N, H, KV_ROPE).transpose(0, 1).reshape(H * N, KV_ROPE)  # [4096, 512]
    A_T = A.transpose(-1, -2)  # [1, 32, S2, S],  [1, 32, 23, 23]
    values_T = (V_T @ C_T).reshape(H, N, B, S)  # [128, 32, 1, 23]
    attn_out_T = values_T @ A_T  # [128, 32, 1, 23]
    attn_out_T = attn_out_T.transpose(0, 1).reshape(H * N, B * S)  # [4096, 23]
    O_out_T = O.T @ attn_out_T  # [3072, 23]
    O_out = O_out_T.reshape(H_DIM, S, B).permute(2, 1, 0)
    return O_out


def f2(A, C, V, O):
    C_T = C.permute(2, 0, 1).reshape(KV_ROPE, B * S)  # [512, 1 *23]
    V_T = V.T.reshape(N, H, KV_ROPE).transpose(0, 1).reshape(H * N, KV_ROPE)  # [128 * 32, 512]
    A_T = A.transpose(-1, -2)  # [1, 32, S2, S],  [1, 32, 23, 23]
    O_T = O.T.reshape(H_DIM, N, H).transpose(1, 2).reshape(H_DIM, H*N)

    values_T = (V_T @ C_T).reshape(H, N, B, S)  # [128, 32, 1, 23]
    attn_out_T = values_T @ A_T  # [128, 32, 1, 23]
    attn_out_T = attn_out_T.reshape(H*N, B * S)  # [4096, 23]
    O_out_T = O_T @ attn_out_T  # [3072, 23]
    O_out = O_out_T.reshape(H_DIM, S, B).permute(2, 1, 0)
    return O_out

def f2_merge(A, C, V, O):
    C_T = C.permute(2, 0, 1).reshape(KV_ROPE, B * S)  # [512, 1 *23]
    V_T = V.T.reshape(N, H, KV_ROPE).transpose(0, 1).reshape(H * N, KV_ROPE)  # [128 * 32, 512]
    A_T = A.transpose(-1, -2)  # [B, N, S, S],  [1, 32, 23, 23]

    O_T = O.T.reshape(H_DIM, N, H).transpose(1, 2).reshape(H_DIM, H*N)
    U = O_T @ V_T  # [3072, 512]

    # values_T = (V_T @ C_T).reshape(H, N, B, S)  # [128, 32, 1, 23]
    # attn_out_T = values_T @ A_T  # [128, 32, 1, 23]
    # attn_out_T = attn_out_T.reshape(H*N, B * S)  # [4096, 23]
    # O_out_T = O_T @ attn_out_T  # [3072, 23]
    # O_out = O_out_T.reshape(H_DIM, S, B).permute(2, 1, 0)

    # original
    # O_T [3072, H*N]
    # V_T [H*N, 512]
    # C_T [512, B*S]
    # A_T [B, N, S, S]
    # output [3072, B*S]

    CT @ A_T # [512, N]


    O_out_T = O_T @ ( (
                              ( (V_T @ C_T).reshape(H, N, B, S) ) @ A_T # [128, 32, 23]
                      ).reshape(H*N, B * S))

    # remove reshape
    O_out_T = O_T @ (
        (
                ( (V_T @ C_T).reshape(H*N, B*S) ) @ A_T.reshape(-1, S, S)  # [H*N, N, B*S]
        ).reshape(H*N, B * S)
    )

    O_out_T = O_T.reshape(3072, 128, 32) @ (
        ( ( (V_T @ C_T).reshape(H, N, B, S) ) @ A_T)
    )

    O_out = O_out_T.T.reshape(B, S, H_DIM)

    return O_out


def f_B(A, C, V, O):
    values = (C @ V).reshape(B, S, N, H)  # [1, 23, 32, 128]
    values = values.transpose(1, 2)  # [1, 32, 23, 128]
    attn_out = (A @ values)  # [1, 32, 23, 128]
    attn_out = attn_out.transpose(1, 2)  # [1, 23, 32, 128]
    attn_out = attn_out.reshape(B, S, N * H)  # [1, 23, 4096]

    attn_out_T = attn_out.transpose(1, 2)
    O_out_T = O.T @ attn_out_T
    O_out = O_out_T.transpose(1, 2)
    return O_out




(values_T @ A_T).permute(2, 1, 3, 0) - (A @ values)


def f_A(A, C, V, O):
    values = (C @ V).reshape(B, S, N, H)  # [1, 23, 32, 128]
    values = values.transpose(1, 2)  # [1, 32, 23, 128]
    attn_out = (A @ values) / 12  # [1, 32, 23, 128]
    attn_out = attn_out.transpose(1, 2)  # [1, 23, 32, 128]
    attn_out = attn_out.reshape(B, S, N*H)  # [1, 23, 4096]
    O_out = attn_out @ O  # [1, 23, 3072]

    return O_out


def f_B(A, C, V, O):
    values = (C @ V).reshape(B, S, N, H)  # [1, 23, 32, 128]
    values = values.transpose(1, 2)  # [1, 32, 23, 128]
    attn_out = (A @ values)  # [1, 32, 23, 128]
    attn_out = attn_out.transpose(1, 2)  # [1, 23, 32, 128]
    attn_out = attn_out.reshape(B, S, N * H)  # [1, 23, 4096]

    attn_out_T = attn_out.transpose(1, 2)
    O_out_T = torch.matmul(O.t(), attn_out_T)
    O_out = O_out_T.transpose(1, 2)
    return O_out



import torch

# Define the dimensions and create matrices
B = 1
S = 23
H = 128
KV_ROPE = 512
H_DIM = 3072
N = 32

A = torch.rand(B, N, S, S)  # Query tensor of shape [1, 32, 23, 23]
C = torch.rand(B, S, KV_ROPE)  # Key tensor of shape [1, 23, 512]
V = torch.rand(KV_ROPE, N*H)  # Value tensor of shape [512, 4096]
O = torch.rand(N*H, H_DIM)  # Output projection matrix [4096, 3072]

# Call the functions
output_f_A = f(A, C, V, O)
output_f_B = calc1(A, C, V, O)

# Check if the outputs are approximately equal
equivalent = torch.allclose(output_f_A, output_f_B)
print("Are the outputs of f_A and f_B approximately equal?", equivalent)




############################

C_T = C.permute(2, 0, 1).reshape(KV_ROPE, B*S)  # [512, 1 *23]
V_T = V.T.reshape(N, H, KV_ROPE).transpose(0, 1).reshape(H*N, KV_ROPE)  # [4096, 512]
values = V_T @ C_T  # [4096, 23]
values = values.reshape(H, N, B, S)  # [128, 32, 1, 23]
attn_out = values @ A  # [128, 32, 1, 23]
attn_out = attn_out.reshape(H*N, B*S)
O_out = O.T @ attn_out
O_out = O_out.reshape(H_DIM, S, B).permute(2, 1, 0)










A_T = A.transpose(1, 2)  # ([1, 23, 32, 23]
CV_T = (C @ V).reshape(B, S, N, H)
attn_out = CV_T  @ A_T



W = V @ O # []
A @ C @ W




######################

def f(A, C, V, O):
    values = (C @ V).reshape(B, S, N, H)  # [1, 23, 32, 128]  -> [128, 23, 32]
    values = values.transpose(1, 2)  # [1, 32, 23, 128]  -> [128, 32, 23, 1]
    attn_out = (A @ values)  #  [1, 32, 23, 23] ->   [128, 32, 23, 1]
    attn_out = attn_out.transpose(1, 2)  # [128 * 32, 23, 1]

    attn_out = attn_out.reshape(B, S, N*H)  # [1, 23, 4096]
    O_out = attn_out @ O  # [1, 23, 3072] -> [3072, 23, 1]
    return O_out

def calc1(A, C, V, O):
    V1 = V.reshape(KV_ROPE, N, H).transpose(0, 1)           # (N, KV_ROPE, H)
    O1 = O.reshape(N, H, H_DIM)                             # (N, H, H_DIM)
    W = (V1 @ O1).reshape(N*KV_ROPE, H_DIM)                 # (N*KV_ROPE, H_DIM)
    C1 = C.reshape(B, 1, S, KV_ROPE)
    T = (A @ C1).transpose(1, 2).reshape(B, S, N*KV_ROPE)   # (B, S, N*KV_ROPE)
    O_out = T @ W                                           # (B, S, H_DIM)
    return O_out

def cal2(A, C, V, O):
    V1 = V.reshape(KV_ROPE, N, H).transpose(0, 1)           # (512, 32, 128) -> (32, 512, 128)
    O1 = O.reshape(N, H, H_DIM)                             # (32, 128, 3072)
    W = (V1 @ O1).reshape(N*KV_ROPE, H_DIM)                 # (32, 512, 3072) -> (32*512, 3072)
    C1 = C.reshape(B, 1, S, KV_ROPE)                        # [1, 23, 512] -> [1, 1, 23, 512]
    T = (A @ C1).transpose(1, 2).reshape(B, S, N*KV_ROPE)   # [1,32, 23, 512] -> [1, 23, 32, 512] -> [1, 23, 32*512]
    O_out = T @ W                                           # (B, S, H_DIM)
    return O_out

def cal_merge(A, C, V, O):
    V1 = V.reshape(KV_ROPE, N, H).transpose(0, 1)           # (512, 32, 128) -> (32, 512, 128)
    O1 = O.reshape(N, H, H_DIM)                             # (32, 128, 3072)
    W = (V1 @ O1).reshape(N*KV_ROPE, H_DIM)                 # (32, 512, 3072) -> (32*512, 3072)


    C1 = C.reshape(B, 1, S, KV_ROPE)                        # [1, 23, 512] -> [1, 1, 23, 512]
    T = (A @ C1).transpose(1, 2).reshape(B, S, N*KV_ROPE)   #  [1, 32, 23, 23]  :  [1,32, 23, 512] -> [1, 23, 32, 512] -> [1, 23, 32*512]
    O_out = T @ W                                           # (B, S, H_DIM)
    return O_out
