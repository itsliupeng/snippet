
import torch

base = 10000000

max_seq_len_cached = 256*1024


for base in [10000, 5000000, 10000000]:
    dim = 128
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len_cached, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    print(base, freqs)
# Different from paper, but it uses a different permutation in order to obtain the same calculation
emb = torch.cat((freqs, freqs), dim=-1)
