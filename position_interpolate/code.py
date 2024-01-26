import torch
import matplotlib.pyplot as plt


# figure 2

# build basis function
d = 4096 // 32
theta = 10000
# Frequency computation,
freqs = 1.0 / (theta ** (torch.arange(0, d, 2)[: (d // 2)].float() / d))
# construct basis function
L = 2048
x = torch.zeros(L)
x[:L] = torch.arange(0, L)
# basis functions
xfreq = torch.outer(x, freqs)
y = torch.randn(x.shape[0])
# do linear regression
X = torch.cat([xfreq.sin(), xfreq.cos()], dim=1)
eps = 0.000
coeffs = torch.linalg.solve(X.t() @ X + torch.eye(X.shape[1]) * eps, X.t() @ y)
x2 = torch.arange(0, 2*L)
xfreq2 = torch.outer(x2, freqs)
X2 = torch.cat([xfreq2.sin(), xfreq2.cos()], dim=1)
y2 = X2 @ coeffs
x3 = torch.arange(25, 75, 0.125)
xfreq3 = torch.outer(x3, freqs)
X3 = torch.cat([xfreq3.sin(), xfreq3.cos()], dim=1)
y3 = X3 @ coeffs
plt.figure(figsize=(16,5))
plt.subplot(1, 3, 1)
plt.plot(x2[:L], y2[:L], "r")
plt.scatter(x, y)
plt.ylabel("attention score $a(s)$")
plt.xlabel("Positional difference $s$")
plt.subplot(1, 3, 2)
plt.plot(x2, y2, "r")
plt.scatter(x, y)
plt.axvline(L, color="k", linestyle="--", linewidth=0.5)
plt.title("Effect of Extrapolation")
plt.xlabel("Positional difference $s$")
plt.subplot(1, 3, 3)
plt.plot(x3, y3, "r")
for i in range(25,75):
    plt.axvline(i, color="k", linestyle="--", linewidth=0.5)
plt.title("Effect of Interpolation")
plt.xlabel("Positional difference $s$")
plt.show()

# figure 5.
# L = 2048
# x = torch.arange(0, 2*L)
# d = 4096 // 32
# theta = 10000
# freqs = 1.0 / (theta ** (torch.arange(0, d, 2)[: (d // 2)].float() / d))
# xfreq = torch.outer(x, freqs)
# mags = (xfreq.sin().cumsum(dim=1).pow(2) + xfreq.cos().cumsum(dim=1).pow(2)).sqrt()
# plt.plot(mags.sum(dim=1)/d)
# plt.axhline(1.0, color='k', linestyle="--")
# plt.xlabel("Positional difference $s$")
# plt.ylabel("$B(s)/d$")
# plt.show()
