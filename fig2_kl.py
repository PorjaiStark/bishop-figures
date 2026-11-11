import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-8, 8, 4001)
dx = x[1] - x[0]
EPS = 1e-300


def gauss(t, m, s):
    return np.exp(-0.5 * ((t - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


p = 0.5 * gauss(x, -2.0, 0.6) + 0.5 * gauss(x, 2.0, 0.6)
p /= p.sum() * dx


def kl(a, b):
    return float(np.sum(np.where(a > 0, a * np.log((a + EPS) / (b + EPS)), 0.0)) * dx)


m_fwd = float(np.sum(x * p) * dx)
s_fwd = float(np.sqrt(np.sum((x - m_fwd) ** 2 * p) * dx))


mus = np.linspace(-4, 4, 321)
sigmas = np.linspace(0.1, 3.0, 291)
best = (np.inf, None, None)
for m in mus:
    for s in sigmas:
        v = kl(gauss(x, m, s), p)
        if v < best[0]:
            best = (v, m, s)
kl_rev, m_rev, s_rev = best

q_fwd, q_rev = gauss(x, m_fwd, s_fwd), gauss(x, m_rev, s_rev)
kl_fwd = kl(p, q_fwd)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
for ax, q, name, val, m, s in (
    (axes[0], q_fwd, "forward  KL(p||q)", kl_fwd, m_fwd, s_fwd),
    (axes[1], q_rev, "reverse  KL(q||p)", kl_rev, m_rev, s_rev),
):
    ax.plot(x, p, color="black", lw=1.6, label="p (target)")
    ax.plot(x, q, color="tab:red", lw=1.6, ls="--", label="q (single Gaussian)")
    ax.set_title(f"{name} = {val:.3f}\nmu {m:.2f}  sigma {s:.2f}", fontsize=9)
    ax.set_xlabel("x")
    ax.legend(fontsize=8, frameon=False)
axes[0].set_ylabel("density")
fig.suptitle("Minimising the two directions of KL gives different answers", fontsize=10)
fig.tight_layout()
fig.savefig("figs/fig2_kl.png", dpi=160)

print("fig2")
print(f"  forward  KL(p||q) = {kl_fwd:.4f}   mu {m_fwd:.4f}  sigma {s_fwd:.4f}")
print(f"  reverse  KL(q||p) = {kl_rev:.4f}   mu {m_rev:.4f}  sigma {s_rev:.4f}")
print(f"  width ratio sigma_fwd / sigma_rev = {s_fwd / s_rev:.2f}")
print(f"  cross-check KL(p||q_rev) = {kl(p, q_rev):.4f}  KL(q_fwd||p) = {kl(q_fwd, p):.4f}")
