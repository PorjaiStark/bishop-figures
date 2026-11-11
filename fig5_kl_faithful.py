import numpy as np
import matplotlib.pyplot as plt

raw = np.genfromtxt("data/faithful.csv", delimiter=",", skip_header=1)
dur = raw[:, 1]
N = len(dur)

x = np.linspace(dur.min() - 1.5, dur.max() + 1.5, 4001)
dx = x[1] - x[0]
EPS = 1e-300


def gauss(t, m, s):
    return np.exp(-0.5 * ((t - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def em1d(v, K, seed, iters=500, reg=1e-8):
    r = np.random.default_rng(seed)
    mu = v[r.choice(len(v), K, replace=False)]
    var = np.full(K, v.var())
    pi = np.full(K, 1.0 / K)
    hist = []
    for _ in range(iters):
        lg = np.stack([np.log(pi[k]) - 0.5 * np.log(2 * np.pi * var[k])
                       - 0.5 * (v - mu[k]) ** 2 / var[k] for k in range(K)], 1)
        mx = lg.max(1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(lg - mx).sum(1))
        hist.append(float(lse.sum()))
        resp = np.exp(lg - lse[:, None])
        Nk = resp.sum(0) + 1e-12
        pi = Nk / len(v)
        mu = (resp * v[:, None]).sum(0) / Nk
        var = (resp * (v[:, None] - mu) ** 2).sum(0) / Nk + reg
        if len(hist) > 2 and abs(hist[-1] - hist[-2]) < 1e-8:
            break
    return pi, mu, np.sqrt(var), hist


runs = [em1d(dur, 2, s) for s in range(20)]
pi, mu_k, sd_k, hist = max(runs, key=lambda r: r[3][-1])
order = np.argsort(mu_k)
pi, mu_k, sd_k = pi[order], mu_k[order], sd_k[order]

p = sum(pi[k] * gauss(x, mu_k[k], sd_k[k]) for k in range(2))
p /= p.sum() * dx


def kl(a, b):
    return float(np.sum(np.where(a > 0, a * np.log((a + EPS) / (b + EPS)), 0.0)) * dx)


m_fwd = float(np.sum(x * p) * dx)
s_fwd = float(np.sqrt(np.sum((x - m_fwd) ** 2 * p) * dx))

mus = np.linspace(x[0], x[-1], 401)
sigmas = np.linspace(0.05, 2.5, 246)
kl_rev, m_rev, s_rev = min(((kl(gauss(x, m, s), p), m, s) for m in mus for s in sigmas))

q_fwd, q_rev = gauss(x, m_fwd, s_fwd), gauss(x, m_rev, s_rev)
kl_fwd = kl(p, q_fwd)

fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7), sharey=True)
for ax, q, name, val, m, s in (
    (axes[0], q_fwd, "forward  KL(p||q)", kl_fwd, m_fwd, s_fwd),
    (axes[1], q_rev, "reverse  KL(q||p)", kl_rev, m_rev, s_rev),
):
    ax.hist(dur, bins=30, density=True, color="0.88", edgecolor="0.6", lw=0.4,
            label=f"{N} eruptions")
    ax.plot(x, p, color="black", lw=1.6, label="p (2-component GMM, EM)")
    ax.plot(x, q, color="tab:red", lw=1.6, ls="--", label="q (single Gaussian)")
    ax.set_title(f"{name} = {val:.3f}\nmu {m:.2f}  sigma {s:.2f}", fontsize=9)
    ax.set_xlabel("eruption duration [min]")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_xlim(x[0], x[-1])
axes[0].set_ylabel("density")
fig.suptitle("Old Faithful eruption durations: one Gaussian against the fitted mixture", fontsize=10)
fig.tight_layout()
fig.savefig("figs/fig5_kl_faithful.png", dpi=160)

print("fig5  Old Faithful, n = %d" % N)
print(f"  data mean {dur.mean():.4f}  sd {dur.std(ddof=1):.4f}  range {dur.min():.2f}-{dur.max():.2f} min")
print(f"  mixture fitted by EM, best of 20 starts, log-likelihood {hist[-1]:.2f}, {len(hist)} iterations")
for k in range(2):
    print(f"    component {k+1}: weight {pi[k]:.4f}  mean {mu_k[k]:.4f}  sd {sd_k[k]:.4f}")
print(f"  forward  KL(p||q) = {kl_fwd:.4f}   mu {m_fwd:.4f}  sigma {s_fwd:.4f}")
print(f"  reverse  KL(q||p) = {kl_rev:.4f}   mu {m_rev:.4f}  sigma {s_rev:.4f}")
print(f"  cross-check KL(p||q_rev) = {kl(p, q_rev):.4f}")
print(f"  width ratio sigma_fwd / sigma_rev = {s_fwd / s_rev:.2f}")
