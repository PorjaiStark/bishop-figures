import numpy as np
import matplotlib.pyplot as plt

raw = np.genfromtxt("data/faithful.csv", delimiter=",", skip_header=1)
dur, wait = raw[:, 1], raw[:, 2]

long_wait = wait[dur > 3.0]
SIGMA = float(long_wait.std(ddof=1))
MU0, S0 = float(wait.mean()), 10.0
N_SHOW = [0, 1, 5, 20]

rng = np.random.default_rng(0)
data = long_wait[rng.permutation(len(long_wait))]


def posterior(n):
    if n == 0:
        return MU0, S0
    prec = 1.0 / S0 ** 2 + n / SIGMA ** 2
    mean = (MU0 / S0 ** 2 + data[:n].sum() / SIGMA ** 2) / prec
    return mean, np.sqrt(1.0 / prec)


grid = np.linspace(50, 100, 600)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.1), sharey=True)
rows = []
for ax, n in zip(axes, N_SHOW):
    m, s = posterior(n)
    dens = np.exp(-0.5 * ((grid - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    ax.plot(grid, dens, color="black", lw=1.6)
    ax.axvline(long_wait.mean(), color="red", ls="--", lw=1.0)
    if n:
        ax.plot(data[:n], np.zeros(n), "|", color="tab:blue", ms=12, mew=1.2)
    ax.set_title(f"N = {n}   mean {m:.2f}  sd {s:.2f}", fontsize=9)
    ax.set_xlabel("mean waiting time [min]")
    rows.append((n, m, s))
axes[0].set_ylabel("density")
fig.suptitle("Waiting time after a long eruption: posterior of the mean as eruptions are observed", fontsize=10)
fig.tight_layout()
fig.savefig("figs/fig1_bayes.png", dpi=160)

m_all, s_all = posterior(len(data))
print("fig1  Old Faithful, eruptions longer than 3 min")
print(f"  n = {len(long_wait)} of {len(wait)}, sample mean {long_wait.mean():.4f}, sd {SIGMA:.4f} min")
print(f"  prior: mean {MU0:.4f} (mean of all eruptions), sd {S0:.1f}")
for n, m, s in rows:
    print(f"    N = {n:3d}   posterior mean {m:.4f}   sd {s:.4f}")
print(f"    N = {len(data):3d}   posterior mean {m_all:.4f}   sd {s_all:.4f}")
print(f"  sd shrink N=0 -> N=20 : {rows[0][2] / rows[-1][2]:.2f}x, N=0 -> N={len(data)} : {S0 / s_all:.2f}x")
print(f"  prior weight left at N=20 : {100 * (1 / S0 ** 2) / (1 / S0 ** 2 + 20 / SIGMA ** 2):.2f}%")
