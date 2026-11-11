"""Figure 1 - Bayesian update of a Gaussian mean (Bishop DLFC, ch.2).

Known variance, unknown mean. Prior N(mu0, s0^2) is conjugate, so the
posterior after N observations is again Gaussian and can be written in
closed form. The figure sweeps N to show the posterior tightening.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

MU_TRUE, SIGMA = 2.0, 1.0      # data-generating truth, sigma assumed known
MU0, S0 = 0.0, 1.0             # prior
N_SHOW = [0, 1, 5, 20]

data = rng.normal(MU_TRUE, SIGMA, size=max(N_SHOW))


def posterior(n):
    """Posterior mean and sd of mu after n observations."""
    if n == 0:
        return MU0, S0
    prec = 1.0 / S0**2 + n / SIGMA**2
    mean = (MU0 / S0**2 + data[:n].sum() / SIGMA**2) / prec
    return mean, np.sqrt(1.0 / prec)


grid = np.linspace(-3, 5, 400)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), sharey=True)
rows = []
for ax, n in zip(axes, N_SHOW):
    m, s = posterior(n)
    dens = np.exp(-0.5 * ((grid - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    ax.plot(grid, dens, color="black", lw=1.6)
    ax.axvline(MU_TRUE, color="red", ls="--", lw=1.0)
    if n:
        ax.plot(data[:n], np.zeros(n), "|", color="tab:blue", ms=12, mew=1.2)
    ax.set_title(f"N = {n}   mean {m:.3f}  sd {s:.3f}", fontsize=9)
    ax.set_xlabel("mu")
    rows.append((n, m, s))
axes[0].set_ylabel("density")
fig.suptitle("Posterior of the mean as data arrives (true mu = 2, sigma = 1 known)", fontsize=10)
fig.tight_layout()
fig.savefig("figs/fig1_bayes.png", dpi=160)

print("fig1  N, posterior mean, posterior sd")
for n, m, s in rows:
    print(f"  {n:3d}  {m:.4f}  {s:.4f}")
print(f"  sd shrink factor N=0 -> N=20 : {rows[0][2] / rows[-1][2]:.2f}x")
