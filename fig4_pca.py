import numpy as np
import matplotlib.pyplot as plt

raw = np.genfromtxt("data/faithful.csv", delimiter=",", skip_header=1)
X = raw[:, 1:3]
N = len(X)


def pca(X):
    mean = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return Vt, S ** 2 / (len(X) - 1), mean


def panel(ax, X, title, xlabel, ylabel, equal):
    V, var, m = pca(X)
    proj = m + np.outer((X - m) @ V[0], V[0])
    resid = np.sqrt(((X - proj) ** 2).sum(1))
    for a, b in zip(X, proj):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="0.78", lw=0.5, zorder=1)
    ax.scatter(X[:, 0], X[:, 1], s=9, color="tab:blue", zorder=2)
    ax.scatter(proj[:, 0], proj[:, 1], s=9, color="tab:red", zorder=3)
    L = 2.2 * np.sqrt(var[0])
    ax.annotate("", m + L * V[0], m - L * V[0], arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.set_title(f"{title}\nPC1 keeps {100*var[0]/var.sum():.2f}%   direction ({V[0][0]:.3f}, {V[0][1]:.3f})",
                 fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if equal:
        ax.set_aspect("equal")
    return V, var, resid


Xs = (X - X.mean(0)) / X.std(0, ddof=1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
V_raw, var_raw, res_raw = panel(axes[0], X, "raw units (minutes)",
                                "eruption duration [min]", "waiting time [min]", False)
V_std, var_std, res_std = panel(axes[1], Xs, "after standardising each variable",
                                "duration [sd]", "waiting time [sd]", True)
fig.suptitle("Old Faithful: PCA gives a different first component depending on the units", fontsize=10)
fig.tight_layout()
fig.savefig("figs/fig4_pca.png", dpi=160)

print("fig4  Old Faithful, n = %d" % N)
print(f"  correlation between duration and waiting = {np.corrcoef(X[:,0], X[:,1])[0,1]:.4f}")
print(f"  sd: duration {X[:,0].std(ddof=1):.4f} min, waiting {X[:,1].std(ddof=1):.4f} min "
      f"(ratio {X[:,1].std(ddof=1)/X[:,0].std(ddof=1):.1f}x)")
print("  raw units        : explained " + ", ".join(f"{v:.4f}" for v in var_raw / var_raw.sum())
      + f"  PC1 ({V_raw[0][0]:.4f}, {V_raw[0][1]:.4f})")
print(f"                     mean residual to PC1 {res_raw.mean():.4f} min")
print("  standardised     : explained " + ", ".join(f"{v:.4f}" for v in var_std / var_std.sum())
      + f"  PC1 ({V_std[0][0]:.4f}, {V_std[0][1]:.4f})")
print(f"                     mean residual to PC1 {res_std.mean():.4f} sd")
print(f"  angle of PC1 to the duration axis: raw {np.degrees(np.arctan2(V_raw[0][1], V_raw[0][0])):.2f} deg, "
      f"standardised {np.degrees(np.arctan2(V_std[0][1], V_std[0][0])):.2f} deg")
