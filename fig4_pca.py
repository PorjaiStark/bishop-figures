import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)


def pca(X):
    mean = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return Vt, S, mean


theta = np.deg2rad(30.0)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
X2 = (rng.normal(size=(200, 2)) * np.array([1.5, 0.45])) @ R.T + np.array([1.0, 0.5])
V2, S2, m2 = pca(X2)
var2 = S2 ** 2 / (len(X2) - 1)
proj = m2 + np.outer((X2 - m2) @ V2[0], V2[0])
resid = np.sqrt(((X2 - proj) ** 2).sum(1))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
ax = axes[0]
for a, b in zip(X2, proj):
    ax.plot([a[0], b[0]], [a[1], b[1]], color="0.75", lw=0.6, zorder=1)
ax.scatter(X2[:, 0], X2[:, 1], s=9, color="tab:blue", zorder=2, label="data")
ax.scatter(proj[:, 0], proj[:, 1], s=9, color="tab:red", zorder=3, label="projection on PC1")
for k, c in ((0, "black"), (1, "0.5")):
    L = 2.5 * np.sqrt(var2[k])
    ax.annotate("", m2 + L * V2[k], m2 - L * V2[k],
                arrowprops=dict(arrowstyle="<->", color=c, lw=1.4))
ax.set_aspect("equal")
ax.set_xlabel("x1"); ax.set_ylabel("x2")
ax.set_title(f"PC1 keeps {100*var2[0]/var2.sum():.1f}% of the variance", fontsize=9)
ax.legend(fontsize=8, frameon=False, loc="upper left")


D, M, N, NOISE = 10, 3, 600, 0.35
W = rng.normal(size=(D, M))
Z = rng.normal(size=(N, M))
X10 = Z @ W.T + NOISE * rng.normal(size=(N, D))
V10, S10, m10 = pca(X10)
var10 = S10 ** 2 / (N - 1)
ratio = var10 / var10.sum()

errs = []
for k in range(D + 1):
    rec = m10 + (X10 - m10) @ V10[:k].T @ V10[:k]
    errs.append(float(((X10 - rec) ** 2).sum(1).mean()))
errs = np.array(errs)

ax = axes[1]
ax.plot(range(D + 1), errs, "o-", color="black", lw=1.4, ms=4)
ax.axvline(M, color="tab:red", ls="--", lw=1.0)
ax.text(M + 0.15, errs[0] * 0.75, f"true latent dim = {M}", color="tab:red", fontsize=8)
ax.set_xlabel("number of components kept")
ax.set_ylabel("mean squared reconstruction error")
ax.set_title("Error drops sharply up to the true dimension, then slowly", fontsize=9)
fig.tight_layout()
fig.savefig("figs/fig4_pca.png", dpi=160)

print("fig4")
print(f"  2-D: variance along PC1 {var2[0]:.4f}, PC2 {var2[1]:.4f} "
      f"({100*var2[0]/var2.sum():.1f}% / {100*var2[1]/var2.sum():.1f}%)")
print(f"  2-D: mean residual to PC1 line = {resid.mean():.4f}, max = {resid.max():.4f}")
print(f"  10-D: explained variance ratio = " + ", ".join(f"{r:.3f}" for r in ratio[:6]) + ", ...")
print(f"  10-D: cumulative at k=3 = {ratio[:3].sum():.4f}")
print(f"  10-D: reconstruction error k=0 {errs[0]:.3f}, k=3 {errs[3]:.3f} "
      f"({100*errs[3]/errs[0]:.1f}% of k=0), k=4 {errs[4]:.3f}, k=10 {errs[10]:.2e}")
print(f"  10-D: drop k=2->3 {errs[2]-errs[3]:.3f} vs k=3->4 {errs[3]-errs[4]:.3f} "
      f"(ratio {(errs[2]-errs[3])/(errs[3]-errs[4]):.1f}x)")
print(f"  noise floor check: D*NOISE^2 = {D*NOISE**2:.3f}")
