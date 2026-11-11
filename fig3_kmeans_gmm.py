import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
K, N_INIT = 3, 20


specs = [((-1.1, 0.0), [[1.60, 0.00], [0.00, 0.22]], 200),
         ((1.1, 0.7), [[0.55, 0.45], [0.45, 0.80]], 200),
         ((0.2, -1.5), [[0.45, 0.00], [0.00, 0.45]], 100)]
X = np.vstack([rng.multivariate_normal(m, c, n) for m, c, n in specs])
truth = np.concatenate([np.full(n, i) for i, (_, _, n) in enumerate(specs)])
N, D = X.shape


def kmeans(X, K, seed, iters=100):
    r = np.random.default_rng(seed)
    mu = X[r.choice(len(X), K, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - mu[None]) ** 2).sum(-1), axis=1)
        new = np.array([X[lab == k].mean(0) if np.any(lab == k) else mu[k] for k in range(K)])
        if np.allclose(new, mu):
            break
        mu = new
    return lab, mu


def log_gauss(X, mu, cov):
    d = X - mu
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, d.T)
    return -0.5 * (sol ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * D * np.log(2 * np.pi)


def em(X, K, seed, init="points", iters=300, reg=1e-6):
    r = np.random.default_rng(seed)
    if init == "points":
        mu = X[r.choice(len(X), K, replace=False)]
        cov = np.array([np.cov(X.T) for _ in range(K)])
    else:
        mu = X.mean(0) + r.normal(0, 2.0, (K, X.shape[1]))
        cov = np.array([np.diag(X.var(0)) * 0.2 for _ in range(K)])
    pi = np.full(K, 1.0 / K)
    hist = []
    for _ in range(iters):
        lg = np.stack([np.log(pi[k]) + log_gauss(X, mu[k], cov[k]) for k in range(K)], 1)
        mx = lg.max(1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(lg - mx).sum(1))
        hist.append(float(lse.sum()))
        resp = np.exp(lg - lse[:, None])
        Nk = resp.sum(0) + 1e-12
        pi = Nk / len(X)
        mu = (resp.T @ X) / Nk[:, None]
        cov = np.array([((resp[:, k, None] * (X - mu[k])).T @ (X - mu[k])) / Nk[k]
                        + reg * np.eye(D) for k in range(K)])
        if len(hist) > 2 and abs(hist[-1] - hist[-2]) < 1e-6:
            break
    return resp, mu, hist


lab_km, mu_km = kmeans(X, K, seed=1)
resp, mu_em, hist = em(X, K, seed=1)

runs = [em(X, K, seed=s, init="wide")[2] for s in range(N_INIT)]
finals = np.array([h[-1] for h in runs])
finals_pts = np.array([em(X, K, seed=s, init="points")[2][-1] for s in range(N_INIT)])
best = max(finals.max(), finals_pts.max())

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
axes[0].scatter(X[:, 0], X[:, 1], c=np.array(["tab:blue", "tab:orange", "tab:green"])[lab_km], s=8)
axes[0].scatter(mu_km[:, 0], mu_km[:, 1], marker="x", c="black", s=90)
axes[0].set_title("K-means: hard assignment", fontsize=9)


order = [int(np.argmin(((mu_em - mu_km[k]) ** 2).sum(1))) for k in range(K)]
order = order if len(set(order)) == K else list(range(K))
base = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17]])
axes[1].scatter(X[:, 0], X[:, 1], c=np.clip(resp[:, order] @ base, 0, 1), s=8)
axes[1].scatter(mu_em[:, 0], mu_em[:, 1], marker="x", c="black", s=90)
axes[1].set_title("GMM + EM: responsibilities as colour mix", fontsize=9)

for h in runs:
    axes[2].plot(h, lw=1.0, color="tab:red" if h[-1] < best - 1.0 else "black", alpha=0.8)
axes[2].set_ylim(best - 120, best + 5)
axes[2].set_xlim(0, 60)
axes[2].set_xlabel("EM iteration")
axes[2].set_ylabel("log-likelihood")
axes[2].set_title(f"{N_INIT} random starts (wide init): red = stuck at a worse optimum", fontsize=9)
for ax in axes[:2]:
    ax.set_xlabel("x1")
axes[0].set_ylabel("x2")
fig.tight_layout()
fig.savefig("figs/fig3_kmeans_gmm.png", dpi=160)

conf = resp.max(1)
print("fig3")
print(f"  points = {N}, clusters = {K}")
print(f"  EM iterations to converge (seed 1) = {len(hist)}, final log-likelihood = {hist[-1]:.2f}")
print(f"  points with max responsibility < 0.90 : {(conf < 0.90).sum()} ({100*(conf<0.90).mean():.1f}%)")
print(f"  points with max responsibility < 0.60 : {(conf < 0.60).sum()} ({100*(conf<0.60).mean():.1f}%)")
print(f"  best log-likelihood over all starts = {best:.2f}")
print(f"  init='points': reach best {(finals_pts > best - 1.0).sum()}/{N_INIT}, "
      f"worst {finals_pts.min():.2f} (gap {best - finals_pts.min():.2f})")
print(f"  init='wide'  : reach best {(finals > best - 1.0).sum()}/{N_INIT}, "
      f"worst {finals.min():.2f} (gap {best - finals.min():.2f})")
from itertools import permutations
km_err = min((np.array(p)[lab_km] != truth).mean() for p in permutations(range(K)))
em_err = min((np.array(p)[resp.argmax(1)] != truth).mean() for p in permutations(range(K)))
print(f"  K-means label error vs truth = {100*km_err:.1f}%")
print(f"  GMM (argmax resp) error vs truth = {100*em_err:.1f}%")
