import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

raw = np.genfromtxt("data/faithful.csv", delimiter=",", skip_header=1)
X = raw[:, 1:3]
N, D = X.shape
Xs = (X - X.mean(0)) / X.std(0, ddof=1)
N_INIT = 20


def kmeans(X, K, seed, iters=300):
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
    s = np.linalg.solve(L, d.T)
    return -0.5 * (s ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * D * np.log(2 * np.pi)


def em(X, K, seed, iters=500, reg=1e-6):
    r = np.random.default_rng(seed)
    mu = X[r.choice(len(X), K, replace=False)]
    cov = np.array([np.cov(X.T) for _ in range(K)])
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
        if len(hist) > 2 and abs(hist[-1] - hist[-2]) < 1e-7:
            break
    return resp, mu, hist


def best_em(X, K, starts=N_INIT):
    return max((em(X, K, s) for s in range(starts)), key=lambda r: r[2][-1])


lab_km, mu_km = kmeans(X, 2, seed=1)
lab_std, _ = kmeans(Xs, 2, seed=1)
resp, mu_em, hist = best_em(X, 2)
lab_em = resp.argmax(1)
runs3 = [em(X, 3, s)[2] for s in range(N_INIT)]
fin3 = np.array([h[-1] for h in runs3])
best3 = fin3.max()


def disagree(a, b, K=2):
    return min((np.array(p)[a] != b).mean() for p in permutations(range(K)))


fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
cols = np.array(["tab:blue", "tab:orange"])
axes[0].scatter(X[:, 0], X[:, 1], c=cols[lab_km], s=14)
axes[0].scatter(mu_km[:, 0], mu_km[:, 1], marker="x", c="black", s=90)
axes[0].set_title("K-means, K = 2: hard assignment", fontsize=9)

order = [int(np.argmin(((mu_em - mu_km[k]) ** 2).sum(1))) for k in range(2)]
order = order if len(set(order)) == 2 else [0, 1]
base = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05]])
axes[1].scatter(X[:, 0], X[:, 1], c=np.clip(resp[:, order] @ base, 0, 1), s=14)
axes[1].scatter(mu_em[:, 0], mu_em[:, 1], marker="x", c="black", s=90)
axes[1].set_title("Gaussian mixture, K = 2: responsibilities", fontsize=9)
for ax in axes[:2]:
    ax.set_xlabel("eruption duration [min]")
axes[0].set_ylabel("waiting time [min]")

for h in runs3:
    axes[2].plot(h, lw=1.0, color="black" if h[-1] > best3 - 0.5 else "tab:red", alpha=0.85)
axes[2].set_ylim(best3 - 40, best3 + 3)
axes[2].set_xlim(0, 120)
axes[2].set_xlabel("EM iteration")
axes[2].set_ylabel("log-likelihood")
axes[2].set_title(f"K = 3, {N_INIT} random starts: red = worse optimum", fontsize=9)
fig.suptitle("Old Faithful: 272 eruptions, duration against waiting time", fontsize=10)
fig.tight_layout()
fig.savefig("figs/fig3_kmeans_gmm.png", dpi=160)

conf = resp.max(1)
print("fig3  Old Faithful, n = %d" % N)
print(f"  duration mean {X[:,0].mean():.3f} sd {X[:,0].std(ddof=1):.3f} | "
      f"waiting mean {X[:,1].mean():.3f} sd {X[:,1].std(ddof=1):.3f} (sd ratio {X[:,1].std()/X[:,0].std():.1f}x)")
print(f"  K-means raw vs standardised disagree on {100*disagree(lab_km, lab_std):.1f}% of points")
print(f"  K-means vs mixture disagree on {100*disagree(lab_km, lab_em):.1f}% of points")
print(f"  mixture weights {resp.mean(0)[0]:.4f} / {resp.mean(0)[1]:.4f}, sizes {np.bincount(lab_em)}")
print(f"  points with max responsibility < 0.90 : {(conf < 0.90).sum()} ({100*(conf<0.90).mean():.1f}%)")
print(f"  points with max responsibility < 0.60 : {(conf < 0.60).sum()} ({100*(conf<0.60).mean():.1f}%)")
fin2 = np.array([em(X, 2, s)[2][-1] for s in range(N_INIT)])
print(f"  K = 2: {N_INIT} starts all reach {fin2.max():.2f} "
      f"({(fin2 > fin2.max()-0.5).sum()}/{N_INIT}), distinct optima {len(np.unique(fin2.round(1)))}")
print(f"  K = 3: best {best3:.2f}, worst {fin3.min():.2f}, reach best {(fin3 > best3-0.5).sum()}/{N_INIT}, "
      f"distinct optima {len(np.unique(fin3.round(1)))}")
lls = [best_em(X, k)[2][-1] for k in range(1, 7)]
print("  log-likelihood by K: " + ", ".join(f"K={k} {v:.1f}" for k, v in zip(range(1, 7), lls)))
print(f"  gain K=1->2 {lls[1]-lls[0]:.1f}, K=2->3 {lls[2]-lls[1]:.1f}, K=3->4 {lls[3]-lls[2]:.1f}")
