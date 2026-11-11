import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

raw = np.genfromtxt("data/faithful.csv", delimiter=",", skip_header=1)
X = raw[:, 1:3]
N, D = X.shape
Xs = (X - X.mean(0)) / X.std(0)


def kmeans(X, K, seed, iters=200):
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
    return -0.5 * (s ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * X.shape[1] * np.log(2 * np.pi)


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
                        + reg * np.eye(X.shape[1]) for k in range(X.shape[1] * 0 + K)])
        if len(hist) > 2 and abs(hist[-1] - hist[-2]) < 1e-7:
            break
    return resp, mu, hist


def best_em(X, K, starts=20):
    runs = [em(X, K, s) for s in range(starts)]
    return max(runs, key=lambda r: r[2][-1])


lab_raw, mu_raw = kmeans(X, 2, seed=1)
lab_std, _ = kmeans(Xs, 2, seed=1)
resp, mu_em, hist = best_em(X, 2)
lab_em = resp.argmax(1)


def disagree(a, b):
    return min((np.array(p)[a] != b).mean() for p in permutations(range(2)))


fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
cols = np.array(["tab:blue", "tab:orange"])
axes[0].scatter(X[:, 0], X[:, 1], c=cols[lab_raw], s=14)
axes[0].scatter(mu_raw[:, 0], mu_raw[:, 1], marker="x", c="black", s=90)
axes[0].set_title("K-means on the raw minutes", fontsize=9)

order = [int(np.argmin(((mu_em - mu_raw[k]) ** 2).sum(1))) for k in range(2)]
order = order if len(set(order)) == 2 else [0, 1]
base = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05]])
axes[1].scatter(X[:, 0], X[:, 1], c=np.clip(resp[:, order] @ base, 0, 1), s=14)
axes[1].scatter(mu_em[:, 0], mu_em[:, 1], marker="x", c="black", s=90)
axes[1].set_title("Gaussian mixture, responsibilities as colour", fontsize=9)
for ax in axes[:2]:
    ax.set_xlabel("eruption duration [min]")
axes[0].set_ylabel("waiting time to the next eruption [min]")

Ks = range(1, 7)
lls = [best_em(X, k)[2][-1] for k in Ks]
axes[2].plot(list(Ks), lls, "o-", color="black", lw=1.4, ms=5)
axes[2].axvline(2, color="tab:red", ls="--", lw=1.0)
axes[2].set_xlabel("number of components K")
axes[2].set_ylabel("log-likelihood of the best start")
axes[2].set_title("the likelihood never stops improving", fontsize=9)
fig.suptitle("Old Faithful: 272 eruptions, duration against waiting time", fontsize=10)
fig.tight_layout()
fig.savefig("figs/fig6_gmm_faithful.png", dpi=160)

conf = resp.max(1)
print("fig6  Old Faithful, n = %d" % N)
print(f"  duration  mean {X[:,0].mean():.3f}  sd {X[:,0].std(ddof=1):.3f} min")
print(f"  waiting   mean {X[:,1].mean():.3f}  sd {X[:,1].std(ddof=1):.3f} min  (sd ratio {X[:,1].std()/X[:,0].std():.1f}x)")
print(f"  K-means raw vs standardised disagree on {100*disagree(lab_raw, lab_std):.1f}% of points")
print(f"  K-means (raw) vs GMM disagree on {100*disagree(lab_raw, lab_em):.1f}% of points")
print(f"  GMM component sizes = {np.bincount(lab_em)}  mixing weights = "
      + ", ".join(f"{w:.3f}" for w in resp.mean(0)))
print(f"  points with max responsibility < 0.90 : {(conf < 0.90).sum()} ({100*(conf<0.90).mean():.1f}%)")
print(f"  points with max responsibility < 0.60 : {(conf < 0.60).sum()} ({100*(conf<0.60).mean():.1f}%)")
print("  log-likelihood by K: " + ", ".join(f"K={k} {v:.1f}" for k, v in zip(Ks, lls)))
print(f"  gain K=1->2 {lls[1]-lls[0]:.1f}, K=2->3 {lls[2]-lls[1]:.1f}, K=3->4 {lls[3]-lls[2]:.1f}")
