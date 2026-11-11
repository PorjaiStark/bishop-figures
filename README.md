# bishop-figures

Plots I made while reading Bishop, *Deep Learning: Foundations and Concepts* —
chapter 2 and the latent variable chapters up to PCA. Reading the equations
wasn't enough to tell whether I had understood them, so I drew them myself.

NumPy and Matplotlib only. EM and PCA are written out by hand instead of
imported, so the responsibilities and the log-likelihood per iteration are
actually visible in the code.

- `fig1_bayes.py` — posterior of a Gaussian mean as N grows
- `fig2_kl.py` — KL(p‖q) against KL(q‖p)
- `fig3_kmeans_gmm.py` — K-means against a Gaussian mixture, and EM getting stuck
- `fig4_pca.py` — PCA as the projection with the smallest residual
- `fig5_kl_faithful.py`, `fig6_gmm_faithful.py` — fig2 and fig3 again on real data

fig1–fig4 use synthetic data with a fixed seed, so I knew the right answer
beforehand and could tell a bug from a misunderstanding. fig5 and fig6 use
`data/faithful.csv`: 272 Old Faithful eruptions, the `faithful` dataset from R.

To run:

```bash
python3 fig1_bayes.py
```

Each script writes one figure into `figs/` and prints the numbers it plotted.
`run_log.txt` is one full run; the seeds are fixed, so it reproduces.
