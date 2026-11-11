# bishop-figures

Plots I made while reading Bishop, *Deep Learning: Foundations and Concepts* —
chapter 2 and the latent variable chapters up to PCA. Reading the equations
wasn't enough to tell whether I had understood them, so I drew them myself.

All four use one dataset: `data/faithful.csv`, 272 eruptions of the Old
Faithful geyser (duration and waiting time in minutes), the `faithful`
dataset from R.

NumPy and Matplotlib only. EM and PCA are written out by hand instead of
imported, so the responsibilities and the log-likelihood per iteration are
actually visible in the code.

- `fig1_bayes.py` — posterior of the mean waiting time as eruptions come in
- `fig2_kl.py` — KL(p‖q) against KL(q‖p), fitting one Gaussian to the durations
- `fig3_kmeans_gmm.py` — K-means against a Gaussian mixture, and EM getting stuck at K = 3
- `fig4_pca.py` — PCA before and after standardising, which changes the answer

To run:

```bash
python3 fig1_bayes.py
```

Each script writes one figure into `figs/` and prints the numbers it plotted.
`run_log.txt` is one full run; the seeds are fixed, so it reproduces.
