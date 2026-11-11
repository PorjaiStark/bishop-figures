# bishop-figures

Figures I wrote to check that I actually understand what I read in
Christopher M. Bishop and Hugh Bishop, *Deep Learning: Foundations and
Concepts* (Springer, 2024) — chapter 2 (Probabilities) and the latent
variable chapters up to PCA.

Everything is NumPy + Matplotlib only. The EM algorithm and PCA are written
out by hand rather than called from a library, so that the quantities the
book talks about (the responsibility matrix, the log-likelihood per
iteration, the reconstruction error) are visible in the code.

| script | figure | what it shows |
|---|---|---|
| `fig1_bayes.py` | `figs/fig1_bayes.png` | posterior of a Gaussian mean tightening as N grows |
| `fig2_kl.py` | `figs/fig2_kl.png` | KL(p‖q) vs KL(q‖p): mean-covering vs mode-seeking |
| `fig3_kmeans_gmm.py` | `figs/fig3_kmeans_gmm.png` | hard vs soft assignment, and EM local optima |
| `fig4_pca.py` | `figs/fig4_pca.png` | PCA as the projection with the smallest residual |

## Run

```bash
python3 fig1_bayes.py      # each script writes its own figure into figs/
```

Every script prints the numbers it plotted; `run_log.txt` is the output of
one full run. All random draws are seeded, so the numbers reproduce.

## Results of that run

- **fig1** — posterior sd 1.000 → 0.218 from N = 0 to N = 20 (4.58× tighter)
- **fig2** — forward KL gives sigma 2.09 (covers both modes), reverse KL gives
  mu −2.00, sigma 0.60 (sits on one mode). Scoring the mode-seeking fit with
  the other direction gives KL(p‖q) = 10.42 against 0.56 for the moment-matched fit
- **fig3** — 32.4% of 500 points have a maximum responsibility below 0.90, so a
  hard label would be hiding real ambiguity. Labelling by argmax of the
  responsibilities is wrong on 12.8% of points against 17.8% for K-means.
  From 20 random starts, EM reaches the best optimum 19/20 times with means
  initialised at data points, 14/20 with a broader initialisation
- **fig4** — on 10-D data generated from a 3-D latent structure, the first three
  components hold 97.1% of the variance, and the error drop at the third
  component is 43.5× the drop at the fourth

Written for the Professional Growth Portfolio of 2103493 ME Portfolio,
Department of Mechanical Engineering, Chulalongkorn University.
