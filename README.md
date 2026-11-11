# bishop-figures

Figures written to check my understanding of Christopher M. Bishop and Hugh
Bishop, *Deep Learning: Foundations and Concepts* (Springer, 2024) — chapter 2
(Probabilities) and the latent variable chapters up to PCA.

NumPy and Matplotlib only. EM and PCA are written out by hand instead of
called from a library, so the responsibility matrix, the log-likelihood per
iteration and the reconstruction error are all visible in the code.

| script | figure | what it shows |
|---|---|---|
| `fig1_bayes.py` | `figs/fig1_bayes.png` | posterior of a Gaussian mean tightening as N grows |
| `fig2_kl.py` | `figs/fig2_kl.png` | KL(p‖q) vs KL(q‖p): mean-covering vs mode-seeking |
| `fig3_kmeans_gmm.py` | `figs/fig3_kmeans_gmm.png` | hard vs soft assignment, and EM local optima |
| `fig4_pca.py` | `figs/fig4_pca.png` | PCA as the projection with the smallest residual |
| `fig5_kl_faithful.py` | `figs/fig5_kl_faithful.png` | fig2 repeated on real data |
| `fig6_gmm_faithful.py` | `figs/fig6_gmm_faithful.png` | fig3 repeated on real data |

fig1–fig4 use synthetic data with a fixed seed, so the right answer is known
in advance and a wrong figure can be told apart from a wrong understanding.
fig5–fig6 then run the same methods on measurements I did not generate:
`data/faithful.csv`, 272 eruptions of the Old Faithful geyser (duration and
waiting time, in minutes), the `faithful` dataset distributed with R.

## Run

```bash
python3 fig1_bayes.py      # each script writes its own figure into figs/
```

Every script prints the numbers it plotted; `run_log.txt` is one full run.
All random draws are seeded, so the numbers reproduce.

## Results of that run

- **fig1** — posterior sd 1.000 → 0.218 from N = 0 to N = 20 (4.58× tighter)
- **fig2** — forward KL gives sigma 2.09 and covers both modes, reverse KL gives
  mu −2.00, sigma 0.60 and sits on one. Scoring the mode-seeking fit with the
  other direction gives 10.42 against 0.56 for the moment-matched fit
- **fig3** — 32.4% of 500 points have a maximum responsibility below 0.90, so a
  hard label would hide real ambiguity. Labelling by argmax of the
  responsibilities is wrong on 12.8% of points against 17.8% for K-means. Over
  20 random starts EM reaches the best optimum 19/20 times from data-point
  initialisation, 14/20 from a broader one
- **fig4** — on 10-D data built from a 3-D latent structure the first three
  components hold 97.1% of the variance, and the error drop at the third
  component is 43.5× the drop at the fourth
- **fig5** — a 2-component mixture fitted to the eruption durations gives
  components at 2.02 ± 0.24 min (weight 0.35) and 4.27 ± 0.44 min (weight 0.65).
  The reverse-KL Gaussian lands on the second component almost exactly
  (4.276, 0.440); the forward-KL one spans both (3.488, 1.139)
- **fig6** — the two groups here barely overlap: only 1 of 272 points has a
  maximum responsibility below 0.90, and K-means and the mixture disagree on
  1.8%. Adding components keeps raising the log-likelihood (+159.5 from K=1 to
  K=2, then +15.8, then +8.4), so the likelihood alone cannot choose K

Written for the Professional Growth Portfolio of 2103493 ME Portfolio,
Department of Mechanical Engineering, Chulalongkorn University.
