# Stage 8C: simultaneous-field calibration on new ID trajectories

The question is: **Can raw ensemble spread be converted into calibrated predictive bands with valid empirical coverage on data that were not used to fit the calibration?**

Horizon-specific split-conformal factors were fitted on 256 new calibration trajectories and frozen before running predictions or computing model-error diagnostics on the independent 256-trajectory audit archive. This is calibration only: the five members, raw-spread machinery, U_rel, and U_rms are unchanged.

## Results and practical interpretation

| Nominal | Mean horizon coverage [95% trajectory-bootstrap CI] | Mean W_rel | Median trajectory-mean W_rel | Secondary pointwise fraction |
| --- | --- | --- | --- | --- |
| 50% | 0.4863 [0.4398, 0.5352] | 1.192 | 1.148 | 0.998457 |
| 70% | 0.7152 [0.6734, 0.7555] | 1.655 | 1.594 | 0.999503 |
| 80% | 0.8160 [0.7797, 0.8488] | 2.036 | 1.961 | 0.999761 |
| 90% | 0.9070 [0.8809, 0.9301] | 2.738 | 2.637 | 0.999910 |
| 95% | 0.9555 [0.9387, 0.9703] | 3.707 | 3.573 | 0.999966 |

At nominal 90%, per-horizon audit coverage ranges from 87.9% to 93.0%. The mean normalized full width grows from 0.683 at t=0.1 to 4.678 at t=1.0. A W_rel of 1 means the RMS full width equals the RMS ensemble-mean field. Factors of 23.82–26.83 times raw spread are required for the 90% simultaneous-field target.

This is informative as an independent coverage audit, but it does **not** establish sharp, practically useful field predictions: the late-rollout bands span several times the predicted field RMS. No application tolerance was predeclared, so utility cannot be certified by coverage alone. The frozen whole-field max-score target is demanding and can be controlled by a small number of cells with large error-to-spread ratios. No factor was changed after looking at audit coverage or width.

Of the 50 pointwise Wilson intervals, 49 contain their nominal level. The exception is conservative coverage at t=0.6, nominal 80%: 85.94% observed, interval [81.15%, 89.67%]. This descriptive count is not a multiple-testing procedure or proof of exact coverage. All five nominal levels lie within their grouped aggregate intervals.

## Frozen ensemble and provenance

Stage 8B revision: `e31221b3a3baaaba3c84e487ef885faa9158b5d8`. Stage 8A committed revision: `4967490593f1f5c125dc8e87abd2387144741954`. The member manifests preserve training-time revision `f095871d842e469f51d49751a7459684a507fd77` plus exact training-source hashes. These distinctions are preserved rather than rewriting historical provenance.

Ensemble manifest SHA-256: `0da21da72cb440d9bfc08563c1308a29d6dd6cc8f85785c5483d1ef6ec02d713`. Stage 8B freeze SHA-256: `3f475d9980c63519e01577678f772e2c50ab64f098e5081059eb698ccc2bba54`. Stage 8B summary SHA-256: `49da86a438651917f124c1793037b0cf5e3775910dd635d2ef7b4f058b9e5c17`. Stage 8C predeclared protocol SHA-256: `6c30d4379b01cb082475dbf9ad18ea65971e916938323902ee9e573efdc254ba`.

| Member | Init/shuffle seeds | Selected checkpoint SHA-256 |
| --- | --- | --- |
| 0 | 2028 / 5029 | `cf8899453cadf2443961fd65daa736269ed671dd26a375fd3edc1159ae56ca33` |
| 1 | 8101 / 8201 | `591ec6819777909579bdc9c4f57ba62f25d91b709609c708c4afef30b78b1903` |
| 2 | 8102 / 8202 | `2f15a50f9a438a3db148d420eede5e15bf25f182ea3fd0682d3be50822253f79` |
| 3 | 8103 / 8203 | `ab95d65c8d340b550ee578c0bcf939df960e1d7a3f9780d2fc42f1434f72b5fd` |
| 4 | 8104 / 8204 | `ee49b65ca83df821c8703d97009ee82283d7bcb864218e5acac5a48925714a46` |

The existing strict loader verifies all five member and ensemble manifests, checkpoint hashes, selected epoch, full training budgets, architecture, ordered train/validation/test splits, optimizer and objective metadata, gate/selection records, finite weights, normalization, original training archive, and exact historical source hashes. The Stage 8C verifier additionally checks the frozen Stage 8B revision and source/summary hashes. A mismatch aborts the phase. All five members are retained.

Architecture remains FNO2d modes=12, width=32, four blocks, projection_width=64, four input channels and one output. Normalization canonical hash: `e413b4e2a87fb0068bce67d3d81684b31d1d7d57b92b5ec96c50b9838c7e0120`. Original training archive hash: `73974ce6d7be46209d22f09e8d9225dd97e59aaf324d55c5328e9f18b375638e`. Field mean/std are 0.12027802015661833/0.2355322120279037. No normalization refit or model training occurred.

## Two new independent ID archives

| Role | Path | Seed | SHA-256 |
| --- | --- | --- | --- |
| calibration | data/stage8/calibration.npz | 8302 | `d1ee3147a68735b7406c9a8afe800cb2b137a942376cfd19f9014b443efa5205` |
| audit | data/stage8/audit.npz | 8303 | `959055fc071134c7bc67098d63303611cb5bd0c29f54b013b901d481f87bd210` |

Both archives use the unchanged Stage 2 public `GenerationConfig`, `generate_dataset`, and `save_dataset`: 256 trajectories, 64×64 grid, 11 snapshots including t=0, final time 1.0. The original ranges remain cx,cy∈[-1,1], nu∈[0.005,0.05], blob width∈[0.4,0.8], amplitude∈[0.5,1.5], one to three blobs, periodic [0,2π)², safety=0.9.

Seeds 8302 and 8303 were verified unused against all existing dataset metadata, member initialization/shuffle seeds, and Stage 8B bootstrap seed. Existing seeds were `[2026, 2028, 4100, 4101, 4110, 4111, 4112, 4113, 4120, 5029, 8101, 8102, 8103, 8104, 8201, 8202, 8203, 8204, 8301]`. Physical-input fingerprints hash coefficients, blob counts, masks, centers, widths, and amplitudes; they exclude archive-local IDs and grid resolution. All 512 new fingerprints are internally unique and disjoint from every old ID/OOD/resolution input and from each other. Historical/OOD archives were opened only for seed/input-identity exclusion, never for fitting, inference, or Stage 8C coverage conclusions.

The calibration archive fits q only. The audit archive fits nothing. Generation-time schema, finite-value, and conservation checks apply to both archives; these are not model-error inspection or tuning. Archive IDs 0–255 are local to each archive. Both archives remain ignored under `data/stage8/`.

| Archive | Minimum reference field | Maximum absolute mass drift |
| --- | --- | --- |
| calibration | -2.933e-12 | 3.553e-15 |
| audit | -3.837e-09 | 3.553e-15 |

Small negative values are numerical undershoots from the unchanged centered/RK4 solver. They were retained. No positivity correction, distribution change, or new long-horizon numerical reference was introduced.

## Rollout, score, quantile, and exact band definition

Each member starts from the same true initial field and feeds back only its own prior prediction. Coefficients remain fixed. The unchanged Stage 8A helper internally computes unused teacher-forced outputs, but only member-wise autoregressive predictions enter this analysis. Physical inverse normalization precedes float64 sample-spread reductions. t=0 is stored for verification and excluded from all calibration, coverage, width, and diagnostic summaries.

```text
mu = mean of the five physical member predictions
sigma = sqrt(sum_m((u_m-mu)^2)/(5-1))                 # ddof=1
epsilon_spread = epsilon_rms = 1e-12 * frozen_field_std
z_i,h(x,y) = abs(truth_i,h(x,y)-mu_i,h(x,y)) / max(sigma_i,h(x,y),epsilon_spread)
R_i,h = max_grid z_i,h                              # one score per trajectory field
k = min(n_cal, ceil((n_cal+1)*nominal_coverage))
q[h,coverage] = kth sorted calibration R[:,h]        # one-based k
lower = mu - q[h,coverage] * sigma                  # raw sigma, no floor in bands
upper = mu + q[h,coverage] * sigma
covered_i,h = every grid cell lies inside [lower,upper], inclusive
W_rel_i,h = RMS(upper-lower) / max(RMS(mu),epsilon_rms)
U_rms = RMS(sigma)
U_rel = U_rms / max(RMS(mu),epsilon_rms)              # unchanged Stage 8B score
```

The frozen numerical floor is `2.355322120279037e-13` physical units. With n_cal=256, the five one-based order indices are **129, 180, 206, 232, 245** for 50%, 70%, 80%, 90%, 95%. No clipping occurs. In general, clipping k to n when the requested level exceeds n/(n+1) cannot provide the usual nominal finite-sample guarantee; this case is tested and does not occur here. No quantile interpolation or post-audit adjustment is used.

Spread-floor activations (cells, summed over ten prediction horizons): calibration **0**, audit **0**, each out of 10,485,760 cells. The per-horizon cell and field counts are recorded in the summary. Because the score floors sigma but the requested band uses raw sigma, the conformal score-set and raw band can differ wherever sigma is below the floor. They coincide on all observed prediction fields here. The general coverage interpretation requires score/band equivalence as well as exchangeability; it must not be extended unconditionally to future floor-activated cases. No formula was silently changed to hide this distinction.

## Horizon-specific factors, frozen before audit inference

| Time | q50 | q70 | q80 | q90 | q95 |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 12.165579 | 17.337164 | 20.107459 | 25.815911 | 30.632849 |
| 0.2 | 12.175185 | 16.476447 | 20.022003 | 26.624498 | 31.368614 |
| 0.3 | 11.393881 | 15.051603 | 18.480910 | 25.109124 | 35.010034 |
| 0.4 | 10.940699 | 15.259722 | 18.453326 | 24.513451 | 32.520896 |
| 0.5 | 10.687102 | 14.228321 | 18.315876 | 26.834111 | 34.499061 |
| 0.6 | 10.965738 | 15.107355 | 19.309664 | 26.790171 | 34.808893 |
| 0.7 | 10.754522 | 15.558189 | 18.921229 | 23.853537 | 29.875108 |
| 0.8 | 10.819662 | 14.949442 | 18.436416 | 23.945945 | 34.943397 |
| 0.9 | 10.533738 | 14.946470 | 18.298377 | 23.823511 | 36.750269 |
| 1.0 | 10.550941 | 14.606339 | 17.683860 | 24.902948 | 33.026567 |

Factor file SHA-256: `380812e0cf516745416238007450558d5a03099888fcf93657a9d1c8aae673cf`. Public factors CSV SHA-256: `6ac5cd408b5066c3b606d8931f811180412791ad04703d1dd56afc7699e9bf73`. The freeze receipt records `audit_predictions_existed_at_freeze=false`. Audit inference requires this receipt and verifies both hashes. Fitting is hard-wired to the verified calibration role; audit or historical role arguments are rejected.

The primary target is one entire field at a fixed horizon. There are 256 calibration scores at each horizon, not 256×4096 independent cell scores. Deep-ensemble correlation does not create additional calibration units.

## Audit coverage, uncertainty intervals, and normalized sharpness

Intervals below are 95% Wilson binomial intervals using 256 trajectory-field events per horizon, conditional on the frozen factors. They quantify audit sampling uncertainty, not uncertainty over newly fitted q, ensembles, or research decisions. They are pointwise intervals, not simultaneous intervals over all 50 reported estimates. Coverage error is empirical minus nominal. Width columns use full two-sided width, not half-width.

| Time | Nominal | Covered / 256 | Field coverage [95% CI] | Error | Mean W_rel | Median W_rel | Secondary pointwise |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 50% | 127 | 0.4961 [0.4354, 0.5569] | -0.0039 | 0.322 | 0.318 | 0.999354 |
| 0.1 | 70% | 182 | 0.7109 [0.6526, 0.7630] | +0.0109 | 0.459 | 0.453 | 0.999840 |
| 0.1 | 80% | 206 | 0.8047 [0.7518, 0.8486] | +0.0047 | 0.532 | 0.526 | 0.999915 |
| 0.1 | 90% | 232 | 0.9062 [0.8643, 0.9362] | +0.0062 | 0.683 | 0.675 | 0.999969 |
| 0.1 | 95% | 244 | 0.9531 [0.9199, 0.9730] | +0.0031 | 0.811 | 0.801 | 0.999985 |
| 0.2 | 50% | 128 | 0.5000 [0.4392, 0.5608] | +0.0000 | 0.577 | 0.572 | 0.999228 |
| 0.2 | 70% | 183 | 0.7148 [0.6567, 0.7667] | +0.0148 | 0.781 | 0.773 | 0.999777 |
| 0.2 | 80% | 214 | 0.8359 [0.7857, 0.8763] | +0.0359 | 0.949 | 0.940 | 0.999903 |
| 0.2 | 90% | 238 | 0.9297 [0.8916, 0.9551] | +0.0297 | 1.261 | 1.250 | 0.999969 |
| 0.2 | 95% | 245 | 0.9570 [0.9247, 0.9758] | +0.0070 | 1.486 | 1.473 | 0.999985 |
| 0.3 | 50% | 128 | 0.5000 [0.4392, 0.5608] | +0.0000 | 0.758 | 0.741 | 0.998826 |
| 0.3 | 70% | 176 | 0.6875 [0.6283, 0.7412] | -0.0125 | 1.002 | 0.978 | 0.999572 |
| 0.3 | 80% | 209 | 0.8164 [0.7644, 0.8590] | +0.0164 | 1.230 | 1.201 | 0.999800 |
| 0.3 | 90% | 231 | 0.9023 [0.8598, 0.9330] | +0.0023 | 1.671 | 1.632 | 0.999924 |
| 0.3 | 95% | 250 | 0.9766 [0.9498, 0.9892] | +0.0266 | 2.330 | 2.275 | 0.999986 |
| 0.4 | 50% | 119 | 0.4648 [0.4047, 0.5260] | -0.0352 | 0.928 | 0.904 | 0.998596 |
| 0.4 | 70% | 180 | 0.7031 [0.6445, 0.7558] | +0.0031 | 1.294 | 1.261 | 0.999532 |
| 0.4 | 80% | 202 | 0.7891 [0.7350, 0.8346] | -0.0109 | 1.565 | 1.525 | 0.999757 |
| 0.4 | 90% | 225 | 0.8789 [0.8332, 0.9134] | -0.0211 | 2.079 | 2.026 | 0.999895 |
| 0.4 | 95% | 240 | 0.9375 [0.9009, 0.9612] | -0.0125 | 2.758 | 2.687 | 0.999954 |
| 0.5 | 50% | 117 | 0.4570 [0.3971, 0.5182] | -0.0430 | 1.096 | 1.055 | 0.998393 |
| 0.5 | 70% | 177 | 0.6914 [0.6323, 0.7448] | -0.0086 | 1.459 | 1.405 | 0.999408 |
| 0.5 | 80% | 211 | 0.8242 [0.7729, 0.8660] | +0.0242 | 1.879 | 1.808 | 0.999728 |
| 0.5 | 90% | 234 | 0.9141 [0.8733, 0.9426] | +0.0141 | 2.752 | 2.649 | 0.999919 |
| 0.5 | 95% | 246 | 0.9609 [0.9296, 0.9786] | +0.0109 | 3.539 | 3.406 | 0.999960 |
| 0.6 | 50% | 129 | 0.5039 [0.4431, 0.5646] | +0.0039 | 1.316 | 1.257 | 0.998472 |
| 0.6 | 70% | 192 | 0.7500 [0.6935, 0.7991] | +0.0500 | 1.813 | 1.732 | 0.999503 |
| 0.6 | 80% | 220 | 0.8594 [0.8115, 0.8967] | +0.0594 | 2.317 | 2.214 | 0.999781 |
| 0.6 | 90% | 236 | 0.9219 [0.8824, 0.9489] | +0.0219 | 3.214 | 3.072 | 0.999914 |
| 0.6 | 95% | 245 | 0.9570 [0.9247, 0.9758] | +0.0070 | 4.176 | 3.991 | 0.999959 |
| 0.7 | 50% | 124 | 0.4844 [0.4238, 0.5454] | -0.0156 | 1.475 | 1.422 | 0.998217 |
| 0.7 | 70% | 191 | 0.7461 [0.6894, 0.7955] | +0.0461 | 2.134 | 2.057 | 0.999505 |
| 0.7 | 80% | 211 | 0.8242 [0.7729, 0.8660] | +0.0242 | 2.595 | 2.501 | 0.999753 |
| 0.7 | 90% | 230 | 0.8984 [0.8553, 0.9297] | -0.0016 | 3.271 | 3.153 | 0.999893 |
| 0.7 | 95% | 245 | 0.9570 [0.9247, 0.9758] | +0.0070 | 4.097 | 3.950 | 0.999949 |
| 0.8 | 50% | 126 | 0.4922 [0.4315, 0.5531] | -0.0078 | 1.668 | 1.607 | 0.998117 |
| 0.8 | 70% | 186 | 0.7266 [0.6689, 0.7775] | +0.0266 | 2.304 | 2.221 | 0.999373 |
| 0.8 | 80% | 210 | 0.8203 [0.7687, 0.8625] | +0.0203 | 2.842 | 2.739 | 0.999709 |
| 0.8 | 90% | 230 | 0.8984 [0.8553, 0.9297] | -0.0016 | 3.691 | 3.558 | 0.999886 |
| 0.8 | 95% | 245 | 0.9570 [0.9247, 0.9758] | +0.0070 | 5.386 | 5.191 | 0.999969 |
| 0.9 | 50% | 124 | 0.4844 [0.4238, 0.5454] | -0.0156 | 1.802 | 1.741 | 0.997772 |
| 0.9 | 70% | 186 | 0.7266 [0.6689, 0.7775] | +0.0266 | 2.556 | 2.470 | 0.999319 |
| 0.9 | 80% | 210 | 0.8203 [0.7687, 0.8625] | +0.0203 | 3.130 | 3.024 | 0.999668 |
| 0.9 | 90% | 234 | 0.9141 [0.8733, 0.9426] | +0.0141 | 4.075 | 3.937 | 0.999868 |
| 0.9 | 95% | 245 | 0.9570 [0.9247, 0.9758] | +0.0070 | 6.286 | 6.073 | 0.999967 |
| 1.0 | 50% | 123 | 0.4805 [0.4200, 0.5415] | -0.0195 | 1.982 | 1.926 | 0.997596 |
| 1.0 | 70% | 178 | 0.6953 [0.6364, 0.7485] | -0.0047 | 2.744 | 2.666 | 0.999202 |
| 1.0 | 80% | 196 | 0.7656 [0.7100, 0.8134] | -0.0344 | 3.322 | 3.228 | 0.999599 |
| 1.0 | 90% | 232 | 0.9062 [0.8643, 0.9362] | +0.0062 | 4.678 | 4.545 | 0.999859 |
| 1.0 | 95% | 241 | 0.9414 [0.9056, 0.9642] | -0.0086 | 6.205 | 6.028 | 0.999944 |

All physical width means/medians, RMS widths, normalized widths, and secondary pointwise means/medians for all 50 horizon/level combinations are in [stage8c_coverage.csv](results/stage8c_coverage.csv). The physical full-width summary at 90% is:

| Time | Mean spatial-average width | Median spatial-average width | Mean RMS width | Median RMS width |
| --- | --- | --- | --- | --- |
| 0.1 | 0.153176 | 0.141051 | 0.183442 | 0.168942 |
| 0.2 | 0.267145 | 0.241753 | 0.338180 | 0.303128 |
| 0.3 | 0.338139 | 0.300021 | 0.446278 | 0.399358 |
| 0.4 | 0.405362 | 0.355846 | 0.552361 | 0.490294 |
| 0.5 | 0.520290 | 0.451645 | 0.726511 | 0.647023 |
| 0.6 | 0.592178 | 0.508710 | 0.842475 | 0.747373 |
| 0.7 | 0.589842 | 0.500641 | 0.851160 | 0.750018 |
| 0.8 | 0.653422 | 0.547601 | 0.953107 | 0.840746 |
| 0.9 | 0.709989 | 0.594203 | 1.044039 | 0.916622 |
| 1.0 | 0.803985 | 0.677773 | 1.189452 | 1.039148 |

For physical average/median widths, first compute each field’s spatial-average full width, then take its mean/median across trajectories. RMS widths and W_rel are computed separately per field. W_rel equals 2*q*U_rel here, and the evaluator checks that identity numerically.

## Reliability curves and grouped interpretation

`runs/stage8/calibration/plots/reliability_by_horizon.png` contains all ten nominal-vs-empirical curves and pointwise Wilson intervals. The diagonal is a visual reference only. `reliability_aggregate.png` gives every horizon equal weight: it first averages the ten events within each trajectory, then averages trajectories and resamples complete trajectory rows 2,000 times (seed 8304) for percentile intervals. The sampling unit remains one trajectory. This aggregate is **not** the probability that all ten fields are simultaneously covered; no simultaneous-time calibration was fitted. Detailed row-level outcomes and bootstrap indices are saved in the ignored audit diagnostics.

The secondary pointwise statistic is the fraction of covered cells within each field, followed by an average across trajectory fields. Its near-unity value does not substitute for the simultaneous-field target. No cell-independent binomial interval or cell-level conformal guarantee is claimed. No pooled-cell standard error is computed.

## Raw spread, calibrated width, and missed failures

Raw sigma is an uncalibrated ensemble dispersion proxy. The calibrated half-width is q[h,level]*sigma. A common positive factor at a fixed horizon leaves trajectory and spatial rank order unchanged. U_rel is still the unchanged Stage 8B primary scalar score; horizon-specific scaling does not repair ranking failures or imply a Bayesian posterior. Ranking across different horizons need not be preserved by different q values.

| Rule | Audit ID | Final physical error | Final U_rel | Final 90% field covered |
| --- | --- | --- | --- | --- |
| predetermined | 0 | 0.03301 | 0.04188 | True |
| largest_final_error | 172 | 0.41290 | 0.08408 | True |
| largest_final_U_rel | 158 | 0.36720 | 0.23006 | True |

The predeclared selections are audit ID 0, independently largest final physical relative L2 error, and independently largest final U_rel. Each has five-panel maps at t=0.1,0.5,1.0: truth, ensemble mean, absolute error, raw spread, and 90% full band width. Ground truth and mean share a color scale, absolute error and raw spread share another, and calibrated full width has its own labeled physical scale because of its much larger magnitude. All selections, including unfavorable ones and any duplicate selected ID, are retained. Figures remain ignored under `runs/stage8/calibration/plots/`.

At t=1.0, **24 of 256** audit fields are not completely covered by the nominal 90% band. Their archive-local IDs are `[2, 7, 31, 52, 69, 97, 116, 125, 127, 133, 139, 144, 170, 178, 179, 197, 204, 213, 216, 218, 219, 229, 248, 255]`. These are band noncoverage cases; a nominal 90% procedure does not promise all trajectories succeed.

For an additional descriptive application of the frozen Stage 8B ranking rule, the most-uncertain ceil(256/5)=52 cases miss **19 of the 52** highest-final-error cases. The worst-error audit trajectory is **172**, with error **0.41290** and U_rel **0.08408**; top-fifth uncertainty selects it: **False**; its final 90% band covers the field: **True**. Top-fifth-error cases in the lowest uncertainty quintile have IDs `[]`. These diagnostics neither fit nor select q and do not convert a coverage result into reliable failure ranking.

The first three final-horizon 90% noncoverage cases in ascending archive-ID order illustrate that nearly all cells can be covered while the whole field fails:

| Audit ID | Relative L2 error | U_rel | W_rel at 90% | Covered cell fraction |
| --- | --- | --- | --- | --- |
| 2 | 0.06524 | 0.04804 | 2.393 | 0.999756 |
| 7 | 0.12472 | 0.14589 | 7.266 | 0.999512 |
| 31 | 0.23476 | 0.13253 | 6.601 | 0.999268 |

## U_rel/U_rms diagnostics

| Time | Calibration mean U_rel | Audit mean U_rel | Calibration mean U_rms | Audit mean U_rms | Audit mean physical error |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.013686 | 0.013238 | 0.003654 | 0.003553 | 0.018306 |
| 0.2 | 0.024520 | 0.023688 | 0.006538 | 0.006351 | 0.033858 |
| 0.3 | 0.034448 | 0.033272 | 0.009148 | 0.008887 | 0.047983 |
| 0.4 | 0.043917 | 0.042411 | 0.011596 | 0.011266 | 0.061106 |
| 0.5 | 0.053117 | 0.051286 | 0.013931 | 0.013537 | 0.073456 |
| 0.6 | 0.062142 | 0.059990 | 0.016178 | 0.015724 | 0.085172 |
| 0.7 | 0.071057 | 0.068573 | 0.018355 | 0.017841 | 0.096355 |
| 0.8 | 0.079911 | 0.077072 | 0.020477 | 0.019901 | 0.107081 |
| 0.9 | 0.088726 | 0.085517 | 0.022551 | 0.021912 | 0.117412 |
| 1.0 | 0.097522 | 0.093934 | 0.024582 | 0.023882 | 0.127404 |

No scalar-score denominator floors activate. U_rel was not redesigned, selected against alternatives, or refitted. It is a scalar error-awareness diagnostic; Stage 8C calibrates the spatial spread field.

## Scientific guardrails and limitations

- The PDE is deterministic. These bands describe model error over a distribution of inputs, not intrinsic stochastic PDE noise.
- Ensemble members are correlated and share data, architecture, normalization, and potential biases. No Bayesian posterior or independent-member noise interpretation is used.
- Split-conformal interpretation is distribution-dependent and requires exchangeable calibration/future trajectories, a frozen predictor, and matching score/band sets. It is marginal over ID input draws at a fixed horizon, not conditional coverage for every physical parameter or difficult trajectory.
- Historical train/validation/test/fresh-ID/OOD data are not calibration data. OOD metadata were used solely to rule out reuse of seeds and physical inputs. No OOD inference or coverage claim is made.
- Whole-field coverage concerns the stored 64×64 grid, not every continuous spatial point. It does not provide joint coverage across time.
- Width matters: very broad bands may achieve coverage while providing little precision. The scalar multiplication preserves ranking failures at fixed horizon.
- Audit intervals quantify sampling error with the fitted factors held fixed. Coverage errors across horizons and nominal levels are correlated; no multiplicity-adjusted claim is made.
- Max standardized residuals are sensitive to individual small-spread cells. The score floor differs from the raw band formula in floor-activated cases; no such activations occurred here, and the general caveat is retained.
- The existing finite-difference/RK4 references have numerical error and tiny negative undershoots. Calibration is to those references, not exact continuous PDE solutions.
- Audit results are reported once with the factors fixed. Reusing these audit data to tune a later interval method would require a new independent audit for a fresh coverage claim.
- The optional global factor comparison was not run. No claim of measured superiority over global calibration is made; horizon-specific calibration remains the requested primary method.

## Reproduction and public/ignored outputs

From the repository root, with the existing frozen Stage 8A/B artifacts, run the phases once in this order:

```bash
MPLCONFIGDIR=/tmp/stage8b-matplotlib .venv/bin/python -m examples.calibrate_stage8c generate
MPLCONFIGDIR=/tmp/stage8b-matplotlib .venv/bin/python -m examples.calibrate_stage8c predict-calibration
MPLCONFIGDIR=/tmp/stage8b-matplotlib .venv/bin/python -m examples.calibrate_stage8c fit
MPLCONFIGDIR=/tmp/stage8b-matplotlib .venv/bin/python -m examples.calibrate_stage8c predict-audit
MPLCONFIGDIR=/tmp/stage8b-matplotlib .venv/bin/python -m examples.calibrate_stage8c evaluate
MPLCONFIGDIR=/tmp/stage8b-matplotlib .venv/bin/python -m pytest -q
```

The phases refuse to overwrite generated archives or frozen outputs. A rerun should use preserved artifacts for verification; regeneration requires an explicitly separate run location or deliberate lifecycle decision, not silent replacement. Generation must begin with unused seeds and absent `data/stage8/`.

New public files (no existing tracked file modified):

- `stage8_uq/calibration.py`: field scores, quantiles, coverage, widths, grouped summaries.
- `stage8_uq/calibration_data.py`: new ID archive generation, roles, seed and input-identity checks.
- `stage8_uq/calibration_plots.py`: predeclared reliability, width, and field figures.
- `examples/calibrate_stage8c.py`: phased provenance-checked workflow and immutable factor freeze.
- `tests/test_stage8c_calibration.py`: calibration and role-separation tests.
- `docs/results/stage8c_freeze.json`: predeclared frozen protocol and provenance.
- `docs/results/stage8c_calibration_factors.csv`: 50 fixed horizon/level factors.
- `docs/results/stage8c_coverage.csv`: all 50 audit rows including intervals, physical/normalized widths, and secondary pointwise coverage.
- `docs/results/stage8c_summary.json`: identities, provenance, grouped results, examples, scalar diagnostics, source/output hashes.
- `docs/stage8c_calibration.md`: this report.

Full new archives are under ignored `data/stage8/`; member arrays, physical means/spreads, raw scores, band endpoints, field-level outcomes, bootstrap indices, plots, factor receipt, logs, and protection snapshots are under ignored `runs/stage8/calibration/`. The public summary links these artifacts by hashes.

## Tests, protected-file verification, and git status

Full regression: **222 passed in 4.83s** (all 198 Stage 1–8B tests plus 24 Stage 8C test cases). Coverage includes unused seeds, physical-input disjointness, frozen Stage 8B/ensemble provenance, normalization-refit prohibition, own-member feedback and fixed coefficients, t=0 exclusion, standardized residuals, floor accounting and raw-band mismatch, field maxima, higher-order quantiles and clipping, monotonic factors, audit-role rejection, fit-only calibration loading, frozen-factor tamper rejection, inclusive whole-field events, secondary pointwise fractions, widths, reliability sample counts, and deterministic whole-trajectory bootstrap.

Protection verification: **405/405 pre-Stage-8C tracked files and local scientific artifacts unchanged**; **329/329 historical protected files unchanged**, with the previously allowed Stage 8A package-discovery registration preserved. All five checkpoint hashes and Stage 8A/B source/manifest definitions passed verification. The frozen factors and every archived prediction/band/source hash match. New datasets and generated outputs remain ignored; neither data nor runs contains tracked files.

Git status: exactly the ten public additions listed above are untracked, with no existing tracked modifications or staged changes. HEAD remains `e31221b3a3baaaba3c84e487ef885faa9158b5d8`. Verification details are in ignored `runs/stage8/calibration/protected_verification.json`; the complete pytest output is `full_pytest.txt`.

Nothing was committed or pushed. The separate jax-differentiable-pde project was not touched. No Stage 8D work or extended-horizon data generation was started.

## NEXT STEPS

A. Calibration provides a scientifically informative independent field-coverage audit, but **coverage alone does not establish useful sharpness**. The late-horizon normalized full widths are several times the ensemble-mean RMS. Preserve this negative practical result rather than describing the procedure as a precise uncertainty estimate. Any application-specific utility claim requires tolerances and validation set in advance.

B. Freeze the exact horizon-specific max-standardized-residual score, epsilon, one-based higher order-statistic convention, raw-spread symmetric band endpoints, five nominal levels, all 50 factors and their receipt hashes, and coverage/width definitions as the Stage 8C baseline. Do not silently floor the band spread or alter factors after audit inspection.

C. Horizon-specific calibration is the predeclared primary method, but it was **not demonstrated to outperform a global method**, because the optional comparison was not performed. The observed q values alone cannot establish that superiority. No extra calibration variants were optimized.

D. The exact Stage 8D question is:

> Does uncertainty grow with autoregressive rollout degradation, and can it provide useful warning before large future rollout error?

E. Keep all Stage 8A members/checkpoints/manifests/seeds, splits, normalization, architecture and training provenance; Stage 8B raw-spread/U_rel/U_rms definitions and association rules; and Stage 8C archive identities, role separation, fixed factors, floor convention, band target, and untouched-audit results frozen. Preserve source and artifact hashes and the ten-step own-member feedback rule.

F. Existing t≤1 trajectories are sufficient for the first Stage 8D analysis of within-horizon difficulty, mean deterioration, and one-/two-step warning. Calibration data have already fitted q, and audit data now have coverage exposure: label their later reuse correctly and never treat them as a new pristine coverage audit after retuning.

G. Longer rollouts could later test warning limits or spread saturation, but these broad short-horizon bands do not by themselves establish useful warning or justify a large automatic extension. First assess warning beyond current error and the practical width tradeoff within the existing ten steps. A later extended experiment should have a separately reviewed question and numerically verified references; **do not generate 16/32/64/128-step data automatically**.

H. Do not claim OOD calibration or a Stage 8E result, retrain/replace members, introduce a Bayesian model, or calibrate on OOD data. Do not tune these factors on the audit set.

I. Preserve the roadmap: **Stage 8D rollout uncertainty → Stage 8E ID-vs-OOD failure awareness**. No roadmap change is justified by coverage achieved with these broad bands. Stage 8D has not started; wait for review.
