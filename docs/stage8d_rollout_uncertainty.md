# Stage 8D: rollout degradation and incremental warning

**Raw uncertainty grows with rollout degradation, but robust incremental warning
beyond current true prediction error is not established on the primary audit set.**
The strong raw association with future error mostly accompanies persistent
trajectory difficulty. This is not proof of zero incremental information: the
intervals allow modest effects, and the smaller secondary set has a positive
Pearson-only pattern that does not consistently replicate.

The scientific question is: **Does uncertainty grow with autoregressive rollout
degradation, and does it provide useful warning of future degradation beyond
simply reflecting the model’s current error?**

Primary audit partial Pearson correlations are 0.071–0.155, with all 17 pointwise
95% intervals including zero. Partial Spearman is 0.016–0.155; only the 0.2→0.4
comparison has a lower endpoint barely above zero (about 0.002), among many
dependent comparisons. Current-error/future-error Pearson correlations are
0.991–1.000. New difficult-tail entries number only 0–4 per comparison, below
the predeclared minimum of five events for discrimination estimates.

**Long-horizon decision: do not proceed to 16/32/64/128-step generation on this
evidence.** Raw U_rel retains cross-trajectory variation through t=1, but robust
conditional warning is unproven, new-entry evidence is too sparse, and the 90%
full bands already reach mean W_rel=4.678 at t=1. These mixed findings do not
justify extending horizons merely to search for a favorable signal.

## Frozen provenance and data roles

Frozen Stage 8C revision: `b84fdab4a601217cd17a7bdbeccf8b693bc99104`; Stage 8B revision: `e31221b3a3baaaba3c84e487ef885faa9158b5d8`; Stage 8A commit: `4967490593f1f5c125dc8e87abd2387144741954`. Training-time source revision remains `f095871d842e469f51d49751a7459684a507fd77`, supplemented by the original exact source-file hashes. The Stage 8D predeclared protocol SHA-256 is `c33674faab4894b0e80ce53e1026e3d339e6013b704ad680d3c1ac12fe8c1d3c`.

Ensemble manifest SHA-256: `0da21da72cb440d9bfc08563c1308a29d6dd6cc8f85785c5483d1ef6ec02d713`. All five selected member and manifest hashes, validation-selected checkpoint records, architecture, seeds, normalization, full budgets, and exact source identities passed the unchanged strict loaders.

| Member | Init/shuffle seeds | Checkpoint SHA-256 |
| --- | --- | --- |
| 0 | 2028 / 5029 | `cf8899453cadf2443961fd65daa736269ed671dd26a375fd3edc1159ae56ca33` |
| 1 | 8101 / 8201 | `591ec6819777909579bdc9c4f57ba62f25d91b709609c708c4afef30b78b1903` |
| 2 | 8102 / 8202 | `2f15a50f9a438a3db148d420eede5e15bf25f182ea3fd0682d3be50822253f79` |
| 3 | 8103 / 8203 | `ab95d65c8d340b550ee578c0bcf939df960e1d7a3f9780d2fc42f1434f72b5fd` |
| 4 | 8104 / 8204 | `ee49b65ca83df821c8703d97009ee82283d7bcb864218e5acac5a48925714a46` |

Architecture is unchanged: FNO2d modes=12, width=32, four blocks, projection width=64; four input channels, one output. Normalization hash: `e413b4e2a87fb0068bce67d3d81684b31d1d7d57b92b5ec96c50b9838c7e0120`. Training archive hash: `73974ce6d7be46209d22f09e8d9225dd97e59aaf324d55c5328e9f18b375638e`.

| Archive | Role | n | SHA-256 |
| --- | --- | --- | --- |
| data/stage8/audit.npz | primary retrospective | 256 | `959055fc071134c7bc67098d63303611cb5bd0c29f54b013b901d481f87bd210` |
| data/stage4/fresh_id.npz | secondary retrospective replication | 64 | `52292b4e91eb9aeb92277f429f0d126ba2a43e76f8f9cda6cd32c2347d12d9e2` |
| data/stage8/calibration.npz | identity checksum only; no warning analysis | 256 | `d1ee3147a68735b7406c9a8afe800cb2b137a942376cfd19f9014b443efa5205` |

Frozen factors SHA-256: `380812e0cf516745416238007450558d5a03099888fcf93657a9d1c8aae673cf`; public factor CSV SHA-256: `6ac5cd408b5066c3b606d8931f811180412791ad04703d1dd56afc7699e9bf73`. All horizon-specific factors and Stage 8C score/band conventions were verified against the frozen public summary and receipt. Nothing was refitted.

The audit archive was already used for Stage 8C coverage reporting. Fresh ID has Stage 4/8B performance exposure. Neither is a pristine prospective benchmark. Stage 8D reuses them for a predeclared retrospective question without fitting a warning model or threshold. Training, checkpoint-selection validation, historical eight-case test, and calibration trajectories are not analyzed. No OOD archive is loaded by the Stage 8D analysis; historical protected files receive checksum-only integrity verification.

Existing member predictions and spread fields remain in their original verified archives. Audit scalars/coverage events are read from the frozen Stage 8C outputs; fresh-ID scalars are read from Stage 8B outputs, and its 90% bands are evaluated by applying the unchanged Stage 8C q values. No new model inference or numerical trajectory generation is needed. Full provenance and source hashes are in [stage8d_freeze.json](results/stage8d_freeze.json) and [stage8d_summary.json](results/stage8d_summary.json).

## Definitions and resampling

All members autoregress their own prior prediction with fixed coefficients.
The ensemble mean is never fed back. Physical-unit reductions retain ddof=1
sample spread. The definitions remain:

```text
E(t) = ||ensemble_mean(t)-truth(t)||_2 / max(||truth(t)||_2, 1e-12)
U_rms(t) = RMS(sample spread(t))
U_rel(t) = U_rms(t) / max(RMS(ensemble_mean(t)), 1e-12*frozen_field_std)
90% band = ensemble_mean(t) ± q[t,90%] * raw sample spread(t)
W_rel90(t) = RMS(full band width) / max(RMS(ensemble_mean(t)), epsilon_rms)
           = 2*q[t,90%]*U_rel(t)
coverage90(t) = every stored grid cell lies in that current-horizon band
```

The Stage 8C residual score uses its frozen spread floor, whereas its band uses
raw spread. That convention is preserved; Stage 8C found no observed floor
activations on calibration/audit fields. No score denominator, band definition,
or calibration quantity is redesigned here.

All analysis arrays have ten horizons, t=0.1,…,1.0. Stored t=0 is removed before
association, growth, distribution, or failure-entry analysis. For each valid
source index h and lag k∈{1,2}, the target is h+k: 9 one-step and 8 two-step pairs.
There are no references beyond t=1.

The independent resampling unit is a complete trajectory row. For each dataset,
2,000 paired bootstrap row samples (seed 8401) are reused across every horizon,
lag, and statistic. Intervals are pointwise 95% percentile intervals. Undefined
bootstrap coefficients are omitted with their counts explicitly reported; this
is not a simultaneous or multiplicity-adjusted procedure. No grid cells or
repeated horizons are treated as independent replicates. No p-values are used.

Partial Pearson procedure: center U(t), E(t+k), and control E(t); project each
of the first two centered variables onto the centered control; subtract those
projections; Pearson-correlate the two residual vectors. This includes an
intercept and one linear control. Partial Spearman first assigns average ranks
to all three original variables, then applies the same projection and ordinary
Pearson correlation to rank residuals. It does not rank residuals again. Both
ranks and projections are recomputed inside each bootstrap sample. A constant
control or residual norm ≤1e-12 times its centered original norm is undefined.

These projections are algebraic descriptive statistics explicitly requested for
partial association, not a trained warning model, tuned risk threshold, or new
calibration. Controlling for true current error is an oracle scientific
comparison: E(t) generally requires a reference unavailable in deployment.
A weak partial result does not make raw uncertainty useless when true error is
unobserved. Neither raw nor partial association establishes causality.

## Mean degradation: descriptive trends, not early warning

| Time | Mean E | Median E | Mean U_rel | Median U_rel | Mean U_rms | Median U_rms | Mean W90 | Median W90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.01831 | 0.01757 | 0.01324 | 0.01307 | 0.00355 | 0.00327 | 0.68349 | 0.67473 |
| 0.2 | 0.03386 | 0.03194 | 0.02369 | 0.02347 | 0.00635 | 0.00569 | 1.26139 | 1.24987 |
| 0.3 | 0.04798 | 0.04428 | 0.03327 | 0.03250 | 0.00889 | 0.00795 | 1.67086 | 1.63196 |
| 0.4 | 0.06111 | 0.05625 | 0.04241 | 0.04131 | 0.01127 | 0.01000 | 2.07926 | 2.02552 |
| 0.5 | 0.07346 | 0.06730 | 0.05129 | 0.04936 | 0.01354 | 0.01206 | 2.75243 | 2.64923 |
| 0.6 | 0.08517 | 0.07798 | 0.05999 | 0.05733 | 0.01572 | 0.01395 | 3.21426 | 3.07169 |
| 0.7 | 0.09636 | 0.08772 | 0.06857 | 0.06610 | 0.01784 | 0.01572 | 3.27143 | 3.15347 |
| 0.8 | 0.10708 | 0.09816 | 0.07707 | 0.07428 | 0.01990 | 0.01756 | 3.69115 | 3.55755 |
| 0.9 | 0.11741 | 0.10737 | 0.08552 | 0.08263 | 0.02191 | 0.01924 | 4.07461 | 3.93693 |
| 1.0 | 0.12740 | 0.11660 | 0.09393 | 0.09126 | 0.02388 | 0.02086 | 4.67846 | 4.54517 |

All four primary means grow over the ten horizons. Mean error rises 0.01831→0.12740, mean U_rel 0.01324→0.09393, and mean W_rel90 0.683→4.678. Fresh-ID mean error rises 0.02094→0.14098, U_rel 0.01461→0.10097, and W_rel90 0.755→5.029. This is evidence of co-growing rollout deterioration and uncertainty, not incremental warning by itself. Both datasets’ complete mean/median tables are in [stage8d_by_horizon.csv](results/stage8d_by_horizon.csv). Figures: `runs/stage8/rollout_uncertainty/{audit,fresh_id}/plots/rollout_trends.png`.

## Current uncertainty and current error versus future error

Primary audit n=256 per row. Each cell gives coefficient [pointwise 95% trajectory-bootstrap interval].

| Source→target | U→future Pearson | U→future Spearman | E→future Pearson | E→future Spearman |
| --- | --- | --- | --- | --- |
| 0.1→0.2 | 0.800 [0.748, 0.849] | 0.846 [0.800, 0.880] | 0.997 [0.996, 0.998] | 0.997 [0.995, 0.998] |
| 0.2→0.3 | 0.780 [0.718, 0.836] | 0.835 [0.786, 0.873] | 0.998 [0.998, 0.999] | 0.998 [0.996, 0.998] |
| 0.3→0.4 | 0.766 [0.695, 0.829] | 0.824 [0.773, 0.865] | 0.999 [0.999, 0.999] | 0.999 [0.998, 0.999] |
| 0.4→0.5 | 0.757 [0.680, 0.824] | 0.818 [0.764, 0.861] | 0.999 [0.999, 1.000] | 0.999 [0.998, 0.999] |
| 0.5→0.6 | 0.751 [0.670, 0.821] | 0.811 [0.758, 0.855] | 0.999 [0.999, 1.000] | 0.999 [0.998, 0.999] |
| 0.6→0.7 | 0.747 [0.663, 0.818] | 0.806 [0.750, 0.852] | 1.000 [0.999, 1.000] | 0.999 [0.999, 0.999] |
| 0.7→0.8 | 0.744 [0.660, 0.817] | 0.803 [0.746, 0.849] | 1.000 [0.999, 1.000] | 0.999 [0.999, 0.999] |
| 0.8→0.9 | 0.743 [0.658, 0.817] | 0.799 [0.741, 0.847] | 1.000 [0.999, 1.000] | 0.999 [0.999, 0.999] |
| 0.9→1.0 | 0.742 [0.656, 0.816] | 0.797 [0.739, 0.846] | 1.000 [0.999, 1.000] | 0.999 [0.999, 1.000] |
| 0.1→0.3 | 0.801 [0.749, 0.848] | 0.846 [0.800, 0.881] | 0.991 [0.989, 0.994] | 0.991 [0.986, 0.993] |
| 0.2→0.4 | 0.781 [0.719, 0.837] | 0.835 [0.787, 0.874] | 0.995 [0.994, 0.997] | 0.994 [0.991, 0.996] |
| 0.3→0.5 | 0.768 [0.697, 0.829] | 0.825 [0.774, 0.866] | 0.997 [0.996, 0.998] | 0.996 [0.993, 0.997] |
| 0.4→0.6 | 0.758 [0.681, 0.825] | 0.818 [0.766, 0.860] | 0.997 [0.997, 0.998] | 0.997 [0.995, 0.998] |
| 0.5→0.7 | 0.752 [0.671, 0.821] | 0.811 [0.758, 0.856] | 0.998 [0.997, 0.999] | 0.998 [0.996, 0.998] |
| 0.6→0.8 | 0.747 [0.666, 0.819] | 0.806 [0.750, 0.851] | 0.998 [0.998, 0.999] | 0.998 [0.996, 0.998] |
| 0.7→0.9 | 0.745 [0.661, 0.818] | 0.803 [0.747, 0.850] | 0.998 [0.998, 0.999] | 0.998 [0.996, 0.998] |
| 0.8→1.0 | 0.743 [0.659, 0.816] | 0.799 [0.742, 0.846] | 0.998 [0.998, 0.999] | 0.998 [0.996, 0.998] |

Raw uncertainty predicts which trajectories will have larger future error in this descriptive sense. However, current error is even more strongly associated with future error; difficulty is highly persistent. The table alone therefore cannot establish additional warning beyond current difficulty.

## Partial/incremental association controlling current error

| Source→target | Audit partial Pearson | Audit partial Spearman | Fresh-ID partial Pearson | Fresh-ID partial Spearman |
| --- | --- | --- | --- | --- |
| 0.1→0.2 | 0.149 [-0.038, 0.302] | 0.054 [-0.108, 0.219] | 0.370 [-0.003, 0.577] | 0.225 [-0.169, 0.501] |
| 0.2→0.3 | 0.155 [-0.036, 0.311] | 0.153 [-0.005, 0.286] | 0.395 [0.077, 0.591] | 0.136 [-0.181, 0.401] |
| 0.3→0.4 | 0.128 [-0.069, 0.295] | 0.106 [-0.047, 0.243] | 0.394 [0.099, 0.580] | 0.253 [-0.027, 0.453] |
| 0.4→0.5 | 0.105 [-0.090, 0.281] | 0.091 [-0.064, 0.232] | 0.386 [0.107, 0.572] | 0.234 [-0.075, 0.457] |
| 0.5→0.6 | 0.090 [-0.100, 0.265] | 0.052 [-0.088, 0.184] | 0.372 [0.100, 0.562] | -0.099 [-0.285, 0.172] |
| 0.6→0.7 | 0.082 [-0.104, 0.252] | 0.050 [-0.096, 0.188] | 0.356 [0.086, 0.546] | 0.109 [-0.173, 0.335] |
| 0.7→0.8 | 0.077 [-0.104, 0.240] | 0.024 [-0.108, 0.161] | 0.338 [0.072, 0.533] | 0.090 [-0.242, 0.333] |
| 0.8→0.9 | 0.074 [-0.101, 0.233] | 0.076 [-0.085, 0.233] | 0.320 [0.053, 0.522] | 0.172 [-0.219, 0.432] |
| 0.9→1.0 | 0.071 [-0.095, 0.226] | 0.016 [-0.133, 0.160] | 0.300 [0.026, 0.512] | 0.165 [-0.242, 0.439] |
| 0.1→0.3 | 0.154 [-0.036, 0.304] | 0.109 [-0.050, 0.263] | 0.386 [0.026, 0.590] | 0.194 [-0.169, 0.500] |
| 0.2→0.4 | 0.149 [-0.045, 0.309] | 0.155 [0.002, 0.293] | 0.400 [0.084, 0.596] | 0.223 [-0.062, 0.475] |
| 0.3→0.5 | 0.125 [-0.071, 0.296] | 0.117 [-0.038, 0.253] | 0.398 [0.103, 0.587] | 0.286 [0.039, 0.473] |
| 0.4→0.6 | 0.105 [-0.089, 0.280] | 0.092 [-0.058, 0.225] | 0.389 [0.112, 0.579] | 0.092 [-0.192, 0.372] |
| 0.5→0.7 | 0.093 [-0.095, 0.266] | 0.066 [-0.075, 0.198] | 0.375 [0.101, 0.569] | 0.021 [-0.203, 0.271] |
| 0.6→0.8 | 0.086 [-0.099, 0.254] | 0.049 [-0.086, 0.187] | 0.360 [0.090, 0.552] | 0.119 [-0.168, 0.360] |
| 0.7→0.9 | 0.083 [-0.095, 0.244] | 0.063 [-0.076, 0.207] | 0.342 [0.073, 0.539] | 0.168 [-0.171, 0.439] |
| 0.8→1.0 | 0.080 [-0.091, 0.238] | 0.061 [-0.088, 0.208] | 0.323 [0.050, 0.530] | 0.220 [-0.188, 0.501] |

No primary partial Pearson interval excludes zero. Only one primary partial-rank interval barely excludes zero, at 0.2→0.4; with dependent horizons and two lags this is not a robust pattern of incremental warning. At the latest one-step pair 0.9→1.0, partial Pearson is 0.071 [-0.095,0.226], and partial Spearman is 0.016 [-0.133,0.160]. These intervals permit modest effects and do not prove the true conditional association is exactly zero.

The secondary fresh-ID set has partial Pearson 0.300–0.400, with 16/17 intervals positive. That favorable Pearson result must not be hidden, but it does not replicate on the larger primary set. Fresh-ID partial rank correlations range −0.099 to 0.286; only 0.3→0.5 has a positive interval. Near-perfect current/future rank agreement leaves small residuals and makes conditional rank estimates sensitive to small rank changes. Audit partial bootstraps are all defined. Fresh-ID partial Spearman has one undefined draw out of 2,000 at each of 0.5→0.6, 0.7→0.8, and 0.9→1.0; these counts remain in the public outputs.

## Current uncertainty versus future error increment

ΔE_k(t)=E(t+k)−E(t). This outcome measures error change rather than future level; it is not conditioned on E(t).

| Source→target | Audit Pearson [CI] | Audit Spearman [CI] | Mean audit ΔE | Fresh Pearson [CI] | Fresh Spearman [CI] |
| --- | --- | --- | --- | --- | --- |
| 0.1→0.2 | 0.799 [0.748, 0.845] | 0.844 [0.796, 0.879] | 0.01555 | 0.807 [0.727, 0.878] | 0.854 [0.753, 0.911] |
| 0.2→0.3 | 0.780 [0.718, 0.834] | 0.832 [0.782, 0.871] | 0.01412 | 0.781 [0.680, 0.870] | 0.838 [0.720, 0.904] |
| 0.3→0.4 | 0.765 [0.691, 0.825] | 0.817 [0.765, 0.860] | 0.01312 | 0.768 [0.654, 0.866] | 0.819 [0.688, 0.898] |
| 0.4→0.5 | 0.752 [0.673, 0.817] | 0.806 [0.752, 0.850] | 0.01235 | 0.761 [0.639, 0.860] | 0.811 [0.682, 0.889] |
| 0.5→0.6 | 0.742 [0.660, 0.809] | 0.795 [0.738, 0.840] | 0.01172 | 0.755 [0.631, 0.855] | 0.800 [0.670, 0.881] |
| 0.6→0.7 | 0.733 [0.649, 0.803] | 0.784 [0.726, 0.832] | 0.01118 | 0.750 [0.625, 0.850] | 0.784 [0.654, 0.868] |
| 0.7→0.8 | 0.726 [0.641, 0.797] | 0.777 [0.717, 0.827] | 0.01073 | 0.743 [0.621, 0.843] | 0.769 [0.628, 0.862] |
| 0.8→0.9 | 0.719 [0.635, 0.790] | 0.766 [0.706, 0.817] | 0.01033 | 0.735 [0.616, 0.835] | 0.759 [0.611, 0.862] |
| 0.9→1.0 | 0.712 [0.630, 0.783] | 0.757 [0.695, 0.809] | 0.00999 | 0.727 [0.603, 0.829] | 0.736 [0.572, 0.845] |
| 0.1→0.3 | 0.797 [0.745, 0.843] | 0.841 [0.793, 0.878] | 0.02968 | 0.812 [0.729, 0.883] | 0.856 [0.756, 0.913] |
| 0.2→0.4 | 0.778 [0.714, 0.832] | 0.829 [0.779, 0.868] | 0.02725 | 0.785 [0.682, 0.871] | 0.835 [0.715, 0.902] |
| 0.3→0.5 | 0.762 [0.688, 0.822] | 0.814 [0.760, 0.857] | 0.02547 | 0.770 [0.659, 0.865] | 0.814 [0.686, 0.892] |
| 0.4→0.6 | 0.749 [0.669, 0.813] | 0.803 [0.749, 0.848] | 0.02407 | 0.761 [0.642, 0.859] | 0.808 [0.680, 0.886] |
| 0.5→0.7 | 0.738 [0.655, 0.806] | 0.791 [0.733, 0.838] | 0.02290 | 0.754 [0.632, 0.853] | 0.795 [0.666, 0.879] |
| 0.6→0.8 | 0.730 [0.645, 0.799] | 0.781 [0.723, 0.829] | 0.02191 | 0.747 [0.627, 0.846] | 0.778 [0.646, 0.864] |
| 0.7→0.9 | 0.722 [0.638, 0.794] | 0.772 [0.712, 0.822] | 0.02106 | 0.739 [0.618, 0.838] | 0.765 [0.623, 0.862] |
| 0.8→1.0 | 0.715 [0.632, 0.786] | 0.761 [0.700, 0.812] | 0.02032 | 0.730 [0.610, 0.832] | 0.748 [0.589, 0.855] |

These unconditional increment associations are positive: primary Pearson 0.712–0.799 and Spearman 0.757–0.844. Difficult trajectories can have both larger current error and larger future increments. Subtracting current error from the outcome does not by itself remove that confounding. Under the same linear projection, residualizing ΔE_k against E(t) yields exactly the same residual as residualizing E(t+k), so the conditional result is still the partial-association table above.

## New-failure entry: insufficient event counts

At each horizon, high error is the top ceil(n/5) stable ranks of E: 52/256 audit or 13/64 fresh-ID trajectories. Exact ties prefer later positions in the frozen trajectory order, preserving Stage 8B. A new entry is not-high at source AND high at target. The detector considers only source-not-high trajectories, then selects the top ceil(n_eligible/5) U_rel scores among them: 41/204 audit, 11/51 fresh ID. Already-high cases are excluded completely. These are relative tail entries, not crossings of an absolute application-error tolerance.

The protocol fixed a minimum of five events and five non-events before inspecting outcomes. Below that minimum AUROC, precision, and recall are null/unstable, even though raw selected counts, hits, and missed IDs are retained for transparency. This avoids turning one or two threshold crossings into a discrimination claim.

| Source→target | Audit events / non-events | Audit selected / hits | Fresh events / non-events | Fresh selected / hits | AUROC / precision / recall |
| --- | --- | --- | --- | --- | --- |
| 0.1→0.2 | 2 / 202 | 41 / 1 | 0 / 51 | 11 / 0 | undefined: rare events |
| 0.2→0.3 | 1 / 203 | 41 / 1 | 0 / 51 | 11 / 0 | undefined: rare events |
| 0.3→0.4 | 3 / 201 | 41 / 3 | 0 / 51 | 11 / 0 | undefined: rare events |
| 0.4→0.5 | 2 / 202 | 41 / 1 | 0 / 51 | 11 / 0 | undefined: rare events |
| 0.5→0.6 | 1 / 203 | 41 / 0 | 0 / 51 | 11 / 0 | undefined: rare events |
| 0.6→0.7 | 1 / 203 | 41 / 1 | 0 / 51 | 11 / 0 | undefined: rare events |
| 0.7→0.8 | 1 / 203 | 41 / 1 | 0 / 51 | 11 / 0 | undefined: rare events |
| 0.8→0.9 | 1 / 203 | 41 / 0 | 0 / 51 | 11 / 0 | undefined: rare events |
| 0.9→1.0 | 0 / 204 | 41 / 0 | 1 / 50 | 11 / 1 | undefined: rare events |
| 0.1→0.3 | 3 / 201 | 41 / 2 | 0 / 51 | 11 / 0 | undefined: rare events |
| 0.2→0.4 | 3 / 201 | 41 / 3 | 0 / 51 | 11 / 0 | undefined: rare events |
| 0.3→0.5 | 4 / 200 | 41 / 3 | 0 / 51 | 11 / 0 | undefined: rare events |
| 0.4→0.6 | 2 / 202 | 41 / 0 | 0 / 51 | 11 / 0 | undefined: rare events |
| 0.5→0.7 | 1 / 203 | 41 / 0 | 0 / 51 | 11 / 0 | undefined: rare events |
| 0.6→0.8 | 2 / 202 | 41 / 2 | 0 / 51 | 11 / 0 | undefined: rare events |
| 0.7→0.9 | 2 / 202 | 41 / 1 | 0 / 51 | 11 / 0 | undefined: rare events |
| 0.8→1.0 | 1 / 203 | 41 / 0 | 1 / 50 | 11 / 1 | undefined: rare events |

Every comparison is below the event threshold: audit 0–4 and fresh ID 0–1 entrants. This is evidence of stable tail membership and insufficient event information, not proof of either good or bad new-failure AUROC. Repeated appearances of the same trajectory across horizons are not independent events. Full entry and missed-ID lists are in [stage8d_new_failures.csv](results/stage8d_new_failures.csv).

## Secondary uncertainty-growth/error-growth association

ΔU(t)=U_rel(t+1)−U_rel(t) and ΔE(t)=E(t+1)−E(t) use the same nine intervals.

| Source→target | Audit Pearson [CI] | Audit Spearman [CI] | Fresh Pearson [CI] | Fresh Spearman [CI] |
| --- | --- | --- | --- | --- |
| 0.1→0.2 | 0.745 [0.666, 0.814] | 0.808 [0.754, 0.854] | 0.685 [0.545, 0.821] | 0.793 [0.647, 0.881] |
| 0.2→0.3 | 0.728 [0.639, 0.805] | 0.789 [0.729, 0.839] | 0.674 [0.511, 0.821] | 0.761 [0.594, 0.863] |
| 0.3→0.4 | 0.719 [0.625, 0.799] | 0.776 [0.710, 0.830] | 0.684 [0.508, 0.829] | 0.738 [0.560, 0.853] |
| 0.4→0.5 | 0.716 [0.622, 0.797] | 0.771 [0.705, 0.825] | 0.700 [0.521, 0.840] | 0.748 [0.573, 0.858] |
| 0.5→0.6 | 0.716 [0.622, 0.796] | 0.767 [0.703, 0.822] | 0.716 [0.539, 0.847] | 0.754 [0.579, 0.864] |
| 0.6→0.7 | 0.716 [0.623, 0.796] | 0.764 [0.700, 0.819] | 0.728 [0.556, 0.850] | 0.741 [0.559, 0.852] |
| 0.7→0.8 | 0.717 [0.625, 0.796] | 0.764 [0.701, 0.818] | 0.734 [0.565, 0.850] | 0.725 [0.546, 0.848] |
| 0.8→0.9 | 0.716 [0.628, 0.794] | 0.764 [0.701, 0.815] | 0.734 [0.565, 0.851] | 0.717 [0.537, 0.847] |
| 0.9→1.0 | 0.714 [0.627, 0.790] | 0.762 [0.700, 0.812] | 0.725 [0.556, 0.847] | 0.688 [0.490, 0.825] |

| Pooled secondary dataset | Pearson [CI] | Spearman [CI] |
| --- | --- | --- |
| audit | 0.723 [0.643, 0.795] | 0.781 [0.727, 0.828] |
| fresh ID | 0.698 [0.546, 0.822] | 0.752 [0.608, 0.852] |

The pooled estimates use whole-trajectory bootstrap rows, retaining all nine increments. Source horizon remains a potential confounder and the pooled calculation is secondary, not 2,304 or 576 independent samples. Same-interval co-growth is not advance warning and is not conditioned on current error.

## Secondary calibrated-band warning diagnostics

At a fixed horizon, W_rel90=2*q90*U_rel exactly, with the same positive q for every trajectory. Consequently its future-error Pearson and Spearman coefficients equal those of U_rel (up to roundoff); calibrated width cannot add a distinct ranking signal. The code independently checks this relationship and retains width/future coefficients and intervals in [stage8d_early_warning.csv](results/stage8d_early_warning.csv). Comparing different horizons is different because q varies with time.

For current whole-field noncoverage, the following descriptive comparison uses binary score 1 for noncoverage and 0 for coverage. Pearson is the point-biserial association; Spearman uses average tied ranks. Neither score is a probability of future failure.

| Source→target | Source noncovered n | Future mean E if noncovered | Future mean E if covered | Noncoverage→future Pearson [CI] | Spearman [CI] |
| --- | --- | --- | --- | --- | --- |
| 0.1→0.2 | 24 | 0.03490 | 0.03375 | 0.019 [-0.070, 0.117] | 0.043 [-0.057, 0.144] |
| 0.2→0.3 | 18 | 0.05272 | 0.04762 | 0.051 [-0.068, 0.179] | 0.042 [-0.078, 0.164] |
| 0.3→0.4 | 25 | 0.06780 | 0.06038 | 0.068 [-0.036, 0.183] | 0.081 [-0.021, 0.178] |
| 0.4→0.5 | 31 | 0.08556 | 0.07179 | 0.116 [0.009, 0.230] | 0.142 [0.033, 0.243] |
| 0.5→0.6 | 22 | 0.10493 | 0.08331 | 0.134 [0.026, 0.253] | 0.144 [0.037, 0.243] |
| 0.6→0.7 | 20 | 0.12158 | 0.09422 | 0.144 [0.028, 0.269] | 0.147 [0.038, 0.252] |
| 0.7→0.8 | 26 | 0.13170 | 0.10430 | 0.147 [0.029, 0.264] | 0.155 [0.047, 0.254] |
| 0.8→0.9 | 26 | 0.14790 | 0.11397 | 0.166 [0.053, 0.279] | 0.188 [0.089, 0.281] |
| 0.9→1.0 | 22 | 0.16875 | 0.12352 | 0.190 [0.073, 0.314] | 0.196 [0.093, 0.291] |
| 0.1→0.3 | 24 | 0.04936 | 0.04784 | 0.017 [-0.072, 0.117] | 0.040 [-0.061, 0.140] |
| 0.2→0.4 | 18 | 0.06725 | 0.06064 | 0.052 [-0.070, 0.180] | 0.043 [-0.076, 0.163] |
| 0.3→0.5 | 25 | 0.08083 | 0.07266 | 0.062 [-0.043, 0.177] | 0.071 [-0.031, 0.171] |
| 0.4→0.6 | 31 | 0.09862 | 0.08332 | 0.111 [0.003, 0.224] | 0.134 [0.026, 0.236] |
| 0.5→0.7 | 22 | 0.11827 | 0.09430 | 0.132 [0.023, 0.252] | 0.139 [0.032, 0.239] |
| 0.6→0.8 | 20 | 0.13512 | 0.10470 | 0.145 [0.028, 0.271] | 0.142 [0.031, 0.248] |
| 0.7→0.9 | 26 | 0.14412 | 0.11439 | 0.146 [0.027, 0.263] | 0.153 [0.044, 0.253] |
| 0.8→1.0 | 26 | 0.15990 | 0.12373 | 0.164 [0.048, 0.277] | 0.182 [0.083, 0.276] |

Primary noncoverage/future Pearson associations are only 0.017–0.190 and rank associations 0.040–0.196; effects are weak and these secondary comparisons are not controlled for current error. Noncoverage also requires knowing the true current field, so it is a retrospective diagnostic rather than an observable online warning without reference data. No current-horizon coverage guarantee is reinterpreted as a guarantee about future error. Secondary fresh-ID results are fully retained in the CSV/JSON.

## Saturation and discrimination spread

| Time | U q10 | U median | U q90 | U q90−q10 | U gap/median | W90 q10 | W90 median | W90 q90 | W90 q90−q10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.00611 | 0.01307 | 0.01988 | 0.01378 | 1.05439 | 0.31525 | 0.67473 | 1.02667 | 0.71142 |
| 0.2 | 0.01092 | 0.02347 | 0.03578 | 0.02485 | 1.05886 | 0.58162 | 1.24987 | 1.90506 | 1.32344 |
| 0.3 | 0.01478 | 0.03250 | 0.05123 | 0.03644 | 1.12143 | 0.74242 | 1.63196 | 2.57256 | 1.83013 |
| 0.4 | 0.01861 | 0.04131 | 0.06635 | 0.04773 | 1.15539 | 0.91253 | 2.02552 | 3.25280 | 2.34026 |
| 0.5 | 0.02253 | 0.04936 | 0.08092 | 0.05839 | 1.18288 | 1.20906 | 2.64923 | 4.34279 | 3.13374 |
| 0.6 | 0.02643 | 0.05733 | 0.09417 | 0.06775 | 1.18171 | 1.41588 | 3.07169 | 5.04574 | 3.62986 |
| 0.7 | 0.03032 | 0.06610 | 0.10697 | 0.07665 | 1.15964 | 1.44651 | 3.15347 | 5.10340 | 3.65689 |
| 0.8 | 0.03423 | 0.07428 | 0.12018 | 0.08595 | 1.15712 | 1.63912 | 3.55755 | 5.75562 | 4.11650 |
| 0.9 | 0.03826 | 0.08263 | 0.13395 | 0.09569 | 1.15812 | 1.82284 | 3.93693 | 6.38227 | 4.55943 |
| 1.0 | 0.04230 | 0.09126 | 0.14659 | 0.10429 | 1.14281 | 2.10683 | 4.54517 | 7.30110 | 5.19426 |

Audit U_rel interdecile spread increases 0.01378→0.10429; its gap/median ratio stays about 1.05–1.18 and ends at 1.143. It does not collapse into a common value or visibly plateau through t=1. Fresh-ID spread increases 0.01458→0.11336 and normalized spread 0.974→1.177. W_rel’s normalized interdecile spread is algebraically the same as U_rel’s at fixed horizon. These describe score variation, not discrimination by themselves. The fixed-horizon U_rel/error rank correlation also remains substantial: audit Spearman 0.846 at t=0.1 and 0.794 at t=1.0; fresh ID 0.846 and 0.780. Complete current-error associations and intervals are retained.

Thus loss of cross-trajectory variation is not the reason to reject strong warning claims here. The limiting evidence is small conditional signal, rare new entries, and very broad bands. Audit final W_rel90 has q10=2.107, median=4.545, q90=7.301: substantial separation remains, but the full widths are large relative to predicted signal RMS. Distribution figures are `runs/stage8/rollout_uncertainty/{audit,fresh_id}/plots/saturation.png`. No behavior beyond t=1 is inferred.

## Predeclared primary examples

| Selection rule | Audit ID | Final E | Final U_rel | Final W90 | E(1.0)−E(0.5) |
| --- | --- | --- | --- | --- | --- |
| predetermined | 0 | 0.03301 | 0.04188 | 2.086 | 0.01586 |
| largest_final_error | 172 | 0.41290 | 0.08408 | 4.187 | 0.17970 |
| largest_final_U_rel | 158 | 0.36720 | 0.23006 | 11.459 | 0.14572 |
| largest_error_increase_0.5_to_1.0 | 172 | 0.41290 | 0.08408 | 4.187 | 0.17970 |
| first_missed_new_failure | 15 | 0.21807 | 0.11370 | 5.663 | 0.10026 |

The largest-final-error and largest-late-increase rules both select ID 172; that duplication is retained rather than replaced with a more favorable example. Its U_rel remains modest despite the highest final error. The missed-entry rule selects lexicographically by lag (1 then 2), source time, and archive-local ID among new entrants not selected by eligible top-fifth uncertainty. It chooses ID 15 at 0.1→0.2: E rises 0.02538→0.04965 while source U_rel=0.01387 and W_rel90=0.716. It crosses into the relative high-error tail but is missed by the frozen selection. There are only two events in this comparison, so it is an illustration, not a stable detector-performance estimate.

Each selected trajectory has a three-panel E(t), U_rel(t), and 90% W_rel(t) plot with separate vertical scales, so wide bands do not visually hide the smaller errors. The missed interval is highlighted. All five plots are retained under `runs/stage8/rollout_uncertainty/audit/plots/`; no examples were chosen solely because uncertainty looks successful.

## Limitations and explicit long-horizon decision

The decision is **no-go for an extended-horizon generation campaign at this stage**.
Evidence in favor of eventual further study exists: raw spread retains current-error
ranking and inter-trajectory variation through t=1, and the secondary set has
moderate conditional Pearson association. Against proceeding now, the primary
partial signal is small and uncertain, conditional ranks are weak, new entries
are too rare for stable metrics, and late bands already span several times the
predicted signal RMS. The ten-step horizon suffices to expose these limitations;
its inability to reveal ultimate breakdown is not by itself a scientific warrant
for generating 16/32/64/128-step references. This decision is not proof that no
longer-horizon signal could ever exist.

Other limitations:

- Both datasets are historically exposed and the estimates are retrospective. The secondary set is smaller; a favorable result there does not override the primary result.
- Shared member architecture/training/normalization induce correlated errors. Five members do not represent a Bayesian posterior or independent intrinsic PDE noise.
- Partial Pearson removes a linear current-error component; partial Spearman removes a linear component of ranks. Neither removes every nonlinear confounder or proves causal warning.
- True E(t) and current noncoverage require reference truth. They are oracle explanatory controls/diagnostics, not deployed warning inputs. No predictive model, out-of-sample incremental forecast test, or warning threshold was trained.
- All 17 source/lag comparisons are dependent; pointwise interval exclusions are not independent confirmations or adjusted hypothesis tests. No p-values or repeated-grid-cell tests are reported.
- Near-unit current/future associations leave little residual variation; conditional estimates are sensitive to small changes. A null-compatible interval cannot exclude every useful effect.
- Relative top-fifth entry can reflect small rank swaps; it is not a fixed absolute error failure. Sparse events prevent stable new-failure metrics under the predeclared rule.
- Saturation quantiles alone measure spread, not utility. Calibrated width preserves fixed-horizon ranking and can be large without being informative enough for an application.
- Bands retain the Stage 8C current-horizon, stored-grid interpretation. No joint-time, continuous-field, future-failure, OOD, or posterior-probability claim follows.
- Physical errors are relative to existing numerical references; no new solver validation or longer references were generated in Stage 8D.

## Reproduction and files

From the repository root, with existing ignored Stage 8A/B/C artifacts:

```bash
MPLCONFIGDIR=/tmp/stage8b-matplotlib .venv/bin/python -m examples.analyze_stage8d
MPLCONFIGDIR=/tmp/stage8b-matplotlib .venv/bin/python -m pytest -q
```

The analysis refuses provenance mismatches and existing output destinations; it
does not regenerate missing frozen inputs. New scalar arrays and bootstrap row
indices are under ignored `runs/stage8/rollout_uncertainty/{audit,fresh_id}/`.
Original member predictions/spreads remain at their verified Stage 8B/C paths.
All figures and verification logs remain ignored.

New public files:

- `stage8_uq/rollout_warning.py`: partial, increment, failure-entry, growth, and distribution statistics.
- `stage8_uq/rollout_warning_io.py`: frozen provenance and strict audit/fresh-ID-only loading.
- `stage8_uq/rollout_warning_plots.py`: fixed trend, distribution, and example plots.
- `examples/analyze_stage8d.py`: read-only-input analysis runner and exports.
- `tests/test_stage8d_rollout_warning.py`: statistical and provenance/role tests.
- `docs/results/stage8d_freeze.json`: predeclared definitions and frozen input hashes.
- `docs/results/stage8d_by_horizon.csv`: complete trends, quantiles, coverage, and current-error association for both sets.
- `docs/results/stage8d_early_warning.csv`: all future, partial, increment, width, and noncoverage coefficients/intervals.
- `docs/results/stage8d_new_failures.csv`: every eligible/event/non-event/selected/hit count, status, and case IDs.
- `docs/results/stage8d_summary.json`: complete numerical results, growth summaries, example IDs, provenance, and source/output hashes.
- `docs/stage8d_rollout_uncertainty.md`: this report.

## Tests, protection, and git status

Full regression: **242 passed in 7.35s** (all 222 Stage 1–8C tests plus 20 Stage 8D cases). Tests cover frozen Stage 8A/B/C provenance and factor tampering, exact source/target extraction, positive/negative error and uncertainty increments, partial Pearson against independent least squares, rank-before-projection with ties, degenerate residuals, shared-confounder removal, deterministic bootstrap reprojection, eligible-only new-failure definitions, rare-event handling, distribution quantiles, time-zero exclusion, fixed-width/ranking equivalence, rejected forbidden archive roles, no fitting/generation calls, strict existing-output loading, and fixed example selection.

Protected verification: **469/469 pre-Stage-8D tracked files and local scientific artifacts unchanged**, plus **329/329 historical protected files unchanged**, with only the already allowed Stage 8A package registration relative to the historical baseline. Frozen factors, Stage 8D source hashes, and derived-array hashes match. The dataset file inventory is unchanged: **zero new data files**, so no extended-horizon references were generated. All new run arrays, bootstrap indices, plots, and logs remain ignored; no data/run files are tracked.

Git status: exactly the eleven public additions listed above are untracked; no pre-existing tracked modifications or staged changes. HEAD remains `b84fdab4a601217cd17a7bdbeccf8b693bc99104`. Verification details and full pytest output are saved under ignored `runs/stage8/rollout_uncertainty/`.

No existing tracked file was modified. No commit or push was made. The separate
jax-differentiable-pde project was not modified. No Stage 8E analysis was started.

## NEXT STEPS

A. **Robust useful incremental warning beyond true current error is not established
on the primary audit archive.** Raw spread remains useful for current difficulty
and associated with future levels/increments, but these observations largely
track persistent difficulty. The favorable secondary partial-Pearson result is
retained as inconsistent evidence, not promoted over the primary result. This
does not establish a zero effect or make U_rel useless when truth is unavailable.

B. **Reject proceeding to extended-horizon generation now.** Persistent raw
trajectory separation supports continuing to study the score, but it does not
overcome weak primary conditional evidence, rare new entries, or broad bands.
Do not search longer horizons merely to obtain a more favorable conclusion.

C. Since extended horizons are not justified at this gate, no 16/32/64/128-step
numerical-reference design is activated or generated. Any later reversal needs
a separately reviewed scientific objective and numerical-verification plan;
no automatic generation is scheduled.

D. Keep all Stage 8A members, checkpoints, manifests, seeds, architecture,
normalization, training/split provenance and own-member feedback; Stage 8B
physical E/U_rms/U_rel and ddof=1 definitions; Stage 8C archive identities,
roles, q factors, floors and band/coverage conventions; and Stage 8D source/hash
identities, ten horizons, lag alignment, partial-rank projection, bootstrap seed
and replicate count, top-fifth/tie/eligibility rules, rare-event minimum,
saturation summaries, and example-selection rules frozen.

E. The exact Stage 8E question is:

> When the model enters an unseen physical regime, does its uncertainty increase in a way that reflects its increased prediction error?

F. After separate approval, use the existing Stage 4 high-diffusivity archive,
the high-velocity archives for all four sign quadrants (pp, pn, np, nn), and the
original historical test/fresh-ID references. Preserve original splits and
report all four quadrants and the aggregate, including unfavorable outcomes.
Those OOD archives were not loaded for Stage 8D analysis.

G. Do not fit calibration on OOD, retrain or replace ensemble members, tune
U_rel to OOD, introduce a Bayesian extension, or assume ID calibration transfers
to OOD without testing. No current band coverage should be relabeled as a
future-failure probability.

H. Preserve **Stage 8E as a distinct OOD failure-awareness study**. The present
negative/limited incremental-warning result does not answer the different
ID-vs-OOD question. Do not start Stage 8E automatically; wait for review.
