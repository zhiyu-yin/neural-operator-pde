# Stage 8E: OOD failure awareness

The frozen ensemble shows useful **regime-level** failure awareness on high velocity, but does not reliably rank individual failures in every quadrant. At t=1, aggregate high-velocity error is 2.81 times fresh ID and U_rel is 1.85 times fresh ID. High diffusivity produces little clear additional error but 1.87 times the uncertainty: conservative shift sensitivity. The evidence supports a mixture of distribution sensitivity and prediction-difficulty awareness, not an error certificate.

## Frozen provenance and archive roles

This is a retrospective study using only existing Stage 4 evaluation archives. The protocol, bootstrap, rank categories and example selection were frozen in [stage8e_freeze.json](results/stage8e_freeze.json) before examining OOD outputs. The committed Stage 8D base is `426331c6c79dc3b05cd4693c684888ad255a9e32`; Stage 8C is `b84fdab4a601217cd17a7bdbeccf8b693bc99104`; Stage 8B is `e31221b3a3baaaba3c84e487ef885faa9158b5d8`. The freeze records every protected tracked-file hash; the ignored pre-study snapshot additionally covers local scientific artifacts.

All five Stage 8A checkpoints, ensemble/member manifests, normalization, architecture and seed metadata passed the inherited Stage 8A–D verifiers. Architecture remains FNO, modes 12, width 32, four blocks, projection width 64. Seed pairs remain (2028,5029), (8101,8201), (8102,8202), (8103,8203), (8104,8204). Physical normalization mean/std remain 0.12027802015661833 / 0.2355322120279037.


| Member | SHA256 |
|---|---|
| 0 | cf8899453cadf2443961fd65daa736269ed671dd26a375fd3edc1159ae56ca33 |
| 1 | 591ec6819777909579bdc9c4f57ba62f25d91b709609c708c4afef30b78b1903 |
| 2 | 2f15a50f9a438a3db148d420eede5e15bf25f182ea3fd0682d3be50822253f79 |
| 3 | ab95d65c8d340b550ee578c0bcf939df960e1d7a3f9780d2fc42f1434f72b5fd |
| 4 | ee49b65ca83df821c8703d97009ee82283d7bcb864218e5acac5a48925714a46 |

The frozen calibration-factor SHA256 is `380812e0cf516745416238007450558d5a03099888fcf93657a9d1c8aae673cf`. All horizon/level factors are reused exactly. Stage 8D source, public summaries, diagnostic definitions and local analysis-array hashes remain unchanged. No models, normalization, warning definitions or calibration factors were fitted or tuned.

| Role | Archive | SHA256 |
|---|---|---|
| fresh_id | data/stage4/fresh_id.npz | 52292b4e91eb9aeb92277f429f0d126ba2a43e76f8f9cda6cd32c2347d12d9e2 |
| high_nu | data/stage4/high_nu.npz | 0932c275af29b0256218b8f1edb13b8437d33a78ad8efd28ebf225f5a49db72c |
| velocity_pp | data/stage4/velocity_pp.npz | 55610b6cc7d1abbf0abce4f3999795ecd1e979ed2bfcacf02d67ff0aba2f186a |
| velocity_pn | data/stage4/velocity_pn.npz | 2dc783971c223fd17cebc0aef1a8b4bd677a208633fcb970eef14932a516cfdf |
| velocity_np | data/stage4/velocity_np.npz | b8659ace32852d65b42b063418963cfa30f6941c32aab12c89961a4cffd4de87 |
| velocity_nn | data/stage4/velocity_nn.npz | 0d1e31e145d9ccb85eee9d149bfeb74e53fb91aeaf9f552fba3ea4c6eebf9196 |
| historical_id | data/dev/trajectories.npz | 73974ce6d7be46209d22f09e8d9225dd97e59aaf324d55c5328e9f18b375638e |

Fresh ID (64 trajectories) is the primary control; historical ID uses only the eight frozen test IDs [51,16,49,17,40,47,22,28] and is exploratory. High nu has 64 trajectories. Each velocity quadrant has 16; the aggregate contains those same 64 trajectories exactly once, with equal quadrant weight. It is not a new archive. ID predictions are reused from verified Stage 8B outputs.

ID coefficient ranges are c_x,c_y in [-1,1], nu in [0.005,0.05]. High nu changes nu to [0.055,0.08]; high velocity uses signed ranges [1.05,1.4] or [-1.4,-1.05], retaining ID nu. pp/pn/np/nn denote the signs of (c_x,c_y). All archives retain the 64×64 grid and original t=0,...,1 states; no reference data or long-horizon data were generated. No OOD archive entered any fitting step. Calibration and independent audit archives retain their previous roles; neither is repurposed as an OOD fitting archive.

## Definitions and inference

All five members start at the same true initial field and retain fixed coefficients. Each member feeds back its own previous prediction, never the ensemble mean. Analysis excludes t=0 and evaluates all ten horizons t=0.1,...,1.0. Error is physical relative L2 error of the ensemble mean. Raw spread is the sample standard deviation across five members (ddof=1); U_rms is its spatial RMS and U_rel its L2 norm divided by the ensemble-mean L2 norm, with the frozen denominator floor. These remain Stage 8B definitions; raw spread is non-Bayesian.

Frozen Stage 8C bands are mean ± q(h,level) × raw spread. The score denominator and width normalization retain epsilon=2.355322120279037e-13; the raw band spread is not floored. Coverage requires every grid point in a trajectory's field at that horizon to be inside the band. W_rel is the full-width L2 norm divided by the mean-field L2 norm; the implementation checks W_rel=2qU_rel for these predictions. This is whole-field, fixed-horizon coverage, not simultaneous coverage over time.

Intervals below are descriptive 95% percentile bootstrap intervals: 2,000 replicates, seed 8501. Trajectories are the independent units; the same row draws apply across repeated horizons. ID and OOD draws are independent, not paired by archive-local index. Aggregate velocity bootstrap samples 16 trajectories within each quadrant, preserving the fixed mixture. Grid cells are never independent replicates. Ratios use means and are null for denominators at or below 1e-12. Boundary coverage bootstrap intervals can collapse at 100%; this is not certainty. Secondary Wilson intervals are retained for individual archives.

## Regime results at t=1

Differences are versus fresh ID. Historical ID is an exploratory reference only.


| Regime | N | Mean E | Median E | Mean U_rel | Median U_rel | Mean U_rms | ΔE | ΔU_rel |
|---|---|---|---|---|---|---|---|---|
| fresh_id | 64 | 0.1410 | 0.1230 | 0.1010 | 0.0963 | 0.0239 | +0.0000 | +0.0000 |
| high_nu | 64 | 0.1484 | 0.1373 | 0.1888 | 0.1660 | 0.0343 | +0.0074 | +0.0879 |
| high_velocity | 64 | 0.3959 | 0.3988 | 0.1872 | 0.1751 | 0.0386 | +0.2549 | +0.0862 |
| velocity_pp | 16 | 0.4454 | 0.4431 | 0.2299 | 0.2192 | 0.0515 | +0.3045 | +0.1289 |
| velocity_pn | 16 | 0.4400 | 0.4001 | 0.2325 | 0.2080 | 0.0443 | +0.2990 | +0.1315 |
| velocity_np | 16 | 0.3130 | 0.2637 | 0.1198 | 0.1163 | 0.0246 | +0.1720 | +0.0189 |
| velocity_nn | 16 | 0.3853 | 0.3766 | 0.1666 | 0.1631 | 0.0340 | +0.2443 | +0.0656 |
| historical_id | 8 | 0.1712 | 0.1926 | 0.1310 | 0.1269 | 0.0380 | +0.0302 | +0.0300 |

All horizons, mean/median E, U_rel and U_rms, U_rel 10/50/90 percentiles, mean intervals, median-difference intervals, mean-ratio intervals and within-horizon associations are in [stage8e_by_horizon.csv](results/stage8e_by_horizon.csv). This includes all quadrants without horizon selection. Complete AUROC intervals, failure-case identities and output hashes are in [stage8e_summary.json](results/stage8e_summary.json).

| OOD regime | Δ mean E [95% CI] | E ratio [95% CI] | Δ mean U_rel [95% CI] | U_rel ratio [95% CI] |
|---|---|---|---|---|
| high_nu | 0.0074 [-0.0151, 0.0308] | 1.0524 [0.9010, 1.2384] | 0.0879 [0.0656, 0.1125] | 1.8703 [1.6158, 2.1938] |
| high_velocity | 0.2549 [0.2235, 0.2853] | 2.8084 [2.4485, 3.2292] | 0.0862 [0.0703, 0.1029] | 1.8541 [1.6420, 2.0992] |
| velocity_pp | 0.3045 [0.2636, 0.3483] | 3.1596 [2.7332, 3.6600] | 0.1289 [0.1020, 0.1590] | 2.2771 [1.9503, 2.6586] |
| velocity_pn | 0.2990 [0.2437, 0.3566] | 3.1212 [2.6183, 3.7126] | 0.1315 [0.0973, 0.1685] | 2.3029 [1.9303, 2.7537] |
| velocity_np | 0.1720 [0.1061, 0.2395] | 2.2198 [1.7124, 2.7910] | 0.0189 [0.0015, 0.0363] | 1.1868 [1.0133, 1.3836] |
| velocity_nn | 0.2443 [0.1909, 0.2941] | 2.7329 [2.2708, 3.2168] | 0.0656 [0.0485, 0.0836] | 1.6497 [1.4479, 1.8954] |

High nu ΔE is 0.0074 with interval [-0.0151,0.0308], whereas ΔU_rel is 0.0879 [0.0656,0.1125]. This does not establish equal error or harmlessness universally, but does not show a clear additional mean error in this sample. It is Case D, not the anticipated Case B. High velocity supports Case A at regime level; the np error ratio 2.22 versus U_rel ratio 1.19 exposes a muted uncertainty response. Its positive final uncertainty difference is small, so it is not literally zero uncertainty increase.

### Primary mean trajectories (all horizons)

| t | ID E | nu E | velocity E | ID U_rel | nu U_rel | velocity U_rel |
|---|---|---|---|---|---|---|
| 0.1 | 0.0209 | 0.0211 | 0.0529 | 0.0146 | 0.0259 | 0.0243 |
| 0.2 | 0.0386 | 0.0388 | 0.1020 | 0.0259 | 0.0464 | 0.0444 |
| 0.3 | 0.0545 | 0.0546 | 0.1479 | 0.0361 | 0.0648 | 0.0630 |
| 0.4 | 0.0692 | 0.0692 | 0.1909 | 0.0458 | 0.0823 | 0.0808 |
| 0.5 | 0.0829 | 0.0830 | 0.2310 | 0.0552 | 0.0994 | 0.0984 |
| 0.6 | 0.0957 | 0.0962 | 0.2686 | 0.0644 | 0.1166 | 0.1159 |
| 0.7 | 0.1078 | 0.1092 | 0.3037 | 0.0735 | 0.1340 | 0.1335 |
| 0.8 | 0.1193 | 0.1221 | 0.3365 | 0.0827 | 0.1518 | 0.1512 |
| 0.9 | 0.1304 | 0.1352 | 0.3672 | 0.0918 | 0.1701 | 0.1691 |
| 1.0 | 0.1410 | 0.1484 | 0.3959 | 0.1010 | 0.1888 | 0.1872 |

![Mean regime trends](../runs/stage8/ood_awareness/plots/regime_trends.png)

## Within-regime awareness and OOD discrimination

These correlations are computed separately at each horizon, never by treating pooled time points as independent samples. AUROC uses larger U_rel as the OOD direction with half credit for ties; it is a descriptive discrimination score, not a calibrated OOD probability.

| Regime | Pearson [95% CI] | Spearman [95% CI] | OOD AUROC [95% CI] |
|---|---|---|---|
| fresh_id | 0.7349 [0.5873, 0.8586] | 0.7799 [0.6306, 0.8795] | — |
| high_nu | 0.6840 [0.5817, 0.7990] | 0.7878 [0.6629, 0.8599] | 0.8296 [0.7620, 0.8940] |
| high_velocity | 0.4642 [0.2849, 0.6418] | 0.4952 [0.2736, 0.6740] | 0.8716 [0.8203, 0.9158] |
| velocity_pp | 0.1737 [-0.2272, 0.6527] | 0.1765 [-0.3810, 0.6800] | 0.9746 [0.9375, 0.9980] |
| velocity_pn | 0.1049 [-0.2853, 0.6184] | 0.2029 [-0.3183, 0.6364] | 0.9678 [0.9277, 0.9941] |
| velocity_np | 0.6468 [0.2083, 0.8783] | 0.4794 [-0.0421, 0.8296] | 0.6553 [0.5234, 0.7715] |
| velocity_nn | 0.3495 [-0.4951, 0.7550] | 0.0265 [-0.5732, 0.5328] | 0.8887 [0.8125, 0.9521] |
| historical_id | 0.6406 [0.1194, 0.9125] | 0.6190 [-0.1507, 1.0000] | — |

Aggregate velocity retains moderate ranking, but pp and pn correlations are weak despite high OOD AUROC. The nn Spearman correlation falls from about 0.62 initially to 0.03 at t=1. The np final rank interval includes zero. Thus successful regime discrimination does not establish useful within-quadrant failure ranking. High-nu AUROC reflects conservative sensitivity to a shift without a demonstrated material error increase.

| t | High-nu AUROC | Velocity AUROC |
|---|---|---|
| 0.1 | 0.8384 [0.7693, 0.9016] | 0.8657 [0.8137, 0.9138] |
| 0.2 | 0.8447 [0.7788, 0.9055] | 0.8777 [0.8267, 0.9233] |
| 0.3 | 0.8411 [0.7746, 0.9028] | 0.8779 [0.8257, 0.9231] |
| 0.4 | 0.8394 [0.7720, 0.9001] | 0.8782 [0.8262, 0.9221] |
| 0.5 | 0.8374 [0.7712, 0.8994] | 0.8777 [0.8254, 0.9221] |
| 0.6 | 0.8333 [0.7661, 0.8958] | 0.8767 [0.8237, 0.9226] |
| 0.7 | 0.8308 [0.7637, 0.8943] | 0.8765 [0.8242, 0.9214] |
| 0.8 | 0.8313 [0.7632, 0.8945] | 0.8750 [0.8242, 0.9190] |
| 0.9 | 0.8306 [0.7624, 0.8948] | 0.8733 [0.8223, 0.9175] |
| 1.0 | 0.8296 [0.7620, 0.8940] | 0.8716 [0.8203, 0.9158] |

## Frozen ID calibration applied to OOD: transfer stress test only

**ID calibration guarantees/interpretations do not automatically transfer under distribution shift.** No factor was adjusted after inspecting coverage. All levels and horizons, coverage intervals, mean/median W_rel and mean-width intervals are in [stage8e_coverage.csv](results/stage8e_coverage.csv).

| Regime | 50% coverage | 70% | 80% | 90% | 95% | Mean W_rel90 |
|---|---|---|---|---|---|---|
| fresh_id | 0.5625 | 0.8125 | 0.8438 | 0.9219 | 0.9531 | 5.029 |
| high_nu | 0.8906 | 0.9375 | 0.9844 | 0.9844 | 1.0000 | 9.405 |
| high_velocity | 0.3438 | 0.5469 | 0.6562 | 0.8750 | 0.9531 | 9.324 |
| velocity_pp | 0.3750 | 0.6875 | 0.6875 | 0.8750 | 1.0000 | 11.451 |
| velocity_pn | 0.2500 | 0.3750 | 0.6250 | 0.8750 | 1.0000 | 11.581 |
| velocity_np | 0.3750 | 0.5625 | 0.7500 | 0.8750 | 0.9375 | 5.968 |
| velocity_nn | 0.3750 | 0.5625 | 0.5625 | 0.8750 | 0.8750 | 8.296 |
| historical_id | 0.2500 | 0.5000 | 0.7500 | 0.8750 | 0.8750 | 6.522 |

At t=1, aggregate velocity coverage is 87.5% at the nominal 90% level (bootstrap interval [79.7%,95.3%]), versus 92.2% fresh ID; its 50–80% bands under-cover substantially. High nu has 98.4% empirical coverage at 90%, but mean full width W_rel is 9.40, compared with 5.03 fresh ID. Aggregate velocity W_rel90 is 9.32. These width norms are many times the mean-field norm and are not practically sharp late-rollout intervals. The apparent 95% aggregate coverage of 95.3% does not imply OOD conformal validity; nn reaches only 87.5%.

![Coverage and width stress test](../runs/stage8/ood_awareness/plots/coverage_width_stress.png)

## Predeclared error/uncertainty categories

For each horizon and each OOD regime, ranks are computed within fresh ID plus that OOD regime: (average-tie rank − 0.5)/pool size. High means rank ≥0.75; low error means rank ≤0.25; modest uncertainty means rank <0.75. Counts below include OOD trajectories only. These are relative descriptive categories, not calibrated failure probabilities or claims of absolute confidence. Each quadrant has its own pooled thresholds, so its counts must not be summed to reconstruct aggregate categories.

| Regime | High E/high U | High E/modest U | Low E/high U | Other |
|---|---|---|---|---|
| high_nu | 14 | 1 | 0 | 49 |
| high_velocity | 20 | 12 | 0 | 32 |
| velocity_pp | 15 | 1 | 0 | 0 |
| velocity_pn | 14 | 2 | 0 | 0 |
| velocity_np | 3 | 7 | 0 | 6 |
| velocity_nn | 10 | 3 | 0 | 3 |

The aggregate final high-error/modest-uncertainty cases must remain in the final story: velocity_pp:0, velocity_pp:4, velocity_np:0, velocity_np:1, velocity_np:4, velocity_np:9, velocity_np:14, velocity_nn:3, velocity_nn:5, velocity_nn:7, velocity_nn:10, velocity_nn:12. This includes the predetermined pp:0 example. No low-error/high-uncertainty cases meet the strict final quartile rule, but that does not negate the high-nu regime-level conservative response.

| t | nu high/high | nu high/modest | nu low/high | velocity high/high | velocity high/modest | velocity low/high |
|---|---|---|---|---|---|---|
| 0.1 | 12 | 3 | 2 | 22 | 9 | 0 |
| 0.2 | 11 | 3 | 3 | 22 | 9 | 0 |
| 0.3 | 12 | 3 | 4 | 24 | 8 | 0 |
| 0.4 | 12 | 3 | 1 | 23 | 9 | 0 |
| 0.5 | 13 | 2 | 1 | 24 | 8 | 0 |
| 0.6 | 12 | 2 | 1 | 23 | 9 | 0 |
| 0.7 | 13 | 2 | 1 | 22 | 10 | 0 |
| 0.8 | 13 | 2 | 1 | 21 | 11 | 0 |
| 0.9 | 14 | 2 | 1 | 20 | 12 | 0 |
| 1.0 | 14 | 1 | 0 | 20 | 12 | 0 |

## Failure-awareness scorecard

Dimensions remain separate; there is no combined pass/fail score.

| Regime | A: error vs ID | B: U_rel vs ID | C: within-regime ranking | D: coverage transfer at t=1 | E: sharpness at t=1 |
|---|---|---|---|---|---|
| fresh ID | Control | Control | Strong final association | 90%: 92.2% | Broad W90=5.03 |
| high nu | No clear increase | Strong increase | Useful positive association | Conservative; 90%: 98.4% | Very broad W90=9.40 |
| velocity aggregate | Clear increase | Clear increase | Moderate; mixture-dependent | Lower at 50–90%; 95% near nominal | Very broad W90=9.32 |
| pp | Clear increase | Clear increase | Weak; interval includes zero | 90%: 87.5%; low levels degrade | Very broad W90=11.45 |
| pn | Clear increase | Clear increase | Weak; interval includes zero | 90%: 87.5%; low levels degrade | Very broad W90=11.58 |
| np | Clear increase | Small increase | Uncertain final rank association | 90%: 87.5%; 95%: 93.8% | Broad W90=5.97; muted response |
| nn | Clear increase | Clear increase | Late rank association vanishes | 90% and 95%: 87.5% | Very broad W90=8.30 |
| historical ID | Exploratory n=8 | Exploratory n=8 | Very uncertain rank association | 90% and 95%: 87.5% | Broad W90=6.52 |

## Fixed spatial examples

The protocol fixed fresh ID:0, high nu:0, pp:0, maximum-final-error aggregate velocity, and maximum-final-U_rel aggregate velocity. The last two selection rules were predeclared, not chosen for favorable agreement. Every plot shows truth, ensemble mean, absolute error, raw spread and frozen-ID 90% full width; truth/mean and error/spread use shared scales within each example. All five happen to be covered at 90%; none was swapped to manufacture a failure or success.

| Rule | Identity | E | U_rel | W_rel90 | Covered90 |
|---|---|---|---|---|---|
| predetermined_fresh | fresh_id:0 | 0.1335 | 0.1024 | 5.100 | True |
| predetermined_high_nu | high_nu:0 | 0.0922 | 0.1245 | 6.203 | True |
| predetermined_velocity | velocity_pp:0 | 0.5252 | 0.1486 | 7.401 | True |
| worst_velocity_error | velocity_pn:10 | 0.6822 | 0.1852 | 9.225 | True |
| highest_velocity_U_rel | velocity_pn:11 | 0.3172 | 0.3680 | 18.327 | True |

![predetermined_fresh](../runs/stage8/ood_awareness/plots/predetermined_fresh.png)

![predetermined_high_nu](../runs/stage8/ood_awareness/plots/predetermined_high_nu.png)

![predetermined_velocity](../runs/stage8/ood_awareness/plots/predetermined_velocity.png)

![worst_velocity_error](../runs/stage8/ood_awareness/plots/worst_velocity_error.png)

![highest_velocity_U_rel](../runs/stage8/ood_awareness/plots/highest_velocity_U_rel.png)

The worst-error case pn:10 (E=0.682) has U_rel=0.185; the maximum-uncertainty case pn:11 has smaller E=0.317 and U_rel=0.368. Predetermined pp:0 has E=0.525 with U_rel=0.149. Broad calibrated bands cover these fields but do not repair raw ranking limitations. Figures are local ignored outputs and are not embedded as tracked binary artifacts.

## Limitations

Independent initial conditions prevent interpreting regime differences as isolated causal coefficient effects. These historical, fresh and OOD datasets have prior performance exposure; this is retrospective evidence, not a new blind test. Quadrant samples have only 16 independent trajectories, and aggregate associations can reflect between-quadrant differences. Bootstrap intervals are descriptive, conditional on the frozen archives and mixture, without multiple-comparison guarantees. All ten horizons are repeated measures; no p-value selection or quadrant removal is used. No extrapolation beyond t=1, new PDEs, Bayesian posterior interpretation, calibrated OOD probabilities or OOD conformal guarantees is supported.

## Verification and files

Full regression: `MPLCONFIGDIR=/tmp/stage8b-matplotlib .venv/bin/python -m pytest -q` — **255 passed in 8.77s**, including 13 Stage 8E tests. Tests cover frozen provenance/archive hashes, archive roles, inference-only calls, ID baseline, deterministic independent trajectory bootstrap, differences, correlations, AUROC, frozen factors, coverage, aggregation and fixed selection.

Protected verification checks all 496 entries in `runs/stage8/ood_awareness/protected_before.json` plus inherited Stage 8A–D verifiers and Stage 8E source, freeze, prediction and analysis hashes. The receipt is `runs/stage8/ood_awareness/protected_verification.json`. Pre-Stage-8E tracked and local protected scientific files remain unchanged. Generated arrays, bootstrap indices, plots, manifests and logs under `runs/stage8/ood_awareness/` are ignored. No existing data archive was changed and no new reference archive was generated.

Ten new review files, with no existing tracked file modified:

- `stage8_uq/ood_awareness.py`
- `stage8_uq/ood_awareness_io.py`
- `stage8_uq/ood_awareness_plots.py`
- `examples/evaluate_stage8e.py`
- `tests/test_stage8e_ood_awareness.py`
- `docs/results/stage8e_freeze.json`
- `docs/results/stage8e_by_horizon.csv`
- `docs/results/stage8e_coverage.csv`
- `docs/results/stage8e_summary.json`
- `docs/stage8e_ood_failure_awareness.md`

Git status: these ten files are untracked review outputs, with no staged changes; main remains at the frozen Stage 8D revision above. **No commit or push was performed. Waiting for review.**

## NEXT STEPS

A. The ensemble demonstrates useful OOD failure awareness at the regime level, with substantial limitations for individual trajectories.

B. Aggregate high velocity supports the claim; pp and pn support regime detection but not robust within-quadrant ranking. np has a muted uncertainty response to substantial additional error; nn loses late ranking. High nu does not establish failure awareness from its uncertainty increase alone.

C. Uncertainty responds to a mixture of shift and difficulty: high-nu conservatism rules out a difficulty-only interpretation, while positive within-regime associations rule out treating it solely as an OOD label.

D. Frozen ID bands retain substantial empirical OOD coverage at high nominal levels, but low-level velocity coverage degrades and late bands are extremely broad. This is limited practical transfer, without OOD coverage validity.

E. Preserve all high-error/modest-uncertainty identities, the muted np response, weak pp/pn correlations, nn late collapse, high-nu conservatism and broad intervals. These are relative confidently-wrong warning signs, not calibrated confidence assertions. Preserve Stage 8D's failure to establish robust incremental warning beyond current true error on the primary audit set.

F. **STOP further UQ experiments.** These results answer the intended question with qualified positive and negative evidence; no sharply defined unresolved question here justifies automatic expansion. Do not add ensemble members, Bayesian operators, longer horizons, PDEs or new calibration methods.

G. After review, recommend a final UQ synthesis connecting frozen ensemble quality, imperfect raw error awareness, ID coverage versus width, limited incremental warning and mixed OOD awareness. That is synthesis of existing evidence, not another scientific stage. No further work is started by this recommendation.

