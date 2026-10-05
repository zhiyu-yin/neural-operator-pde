# Stage 8B: raw ensemble error awareness

Raw U_rel contains useful information about actual error on the 64-trajectory
fresh-ID archive: horizon-specific Pearson correlations are 0.709–0.776 and
Spearman correlations are 0.780–0.846. All pointwise bootstrap intervals are
positive. This justifies a separate calibration study using the predeclared
U_rel score, but does not establish calibrated coverage or dependable detection
of every failure. The worst final-error trajectory is missed by the top-fifth
uncertainty detector.

This is a **retrospective association study**, not a pristine prospective
benchmark. Stage 4 fresh ID existed before UQ and has historical performance
exposure. The original eight-trajectory test set is explicitly exploratory.
No OOD predictions or OOD conclusions enter this study.

## Frozen inputs and provenance

All five members passed strict Stage 8A loading before the fresh-ID archive was
loaded and rollouts began. Verification checks checkpoint, member manifest,
ensemble manifest, architecture, seeds, split IDs, normalization, training
archive, selection/gate/protocol records, finite weights, full training budgets,
and exact training-source hashes. A changed member or manifest aborts execution.
No member was dropped, replaced, retrained, or selected using these outcomes.

The complete frozen metadata are in [stage8b_freeze.json](results/stage8b_freeze.json).

Stage 8A committed revision: `4967490593f1f5c125dc8e87abd2387144741954`. Member manifests record training-time source revision `f095871d842e469f51d49751a7459684a507fd77` because Stage 8A source was uncommitted during training; their exact source-file hashes resolve that distinction and remain verified.

Ensemble manifest SHA-256: `0da21da72cb440d9bfc08563c1308a29d6dd6cc8f85785c5483d1ef6ec02d713`.

| Member | Init seed | Shuffle seed | Checkpoint SHA-256 |
| --- | --- | --- | --- |
| 0 | 2028 | 5029 | `cf8899453cadf2443961fd65daa736269ed671dd26a375fd3edc1159ae56ca33` |
| 1 | 8101 | 8201 | `591ec6819777909579bdc9c4f57ba62f25d91b709609c708c4afef30b78b1903` |
| 2 | 8102 | 8202 | `2f15a50f9a438a3db148d420eede5e15bf25f182ea3fd0682d3be50822253f79` |
| 3 | 8103 | 8203 | `ab95d65c8d340b550ee578c0bcf939df960e1d7a3f9780d2fc42f1434f72b5fd` |
| 4 | 8104 | 8204 | `ee49b65ca83df821c8703d97009ee82283d7bcb864218e5acac5a48925714a46` |

Architecture: FNO2d modes=12, width=32, blocks=4, projection_width=64; four inputs and one output. The exact ordered 48/8/8 train/validation/test splits and normalization values are frozen in the JSON, identical across all members.

Normalization canonical SHA-256: `e413b4e2a87fb0068bce67d3d81684b31d1d7d57b92b5ec96c50b9838c7e0120`. Training archive SHA-256: `73974ce6d7be46209d22f09e8d9225dd97e59aaf324d55c5328e9f18b375638e`. No normalization is refitted.

| Evaluation archive | SHA-256 | Trajectories and role |
| --- | --- | --- |
| data/stage4/fresh_id.npz | `52292b4e91eb9aeb92277f429f0d126ba2a43e76f8f9cda6cd32c2347d12d9e2` | 64; primary retrospective |
| data/dev/trajectories.npz | `73974ce6d7be46209d22f09e8d9225dd97e59aaf324d55c5328e9f18b375638e` | IDs 51,16,49,17,40,47,22,28 only; exploratory |

Fresh ID uses all 64 trajectories, seed 4100, cx and cy in [-1,1], nu in [0.005,0.05], amplitude [0.5,1.5], width [0.4,0.8], grid 64×64. The original archive is read to select only its frozen test IDs; training and validation trajectories never enter inference or association arrays. Archive-local IDs are not identities shared between archives. No new data were generated.

## Definitions and statistical guardrails

All members start at the same true initial field. Each member autoregresses its
own normalized prediction, with fixed cx, cy, nu; the ensemble mean is never
fed back. The unchanged Stage 8A rollout helper also computes unused teacher-forced
outputs internally; Stage 8B retains and analyzes only autoregressive outputs.
Physical inverse normalization precedes float64 ensemble reductions.

For M=5, physical predictions u_m, ground truth u, and G grid cells:

```text
mu = mean_m(u_m)
s = sqrt(sum_m((u_m - mu)^2) / (M - 1))
U_rms = sqrt(sum_grid(s^2) / G)
U_rel = U_rms / max(sqrt(sum_grid(mu^2) / G), 1e-12 * frozen_field_std)
error = sqrt(sum_grid((mu-u)^2)) / max(sqrt(sum_grid(u^2)), 1e-12)
```

The U_rel denominator uses predictions only. Its frozen floor is
2.355322120279037e-13; neither evaluation set activates it. The physical error
uses the unchanged Stage 4 L2 denominator floor. All member predictions were
finite. Raw spread is an uncalibrated uncertainty proxy, not a confidence
interval or an assumed posterior standard deviation.

The initial state t=0 is retained in full arrays for verification but excluded
from every association, binning, discrimination, trend, and spatial summary.
The only analysis horizons are 0.1,…,1.0. There are 64 independent trajectory
units per fresh-ID horizon, not 640 independent trajectory/horizon replicates.

Pearson measures linear association; Spearman uses average ranks for ties.
Intervals are pointwise 95% percentile bootstrap intervals from 2,000 paired
trajectory resamples, NumPy seed 8301. The same sampled trajectory rows preserve
all their horizons. No grid-cell bootstrap, pooled independent-observation
p-value, significance selection, or simultaneous-coverage claim is made.
Degenerate correlations return an explicit undefined status and invalid-bootstrap
count; no correlation is forced. All observed horizon correlations here are
defined. The multiple horizon and early-warning estimates are dependent.

Five approximately equal uncertainty quintiles are assigned independently at
each horizon using stable ascending ranks (ties follow frozen trajectory order).
Fresh-ID sizes are 13/13/13/13/12; historical sizes are 2/2/2/1/1. Boundaries are
not tuned to outcomes. High error and high uncertainty each select the largest
ceil(n/5) ranks: 13/64 (20.3125%) or 2/8 (25%). Exact ties select later positions
in the fixed trajectory order. AUROC uses half credit for tied scores.

## Primary fixed-horizon association: fresh ID

| Time | n | Pearson [95% CI] | Spearman [95% CI] |
| --- | --- | --- | --- |
| 0.1 | 64 | 0.776 [0.697, 0.864] | 0.846 [0.744, 0.903] |
| 0.2 | 64 | 0.738 [0.635, 0.844] | 0.829 [0.709, 0.896] |
| 0.3 | 64 | 0.720 [0.598, 0.836] | 0.818 [0.685, 0.893] |
| 0.4 | 64 | 0.711 [0.578, 0.831] | 0.817 [0.682, 0.895] |
| 0.5 | 64 | 0.709 [0.571, 0.833] | 0.807 [0.673, 0.890] |
| 0.6 | 64 | 0.710 [0.563, 0.836] | 0.794 [0.651, 0.879] |
| 0.7 | 64 | 0.715 [0.567, 0.839] | 0.781 [0.633, 0.874] |
| 0.8 | 64 | 0.721 [0.567, 0.844] | 0.780 [0.628, 0.876] |
| 0.9 | 64 | 0.728 [0.569, 0.849] | 0.781 [0.633, 0.876] |
| 1.0 | 64 | 0.735 [0.575, 0.854] | 0.780 [0.628, 0.873] |

## Rollout trends: a separate question

Mean error and uncertainty both increase throughout rollout. This answers the
mean deterioration question; the fixed-horizon table above answers whether
uncertainty ranks difficult trajectories without relying on time variation.

| Time | Mean error | Median error | Mean U_rel | Median U_rel | Mean U_rms |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.020941 | 0.019252 | 0.014614 | 0.014971 | 0.003635 |
| 0.2 | 0.038600 | 0.034066 | 0.025877 | 0.026122 | 0.006457 |
| 0.3 | 0.054531 | 0.047761 | 0.036096 | 0.036848 | 0.008996 |
| 0.4 | 0.069197 | 0.059941 | 0.045785 | 0.046054 | 0.011365 |
| 0.5 | 0.082856 | 0.071581 | 0.055178 | 0.054506 | 0.013617 |
| 0.6 | 0.095678 | 0.084403 | 0.064401 | 0.061708 | 0.015783 |
| 0.7 | 0.107797 | 0.094372 | 0.073538 | 0.070407 | 0.017882 |
| 0.8 | 0.119325 | 0.102938 | 0.082653 | 0.079817 | 0.019930 |
| 0.9 | 0.130360 | 0.114073 | 0.091792 | 0.088143 | 0.021939 |
| 1.0 | 0.140981 | 0.123033 | 0.100966 | 0.096292 | 0.023911 |

Figure: `runs/stage8/error_awareness/fresh_id/rollout_trends.png` contains separate horizon-vs-mean-error and horizon-vs-mean-U_rel panels.

## Uncertainty quintiles

At each horizon the full mean/median error table is saved in [stage8b_quintiles.csv](results/stage8b_quintiles.csv). The primary aggregate below gives each horizon equal weight; uncertainty ranks are computed within each horizon. Intervals resample whole trajectories with fixed observed bin labels, then recompute each horizon-bin mean and average across horizons. Thus these intervals are conditional on the observed bin assignment. Resamples missing any horizon-bin are counted and omitted, not filled in. The eight-case exploratory aggregate can have many such draws and is correspondingly fragile.

| Quintile | Mean error [95% CI] | Mean of horizon median errors | Invalid bootstrap draws |
| --- | --- | --- | --- |
| 1 | 0.03958 [0.03282, 0.04691] | 0.03687 | 0 |
| 2 | 0.06401 [0.05355, 0.07846] | 0.05825 | 0 |
| 3 | 0.09135 [0.07629, 0.10853] | 0.07955 | 0 |
| 4 | 0.10794 [0.09041, 0.12698] | 0.10668 | 0 |
| 5 | 0.13070 [0.11738, 0.14375] | 0.13118 | 0 |

Aggregate mean error increases across all five bins. Within individual horizons, mean errors are monotonic at 8/10 horizons; 0.2 and 0.3 are exceptions. These exceptions are retained. The JSON also provides a sensitivity summary that first averages each trajectory’s errors during its visits to each bin, then gives each contributing trajectory equal weight. That different estimand has unequal horizon exposure and remains secondary.

Figure: `runs/stage8/error_awareness/fresh_id/quintiles.png`; error bars are bootstrap uncertainty in the bin mean, not predictive intervals.

## High-error discrimination

| Time | AUROC | Hits / 13 | Precision | Recall |
| --- | --- | --- | --- | --- |
| 0.1 | 0.882 | 6/13 | 0.462 | 0.462 |
| 0.2 | 0.867 | 7/13 | 0.538 | 0.538 |
| 0.3 | 0.861 | 7/13 | 0.538 | 0.538 |
| 0.4 | 0.854 | 7/13 | 0.538 | 0.538 |
| 0.5 | 0.840 | 6/13 | 0.462 | 0.462 |
| 0.6 | 0.828 | 6/13 | 0.462 | 0.462 |
| 0.7 | 0.817 | 6/13 | 0.462 | 0.462 |
| 0.8 | 0.813 | 6/13 | 0.462 | 0.462 |
| 0.9 | 0.810 | 7/13 | 0.538 | 0.538 |
| 1.0 | 0.845 | 7/13 | 0.538 | 0.538 |

AUROC is 0.810–0.882, but precision/recall are only 0.462–0.538. Equal selected/positive counts make precision and recall equal. Random selection would have expected precision/recall 13/64≈0.203. These are descriptive ranking diagnostics, not calibrated probabilities.

## Early-warning association

Each row compares U_rel at the source time with physical error at a future time across the same 64 trajectories. Only existing reference horizons are used.

| Source | Target | Steps ahead | Pearson [95% CI] | Spearman [95% CI] |
| --- | --- | --- | --- | --- |
| 0.1 | 0.2 | 1 | 0.794 [0.719, 0.869] | 0.852 [0.756, 0.907] |
| 0.2 | 0.3 | 1 | 0.755 [0.654, 0.850] | 0.833 [0.716, 0.899] |
| 0.3 | 0.4 | 1 | 0.735 [0.615, 0.841] | 0.826 [0.696, 0.901] |
| 0.4 | 0.5 | 1 | 0.724 [0.595, 0.838] | 0.823 [0.698, 0.897] |
| 0.5 | 0.6 | 1 | 0.721 [0.584, 0.839] | 0.804 [0.668, 0.887] |
| 0.6 | 0.7 | 1 | 0.722 [0.579, 0.840] | 0.796 [0.655, 0.881] |
| 0.7 | 0.8 | 1 | 0.725 [0.579, 0.844] | 0.783 [0.633, 0.877] |
| 0.8 | 0.9 | 1 | 0.730 [0.579, 0.848] | 0.785 [0.635, 0.879] |
| 0.9 | 1.0 | 1 | 0.736 [0.580, 0.851] | 0.785 [0.640, 0.877] |
| 0.1 | 0.3 | 2 | 0.804 [0.731, 0.874] | 0.852 [0.757, 0.907] |
| 0.2 | 0.4 | 2 | 0.767 [0.668, 0.856] | 0.838 [0.724, 0.904] |
| 0.3 | 0.5 | 2 | 0.746 [0.631, 0.846] | 0.831 [0.706, 0.901] |
| 0.4 | 0.6 | 2 | 0.735 [0.610, 0.843] | 0.818 [0.690, 0.895] |
| 0.5 | 0.7 | 2 | 0.731 [0.598, 0.845] | 0.805 [0.668, 0.886] |
| 0.6 | 0.8 | 2 | 0.731 [0.592, 0.846] | 0.797 [0.657, 0.884] |
| 0.7 | 0.9 | 2 | 0.733 [0.590, 0.848] | 0.788 [0.638, 0.881] |
| 0.8 | 1.0 | 2 | 0.737 [0.589, 0.851] | 0.788 [0.643, 0.882] |

Future-error associations are positive, but persistent trajectory difficulty and correlation with current error can explain them. This does not show causal warning, incremental predictive value beyond current error, or guaranteed detection of future degradation. No longer trajectories were generated.

## Secondary pooled and horizon-demeaned checks

| Dataset/check | Pearson [95% CI] | Spearman [95% CI] |
| --- | --- | --- |
| fresh ID/pooled; horizon confounded | 0.839 [0.759, 0.909] | 0.902 [0.851, 0.940] |
| fresh ID/horizon demeaned | 0.721 [0.575, 0.842] | 0.789 [0.641, 0.885] |
| historical exploratory/pooled; horizon confounded | 0.847 [0.714, 0.955] | 0.855 [0.737, 0.957] |
| historical exploratory/horizon demeaned | 0.670 [0.264, 0.928] | 0.638 [0.120, 0.955] |

Pooled coefficients are inflated relative to fixed-horizon results because horizon is a confounder. Bootstrap samples retain whole trajectories. The demeaned check subtracts each horizon’s trajectory mean separately from each variable before flattening; it repeats that centering inside every resample. Its Spearman coefficient ranks the demeaned values, not the original within-horizon ranks. Neither secondary check replaces the primary analysis.

## Spatial diagnostics and missed failures

| Selection rule | Fresh-ID trajectory | Final error | Final U_rel | Field Spearman |
| --- | --- | --- | --- | --- |
| predetermined | 0 | 0.13350 | 0.10239 | 0.793 |
| highest_final_error | 13 | 0.30387 | 0.08282 | 0.589 |
| highest_final_uncertainty | 55 | 0.25100 | 0.21910 | 0.712 |

The predetermined example is ID 0; the other rules independently select the maximum actual final error (13) and maximum final uncertainty (55). All three final-horizon four-panel maps are under `runs/stage8/error_awareness/fresh_id/spatial_*.png`. Ground truth and mean share a physical color scale; absolute error and spread share another. The highest-error map visibly underrepresents the error magnitude with spread and has spatial mismatch despite positive rank association.

Spatial rank correlations are computed separately for each trajectory/horizon field. Median field Spearman increases from 0.442 at t=0.1 to 0.726 at t=1.0, with all 64 fields defined per horizon. Grid cells are not independent replicates; no spatial p-values are reported. Complete horizon medians are in the JSON.

For a narrow operational “confidently wrong” diagnostic, the implementation checks top-fifth error intersecting the bottom uncertainty quintile within each horizon. No fresh-ID or historical cases meet that definition. This does **not** imply there are no missed failures: trajectory 13 has final error 0.30387 with U_rel 0.08282, only 24th of 64 in ascending uncertainty order (41st in descending order), and is missed by the top-fifth detector. Six of 13 high-error cases are missed at t=1.0. Shared ensemble bias can produce substantial errors at modest spread. Raw spread magnitude must not be interpreted as coverage.

## Exploratory historical ID

| Time | n | Pearson [95% CI] | Spearman [95% CI] | Mean error | Mean U_rel | AUROC | Precision=recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 8 | 0.807 [0.435, 0.983] | 0.762 [0.114, 1.000] | 0.02497 | 0.01824 | 1.000 | 1.000 |
| 0.2 | 8 | 0.787 [0.454, 0.967] | 0.833 [0.342, 1.000] | 0.04605 | 0.03285 | 1.000 | 1.000 |
| 0.3 | 8 | 0.760 [0.436, 0.959] | 0.810 [0.284, 1.000] | 0.06517 | 0.04624 | 1.000 | 1.000 |
| 0.4 | 8 | 0.735 [0.399, 0.951] | 0.810 [0.284, 1.000] | 0.08293 | 0.05899 | 1.000 | 1.000 |
| 0.5 | 8 | 0.712 [0.357, 0.946] | 0.810 [0.284, 1.000] | 0.09961 | 0.07137 | 1.000 | 1.000 |
| 0.6 | 8 | 0.693 [0.333, 0.939] | 0.810 [0.284, 1.000] | 0.11538 | 0.08352 | 1.000 | 1.000 |
| 0.7 | 8 | 0.677 [0.280, 0.932] | 0.762 [0.216, 1.000] | 0.13035 | 0.09552 | 0.833 | 0.500 |
| 0.8 | 8 | 0.663 [0.236, 0.928] | 0.667 [0.089, 1.000] | 0.14460 | 0.10741 | 0.833 | 0.500 |
| 0.9 | 8 | 0.651 [0.187, 0.925] | 0.524 [-0.200, 1.000] | 0.15821 | 0.11922 | 0.833 | 0.500 |
| 1.0 | 8 | 0.641 [0.144, 0.924] | 0.619 [-0.064, 1.000] | 0.17122 | 0.13095 | 0.917 | 0.500 |

The historical set is small, heavily inspected, and exploratory. Late-horizon Spearman intervals include zero; none of its ten per-horizon quintile-mean sequences is monotonic. Its full bin, early-warning, spatial, and trend results remain in the JSON/CSVs, including unfavorable results. Historical early-warning Spearman spans approximately 0.524–0.881, with broad intervals. These eight cases cannot override the primary 64-case findings.

## Limitations

Five members with shared training data, architecture, and normalization need not
have independent errors and can share bias. Retrospective exposure limits
prospective generalization. The bootstrap assumes exchangeable trajectories
within each archive; it does not quantify uncertainty over retrained ensembles,
architectures, numerical-reference error, or prior research choices. Intervals
are pointwise, not multiplicity-adjusted. High-error definitions are relative to
each horizon, not absolute application tolerances. Shared amplitude and
normalization effects can influence both scores. Binning loses information and
is sensitive to small sample sizes. Spatial similarity can reflect the common
support of the physical field without correctly estimating error magnitude.
Early-warning association is not incremental or causal evidence. No inference
extends past t=1.0 or to OOD regimes.

## Reproduction, tests, and protected files

From the repository root:

```bash
MPLCONFIGDIR=/tmp/stage8b-matplotlib .venv/bin/python -m examples.analyze_stage8b
MPLCONFIGDIR=/tmp/stage8b-matplotlib .venv/bin/python -m pytest -q
```

The evaluation requires the existing ignored Stage 4/6/8A artifacts. It fails
on mismatched frozen inputs and never generates replacements. Full physical
member arrays, means, spreads, U_rms, U_rel, floor flags, truth, errors, coefficients,
IDs, times, and plots are retained under ignored `runs/stage8/error_awareness/`.
The summary hashes prediction archives and analysis source files.

Full regression result: **198 passed in 4.85s**, covering all 182 Stage 1–8A tests
plus 16 Stage 8B tests. Tests cover provenance and archive tampering, physical
errors, t=0 exclusion, Pearson/Spearman, tie ranks, deterministic trajectory
bootstrap, quintiles, no duplicate bin membership, top-fifth detection,
early-warning alignment, ID-only roles, absence of fitting, and field-wise
spatial statistics. Existing tests verify own-member feedback and fixed
coefficient channels.

Protected verification: **142/142** Stage 8B baseline snapshot files unchanged,
including every pre-existing tracked file and Stage 8A local artifacts;
**329/329** historical protected files unchanged, with the previously allowed
Stage 8A package-discovery registration still the sole historical packaging
difference. Verification records and pytest output remain in the ignored run
folder. No existing tracked file was modified; the separate
jax-differentiable-pde project was not touched.

New public files:

- `stage8_uq/awareness.py`: statistical definitions and analysis.
- `examples/analyze_stage8b.py`: strict inputs, inference, summaries, figures.
- `tests/test_stage8b_awareness.py`: 16 regression tests.
- `docs/results/stage8b_freeze.json`: frozen input identities and rules.
- `docs/results/stage8b_summary.json`: complete compact results.
- `docs/results/stage8b_by_horizon.csv`: both datasets, association/trends/detection.
- `docs/results/stage8b_quintiles.csv`: every horizon/quintile mean and median.
- `docs/results/stage8b_early_warning.csv`: both datasets and both future lags.
- `docs/stage8b_error_awareness.md`: this report.

Git status at completion consists of these nine untracked additions; no tracked
modifications, staging, commit, or push. Generated prediction arrays and figures
are ignored. Stage 8B is ready for review.

## NEXT STEPS

A. **Proceed to a separately reviewed Stage 8C calibration study.** The raw score
has consistently positive fixed-horizon association and useful, imperfect
high-error discrimination. This is enough information to investigate calibration,
not evidence that calibrated coverage will succeed.

B. **Freeze U_rel** as the Stage 8C score. It was the predeclared primary score,
has no floor activations or technical failure, and is informative here. Retain
U_rms as the physical-unit companion. No score search or substitution based on
which coefficient looks best was performed.

C. Freeze all five checkpoints, member/ensemble manifests and hashes, seed table,
architecture, ordered splits, normalization and floor, training archive identity,
Stage 8A source provenance, Stage 8B analysis source hashes and output archive
hashes. Also freeze the member-wise feedback rule, ddof=1 spread, physical error,
U_rel/U_rms definitions, ten horizons, ID archive roles, trajectory-bootstrap
seed/replicate count, rank/tie/bin rules, top-fifth rule, lag alignment, and
spatial-selection rules. Any future methodological change must be versioned.

D. The exact Stage 8C question is:

> Can raw ensemble spread be converted into calibrated predictive intervals with valid empirical coverage on data not used to fit the calibration?

E. If Stage 8C is approved, predeclare and generate a new independent
**calibration-only ID archive** using the original physical and initial-condition
ranges, 64×64 grid, and existing ten-step time grid. A concrete next design is
256 calibration trajectories with a new fixed seed (e.g. 8302), plus a separately
seeded, never-fit 256-trajectory ID coverage audit (e.g. 8303), after verifying
those seeds are unused. Both remain disjoint from original training, validation,
historical/fresh test archives, and OOD data. Set coverage levels, calibration
method, interval target (pointwise versus field/trajectory simultaneous), horizon
handling, and audit criteria before generating or inspecting outcomes. The
sample sizes are a proposed design, not a coverage guarantee. No datasets have
been generated here, and Stage 8C has not started.

F. Do not yet make OOD failure-awareness conclusions, generate long-horizon
references, introduce a Bayesian neural operator, replace ensemble members, or
fit calibration with test/OOD data.

G. The overall ranking is useful here, but the missed worst-error trajectory
shows that simple positive scaling cannot repair ranking failures: it preserves
rank order. If independent validation shows poor ranking, calibration may still
be meaningful for marginal coverage, possibly at the cost of wide intervals;
it cannot establish adaptive failure awareness by scaling alone. Stage 8C should
explicitly separate coverage, interval width, and conditional/missed-failure
behavior, and report a failure of simple scaling if that is the outcome.

H. Preserve the roadmap: **Stage 8C calibration → Stage 8D rollout uncertainty →
Stage 8E ID-vs-OOD failure awareness**. These results give no strong scientific
reason to change it. Wait for review before starting Stage 8C.
