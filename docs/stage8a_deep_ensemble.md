# Stage 8A: a reproducible Model D deep ensemble

## Motivation and scope

The broader question is: **Can a neural operator recognize when its own PDE
prediction or rollout is becoming unreliable?** Stage 8A establishes a technically
verified five-member ensemble and raw spread machinery. It does not answer that
scientific question. There is no calibration, Pearson/Spearman analysis, OOD
inference, new calibration archive, or long-horizon reference generation here.
The separate jax-differentiable-pde project was not modified or merged.

## Frozen pre-UQ baseline

The pre-UQ baseline is commit
`f095871d842e469f51d49751a7459684a507fd77`, “Finalize neural-operator scientific
ML portfolio”. Before source edits, the checkout was clean on main, local
origin/main matched HEAD, and GitHub main was verified directly. Annotated tag
`v1.0.0`, message “Pre-UQ neural-operator PDE portfolio release”, was created and
pushed as authorized. Both local and remote peeled tag targets were verified
against that exact commit. The annotated tag object is
`397be414396c0f6e08b25b3b2bfabc62348bb2c6`.

Stage 8A source and results are uncommitted pending review. No branch push was
made during Stage 8A. Historical numerical/model/evaluation behavior is unchanged.
The sole edit to an existing tracked file registers `stage8_uq*` in setuptools
package discovery in pyproject.toml; the exact allowed edit is checked separately.

## Exact Model D protocol

The frozen [Stage 6 protocol](stage6_matched_control.md) remains the source of truth:

- FNO2d: 12 retained modes, width 32, four spectral-plus-local/GELU blocks,
  pointwise projection width 64. Four input channels, one output channel.
- Inputs `(B,4,N,N)`: normalized field, cx, cy, nu. Output `(B,1,N,N)`:
  normalized next field. Prediction interval 0.1; training grid 64×64.
- 1,186,209 parameter tensor elements, equivalent to 2,365,857 real scalars
  because spectral weights are complex.
- One-step normalized-space field MSE only; no mass penalty or rollout loss.
- Adam lr=0.0005, betas=(0.9,0.999), eps=1e-8, weight_decay=0; no scheduler.
- Batch size 8, shuffle=True, num_workers=0, drop_last=False.
- Exactly 60 epochs, 480 pairs/epoch, 60 updates/epoch, 3,600 updates,
  28,800 supervised fields. No performance-based early stopping.
- Checkpoint selection: minimum validation final 10-step autoregressive mean
  physical relative L2; then validation mass error; then earlier epoch on exact ties.
- CPU, four threads, deterministic PyTorch algorithms.

All members use exactly these ordered trajectory IDs:

```json
{
  "train": [
    0,
    50,
    21,
    20,
    38,
    10,
    45,
    59,
    29,
    31,
    35,
    8,
    58,
    46,
    13,
    34,
    30,
    33,
    61,
    44,
    11,
    32,
    7,
    53,
    26,
    52,
    42,
    37,
    6,
    12,
    36,
    54,
    2,
    23,
    18,
    39,
    15,
    24,
    1,
    63,
    4,
    62,
    57,
    19,
    56,
    27,
    55,
    14
  ],
  "validation": [
    60,
    41,
    5,
    9,
    48,
    25,
    43,
    3
  ],
  "test": [
    51,
    16,
    49,
    17,
    40,
    47,
    22,
    28
  ]
}
```

The original eight validation trajectories select checkpoints only. The original
eight test trajectories are used only after all member selection is frozen, for a
basic historical-ID technical check. The Stage 4 fresh-ID, high-diffusivity,
high-velocity, and resolution archives are not used to fit or select anything.
No test/OOD score controls ensemble membership.

Normalization is copied from the frozen artifact; `Normalization.fit` is never
called by Stage 8 code:

```json
{
  "field_mean": 0.12027802015661833,
  "field_std": 0.2355322120279037,
  "coefficient_mean": [
    -0.12289070488872648,
    -0.019631126517232585,
    0.031608018258263024
  ],
  "coefficient_std": [
    0.5691624809559207,
    0.5642181456221029,
    0.01183780749459938
  ]
}
```

Normalization canonical JSON SHA-256:
`e413b4e2a87fb0068bce67d3d81684b31d1d7d57b92b5ec96c50b9838c7e0120`.
Training archive SHA-256:
`73974ce6d7be46209d22f09e8d9225dd97e59aaf324d55c5328e9f18b375638e`.

## Members, selection, and measured runtime

Member 0 references the historical `runs/stage6/best.pt` directly; it was neither
retrained nor overwritten. Its recorded runtime describes the old D run. All four
new members use their predeclared initialization and shuffle seeds. No trajectory
bootstrapping, distribution change, or favorable-seed replacement was performed.

| Member | Init seed | Shuffle seed | Selected epoch | Validation final AR relative L2 | Full-run seconds | Gate seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2028 | 5029 | 50 | 0.1259710821 | 507.066 | 18.389 |
| 1 | 8101 | 8201 | 54 | 0.1281256832 | 505.926 | 19.434 |
| 2 | 8102 | 8202 | 41 | 0.1391000709 | 511.745 | 16.497 |
| 3 | 8103 | 8203 | 55 | 0.1445422052 | 529.736 | 20.284 |
| 4 | 8104 | 8204 | 59 | 0.1555041149 | 545.670 | 22.286 |

Every member completed the 60-epoch protocol. Selection of an earlier checkpoint
does not shorten the training budget. Each new member first runs the unchanged
Stage 6 training-only tiny gate, discards its gate-trained weights, then restarts
full training from its declared deterministic initialization. Gate failure would
stop the run and record a failure rather than substitute a new seed.

Four new full runs total **2093.077 seconds**;
new-member gates total **78.501 seconds**.
The ensemble orchestration wall time, including registration and final loading
checks, was **2172.570 seconds**
(**36.21 minutes**). Member 0 historical time is excluded.
These are machine-specific CPU measurements, not guaranteed future timings.

## Sanity checks and regression verification

The final pretraining suite passed **182 tests**, including all 167 historical
tests, before full training was permitted. It covers real member-0 loading,
architecture/normalization equality, different deterministic initialization,
nonshared parameter storage, independent optimizer state, independent DataLoader
generators, and inability of one member update to change another.

Controlled tensors verify physical-unit means, sample spread with ddof=1, zero
spread for identical predictions, positive spread for distinct predictions,
RMS score arithmetic/floor activation, member-specific feedback, and fixed
coefficient channels. Tests forbid normalization refitting and verify train/validation
roles and full-budget selection. Technical gate failure blocks full training.

Final regression result: **182 passed in 2.90s**.
Historical verification: **329 files unchanged**;
pyproject.toml differs only by the exact package-discovery registration. The
protected snapshot covers tracked Stage 1–7 sources/docs/results plus historical
local scientific artifacts. No historical source/test behavior was changed.

### Training-only gate measurements

The unchanged gate requires final/initial normalized MSE <= 0.05 and final
physical relative L2 <= 0.05, checked every 25 updates up to 300. Member 0's
measurements are historical; only gates 1–4 were run in Stage 8A.

| Member | Gate updates | Final/initial MSE | Final physical relative L2 | Result |
| --- | ---: | ---: | ---: | --- |
| 0 | 150 | 0.002811 | 0.046422 | Pass |
| 1 | 150 | 0.003025 | 0.047036 | Pass |
| 2 | 125 | 0.003154 | 0.049225 | Pass |
| 3 | 150 | 0.001802 | 0.035909 | Pass |
| 4 | 150 | 0.002565 | 0.044732 | Pass |

There were finite late-epoch loss/validation fluctuations, including in members
2 and 4. These did not trigger early termination or seed replacement; their
validation-selected checkpoints passed the same technical checks as all others.

## Prediction tensors and raw spread

At one step, the physical-unit member stack is `(5,B,1,N,N)` and mean/spread are
`(B,1,N,N)`. The existing frozen inverse normalization is applied to each model
output before float64 ensemble reductions. For physical predictions u_m:

```text
ensemble_mean = sum_m u_m / M
ensemble_spread = sqrt(sum_m (u_m - ensemble_mean)^2 / (M - 1))
```

The implementation uses an algebraically equivalent shifted calculation to avoid
roundoff spread for exactly identical predictions. It uses the sample-standard-
deviation convention (ddof=1): sample variance is unbiased under independent
sampling assumptions, but the square root is not itself exactly unbiased.

**Raw ensemble spread is NOT calibrated.** It is an epistemic uncertainty proxy,
not a confidence interval, calibrated interval, or posterior standard deviation.
No Gaussian predictive-distribution assumption is made.

## Scalar spread scores

For each trajectory and horizon:

```text
U_rms = sqrt(mean_x,y(spread**2))
epsilon_rms = 1e-12 * frozen_field_std
U_rel = U_rms / max(sqrt(mean_x,y(ensemble_mean**2)), epsilon_rms)
```

U_rel is the primary dimensionless score; U_rms is its physical-unit companion.
The target is never used in the uncertainty denominator. A boolean floor flag is
recorded. Epsilon is **2.355322120279037e-13** and there were
**0** floor activations in the basic rollout check.
Scores are defined in RMS form so the floor does not depend on grid resolution.
No formal uncertainty/error association analysis is performed in Stage 8A.

## Basic historical-ID rollout check

Evaluation used only the frozen original ID test IDs after all five member
manifests were complete. This is a technical check on a historically inspected
evaluation set, not a pristine prospective benchmark.

Each model calls the unchanged Stage 4 rollout function separately and feeds back
its own previous prediction. Coefficient channels stay fixed. The ensemble mean
is never fed back into all models. All initial states equal the numerical initial
condition, giving exactly zero mean prediction error and spread at time zero.

Observed shapes:

- One-step member predictions: `[5, 8, 1, 64, 64]`.
- Member rollouts: `[5, 8, 11, 1, 64, 64]`.
- Rollout mean/spread: `[8, 11, 1, 64, 64]`.

The 11 states represent the initial condition plus ten 0.1 steps through t=1.
The first rollout prediction exactly matches the separately computed one-step
member prediction. All five members were finite; no member was dropped.

The following numbers are means over the eight test trajectories, provided as
raw technical diagnostics rather than evidence of error awareness:

| Step | Time | Mean U_rms | Mean U_rel | Ensemble-mean physical relative L2 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.0 | 0 | 0 | 0 |
| 1 | 0.1 | 0.005663967 | 0.018238869 | 0.024974678 |
| 5 | 0.5 | 0.021631619 | 0.071371273 | 0.099606738 |
| 10 | 1.0 | 0.037978995 | 0.13095184 | 0.17121562 |

The full per-trajectory/per-horizon arrays remain in ignored artifacts. No
correlation coefficient, significance test, calibration fit, or OOD conclusion
is inferred from this small table.

## Provenance and artifact locations

Each member manifest records ID, seeds, initial parameter hash, selected checkpoint
hash/epoch, full budget, optimizer, architecture, exact splits, normalization
statistics/hash, training archive hash, validation metrics, runtimes, versions,
source revision and source-file hashes, and parameter counts. For new members,
source revision remains the pre-UQ commit because source changes are uncommitted;
SHA-256 maps identify the exact new source. The ensemble manifest hashes all five
member manifests. Strict loading checks checkpoint/protocol/selection/gate hashes,
frozen metadata, full budgets, and finite weights.

- `runs/stage8/member_0/`: registration and copied historical metadata;
  checkpoint remains `runs/stage6/best.pt`.
- `runs/stage8/member_1/` through `member_4/`: protocol, gate, 60-epoch history,
  selected checkpoint, selection record, and completed manifest.
- `runs/stage8/ensemble_manifest.json`: complete member-manifest hashes and timing.
- `runs/stage8/id_check/raw_predictions.npz`: physical one-step/rollout members,
  means, sample spreads, scores, floor flags, IDs, times, coefficients, and mean errors.
- `runs/stage8/id_check/summary.json`: compact technical diagnostics.
- `runs/stage8/sanity*`, `final_pytest.txt`, `protected_before.json`, and
  `protected_verification.json`: local verification records.

The existing `/runs/` and `/data/` ignore rules protect all generated artifacts.
No calibration archive was generated under `data/stage8/` or elsewhere.

### Member 0

Initial parameter hash:
```text
6569c2e401830b7eb13ddd91e1f118005b4863f67a188952f7b5e83fde8a4884
```
Selected checkpoint SHA-256:
```text
cf8899453cadf2443961fd65daa736269ed671dd26a375fd3edc1159ae56ca33
```

### Member 1

Initial parameter hash:
```text
6a1272b02d77c4ef76c3c9161a06d53351401faa3fc3c035c9153c0c4ace2cc9
```
Selected checkpoint SHA-256:
```text
591ec6819777909579bdc9c4f57ba62f25d91b709609c708c4afef30b78b1903
```

### Member 2

Initial parameter hash:
```text
9af41723a0cf56c6650433e8a7f86af97cbb43333dcd5f6f32d640c1b293c0df
```
Selected checkpoint SHA-256:
```text
2f15a50f9a438a3db148d420eede5e15bf25f182ea3fd0682d3be50822253f79
```

### Member 3

Initial parameter hash:
```text
54263200a0fcfc17e1a0e91edafa9c87b5dd627309c1b41c246c3a18ead19182
```
Selected checkpoint SHA-256:
```text
ab95d65c8d340b550ee578c0bcf939df960e1d7a3f9780d2fc42f1434f72b5fd
```

### Member 4

Initial parameter hash:
```text
83e8dfe196cb8391bd1fa25d60299b153d0de77e4b3c275a1c3f7c9a572d23cd
```
Selected checkpoint SHA-256:
```text
ee49b65ca83df821c8703d97009ee82283d7bcb864218e5acac5a48925714a46
```

## Limitations and scientific/statistical guardrails

Five members provide a small, correlated sample of optimization outcomes. Shared
architecture/data/normalization create common biases; the models may be confidently
wrong. Distinct seeds do not guarantee independent errors. Reusing D preserves
its valid validation-only selection but does not undo the historical exposure
of the research design to reported evaluation outcomes.

No test/OOD data may fit future calibration. Historical validation already served
checkpoint selection and is not automatically a clean calibration set. Spatial
cells are not independent statistical replicates, and repeated horizons within a
trajectory require grouped analysis. Spatial maps are secondary diagnostics.
Nonfinite member predictions raise a technical failure rather than silently
shrinking the ensemble. Scientifically weak but finite members remain included.
Failure of spread to track error is a valid outcome; all predeclared OOD regimes
must eventually be reported, including unfavorable results.

Current ground truth ends at t=1. Later 16/32/64/128-step studies would need new
numerical references through t=1.6/3.2/6.4/12.8 with numerical verification. None
were generated. Finite-difference reference error and shared amplitude effects
can influence later uncertainty/error analysis.

**No formal error-awareness conclusion is made yet.**

## NEXT STEPS

A. The five-member ensemble is technically valid enough to freeze: all declared
members completed their protocols, strict artifact checks passed, and basic ID
predictions/rollouts were finite with correct tensor and spread behavior. This is
technical validity, not proof of reliable uncertainty estimates.

B. Freeze/reuse all five selected checkpoint hashes, member and ensemble manifests,
initial hashes, seed table, exact split/normalization metadata, source hashes,
and raw ID outputs. Do not retrain, replace, or reseed members based on later errors.

C. No observed member requires investigation for technical failure. Different
validation performance alone is not grounds for removal.

D. Stage 8B asks: **Does raw ensemble spread contain useful information about actual
prediction error?**

E. Predeclare trajectory/horizon-level Pearson association and Spearman rank
association, uncertainty binning, and high-error/high-uncertainty discrimination.
Analyze within horizon and use trajectory-grouped resampling for repeated horizons;
exclude the trivial initial state. Define binning/threshold rules before examining
favorable results. Spatial uncertainty/error maps are secondary diagnostics.
Treat historical ID results as exploratory rather than a new untouched benchmark.

F. Do not begin calibration or confidence-interval claims, long-horizon data
generation, an OOD conclusion before Stage 8E, or a Bayesian neural-operator extension.
The new calibration-only archive remains reserved for Stage 8C.

G. No roadmap change is justified by these technical checks. Proceed to Stage 8B
only after review; stages 8C–8E retain their planned separate roles. Stage 8B was
not started automatically.
