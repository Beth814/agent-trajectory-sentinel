# Sizing the healthy calibration fold for the conditional detector

The conditional design was held back on the grounds that its second stage is
calibrated on too few episodes: a gate opening on a quarter of healthy runs
leaves only a quarter of the calibration fold for the adoption threshold, and
in the three goal corpora that quarter is 6 to 14 episodes. Before building the
complementary design — a flat detector and a conditional `g` to `c` path
sharing one budget — the question was how many healthy episodes it would take
to calibrate that second stage properly.

The answer is that the second stage is not the expensive part, and sizing it is
not what has been blocking the design. The requirement is the same as the flat
detector's, and the flat detector does not meet it either.

## What governs the precision

Both thresholds are rank statistics. A threshold placed at the `k`-th largest
of `n` healthy calibration values fires on a fresh healthy episode with
probability `Beta(k, n + 1 - k)` — a distribution that depends on `k` and `n`
alone and not at all on how the scores are distributed. Its coefficient of
variation is about `1 / sqrt(k)`. So the precision of a false-alarm calibration
is set by **how many calibration episodes sit above the final threshold**,
not by the size of the fold underneath it.

For the flat detector that count is `0.05 x N`. For the two-stage detector it
is `p2 x (p1 x N)`, and since `p1 x p2 = 0.05` by construction, it is
`0.05 x N` as well — the same number, for every allocation of the budget
between the stages.

| allocation | n_calib | exceedances above final threshold | realized FA, 95th pct | CV |
|---|---|---|---|---|
| flat 0.05 | 320 | 16 | 0.0712 | 0.243 |
| gate 0.25 / cond 0.20 | 320 | 16 | 0.0705 | 0.243 |
| gate 0.50 / cond 0.10 | 320 | 16 | 0.0708 | 0.244 |
| gate 1.00 / cond 0.05 | 320 | 16 | 0.0710 | 0.243 |

The conditional fold is small, but the rate it has to resolve is
correspondingly coarse: a fold of 80 resolving 20% is exactly as precise as a
fold of 320 resolving 5%. Splitting the budget across two stages costs nothing
in calibration precision. The earlier reading — that the conditional stage
needed several times more healthy data than the flat one — was wrong.

## Where the corpora actually stand

| design | n_calib | exceedances | median realized FA | 95th pct | P(realized ≤ 5%) |
|---|---|---|---|---|---|
| langgraph, hierarchy 0.25/0.20 | 24 | 1 | 0.024 | 0.102 | 0.765 |
| autogen, hierarchy 0.25/0.20 | 54 | 2 | 0.029 | 0.079 | 0.794 |
| long_drift, hierarchy 0.25/0.20 | 24 | 1 | 0.024 | 0.102 | 0.766 |
| deployment, flat 0.05 (frozen) | 80 | 4 | 0.046 | 0.094 | 0.572 |

The conditional thresholds in the goal corpora are being set by the single
largest conditional value in the fold, which is not a calibration. The frozen
flat detector is better placed but not by much: its 5% threshold is the 4th
largest of 80, so the rate it actually achieves on fresh healthy episodes has a
95th percentile of 9.4%.

This is a statement about finite-sample calibration, not a relaxation of the
budget. The 5% target stands. What the table shows is how much evidence the
current folds provide that a given deployment meets it — and the honest answer
is that a nominal 5% threshold read off 80 episodes supports a claim of "no
worse than about 10% with 95% confidence", not a claim of 5%.

## What it would take

Calibrating at a nominal 5% puts the median realized rate at 5%, so half of all
deployments overshoot by construction. Holding the budget with confidence means
accepting a bounded overshoot. The fold needed for each tolerance:

| realized FA ≤ tolerance × nominal | exceedances | n_calib | 95th pct |
|---|---|---|---|
| 2.0× | 4 | 80 | 0.095 |
| 1.5× | 13 | 260 | 0.074 |
| 1.3× | 32 | 640 | 0.065 |
| 1.2× | 70 | 1400 | 0.060 |
| 1.1× | 267 | 5340 | 0.055 |

A separate and much weaker constraint is resolution: the conditional threshold
exists at all only when `floor(p2 x p1 x N) >= 1`, i.e. `N >= 20` for any
allocation. Below that the quantile helper falls back to a threshold above
every calibration value, and the realized conditional rate becomes `1/(n+1)`
regardless of the rate that was requested. All three goal corpora clear
resolution and fail precision.

## The collection cost is mostly an allocation choice

Under the split the studies use today — 30% development holdout, then 50% fit,
20% scale, 30% calibration of the remainder — only 21% of a healthy corpus
reaches the calibration fold, so every calibration episode costs about five
collected ones. That proportion is not required. The fit fold estimates a
readout and the scale fold a dispersion; both are adequate at their current
sizes and neither has any reason to grow with the false-alarm budget. Holding
them fixed and giving every further healthy episode to calibration:

| tolerance | n_calib | healthy episodes, proportional folds | healthy episodes, fixed fit + scale |
|---|---|---|---|
| 2.0× | 80 | 381 | 297 |
| 1.5× | 260 | 1239 | 477 |
| 1.3× | 640 | 3048 | 857 |
| 1.2× | 1400 | 6667 | 1617 |

The deployment corpus in hand is 297 healthy episodes split 140 fit / 40 scale
/ 80 threshold / 37 development. Reallocated under a fixed-fold design it gives
the same 80 calibration episodes it already has — the 2.0× row. Reaching 1.5×
needs 477 healthy episodes per deployment, an additional 180 on top of what
exists.

## Consequence for the complementary design

The conditional design is not blocked on a conditional-fold problem specific to
it. It is blocked on the same calibration evidence the flat detector is
blocked on, and it should be built when that evidence exists rather than
before. The gating requirement is **260 calibration episodes, or 477 healthy
episodes per deployment under a fixed fit and scale fold**, for a 1.5×
tolerance at 95% confidence.

Until then the corrected disjoint fit / scale / calibration procedure stays as
it is, the 5% total budget is unchanged, and the complementary design remains
unrun.

Derivation: `_internal/deployment_healthy_collection/size_conditional_calibration.py`;
figures in `conditional_calibration_sizing.json`.
