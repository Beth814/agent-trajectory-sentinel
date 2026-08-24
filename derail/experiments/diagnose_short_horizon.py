"""Is `_WASHOUT = 3` responsible for the short-episode misses?

Diagnosis only, on development data. The washout question was closed once
before on `real_research7b` as a negative for episode AUC and lead time; this
re-opens only the specific case the framework corpora exposed -- injected
episodes with tau = 2 and T = 4-5, where a three-step washout leaves one or
two scored steps for the CUSUM to work with.

Sweeps the washout over 0..3 and, independently, the CUSUM slack k, refitting
the whole per-deployment calibration at each setting so nothing is compared
against a monitor fit under a different rule. Reports what each setting costs
in healthy false alarms as well as what it buys in detection, because a
washout of 0 also lets the reservoir's untrained transient into the healthy
null.

Writes:
  results/tables/short_horizon_sweep.csv
  results/short_horizon_diagnosis.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from derail.evaluation.metrics import first_alarm
from derail.experiments.diagnose_framework_misses import (FA_BUDGET,
                                                          FRAMEWORKS,
                                                          SHIPPED_CHANNELS,
                                                          load_corpus)
from derail.monitor import esn as esn_module
from derail.monitor.deployment_calibration import calibrate

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = REPO_ROOT / "results" / "tables" / "short_horizon_sweep.csv"
OUT_REPORT = REPO_ROOT / "results" / "short_horizon_diagnosis.md"

WASHOUTS = (0, 1, 2, 3)
#: The reference value the CUSUM subtracts from every increment. At the
#: shipped 0.5 a single scored step whose evidence is worth less than 0.5
#: leaves the statistic at exactly zero.
SLACKS = (0.0, 0.1, 0.25, 0.5)


def _score(monitor, episode) -> np.ndarray:
    monitor.start_episode()
    return np.array([monitor.score_step(episode.X[t])
                     for t in range(episode.T)])


def _evaluate(healthy: list, injected: list, *, cusum_k: float) -> dict:
    cal = calibrate(healthy, channels=SHIPPED_CHANNELS, fa_budget=FA_BUDGET,
                    seed=0, cusum_k=cusum_k)
    test = [ep for ep in healthy if ep.episode_id in cal.test_ids]
    fa = sum(1 for ep in test
             if first_alarm(_score(cal.monitor, ep), cal.theta) is not None)
    per_class: dict[str, list[int]] = {}
    detected = 0
    for ep in injected:
        alarm = first_alarm(_score(cal.monitor, ep), cal.theta)
        hit = int(alarm is not None and alarm >= ep.tau)
        detected += hit
        per_class.setdefault(ep.failure_class, []).append(hit)
    silent = sum(1 for ep in injected
                 if _score(cal.monitor, ep).max() == 0.0)
    return {"theta": cal.theta, "n_test": len(test), "n_fa": fa,
            "fa_rate": fa / len(test), "n_injected": len(injected),
            "detection_rate": detected / len(injected),
            "n_all_zero_injected": silent,
            "per_class": {c: (sum(v), len(v)) for c, v in per_class.items()}}


def main() -> None:
    original_washout = esn_module._WASHOUT
    rows = []
    try:
        for corpus, framework in FRAMEWORKS.items():
            episodes, _ = load_corpus(corpus)
            healthy = [ep for ep in episodes if ep.is_healthy]
            injected = [ep for ep in episodes if not ep.is_healthy]
            for washout in WASHOUTS:
                for slack in SLACKS:
                    esn_module._WASHOUT = washout
                    try:
                        m = _evaluate(healthy, injected, cusum_k=slack)
                    except ValueError as exc:       # degenerate threshold
                        rows.append({"framework": framework,
                                     "washout": washout, "cusum_k": slack,
                                     "error": str(exc)[:120]})
                        continue
                    row = {"framework": framework, "washout": washout,
                           "cusum_k": slack, "theta": m["theta"],
                           "n_healthy_test": m["n_test"], "n_fa": m["n_fa"],
                           "healthy_fa_rate": m["fa_rate"],
                           "n_injected": m["n_injected"],
                           "detection_rate": m["detection_rate"],
                           "n_all_zero_injected": m["n_all_zero_injected"]}
                    for cls, (hit, n) in sorted(m["per_class"].items()):
                        row[f"det_{cls}"] = hit / n
                    rows.append(row)
                    print(f"[short-horizon] {framework} washout={washout} "
                          f"k={slack}: det={m['detection_rate']:.3f} "
                          f"fa={m['fa_rate']:.3f} theta={m['theta']:.2f}")
    finally:
        esn_module._WASHOUT = original_washout

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    lines = ["# Short-horizon diagnosis: washout and CUSUM slack",
             "",
             "**SUPPORTING / diagnostic evidence, on development data.** "
             "Every row refits the whole per-deployment calibration at that "
             "setting, on the same healthy fold, at the same 5% budget. "
             "Shipped setting is washout 3, k 0.5. Nothing here selects a "
             "value: a setting chosen because it maximises a column of this "
             "table would be tuned on the corpus it is measured on.",
             ""]
    for framework in sorted(df["framework"].unique()):
        sub = df[df["framework"] == framework]
        lines += [f"## {framework}", "",
                  "| washout | k | theta | detection | healthy FA | "
                  "injected episodes scoring 0 |", "|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            if "error" in r and isinstance(r.get("error"), str):
                lines.append(f"| {int(r['washout'])} | {r['cusum_k']} | "
                             f"refused | -- | -- | {r['error']} |")
                continue
            lines.append(
                f"| {int(r['washout'])} | {r['cusum_k']} | {r['theta']:.2f} | "
                f"{r['detection_rate']:.3f} | {r['healthy_fa_rate']:.3f} | "
                f"{int(r['n_all_zero_injected'])} |")
        lines.append("")
    OUT_REPORT.write_text("\n".join(lines), "utf-8")
    print(f"[short-horizon] wrote {OUT_CSV}")
    print(f"[short-horizon] wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
