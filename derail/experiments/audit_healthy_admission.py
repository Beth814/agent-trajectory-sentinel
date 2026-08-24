"""What actually got into the healthy null every threshold is derived from?

A calibration population has to be admitted by a rule fixed in advance, and
the rule has to be reported with its numbers: how many runs were attempted,
how many admitted, how many rejected and why, and what tasks and shapes the
survivors cover. Otherwise a threshold is a statement about a hand-shaped
sample rather than about the deployment.

The framework corpora admit healthy runs by an OUTCOME-INDEPENDENT rule
(`collect_framework_real_traces.check_admission`), which is the right
default: it cannot quietly drop a difficult-but-valid healthy run for
scoring badly. What it also does is never ask the task verifier, so every
healthy episode carries `success = None` and nothing in the corpus says
whether the agent finished its task.

Both tasks in these corpora ship a `success_fn`, and it reads only the
recorded steps and the final response. This module runs it offline over the
committed traces, so the composition of the healthy null is reported rather
than assumed. It changes no label and admits or rejects nothing: the corpus
is what it is, and this says what is in it.

Writes:
  results/tables/healthy_admission_audit.csv
  results/healthy_admission_audit.md
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from derail.harness.tasks import REAL_TASKS

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES_ROOT = REPO_ROOT / "traces"
OUT_CSV = REPO_ROOT / "results" / "tables" / "healthy_admission_audit.csv"
OUT_REPORT = REPO_ROOT / "results" / "healthy_admission_audit.md"

CORPORA = ("langgraph7b_real2", "autogen7b_real2")
TASK_BY_NAME = {t.name: t for t in REAL_TASKS}

#: The ESN emits no score until this many steps have passed, so a healthy
#: episode shorter than this contributes nothing to the null it is counted
#: in. Imported rather than restated so the two cannot drift apart.
from derail.monitor.esn import _WASHOUT  # noqa: E402


def _final_response(steps: list[dict]) -> str:
    """The last step's own prose -- what a verifier reads as the answer."""
    for step in reversed(steps):
        text = str(step.get("text", "") or "")
        if text.strip():
            return text
    return ""


def audit_corpus(corpus: str) -> tuple[list[dict], dict]:
    corpus_dir = TRACES_ROOT / corpus
    manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
    rejected_path = corpus_dir / "landing_failures.json"
    rejected = (json.loads(rejected_path.read_text("utf-8"))
                if rejected_path.exists() else [])

    rows = []
    for entry in manifest:
        if entry["failure_class"] is not None:
            continue
        steps = [json.loads(line) for line in
                 (corpus_dir / entry["file"]).read_text("utf-8").splitlines()
                 if line.strip()]
        task = TASK_BY_NAME[entry["task"]]
        verified = task.verify(_final_response(steps), steps)
        rows.append({
            "dataset": corpus, "episode_id": entry["episode_id"],
            "task": entry["task"], "model": entry["model"],
            "T": len(steps),
            "admitted_because": entry.get("accepted_because"),
            "success_recorded_at_collection": entry.get("success"),
            "success_reverified_offline": verified,
            "scores_any_step": len(steps) > _WASHOUT,
        })

    summary = {
        "corpus": corpus,
        "healthy_admitted": len(rows),
        "healthy_rejected": sum(1 for r in rejected
                                if r.get("failure_class") is None),
        "injected_rejected": sum(1 for r in rejected
                                 if r.get("failure_class") is not None),
        "rejection_reasons": Counter(r.get("reason") for r in rejected),
    }
    return rows, summary


def main() -> None:
    all_rows, summaries = [], []
    for corpus in CORPORA:
        rows, summary = audit_corpus(corpus)
        all_rows += rows
        summaries.append(summary)

    df = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    lines = ["# What is in the healthy null",
             "",
             "**SUPPORTING / diagnostic evidence.** Nothing here relabels, "
             "admits or rejects an episode. It reports the composition of the "
             "healthy populations every per-deployment threshold is derived "
             "from, by re-running each task's own `success_fn` over the "
             "committed traces.",
             "",
             "Healthy admission in these corpora is outcome-independent "
             "(`collect_framework_real_traces.check_admission`): a healthy run "
             "is admitted for existing, never for scoring well. That is the "
             "right rule -- it cannot silently drop difficult-but-valid "
             "behaviour -- but it also never consults the verifier, so the "
             "corpus records `success = None` for every healthy episode and "
             "says nothing about whether the agent finished the task.",
             ""]
    for summary in summaries:
        sub = df[df["dataset"] == summary["corpus"]]
        verified = sub["success_reverified_offline"]
        lines += [f"## {summary['corpus']}", "",
                  f"- healthy attempted: "
                  f"{summary['healthy_admitted'] + summary['healthy_rejected']}"
                  f"; admitted **{summary['healthy_admitted']}**; rejected "
                  f"**{summary['healthy_rejected']}**",
                  f"- injected rejected at admission (for contrast): "
                  f"{summary['injected_rejected']}",
                  f"- admission reason, all healthy episodes: "
                  f"`{sub['admitted_because'].iloc[0]}`",
                  f"- task mix: " + ", ".join(
                      f"{k} {v}" for k, v in
                      sorted(Counter(sub['task']).items())),
                  f"- model: " + ", ".join(sorted(set(sub["model"]))),
                  f"- trajectory length: min {sub['T'].min()}, median "
                  f"{int(sub['T'].median())}, max {sub['T'].max()}",
                  f"- **verifier re-run offline: {int((verified == True).sum())} "  # noqa: E712
                  f"pass, {int((verified == False).sum())} fail, "  # noqa: E712
                  f"{int(verified.isna().sum())} unverifiable**",
                  f"- episodes too short to produce any score "
                  f"(T <= washout {_WASHOUT}): "
                  f"**{int((~sub['scores_any_step']).sum())}**",
                  ""]
        if summary["rejection_reasons"]:
            lines.append("Rejection reasons recorded at collection:")
            lines += [f"- {reason}: {n}" for reason, n
                      in summary["rejection_reasons"].most_common()]
            lines.append("")

    lines += ["## Trajectory length against verified success", "",
              "| corpus | verifier | n | T min | T median | T max | "
              "too short to score |", "|---|---|---|---|---|---|---|"]
    for corpus in CORPORA:
        sub = df[df["dataset"] == corpus]
        for label, mask in (("pass", sub["success_reverified_offline"] == True),   # noqa: E712
                            ("fail", sub["success_reverified_offline"] == False),  # noqa: E712
                            ("unverifiable",
                             sub["success_reverified_offline"].isna())):
            group = sub[mask]
            if group.empty:
                continue
            lines.append(
                f"| {corpus} | {label} | {len(group)} | {group['T'].min()} | "
                f"{int(group['T'].median())} | {group['T'].max()} | "
                f"{int((~group['scores_any_step']).sum())} |")
    lines.append("")
    OUT_REPORT.write_text("\n".join(lines), "utf-8")
    print(df.groupby(["dataset", "success_reverified_offline"],
                     dropna=False).size().to_string())
    for path in (OUT_CSV, OUT_REPORT):
        print(f"[healthy-admission] wrote {path}")


if __name__ == "__main__":
    main()
