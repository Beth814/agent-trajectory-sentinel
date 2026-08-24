"""Keep the pre-fix framework baseline independently executable.

The pre-fix result -- LangGraph detection 0.625 at 0.000 healthy false
alarms, AutoGen 0.913 at 0.125, and the zero-shot arm beside them -- is the
control every later E1 number is judged against. It has to stay runnable from
the tree, not merely protected by a recorded hash, and it must never be
satisfied by new detector logic wearing the old name.

Two things make that true and this tool checks both:

  * the freeze artifact reconstructs its monitor from ITS OWN recorded
    hyperparameters, so a later change to an `ESNEnsembleMonitor` default
    cannot reinterpret it (`load_frozen_monitor`);
  * every code file that can move either arm's numbers is pinned here by git
    blob hash, and a changed revision is refused unless it has been PROVEN to
    reconstruct the byte-identical monitor.

`--prove` is the only way to widen that allowlist, and it fails unless the
reconstruction digest matches. There is no flag to skip the check.

    py -m devtools.baseline_lock --write     # record the lock (once)
    py -m devtools.baseline_lock --check     # gate: still reproducible?
    py -m devtools.baseline_lock --prove     # accept a proven-equal revision
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = REPO_ROOT / "results" / "framework_monitor_freeze.json"
LOCK_PATH = REPO_ROOT / "results" / "framework_monitor_freeze_compat.json"

#: Everything that can change either published arm. The first five are the
#: freeze's own dependency chain; the rest are what the CALIBRATED arm adds,
#: which the freeze artifact never covered because it describes one monitor
#: rather than the study that used it.
DEPENDENCY_FILES = (
    "derail/monitor/esn.py",
    "derail/common.py",
    "derail/telemetry/adapter.py",
    "derail/telemetry/events.py",
    "derail/evaluation/metrics.py",
    "derail/monitor/deployment_calibration.py",
    "derail/experiments/framework_monitor_freeze.py",
    "derail/experiments/run_framework_real_tool_analysis.py",
    "derail/experiments/run_framework_generalized_monitor_eval.py",
)

#: The published artifacts this baseline consists of. Named so the lock says
#: what it is protecting, and so a reader can find the numbers.
PUBLISHED = (
    "results/framework_real_tool_report.md",
    "results/framework_generalized_monitor_report.md",
    "results/tables/framework_generalized_monitor_summary.csv",
    "results/tables/framework_generalized_monitor_alarms.csv",
    "results/tables/framework_real_tool_alarms.csv",
)

#: Corpora the digest scores. The zero-shot arm's own scoring targets, so the
#: digest covers the monitor as it was actually used, not a synthetic probe.
DIGEST_CORPORA = ("langgraph7b_real", "autogen7b_real")

#: Same quantisation the freeze uses on feature arrays, for the same reason:
#: these scores run through log and ridge solves, whose last place differs
#: between platform libm builds, and the digest has to travel off the machine
#: that recorded it.
DIGEST_DP = 6

MONITOR_VERSION = "esn_cusum_max[e,m]/v1-prefix"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                          text=True, timeout=10).stdout.strip()


def _blob(path: str) -> str:
    return _git("rev-parse", f"HEAD:{path}")


def _dirty(paths: tuple[str, ...]) -> list[str]:
    """Dependency files with uncommitted edits.

    Blob hashes name what is COMMITTED. Without this a working-tree edit to a
    dependency would run against the lock while the lock still described the
    committed revision -- the check would pass on code nobody has recorded.
    """
    out = _git("status", "--porcelain", "--", *paths)
    return sorted({line[3:].strip().strip('"') for line in out.splitlines()
                   if line.strip()})


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def reconstruction_digest() -> str:
    """SHA-256 over the frozen monitor's own per-step scores.

    Stronger than any source hash: it is the monitor's behaviour on the
    episodes it was published against. Two trees agreeing here reconstruct
    the same detector whatever their source looks like.
    """
    from derail.experiments.framework_monitor_freeze import load_frozen_monitor
    from derail.telemetry.adapter import episode_from_trace

    monitor, _, _ = load_frozen_monitor()
    h = hashlib.sha256()
    for corpus in DIGEST_CORPORA:
        corpus_dir = REPO_ROOT / "traces" / corpus
        manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
        for entry in sorted(manifest, key=lambda e: e["episode_id"]):
            steps = [json.loads(line) for line in
                     (corpus_dir / entry["file"]).read_text("utf-8").splitlines()
                     if line.strip()]
            ep = episode_from_trace(steps, entry["episode_id"],
                                    use_sentence_transformers=False,
                                    extended=True)
            monitor.start_episode()
            scores = np.array([monitor.score_step(ep.X[t])
                               for t in range(ep.X.shape[0])])
            h.update(entry["episode_id"].encode("utf-8"))
            h.update(np.ascontiguousarray(
                np.round(scores, DIGEST_DP) + 0.0).tobytes())
    return h.hexdigest()


def build_lock() -> dict:
    dirty = _dirty(DEPENDENCY_FILES)
    if dirty:
        raise SystemExit(
            f"[baseline-lock] REFUSED: {', '.join(dirty)} have uncommitted "
            f"edits. A lock records committed blob hashes, so writing one now "
            f"would pin revisions that are not what this tree runs.")
    return {
        "purpose": "Keeps the pre-fix framework baseline executable and "
                   "unambiguous while the detector evolves. The freeze "
                   "artifact itself is immutable; this file records what has "
                   "been proven equivalent to it.",
        "monitor_version": MONITOR_VERSION,
        "freeze_artifact": FREEZE_PATH.relative_to(REPO_ROOT).as_posix(),
        "freeze_sha256": _sha256_file(FREEZE_PATH),
        "published_artifacts": list(PUBLISHED),
        "baseline_commit": _git("rev-parse", "HEAD"),
        "dependency_blob_hashes": {p: _blob(p) for p in DEPENDENCY_FILES},
        "compatible_dependency_blob_hashes": {},
        "reconstruction_digest": reconstruction_digest(),
        "digest_corpora": list(DIGEST_CORPORA),
        "digest_decimals": DIGEST_DP,
    }


def check(lock: dict) -> list[str]:
    """Everything wrong with the current tree, as a list of complaints."""
    problems = []
    if lock["freeze_sha256"] != _sha256_file(FREEZE_PATH):
        problems.append(
            f"{FREEZE_PATH.name} has changed since the lock was written; the "
            f"freeze is meant to be immutable")
    for path in _dirty(DEPENDENCY_FILES):
        problems.append(
            f"{path} has uncommitted edits, so no blob hash describes what "
            f"this tree would actually run")
    allowed = lock.get("compatible_dependency_blob_hashes", {})
    for path, pinned in lock["dependency_blob_hashes"].items():
        current = _blob(path)
        if current == pinned or current in allowed.get(path, ()):
            continue
        problems.append(
            f"{path} is at {current or '(untracked)'}, which is neither the "
            f"baseline revision {pinned} nor a proven-equivalent one -- run "
            f"--prove")
    if problems:
        return problems                    # a digest now would just re-raise
    digest = reconstruction_digest()
    if digest != lock["reconstruction_digest"]:
        problems.append(
            f"the frozen monitor no longer reproduces its own scores "
            f"({lock['reconstruction_digest'][:12]}... -> {digest[:12]}...); "
            f"the pre-fix baseline is NOT reproducible from this tree")
    return problems


def prove(lock: dict) -> dict:
    """Record the current revisions as equivalent, if they demonstrably are."""
    dirty = _dirty(DEPENDENCY_FILES)
    if dirty:
        raise SystemExit(
            f"[baseline-lock] REFUSED: {', '.join(dirty)} have uncommitted "
            f"edits; commit them before proving, or the proof records a blob "
            f"hash for code that was never run.")
    digest = reconstruction_digest()
    if digest != lock["reconstruction_digest"]:
        raise SystemExit(
            f"[baseline-lock] REFUSED: this tree reconstructs a different "
            f"monitor ({digest[:12]}... vs the locked "
            f"{lock['reconstruction_digest'][:12]}...). A revision that "
            f"changes the frozen monitor's behaviour cannot be recorded as "
            f"compatible with it -- give the new detector its own freeze.")
    allowed = lock.setdefault("compatible_dependency_blob_hashes", {})
    added = []
    for path, pinned in lock["dependency_blob_hashes"].items():
        current = _blob(path)
        if current and current != pinned and current not in allowed.get(path, ()):
            allowed.setdefault(path, []).append(current)
            added.append(f"{path} -> {current[:12]}")
    lock["proofs"] = lock.get("proofs", []) + [{
        "commit": _git("rev-parse", "HEAD"),
        "digest": digest,
        "added": added,
    }]
    return lock


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--prove", action="store_true")
    args = ap.parse_args(argv)

    if args.write:
        if LOCK_PATH.exists():
            raise SystemExit(
                f"[baseline-lock] {LOCK_PATH.name} already exists -- it "
                f"records a baseline that has already been established. Use "
                f"--prove to widen it, never --write to replace it.")
        LOCK_PATH.write_text(json.dumps(build_lock(), indent=2) + "\n", "utf-8")
        print(f"[baseline-lock] wrote {LOCK_PATH}")
        return

    lock = json.loads(LOCK_PATH.read_text("utf-8"))
    if args.prove:
        LOCK_PATH.write_text(json.dumps(prove(lock), indent=2) + "\n", "utf-8")
        print(f"[baseline-lock] proven equivalent; updated {LOCK_PATH}")
        return

    problems = check(lock)
    for problem in problems:
        print(f"[baseline-lock] {problem}")
    if problems:
        raise SystemExit(1)
    print(f"[baseline-lock] pre-fix baseline {lock['monitor_version']} "
          f"reproduces: digest {lock['reconstruction_digest'][:12]}... over "
          f"{len(lock['digest_corpora'])} corpora, "
          f"{len(lock['dependency_blob_hashes'])} dependencies pinned")


if __name__ == "__main__":
    main()
