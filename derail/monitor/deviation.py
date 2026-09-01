"""Two-sided per-dimension deviation from the healthy null, level and spread.

Two things the frozen monitor cannot see, both found by per-dim AUC on raw
telemetry:

  level    several dims separate the open classes almost perfectly in the LOWER
           tail. After a goal hijack the agent locks onto one narrow topic, so
           results become more self-similar and steps get shorter; under a tool
           cascade, error strings replace documents, so latency and output
           length drop. A one-sided statistic reads those runs as healthier
           than healthy.
  spread   on the long-drift corpus post-onset steps are not extreme at all,
           they are unusually UNIFORM. No per-step deviation can see a collapse
           in variability, so the windowed standard deviation of each dim is
           scored as a feature in its own right.

Each feature is scored against its own healthy null with a SEPARATE scale per
tail: a shared scale in a skewed dim lets the fat side set the divisor and
hides excursions into the thin one. Nothing here sees a label, a future step,
or a framework.
"""
from __future__ import annotations

import collections

import numpy as np

from derail.monitor.esn import _WASHOUT

#: Interpretable derived dims. The 32 semantic embedding dims are excluded:
#: individually they carry no meaning, so a per-dim tail on them is noise.
DERIVED_DIMS = tuple(range(32, 36)) + (40, 41, 42) + tuple(range(43, 64))
WINDOW = 3


class DeviationMonitor:
    def __init__(self, dims=DERIVED_DIMS, combiner: str = "topk",
                 statistic: str = "cusum", top_k: int = 3, alpha: float = 0.05,
                 feature: str = "both", window: int = WINDOW):
        self.dims = tuple(dims)
        self.combiner = combiner
        self.statistic = statistic
        self.top_k = top_k
        self.alpha = alpha
        self.feature = feature
        self.window = window
        self._scales: dict[tuple[str, int], tuple | None] = {}
        self._drift = 0.0
        self.start_episode()

    def _features(self, ep):
        """Per-step (level, windowed spread) for one episode. Causal."""
        buf = collections.deque(maxlen=self.window)
        levels, spreads = [], []
        for t in range(ep.T):
            row = ep.X[t, list(self.dims)]
            buf.append(row)
            levels.append(row)
            stacked = np.asarray(buf)
            spreads.append(stacked.std(axis=0) if stacked.shape[0] > 1
                           else np.zeros(len(self.dims)))
        return np.asarray(levels), np.asarray(spreads)

    def _tail_scales(self, col):
        centre = float(np.median(col))
        lo, hi = centre - col[col < centre], col[col > centre] - centre
        s_lo = float(np.quantile(lo, 1 - self.alpha)) if lo.size else 0.0
        s_hi = float(np.quantile(hi, 1 - self.alpha)) if hi.size else 0.0
        if s_lo <= 1e-9 and lo.size:
            s_lo = float(lo.max())
        if s_hi <= 1e-9 and hi.size:
            s_hi = float(hi.max())
        return None if max(s_lo, s_hi) <= 1e-9 else (centre, s_lo, s_hi)

    def fit(self, healthy) -> "DeviationMonitor":
        width = healthy[0].X.shape[1]
        self.dims = tuple(d for d in self.dims if d < width)
        lv, sp = [], []
        for ep in healthy:
            if ep.T <= _WASHOUT:
                continue
            a, b = self._features(ep)
            lv.append(a[_WASHOUT:])
            sp.append(b[_WASHOUT:])
        lv, sp = np.concatenate(lv), np.concatenate(sp)
        for i, d in enumerate(self.dims):
            self._scales[("level", d)] = self._tail_scales(lv[:, i])
            self._scales[("spread", d)] = self._tail_scales(sp[:, i])
        return self

    def fit_reference(self, healthy) -> "DeviationMonitor":
        """CUSUM drift constant: the healthy median step surprise."""
        vals = []
        for ep in healthy:
            self.start_episode()
            for t in range(ep.T):
                self._buf.append(np.asarray(ep.X[t])[list(self.dims)])
                vals.append(self._surprise())
        finite = [v for v in vals if np.isfinite(v)]
        self._drift = float(np.median(finite)) if finite else 0.0
        return self

    def start_episode(self) -> None:
        self._buf = collections.deque(maxlen=self.window)
        self._state = 0.0
        self._t = 0

    def _score_one(self, kind, d, value):
        ref = self._scales.get((kind, d))
        if ref is None:
            return None
        centre, s_lo, s_hi = ref
        delta = float(value) - centre
        scale = s_hi if delta >= 0 else s_lo
        return None if scale <= 1e-9 else abs(delta) / scale

    def _surprise(self) -> float:
        stacked = np.asarray(self._buf)
        level = stacked[-1]
        spread = stacked.std(axis=0) if stacked.shape[0] > 1 else None
        out = []
        for i, d in enumerate(self.dims):
            if self.feature in ("level", "both"):
                v = self._score_one("level", d, level[i])
                if v is not None:
                    out.append(v)
            if self.feature in ("spread", "both") and spread is not None:
                v = self._score_one("spread", d, spread[i])
                if v is not None:
                    out.append(v)
        if not out:
            return 0.0
        arr = np.asarray(out)
        if self.combiner == "max":
            return float(arr.max())
        if self.combiner == "topk":
            return float(np.sort(arr)[::-1][:min(self.top_k, arr.size)].mean())
        return float(arr.mean())

    def score_step(self, x: np.ndarray) -> float:
        self._buf.append(np.asarray(x)[list(self.dims)])
        s = self._surprise()
        self._t += 1
        if self._t <= _WASHOUT:
            return -np.inf
        if self.statistic == "cusum":
            self._state = max(0.0, self._state + s - self._drift)
            return self._state
        return s

    def episode_scores(self, ep) -> np.ndarray:
        self.start_episode()
        return np.array([self.score_step(ep.X[t]) for t in range(ep.T)])
