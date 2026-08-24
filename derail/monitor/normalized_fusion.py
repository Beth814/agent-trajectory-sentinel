"""Fuse per-channel detectors on a common healthy-null scale.

`ChannelMaxESNMonitor` takes the max of each channel's CUSUM state. Those
states are not on the same scale -- measured on the framework corpora the
semantic channel wins 93-96% of steps and the fused threshold is identical to
the semantic channel's own, so the metadata channel is silenced by
construction rather than by being uninformative. Two changes here, and they
are separate:

  common scale   each channel's causal stream is mapped through the location
                 and dispersion of THAT channel's own healthy null, so the
                 max compares like with like;
  fused          the max of two calibrated channels is not itself calibrated.
  calibration    Taking a max changes the null, and the channels are
                 correlated, so a 5% budget per channel does not give 5%
                 fused. The threshold is therefore picked on the FUSED
                 statistic, from healthy episodes that set neither the fit
                 nor the scales.

Everything is causal and label-free. The scales and the threshold are frozen
before serving; at serving time step t uses only x_0..x_t. No failure label,
onset or class reaches any of it -- `fit`, `fit_channel_scales` and
`pick_fused_threshold` all take healthy episodes only.

Three per-channel statistics are selectable because they carry different
length behaviour, which is the other half of the problem: `cusum` is the
running CUSUM the shipped monitor uses and grows with the number of scored
steps; `rate` divides it by the steps scored so far; `increment` drops
accumulation and uses the step's own evidence. All three are computable at
step t from x_0..x_t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from derail.common import Episode, Standardizer
from derail.evaluation.metrics import pick_threshold
from derail.monitor.esn import _WASHOUT, ESNEnsembleMonitor, OnlineMonitor

#: Per-channel causal statistics this monitor can fuse.
STATISTICS = ("cusum", "rate", "increment")

#: Dispersion floor. A channel whose healthy null is a single repeated value
#: has no scale to divide by; without this it would explode to infinity and
#: win every fusion for a reason that has nothing to do with evidence.
_MIN_DISPERSION = 1e-9


@dataclass(frozen=True)
class ChannelScale:
    """Location and dispersion of one channel's healthy null.

    Robust statistics rather than mean and standard deviation: these streams
    are heavy-tailed by construction -- a CUSUM's whole purpose is to grow
    without bound when something is wrong -- and a mean would be set by the
    tail it is supposed to measure against.
    """

    median: float
    dispersion: float
    n_steps: int

    def to_common_scale(self, value: float) -> float:
        return (value - self.median) / max(self.dispersion, _MIN_DISPERSION)


def scoreable(episode: Episode) -> bool:
    """Does the detector get at least one scoring opportunity here?

    An episode shorter than the washout produces no score at any step. It
    must not enter a healthy null as a run of zeros: those are the absence of
    a measurement, not evidence of quiet healthy behaviour, and they drag a
    threshold quantile toward zero.
    """
    return episode.T > _WASHOUT


class NormalizedFusionMonitor(OnlineMonitor):
    """Per-channel ESN-CUSUM detectors fused on a shared healthy-null scale."""

    def __init__(self, standardizer: Standardizer, *,
                 channels: tuple[str, ...] = ("e", "m"),
                 statistic: str = "rate", K: int = 8, seed: int = 0,
                 name: str | None = None, **esn_kwargs) -> None:
        if statistic not in STATISTICS:
            raise ValueError(f"statistic must be one of {STATISTICS}, "
                             f"got {statistic!r}")
        self.name = name or f"normalized_fusion[{'+'.join(channels)}]/{statistic}"
        self.channels = tuple(channels)
        self.statistic = statistic
        self.subs = [
            ESNEnsembleMonitor(standardizer, channels=(c,), K=K, cusum=True,
                               seed=seed * 100 + i, **esn_kwargs)
            for i, c in enumerate(self.channels)
        ]
        self.scales: list[ChannelScale] | None = None
        self._i = 0
        self._last_channel_values: list[float] = []

    # ------------------------------------------------------------- fitting

    def fit(self, healthy_episodes: list[Episode]) -> None:
        """Fit each channel's reservoir and readout. Healthy episodes only."""
        for sub in self.subs:
            sub.fit(healthy_episodes)

    def fit_channel_scales(self, healthy_episodes: list[Episode]) -> None:
        """Learn each channel's healthy null, on episodes the fit never saw.

        Fitting the scales on the same episodes that fit the readout would
        measure the residual the readout has already minimised, and every
        channel would look equally quiet.
        """
        usable = [ep for ep in healthy_episodes if scoreable(ep)]
        if not usable:
            raise ValueError(
                "no scoreable healthy episode to learn channel scales from: "
                f"every episode is shorter than the washout ({_WASHOUT})")
        per_channel: list[list[float]] = [[] for _ in self.subs]
        for ep in usable:
            for index, values in enumerate(self._raw_channel_streams(ep)):
                per_channel[index].extend(values[_WASHOUT:])
        scales = []
        for values in per_channel:
            arr = np.asarray(values, dtype=float)
            q25, q50, q75 = np.quantile(arr, [0.25, 0.50, 0.75])
            scales.append(ChannelScale(median=float(q50),
                                       dispersion=float(q75 - q25),
                                       n_steps=int(arr.size)))
        self.scales = scales

    # ------------------------------------------------------------ scoring

    def start_episode(self) -> None:
        for sub in self.subs:
            sub.start_episode()
        self._i = 0
        self._last_channel_values = []

    def score_step(self, x_t: np.ndarray) -> float:
        if self.scales is None:
            raise RuntimeError(
                f"{self.name}: fit_channel_scales() must be called before "
                f"scoring -- fusing unnormalized channels is the defect this "
                f"monitor exists to fix")
        scored_so_far = max(0, self._i - _WASHOUT + 1)
        values = []
        for sub, scale in zip(self.subs, self.scales):
            cusum = sub.score_step(x_t)
            if self.statistic == "cusum":
                raw = cusum
            elif self.statistic == "rate":
                raw = cusum / max(scored_so_far, 1)
            else:
                raw = sub.last_raw_increment
            values.append(scale.to_common_scale(raw))
        self._i += 1
        self._last_channel_values = values
        return max(values)

    def score_stream(self, episode: Episode) -> np.ndarray:
        self.start_episode()
        return np.array([self.score_step(episode.X[t]) for t in range(episode.T)])

    @property
    def last_winning_channel(self) -> str | None:
        """Which channel the most recent fusion took its value from."""
        if not self._last_channel_values:
            return None
        return self.channels[int(np.argmax(self._last_channel_values))]

    # ----------------------------------------------------------- internals

    def _raw_channel_streams(self, episode: Episode) -> list[np.ndarray]:
        """Per-channel causal stream, before normalization.

        Runs the real scoring path rather than a reconstruction of it, so the
        scales are learned from exactly the quantity that will be normalized.
        """
        for sub in self.subs:
            sub.start_episode()
        out = [np.zeros(episode.T) for _ in self.subs]
        for t in range(episode.T):
            scored_so_far = max(0, t - _WASHOUT + 1)
            for index, sub in enumerate(self.subs):
                cusum = sub.score_step(episode.X[t])
                if self.statistic == "cusum":
                    out[index][t] = cusum
                elif self.statistic == "rate":
                    out[index][t] = cusum / max(scored_so_far, 1)
                else:
                    out[index][t] = sub.last_raw_increment
        return out


def pick_fused_threshold(monitor: NormalizedFusionMonitor,
                         healthy_episodes: list[Episode], *,
                         fa_budget: float = 0.05) -> tuple[float, int, int]:
    """Threshold for the FUSED statistic, from healthy episodes only.

    Returns (theta, n_used, n_skipped_unscoreable). The skipped count is
    returned rather than swallowed: an episode the detector cannot score is
    not a quiet healthy episode, and silently folding it in as a zero is how
    a threshold ends up set by a distribution that is mostly absence of
    measurement.
    """
    streams, skipped = [], 0
    for ep in healthy_episodes:
        if not scoreable(ep):
            skipped += 1
            continue
        streams.append(monitor.score_stream(ep))
    if not streams:
        raise ValueError("no scoreable healthy episode to threshold on")
    theta = float(pick_threshold(streams, fa_budget=fa_budget,
                                 warn_infeasible=False))
    if not np.isfinite(theta):
        raise ValueError(f"degenerate fused threshold ({theta!r})")
    return theta, len(streams), skipped


if __name__ == "__main__":
    from derail.telemetry.adapter import episode_from_trace

    def _steps(seed: int, n: int = 10) -> list[dict]:
        rng = np.random.default_rng(seed)
        return [{"text": f"step {t} {rng.integers(0, 1000)}",
                 "token_logprobs": [-0.1] * 10, "action": "tool_call",
                 "latency_s": float(rng.uniform(0.3, 0.8)), "output_tokens": 10,
                 "error": False} for t in range(n)]

    healthy = [episode_from_trace(_steps(i), f"h{i}",
                                  use_sentence_transformers=False, extended=True)
               for i in range(60)]
    monitor = NormalizedFusionMonitor(Standardizer().fit(healthy[:40]),
                                      channels=("e", "m"), statistic="rate")
    monitor.fit(healthy[:40])
    monitor.fit_channel_scales(healthy[40:50])
    theta, used, skipped = pick_fused_threshold(monitor, healthy[50:])
    assert used == 10 and skipped == 0, (used, skipped)
    assert np.isfinite(theta)
    stream = monitor.score_stream(healthy[50])
    assert stream.shape == (healthy[50].T,)
    assert monitor.last_winning_channel in ("e", "m")
    short = episode_from_trace(_steps(999, n=2), "short",
                               use_sentence_transformers=False, extended=True)
    assert not scoreable(short)
    print("PASS normalized_fusion.py smoke test")
