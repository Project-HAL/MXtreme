"""Parameter dataclasses for the burst and activity stages of the pipeline.

These replace the loose module-level constants that used to live in ``mxtreme.constants``. Keeping the
knobs in small frozen dataclasses makes a run's settings a single object you can pass around, log
alongside the outputs (see :meth:`mxtreme.bursting.detection.BurstSet`), and default sensibly without
reaching into lab-specific globals.

Defaults mirror the historical ``constants.py`` values, so behaviour is unchanged unless you override.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BurstDetectParams:
    """Settings for the ``"isi_rate"`` burst-detection method.

    The method first groups spikes into candidate bursts with an ISI-N threshold, then picks peaks
    within each group by array-wide rate. Peaks at or above ``burst_thresh`` are ``"network"`` bursts;
    peaks between ``noise_thresh`` and ``burst_thresh`` are ``"mini"`` bursts.

    :param n: Number of spikes ``N`` in the ISI-N window (minimum spikes comprising a burst).
    :param noise_thresh: Lower rate bound (fraction of channels active) below which peaks are ignored.
    :param burst_thresh: Rate (fraction of channels active) separating ``mini`` from ``network`` peaks.
    :param min_dist_bins: Minimum separation between detected peaks, in bins.
    """

    n: int = 300
    noise_thresh: float = 0.05
    burst_thresh: float = 0.15
    min_dist_bins: int = 30


@dataclass(frozen=True)
class BurstFeatureParams:
    """Settings for per-burst feature extraction.

    :param onset_thresh_pct: Fraction of the 0->peak spike count that marks burst onset.
    :param offset_thresh_pct: Fraction of the peak->0 spike count that marks burst offset.
    """

    onset_thresh_pct: float = 0.1
    offset_thresh_pct: float = 0.05


@dataclass(frozen=True)
class ActivityParams:
    """Settings for the per-channel activity metrics in :mod:`mxtreme.analysis.activity`.

    :param isi_threshold_ms: Upper bound on inter-spike intervals, in milliseconds. Intervals at or
        above this are dropped before taking a channel's median ISI, replicating the MaxLab ISI
        analysis -- without it, the long gaps between bursts dominate the median.
    """

    isi_threshold_ms: float = 200.0
