"""Burst detection and per-burst feature extraction.

This package replaces the pre-refactor ``mxtreme.burst`` module and the ``mxtreme.processing`` burst
drivers. It is experiment-agnostic: detection consumes a :class:`Recording <mxtreme.recording.Recording>`
plus :class:`~mxtreme.params.BurstDetectParams`, and phase labels come from an injected
:class:`~mxtreme.phases.Phases` (default: a single ``"full"`` phase).

Typical use::

    from mxtreme.recording import Recording
    from mxtreme.bursting import BurstDetector
    from mxtreme.params import BurstDetectParams, BurstFeatureParams

    rec = Recording(0, exp_data)                       # optionally Recording(..., phases=...)
    bursts = BurstDetector(BurstDetectParams()).detect(rec)
    bursts.extract_features(rec, BurstFeatureParams())
    df = bursts.to_dataframe()
"""

from mxtreme.bursting.burst import Burst
from mxtreme.bursting.detection import BurstDetector, BurstSet

__all__ = ["Burst", "BurstDetector", "BurstSet"]
