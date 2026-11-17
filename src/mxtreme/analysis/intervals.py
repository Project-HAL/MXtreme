"""Named intervals of a recording, cut by its bursts, for per-interval analyses.

The per-channel analyses in :mod:`mxtreme.analysis.activity` are computed *per phase*, and a
:class:`~mxtreme.phases.Phases` is nothing more than an ordered list of named frame intervals.
So "the firing rate of every channel between burst 3 and burst 4" needs no new analysis: it is
:func:`~mxtreme.analysis.activity.firing_rate` on a recording whose phases are the gaps between
its bursts. This module builds those phases.

- :func:`burst_phases` turns a burst table into phases of one of three kinds: the bursts
  themselves (``"burst"``), the gaps between successive bursts (``"inter"``), or each burst
  together with the gap that follows it (``"cycle"``).
- :func:`window_phases` names arbitrary frame windows, for intervals someone picked by hand.
- :func:`stim_windows` finds the stimulation episodes in a recording's stimulated frames, so a
  stimulated stretch can be shown, or cut around, like a burst.

Typical use::

    rec = Recording(0, load_preprocessed(paths.npz, skip=("spike_bin",)))
    bursts = BurstSet.from_csv(paths.burst_stats).to_dataframe()
    rec.phases = burst_phases(bursts, kind="inter", start_frame=..., end_frame=...)
    rates = firing_rate(rec, save=False)          # channel, phase ("post_B1", ...), firing_rate_hz
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from mxtreme.phases import Phase, Phases

#: What :func:`burst_phases` can cut a recording into.
PHASE_KINDS = ("burst", "inter", "cycle")

#: Columns :func:`burst_windows` returns, in order.
WINDOW_COLUMNS = ("id", "kind", "start_frame", "end_frame", "peak_frame")


def burst_windows(bursts: pd.DataFrame, *, edges: str = "onset", kinds: Sequence[str] | None = ("network",)) -> pd.DataFrame:
    """The time window of every burst that counts, sorted by start.

    :param bursts: A :meth:`BurstSet.to_dataframe <mxtreme.bursting.BurstSet.to_dataframe>` table. Rows whose ``ignore`` is true are left out.
    :param edges: ``"onset"`` bounds a burst by its ``onset_frame`` / ``offset_frame`` where
        feature extraction set them (falling back to the ISI-N group for a burst it did not reach);
        ``"group"`` always uses ``group_start_frame`` / ``group_stop_frame``.
    :param kinds: Which burst kinds count, e.g. ``("network",)`` (the default) or
        ``("network", "mini")``; ``None`` for every kind.
    :returns: Columns :data:`WINDOW_COLUMNS`.
    """
    if edges not in ("onset", "group"):
        raise ValueError(f"edges must be 'onset' or 'group', not {edges!r}")
    df = bursts.copy()
    if df.empty:
        return pd.DataFrame(columns=list(WINDOW_COLUMNS))
    if "ignore" in df.columns:
        df = df[~df["ignore"].astype(bool)]
    if kinds is not None and "kind" in df.columns:
        df = df[df["kind"].isin(list(kinds))]
    start = df["group_start_frame"].to_numpy(dtype=np.int64)
    end = df["group_stop_frame"].to_numpy(dtype=np.int64)
    if edges == "onset" and "onset_frame" in df.columns:
        onset = df["onset_frame"].to_numpy(dtype=np.int64)
        offset = df["offset_frame"].to_numpy(dtype=np.int64)
        has = (onset >= 0) & (offset >= onset)
        start = np.where(has, onset, start)
        end = np.where(has, offset, end)
    out = pd.DataFrame({
        "id": df["id"].to_numpy(),
        "kind": df["kind"].to_numpy() if "kind" in df.columns else "network",
        "start_frame": start,
        "end_frame": np.maximum(end, start),
        "peak_frame": df["peak_frame"].to_numpy(dtype=np.int64) if "peak_frame" in df.columns else (start + end) // 2,
    })
    return out.sort_values(["start_frame", "id"], kind="stable").reset_index(drop=True)


def burst_phases(
    bursts: pd.DataFrame,
    *,
    kind: str = "inter",
    start_frame: int | None = None,
    end_frame: int | None = None,
    edges: str = "onset",
    kinds: Sequence[str] | None = ("network",),
) -> Phases:
    """Cut a recording into phases at its bursts.

    Bursts are numbered ``1..n`` in time order among those that count (see ``kinds``), and the
    phase names carry that number so a result can be read back against the burst table:

    - ``kind="burst"``: ``B1, B2, ...`` -- each burst's own window.
    - ``kind="inter"``: ``post_B1, post_B2, ...`` -- from the end of burst *i* to the start of
      burst *i+1*. The stretch before the first burst (``pre_B1``) and after the last one
      (``post_Bn``) are included when ``start_frame`` / ``end_frame`` say where the recording
      begins and ends; without them only the gaps between bursts are made.
    - ``kind="cycle"``: ``cycle_B1, ...`` -- burst *i* together with the gap after it, up to the
      start of burst *i+1*. The last burst's cycle runs to ``end_frame`` when given.

    :param bursts: A burst table (see :func:`burst_windows`).
    :param kind: One of :data:`PHASE_KINDS`.
    :param start_frame: First frame of the recording, for the leading interval.
    :param end_frame: Last frame of the recording, for the trailing interval.
    :param edges: As for :func:`burst_windows`.
    :param kinds: As for :func:`burst_windows`.
    :returns: The phases, in time order; empty when there are no bursts to cut at.
    """
    if kind not in PHASE_KINDS:
        raise ValueError(f"kind must be one of {PHASE_KINDS}, not {kind!r}")
    w = burst_windows(bursts, edges=edges, kinds=kinds)
    starts = w["start_frame"].to_numpy(dtype=np.int64)
    ends = w["end_frame"].to_numpy(dtype=np.int64)
    phases: list[Phase] = []
    n = len(w)
    if kind == "burst":
        phases = [Phase(f"B{i + 1}", int(starts[i]), int(ends[i])) for i in range(n)]
    elif kind == "inter":
        if n and start_frame is not None and start_frame < starts[0]:
            phases.append(Phase("pre_B1", int(start_frame), int(starts[0]) - 1))
        for i in range(n - 1):
            a, b = int(ends[i]) + 1, int(starts[i + 1]) - 1
            if b >= a:
                phases.append(Phase(f"post_B{i + 1}", a, b))
        if n and end_frame is not None and end_frame > ends[-1]:
            phases.append(Phase(f"post_B{n}", int(ends[-1]) + 1, int(end_frame)))
    else:
        for i in range(n):
            b = int(starts[i + 1]) - 1 if i + 1 < n else (int(end_frame) if end_frame is not None else int(ends[i]))
            phases.append(Phase(f"cycle_B{i + 1}", int(starts[i]), max(b, int(starts[i]))))
    return Phases(tuple(phases))


def window_phases(windows: Iterable[tuple[str, int, int]]) -> Phases:
    """Phases from explicit ``(name, start_frame, end_frame)`` windows, in the order given.

    For intervals someone chose by hand rather than ones a burst table implies; names must be
    distinct so a per-phase table can be read back by them.
    """
    phases = [Phase(str(name), int(a), int(b)) for name, a, b in windows]
    names = [p.name for p in phases]
    if len(set(names)) != len(names):
        raise ValueError(f"window names must be distinct, got {names}")
    return Phases(tuple(phases))


def stim_windows(stim_frames, samp_rate: float, *, gap_sec: float = 0.05) -> np.ndarray:
    """Stimulation episodes from a recording's stimulated frames: ``(n, 2)`` of first and last frame.

    ``stim_frames`` (``Recording.stim_frames``, from the raw file's stimulation record) lists
    every frame a stimulus was being delivered on, so a 50 ms pulse train at 20 kHz is a
    thousand consecutive frames. Frames closer than ``gap_sec`` are one episode; an inter-pulse
    gap inside a train is far shorter than the time between trains, so the default separates
    trains without splitting them.
    """
    frames = np.unique(np.asarray(stim_frames, dtype=np.int64).ravel()) if stim_frames is not None else np.zeros(0, dtype=np.int64)
    if frames.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    gap = max(1, round(gap_sec * samp_rate))
    breaks = np.nonzero(np.diff(frames) > gap)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, frames.size - 1]
    return np.column_stack([frames[starts], frames[ends]])
