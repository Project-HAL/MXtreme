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
- :func:`annotated_bursts` reads a recording's burst table and folds in the corrections people
  have recorded against it in the transaction log (``burst.relabelled`` / ``burst.added``), so an
  interval analysis sees the bursts as the last person to look at them judged them.

Typical use::

    rec = Recording(0, load_preprocessed(paths.npz, skip=("spike_bin",)))
    bursts = annotated_bursts(config, paths)
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

#: Burst kinds a burst table can carry, plus the value a person uses to set one aside.
BURST_KINDS = ("network", "mini", "ignore")

#: Columns :func:`burst_windows` returns, in order.
WINDOW_COLUMNS = ("id", "kind", "start_frame", "end_frame", "peak_frame", "source")


def burst_windows(bursts: pd.DataFrame, *, edges: str = "onset", kinds: Sequence[str] | None = ("network",)) -> pd.DataFrame:
    """The time window of every burst that counts, sorted by start.

    :param bursts: A burst table (:meth:`BurstSet.to_dataframe <mxtreme.bursting.BurstSet.to_dataframe>`,
        or :func:`annotated_bursts`). Rows whose ``ignore`` is true are left out.
    :param edges: ``"onset"`` bounds a burst by its ``onset_frame`` / ``offset_frame`` where
        feature extraction set them (falling back to the ISI-N group for a burst it did not reach);
        ``"group"`` always uses ``group_start_frame`` / ``group_stop_frame``.
    :param kinds: Which burst kinds count, e.g. ``("network",)`` (the default) or
        ``("network", "mini")``; ``None`` for every kind.
    :returns: Columns :data:`WINDOW_COLUMNS`; ``source`` is ``"detected"`` unless the table says
        otherwise (``"manual"`` for a burst added by hand).
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
        "source": df["source"].to_numpy() if "source" in df.columns else "detected",
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
    gap = max(1, int(round(gap_sec * samp_rate)))
    breaks = np.nonzero(np.diff(frames) > gap)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, frames.size - 1]
    return np.column_stack([frames[starts], frames[ends]])


# --- corrections from the transaction log ---------------------------------------------------------


def burst_edits(transactions, recording_id) -> list:
    """The ``burst.relabelled`` / ``burst.added`` records about one recording, in log order.

    Matched on the recording's identity (batch, chip, well, DIV, experiment label), not on the
    path in the record's data, so a store that has moved keeps its corrections.
    """
    rid = recording_id
    out = []
    for tx in transactions:
        if tx.op not in ("burst.relabelled", "burst.added"):
            continue
        if str(tx.batch_id) != str(rid.batch_id) or str(tx.chip) != str(rid.chip):
            continue
        if tx.well is None or int(tx.well) != int(rid.well) or tx.div is None or int(tx.div) != int(rid.div):
            continue
        if str(tx.experiment or "") != str(rid.experiment or ""):
            continue
        out.append(tx)
    return out


def apply_burst_edits(bursts: pd.DataFrame, edits: Iterable) -> pd.DataFrame:
    """Fold a recording's corrections over its burst table, in the order they were recorded.

    A ``burst.relabelled`` record sets the burst's ``kind`` (``"ignore"`` sets its ``ignore`` flag
    and leaves the kind as detected); a later record about the same burst wins, so a correction is
    undone by recording the opposite. A ``burst.added`` record appends a row with the next free
    id, ``source="manual"``, its onset/offset as both the feature edges and the group edges, and a
    peak at the midpoint; detection's per-burst features are left at their sentinels.

    :returns: A new table with a ``source`` column (``"detected"`` / ``"manual"``) and every row's
        ``ignore`` as a bool.
    """
    df = bursts.copy()
    if "source" not in df.columns:
        df["source"] = "detected"
    if "ignore" not in df.columns:
        df["ignore"] = False
    df["ignore"] = df["ignore"].astype(bool)
    next_id = int(df["id"].max()) + 1 if len(df) else 1
    for tx in edits:
        data = tx.data or {}
        if tx.op == "burst.relabelled":
            kind = data.get("kind")
            mask = df["id"].astype(int) == int(data.get("burst_id", -1))
            if not mask.any() or kind not in BURST_KINDS:
                continue
            if kind == "ignore":
                df.loc[mask, "ignore"] = True
            else:
                df.loc[mask, "ignore"] = False
                df.loc[mask, "kind"] = kind
        elif tx.op == "burst.added":
            try:
                onset, offset = int(data["onset_frame"]), int(data["offset_frame"])
            except (KeyError, TypeError, ValueError):
                continue
            kind = data.get("kind", "network")
            if kind not in ("network", "mini"):
                kind = "network"
            row = {col: _sentinel(df[col]) for col in df.columns}
            row.update({
                "id": next_id, "kind": kind, "peak_frame": (onset + offset) // 2, "peak_amp": -1.0,
                "group_start_frame": onset, "group_stop_frame": offset, "onset_frame": onset,
                "offset_frame": offset, "duration_frames": offset - onset, "n_spikes": -1,
                "ignore": False, "source": "manual",
            })
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            next_id += 1
    return df


def _sentinel(column: pd.Series):
    """A blank value of a column's kind, for a row that has no measured value of it."""
    if column.dtype.kind in "iu":
        return -1
    if column.dtype.kind == "f":
        return -1.0
    if column.dtype.kind == "b":
        return False
    return ""


def annotated_bursts(config, paths) -> pd.DataFrame:
    """One recording's burst table as the transaction log last left it.

    :param config: The store's :class:`~mxtreme.config.Config`.
    :param paths: The recording's :class:`~mxtreme.paths.RecordingPaths`; a recording with no burst
        CSV yet gives an empty table, which corrections can still add to.
    :returns: See :func:`apply_burst_edits`.
    """
    from mxtreme import transactions
    from mxtreme.bursting.burst import _COLUMNS

    if paths.burst_stats is not None and paths.burst_stats.is_file():
        df = pd.read_csv(paths.burst_stats)
    else:
        df = pd.DataFrame(columns=_COLUMNS)
    edits = burst_edits(transactions.read(config), paths.recording_id)
    return apply_burst_edits(df, edits)
