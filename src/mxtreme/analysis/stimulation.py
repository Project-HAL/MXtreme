"""Stimulation delivered to a culture.

Recording level: :func:`stim_events` lists a recording's stimulation pulses, and :func:`stim_delivered`
totals them per phase.

Group level: :func:`summarize_stimulation` / :func:`plot_stimulation_summary` take a ``RecordingID``,
``CultureID``, ``CultureSelector`` or several selectors, as in :mod:`~mxtreme.analysis.activity`.

.. note:: A stimulation event is one whose message carries both a ``start_stimulation`` and a
   ``phase_us`` key -- maxlab closed-loop-stimulation names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mxtreme import io
from mxtreme.config import Config
from mxtreme.recording import Recording
from mxtreme.analysis._group import SummarySpec, culture_summary, fill_docs, plot_summary, summarize

__all__ = ["stim_events", "stim_delivered", "summarize_stimulation", "plot_stimulation_summary"]


def _recording_output(rec, analysis_dir, suffix: str) -> Path:
    out_dir = Path(analysis_dir) / "stimulation" / str(rec.batch_id) / str(rec.chip) / f"well{rec.well}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / io.recording_file_name(rec.DIV, rec.plate_date, rec.chip, rec.batch_id, rec.well,
                                            suffix, rec.experiment or "")


# --- recording level ------------------------------------------------------------------------------


def stim_events(rec: Recording, analysis_dir=None, *, save: bool = True) -> pd.DataFrame:
    """Every stimulation pulse in one recording.

    Messages that aren't dicts (older stores hold plain strings) are skipped rather than raising.

    :param rec: The recording.
    :param analysis_dir: Analysis output root; saved there as ``..._stim_events.csv`` when ``save`` is on.
    :param save: Write the result (needs ``analysis_dir``).
    :returns: One row per pulse: ``frame``, ``time_sec``, ``pulse_phase_us`` (maxlab's ``phase_us``),
        and ``phase`` -- the recording phase it falls in (``None`` in a gap between phases).
    """
    events = rec.event_df
    is_stim = events["eventmessage"].map(
        lambda d: isinstance(d, dict) and "phase_us" in d and "start_stimulation" in d
    )
    stim = events[is_stim]
    frames = stim["eventtime"].to_numpy(dtype=np.int64)
    df = pd.DataFrame({
        "frame": frames,
        "time_sec": frames / rec.samp_rate,
        "pulse_phase_us": stim["eventmessage"].map(lambda d: d.get("phase_us", np.nan)).astype(float).to_numpy(),
        "phase": [rec.phases.label_for(f) for f in frames],
    })

    if save and analysis_dir is not None:
        path = _recording_output(rec, analysis_dir, "stim_events.csv")
        df.to_csv(path, index=False)
        print(f"Saved stimulation events to {path}")
    return df


def stim_delivered(rec: Recording, analysis_dir=None, *, save: bool = True) -> pd.DataFrame:
    """Stimulation delivered in each phase of one recording.

    :param rec: The recording.
    :param analysis_dir: Analysis output root; saved there as ``..._stim_delivered.csv`` when ``save``
        is on.
    :param save: Write the result (needs ``analysis_dir``).
    :returns: One row per phase: ``phase``, ``n_pulses``, ``total_stim_ms`` (summed pulse phase,
        µs -> ms), and ``phase_dur_min`` (the phase's duration).
    """
    events = stim_events(rec, save=False)
    rows = []
    for p in rec.phases:
        in_phase = events["frame"].between(p.start_frame, p.end_frame, inclusive="both")
        rows.append({
            "phase": p.name,
            "n_pulses": int(in_phase.sum()),
            "total_stim_ms": events.loc[in_phase, "pulse_phase_us"].sum() / 1000,
            "phase_dur_min": p.n_frames / rec.samp_rate / 60,
        })
    df = pd.DataFrame(rows, columns=["phase", "n_pulses", "total_stim_ms", "phase_dur_min"])

    if save and analysis_dir is not None:
        path = _recording_output(rec, analysis_dir, "stim_delivered.csv")
        df.to_csv(path, index=False)
        print(f"Saved stimulation delivered to {path}")
    return df


# --- group level ----------------------------------------------------------------------------------

_SPEC = SummarySpec(
    title="Stimulation",
    metrics=[
        ("total_stim_ms", None, "Total stim time (ms)",  "Stimulation Delivered"),
        ("phase_dur_min", None, "Window duration (min)", "Recorded Window"),
    ],
    grid={"figsize": (11, 4.5)},
)


def _stim_rows(rp) -> list[dict]:
    rec = Recording(0, io.load_preprocessed(rp.npz))
    return [{"div": rp.recording_id.div, "experiment": rp.recording_id.experiment or "", **row}
            for row in stim_delivered(rec, save=False).to_dict("records")]


def _culture_stim_summary(cpath, analysis_dir: Path, use_existing: bool = True) -> pd.DataFrame:
    """Per-culture stimulation delivered (one row per recording x phase), cached."""
    return culture_summary(cpath, analysis_dir, "stimulation", "stim_summary", _stim_rows, use_existing)


def summarize_stimulation(target, config: Config, *, phase: str | None = None,
                          use_existing: bool = True) -> pd.DataFrame:
    """Table of stimulation delivered: one row per recording and phase (columns as in
    :func:`stim_delivered`).

    Identity columns lead: ``batch_id, culture_id, chip, well, div, experiment, phase``, plus ``group``
    when several selectors are given.

    {target}
    :param phase: Keep only this phase; ``None`` keeps every phase.
    :param use_existing: Reuse cached per-culture summaries; ``False`` recomputes the target's
        recordings.
    :returns: The summary table.
    """
    return summarize(target, config, _culture_stim_summary, phase=phase, use_existing=use_existing)[0]


def plot_stimulation_summary(target, config: Config, *, split_by: str = "phase", phase: str | None = None,
                             error: str = "sem", show_plot: bool = True, save_path=None):
    """Two-panel stimulation summary: stimulation delivered and window duration, vs DIV.

    {plot_doc}
    """
    return plot_summary(target, config, _culture_stim_summary, _SPEC, split_by=split_by, phase=phase,
                        error=error, show_plot=show_plot, save_path=save_path)


fill_docs(summarize_fns=(summarize_stimulation,), plot_fns=((plot_stimulation_summary, None),))
