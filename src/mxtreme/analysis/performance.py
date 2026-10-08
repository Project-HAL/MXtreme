"""Performance: a learning curve from a user-chosen score of each recording's network bursts.

The score comes from a pluggable ``objective_fn(burst_df) -> float``, so users studying a different
task supply their own. The shipped default, :func:`default_direction_objective`, scores burst
propagation toward the side each culture was trained to burst from (read from the recording's
``exp_condition``).

Recording level: :func:`performance_score` scores one recording's bursts, per phase.

Group level: :func:`summarize_performance` / :func:`plot_performance_summary` take a ``RecordingID``,
``CultureID``, ``CultureSelector`` or several selectors, as in :mod:`~mxtreme.analysis.activity`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from mxtreme import io
from mxtreme.config import Config
from mxtreme.recording import Recording
from mxtreme.analysis.activity import _bursts_by_phase
from mxtreme.analysis._group import SummarySpec, culture_summary, fill_docs, plot_summary, summarize

__all__ = [
    "performance_score", "default_direction_objective",
    "summarize_performance", "plot_performance_summary",
]


def _trained_side(condition) -> str | None:
    """The side a culture was trained to burst from, as ``'left'`` / ``'right'`` / ``None``.

    An experimental condition is a ``[left, right]`` pair (``Recording.exp_condition``). The *lower*
    number marks the trained side, so ``[0, 2]`` and ``[2, 1]`` are left-trained while ``[2, 0]`` and
    ``[1, 0]`` are right-trained. Equal values carry no target (e.g. an untrained ``[0, 0]`` control).

    Experiment-specific: to be replaced by a user-defined mapping.
    """
    if condition is None:
        return None
    pair = np.asarray(condition).ravel()
    if pair.size != 2:  # unset, or a legacy dict-shaped condition
        return None
    left, right = pair.tolist()
    if left == right:
        return None
    return "left" if left < right else "right"


def default_direction_objective(burst_df: pd.DataFrame) -> float:
    """Default performance objective: the fraction of bursts propagating toward the trained side.

    Uses the ``trained_side`` column that :func:`performance_score` attaches -- ``origin_x < peak_x``
    counts for ``'left'``, ``origin_x > peak_x`` for ``'right'`` -- so left- and right-trained cultures
    score on the same "fraction toward the target" scale and can be pooled.

    - Column **absent** (the condition is unknown, e.g. called on a burst CSV directly): the
      paradigm-free left-to-right fraction.
    - Column present and ``None`` (the culture has no trained side): ``nan``.

    :param burst_df: Bursts for one phase. Must contain ``origin_x`` and ``peak_x``.
    :returns: A fraction in ``[0, 1]``, or ``nan`` for no bursts or no trained side.
    """
    if len(burst_df) == 0:
        return np.nan
    if "trained_side" not in burst_df.columns:
        return float((burst_df["origin_x"] < burst_df["peak_x"]).mean())

    side = burst_df["trained_side"].iloc[0]
    if side == "left":
        return float((burst_df["origin_x"] < burst_df["peak_x"]).mean())
    if side == "right":
        return float((burst_df["origin_x"] > burst_df["peak_x"]).mean())
    return np.nan


# --- recording level ------------------------------------------------------------------------------


def _objective_name(objective_fn) -> str:
    name = getattr(objective_fn, "__name__", type(objective_fn).__name__)
    return re.sub(r"[^A-Za-z0-9_]+", "", name) or "objective"


def performance_score(rec: Recording, burst_df: pd.DataFrame, analysis_dir=None, *,
                      objective_fn=default_direction_objective, save: bool = True) -> pd.DataFrame:
    """Score one recording's network bursts, per phase, with ``objective_fn``.

    Each phase's bursts are handed to ``objective_fn`` with a ``trained_side`` column derived from the
    recording's ``exp_condition``, so the default objective can score toward the culture's own target;
    objectives that ignore the column are unaffected. A phase with no network bursts scores ``nan``
    without calling the objective.

    :param rec: The recording.
    :param burst_df: Its bursts (only ``kind == "network"`` rows are scored).
    :param analysis_dir: Analysis output root; saved there as ``..._performance_<objective>.csv`` when
        ``save`` is on.
    :param objective_fn: ``callable(burst_df) -> float``; defaults to
        :func:`default_direction_objective`.
    :param save: Write the result (needs ``analysis_dir``).
    :returns: One row per phase: ``phase, n_bursts, condition, trained_side, score``.
    """
    side = _trained_side(rec.exp_condition)
    condition = None if rec.exp_condition is None else str(np.asarray(rec.exp_condition).ravel().tolist())
    rows = []
    for phase, bursts in _bursts_by_phase(rec, burst_df).items():
        rows.append({
            "phase": phase,
            "n_bursts": len(bursts),
            "condition": condition,
            "trained_side": side,
            # assign() scores a copy -- the group is a slice, so writing to it would warn.
            "score": float(objective_fn(bursts.assign(trained_side=side))) if len(bursts) else np.nan,
        })
    df = pd.DataFrame(rows, columns=["phase", "n_bursts", "condition", "trained_side", "score"])

    if save and analysis_dir is not None:
        out_dir = Path(analysis_dir) / "performance" / str(rec.batch_id) / str(rec.chip) / f"well{rec.well}"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"performance_{_objective_name(objective_fn)}.csv"
        path = out_dir / io.recording_file_name(rec.DIV, rec.plate_date, rec.chip, rec.batch_id, rec.well,
                                                suffix, rec.experiment or "")
        df.to_csv(path, index=False)
        print(f"Saved performance scores to {path}")
    return df


# --- group level ----------------------------------------------------------------------------------


def _spec(score_label: str) -> SummarySpec:
    return SummarySpec(title="Performance", metrics=[("score", None, score_label, "Performance")],
                       grid={"ncols": 1, "figsize": (8, 5)})


def _culture_performance_summary(cpath, analysis_dir: Path, use_existing: bool = True,
                                 objective_fn=default_direction_objective) -> pd.DataFrame:
    """Per-culture performance scores (one row per recording x phase), cached per objective.

    The cache is named after ``objective_fn``, so scores under different objectives never mix. Two
    different functions sharing a name (e.g. two lambdas) do share a cache: pass
    ``use_existing=False`` after changing one.
    """
    def rows(rp):
        rec = Recording(0, io.load_preprocessed(rp.npz))
        bursts = pd.read_csv(rp.require_burst_stats())
        df = performance_score(rec, bursts, objective_fn=objective_fn, save=False)
        return [{"div": rp.recording_id.div, "experiment": rp.recording_id.experiment or "", **row}
                for row in df.to_dict("records")]

    name = f"performance_{_objective_name(objective_fn)}"
    return culture_summary(cpath, analysis_dir, "performance", name, rows, use_existing)


def summarize_performance(target, config: Config, *, objective_fn=default_direction_objective,
                          phase: str | None = None, use_existing: bool = True) -> pd.DataFrame:
    """Table of performance scores: one row per recording and phase (columns as in
    :func:`performance_score`).

    Identity columns lead: ``batch_id, culture_id, chip, well, div, experiment, phase``, plus ``group``
    when several selectors are given. Needs burst detection to have been run on every recording.

    {target}
    :param objective_fn: ``callable(burst_df) -> float``; defaults to
        :func:`default_direction_objective`.
    :param phase: Keep only this phase; ``None`` keeps every phase.
    :param use_existing: Reuse cached per-culture scores; ``False`` recomputes the target's recordings.
    :returns: The summary table.
    """
    def summary(cpath, analysis_dir, use):
        return _culture_performance_summary(cpath, analysis_dir, use, objective_fn)
    return summarize(target, config, summary, phase=phase, use_existing=use_existing)[0]


def plot_performance_summary(target, config: Config, *, objective_fn=default_direction_objective,
                             score_label: str = "Score", split_by: str = "phase",
                             phase: str | None = None, error: str = "sem",
                             show_plot: bool = True, save_path=None):
    """Learning curve: the performance score vs DIV.

    The y-axis follows the data rather than being pinned to ``[0, 1]``: the objective is pluggable,
    and only the default happens to be a fraction.

    {plot_doc}
    :param objective_fn: ``callable(burst_df) -> float``; defaults to
        :func:`default_direction_objective`.
    :param score_label: Y-axis label (should match the objective's units).
    """
    def summary(cpath, analysis_dir, use):
        return _culture_performance_summary(cpath, analysis_dir, use, objective_fn)
    return plot_summary(target, config, summary, _spec(score_label), split_by=split_by, phase=phase,
                        error=error, show_plot=show_plot, save_path=save_path)


fill_docs(summarize_fns=(summarize_performance,), plot_fns=((plot_performance_summary, None),))
