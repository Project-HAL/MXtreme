"""Shared machinery for the group-level analysis functions (``summarize_*`` / ``plot_*_summary``).

Every topic module has the same shape:

- a *rows function* turns one recording (a :class:`~mxtreme.paths.RecordingPaths`) into summary rows,
  one per phase, each carrying ``div``, ``experiment``, ``phase`` and the metrics;
- :func:`culture_summary` caches those rows per culture, computing only recordings it hasn't seen;
- :func:`summarize` resolves any target (see :mod:`mxtreme.analysis._targets`) into a tidy table;
- :func:`plot_summary` draws that table at the target's level with
  :func:`~mxtreme.analysis._plotting.plot_metric_grid`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from mxtreme.config import Config
from mxtreme.analysis._paths import _summary_paths, stamp_identity
from mxtreme.analysis._plotting import (experiment_label, is_unphased, phase_tag, plot_metric_grid,
                                        sort_phases)
from mxtreme.analysis._stats import aggregate_by_div_phase
from mxtreme.analysis._targets import CULTURE, GROUPS, RECORDING, SELECTION, resolve_target

IDENTITY_COLS = ["group", "batch_id", "culture_id", "chip", "well", "div", "experiment", "phase"]
ERRORS = {"sem": "SEM", "std": "SD"}


def _key(rp) -> tuple[int, str]:
    return rp.recording_id.div, rp.recording_id.experiment or ""


def _read_summary_cache(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    try:
        cached = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return None
    if "experiment" not in cached.columns or "div" not in cached.columns:
        return None  # written before recordings carried labels: can't be matched per recording
    cached["experiment"] = cached["experiment"].fillna("").astype(str)
    return cached


def _matches(column: pd.Series, value) -> pd.Series:
    """Which cached values equal ``value`` -- numerically for numbers (``2`` == ``2.0`` read back from
    CSV), as text otherwise."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = pd.to_numeric(column, errors="coerce")
        return pd.Series(np.isclose(numeric, float(value)), index=column.index)
    return column.astype(str).eq(str(value))


def _cached_keys(df: pd.DataFrame) -> list[tuple[int, str]]:
    return list(zip(df["div"].astype(int), df["experiment"]))


def culture_summary(cpath, analysis_dir: Path, category: str, name: str,
                    rows_fn: Callable, use_existing: bool = True,
                    stamp: dict | None = None) -> pd.DataFrame:
    """Rows for ``cpath``'s recordings from the per-culture CSV cache, computing what's missing.

    The cache (``<analysis_dir>/<category>/<batch>/<chip>/well<N>/<cid>_<name>.csv``) holds every
    recording of the culture computed so far, keyed by ``(div, experiment)``. So resolving only part
    of a culture (one recording, or a selector's DIV filter) neither recomputes nor discards the rest.

    :param rows_fn: ``rows_fn(recording_paths) -> list[dict]``, one dict per phase with ``div``,
        ``experiment``, ``phase`` and the metrics.
    :param use_existing: ``False`` recomputes ``cpath``'s recordings.
    :param stamp: Settings the rows were computed with, ``{column: value}``. Each row carries them,
        and a cached recording whose stamped values differ is recomputed -- so changing a setting
        never silently reuses rows computed under the old one.
    """
    cid = cpath.culture_id
    save_dir, csv_path = _summary_paths(cpath, analysis_dir, category, name)
    stamp = stamp or {}

    cached = _read_summary_cache(csv_path)
    have = set()
    if cached is not None:
        current = pd.Series(True, index=cached.index)
        for column, value in stamp.items():
            current &= _matches(cached[column], value) if column in cached.columns else False
        have = {k for k, ok in zip(_cached_keys(cached), current) if ok}

    keys = [_key(rp) for rp in cpath.recordings]
    todo = [rp for rp, key in zip(cpath.recordings, keys) if not use_existing or key not in have]

    if todo:
        print(f"Computing {name} for {cid}: {len(todo)} recording(s)")
        fresh = pd.DataFrame([{**row, **stamp} for rp in todo for row in rows_fn(rp)])
        if not fresh.empty:
            fresh.insert(0, "culture_id", str(cid))
        redone = {_key(rp) for rp in todo}
        parts = []
        if cached is not None:
            parts.append(cached[[k not in redone for k in _cached_keys(cached)]])
        parts.append(fresh)
        parts = [p for p in parts if not p.empty]
        cached = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not cached.empty:
            cached = cached.sort_values(["div", "experiment"], kind="stable").reset_index(drop=True)
        os.makedirs(save_dir, exist_ok=True)
        cached.to_csv(csv_path, index=False)

    if cached is None or cached.empty:
        return pd.DataFrame()
    wanted = set(keys)
    return cached[[k in wanted for k in _cached_keys(cached)]].reset_index(drop=True)


def pick_phase(df: pd.DataFrame, phase: str | None) -> str:
    """``phase`` validated against ``df``, or the first phase in the sequence when ``None``."""
    available = sort_phases(df["phase"].dropna().unique())
    if not available:
        raise ValueError("No phases in the summary; nothing to plot.")
    if phase is None:
        return available[0]
    if phase not in available:
        raise ValueError(f"No data for phase {phase!r} (available: {', '.join(available)}).")
    return phase


def summarize(target, config: Config, summary_fn: Callable, *, phase=None, use_existing=True):
    """Tidy table for ``target``: ``summary_fn(cpath, analysis_dir, use_existing)`` per culture, stamped
    with identity columns (and ``group`` when several selectors are compared).

    :returns: ``(table, resolved target)``.
    """
    t = resolve_target(target, config)
    frames = []
    for label, cultures in t.groups.items():
        for cpath in cultures:
            df = summary_fn(cpath, config.analysis_dir, use_existing)
            if df.empty:
                continue
            frames.append(stamp_identity(df, cpath.culture_id).assign(
                batch_id=cpath.culture_id.batch_id, group=label
            ))

    if not frames:
        return pd.DataFrame(columns=[c for c in IDENTITY_COLS if c != "group" or t.level == GROUPS]), t
    df = pd.concat(frames, ignore_index=True)
    if phase is not None:
        df = df[df["phase"] == pick_phase(df, phase)]
    if t.level != GROUPS:
        df = df.drop(columns="group")
    lead = [c for c in IDENTITY_COLS if c in df.columns]
    return df[lead + [c for c in df.columns if c not in lead]].reset_index(drop=True), t


@dataclass(frozen=True)
class SummarySpec:
    """How one topic module's summary is plotted.

    :param title: Figure title, e.g. ``"Spiking Activity"``.
    :param metrics: ``(y_col, within_err_col | None, y_label, panel_title)`` per panel. The error
        column is the within-recording spread drawn at the recording and culture levels.
    :param within: What that spread is, for the figure title (e.g. ``"SD across channels"``);
        ``None`` when no metric has one.
    """

    title: str
    metrics: list
    within: str | None = None
    grid: dict = field(default_factory=dict)  # extra plot_metric_grid kwargs (ncols, figsize)


def _ordered_by_series(stats: pd.DataFrame, order: list) -> pd.DataFrame:
    rank = {s: i for i, s in enumerate(order)}
    return stats.assign(_rank=stats["series"].map(rank)).sort_values(["_rank", "div"]).drop(columns="_rank")


def plot_summary(target, config: Config, summary_fn: Callable, spec: SummarySpec, *,
                 split_by="phase", phase=None, error="sem", show_plot=True, save_path=None):
    """Draw ``spec``'s metric grid for ``target`` at its level (see the topic modules' docstrings)."""
    if split_by not in ("phase", "experiment"):
        raise ValueError(f"split_by must be 'phase' or 'experiment', not {split_by!r}")
    if error not in ERRORS:
        raise ValueError(f"error must be 'sem' or 'std', not {error!r}")

    df, t = summarize(target, config, summary_fn)
    if df.empty:
        print(f"Nothing to plot for {t.name}.")
        return None

    within = spec.metrics
    value_cols = [m[0] for m in within]
    population = [(y, f"{error}_{y}", label, title) for (y, _e, label, title) in within]
    err_name = ERRORS[error]
    title = f"{spec.title} — {t.name}"
    within_note = f"\nerror bars: {spec.within}" if spec.within else ""
    grid = dict(spec.grid, save_path=save_path, show_plot=show_plot)

    # Splitting by experiment (or comparing groups) fixes one phase; otherwise phases are the series.
    if split_by == "experiment" or t.level == GROUPS or phase is not None:
        unphased = is_unphased(df["phase"].dropna().unique())
        phase = pick_phase(df, phase)
        df = df[df["phase"] == phase]
        title += phase_tag(phase, unphased)

    if t.level in (RECORDING, CULTURE):
        # A single recording puts every phase at the same DIV: spread them so each bar is visible.
        dodge = 0.12 if t.level == RECORDING else 0.0
        if split_by == "phase":
            return plot_metric_grid(df, within, suptitle=title + within_note, dodge=dodge, **grid)
        df = df.assign(series=df["experiment"].map(experiment_label))
        return plot_metric_grid(df, within, series_col="series", suptitle=title + within_note,
                                dodge=dodge, **grid)

    n_cultures = df["culture_id"].nunique()
    if t.level == SELECTION:
        suptitle = f"{title}\nmean ± {err_name}, n = {n_cultures} cultures"
        if split_by == "phase":
            stats = aggregate_by_div_phase(df, value_cols, stat=error)
            return plot_metric_grid(stats, population, overlay_df=df, err_label=err_name,
                                    suptitle=suptitle, **grid)
        df = df.assign(series=df["experiment"].map(experiment_label))
        stats = aggregate_by_div_phase(df, value_cols, group_cols=("series", "div"), stat=error)
        return plot_metric_grid(stats, population, overlay_df=df, series_col="series",
                                err_label=err_name, suptitle=suptitle, **grid)

    # GROUPS: one mean ± error series per group (per group and label when splitting by experiment).
    series = df["group"]
    if split_by == "experiment":
        series = series + " · " + df["experiment"].map(experiment_label)
    df = df.assign(series=series)
    stats = _ordered_by_series(
        aggregate_by_div_phase(df, value_cols, group_cols=("series", "div"), stat=error),
        list(dict.fromkeys(df["series"])),
    )
    counts = ", ".join(f"{g}: n = {n}"
                       for g, n in df.groupby("group", sort=False)["culture_id"].nunique().items())
    return plot_metric_grid(stats, population, series_col="series", err_label=err_name,
                            suptitle=f"{title}\nmean ± {err_name}  ({counts})", **grid)


TARGET_DOC = """:param target: What to analyse -- a :class:`~mxtreme.identity.RecordingID` (one recording), a
        :class:`~mxtreme.identity.CultureID` (one culture over DIV), a
        :class:`~mxtreme.identity.CultureSelector` (a group of cultures), or several selectors to
        compare -- a ``dict`` of ``{name: selector}`` or a ``list`` (named "Group 1", "Group 2", ...).
    :param config: The :class:`~mxtreme.config.Config` describing the store; summaries are cached
        under ``config.analysis_dir``."""

PLOT_DOC = """What is drawn depends on ``target``:

    - **RecordingID** -- that recording's values, one point per phase{within_clause}.
    - **CultureID** -- the culture over DIV{within_clause}; one series per phase, or per experiment
      label with ``split_by="experiment"``.
    - **CultureSelector** -- a faint line per culture under the across-culture mean ± ``error``.
    - **several selectors** -- one mean ± ``error`` series per group, on one phase, so the groups can
      be compared directly.

    {target}
    :param split_by: ``"phase"`` (default) -- one series per phase; ``"experiment"`` -- one series per
        experiment label, on a single phase. For several selectors, ``"experiment"`` splits each group
        by label.
    :param phase: The phase to plot. Required in effect when splitting by experiment or comparing
        groups (defaults to the first phase); otherwise ``None`` plots every phase.
    :param error: Spread of the population mean -- ``"sem"`` (default) or ``"std"``.
    :param show_plot: Show the figure.
    :param save_path: Full path (including filename) to save the figure to.
    :returns: The matplotlib Figure (``None`` if there is nothing to plot)."""


def fill_docs(summarize_fns=(), plot_fns=()):
    """Fill the shared ``{target}`` / ``{plot_doc}`` placeholders of group-level docstrings.

    :param summarize_fns: Functions whose docstring holds ``{target}``.
    :param plot_fns: ``(function, within)`` pairs whose docstring holds ``{plot_doc}``; ``within`` names
        the within-recording spread drawn at the recording/culture levels, or ``None``.
    """
    for fn in summarize_fns:
        fn.__doc__ = fn.__doc__.replace("{target}", TARGET_DOC)
    for fn, within in plot_fns:
        clause = f", ± SD across {within}" if within else ""
        fn.__doc__ = fn.__doc__.replace(
            "{plot_doc}", PLOT_DOC.replace("{within_clause}", clause).replace("{target}", TARGET_DOC)
        )
