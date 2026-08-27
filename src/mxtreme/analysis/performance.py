"""Performance / learning-curve analysis.

A *performance* summary tracks a single user-chosen scalar per recording (per DIV, per phase) so it
can be plotted as a learning curve over development. The scalar is produced by a pluggable
``objective_fn(burst_df) -> float``. Users studying a different task supply their own objective.

The shipped default scores burst propagation *toward the side the culture was trained to burst from*,
read from each recording's ``exp_condition`` (see :func:`trained_side`). Scoring both directions on the
same "fraction toward the target" scale is what makes scores comparable across cultures -- pooling a
left-trained and a right-trained culture under a single fixed direction would average two opposing
conventions and pin the population mean near 0.5. When the condition is unknown the default falls back
to a paradigm-free left-to-right propagation fraction.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

from mxtreme.analysis._paths import _summary_paths, load_population_summaries
from mxtreme.analysis._plotting import plot_metric_grid
from mxtreme.analysis._stats import aggregate_by_div_phase


def trained_side(condition) -> str | None:
    """The side a culture was trained to burst from, as ``'left'`` / ``'right'`` / ``None``.

    An experimental condition is a ``[left, right]`` pair (``Recording.exp_condition``, stored in each
    preprocessed ``.npz``). The *lower* of the two numbers marks the trained side, so ``[0, 2]`` and
    ``[2, 1]`` are left-trained while ``[2, 0]`` and ``[1, 0]`` are right-trained. Equal values carry no
    lower number and therefore no target (e.g. an untrained ``[0, 0]`` control).

    :param condition: A condition pair, or ``None`` for a recording that has none.
    :returns: ``'left'``, ``'right'``, or ``None`` when there is no trained side to score against.
    """
    if condition is None:
        return None

    pair = np.asarray(condition).ravel()
    if pair.size != 2:  # unset, or a legacy dict-shaped condition
        return None

    left, right = pair.tolist()
    if left == right:
        return None
    return 'left' if left < right else 'right'


def default_direction_objective(burst_df: pd.DataFrame) -> float:
    """Default performance objective: burst propagation toward the trained side.

    Returns the fraction of bursts propagating toward the side named by the group's ``trained_side``
    column -- ``origin_x < peak_x`` for ``'left'``, ``origin_x > peak_x`` for ``'right'``. Both are on
    the same "fraction toward the target" scale, so left- and right-trained cultures can be compared
    and pooled directly. :func:`performance_summary` attaches that column from each recording's
    ``exp_condition``.

    Two distinct ways the target can be missing:

    - the column is **absent** -- the condition is *unknown* (e.g. called directly on a burst CSV), so
      this falls back to the paradigm-free left-to-right fraction it has always returned;
    - the column is present and ``None`` -- the culture is *known* to have no trained side, so there is
      no direction to score and the result is ``nan``.

    :param burst_df: Bursts for one (DIV, phase) group. Must contain ``origin_x`` and ``peak_x``.
    :returns: Fraction in ``[0, 1]``, or ``nan`` for an empty group or one with no trained side.
    """

    if len(burst_df) == 0:
        return np.nan

    if 'trained_side' not in burst_df.columns:
        return float((burst_df['origin_x'] < burst_df['peak_x']).mean())

    side = burst_df['trained_side'].iloc[0]
    if side == 'left':
        return float((burst_df['origin_x'] < burst_df['peak_x']).mean())
    if side == 'right':
        return float((burst_df['origin_x'] > burst_df['peak_x']).mean())
    return np.nan


def performance_summary(
    cpath,
    analysis_dir: Path,
    *,
    objective_fn=default_direction_objective,
    use_existing=True,
    show_plot=True,
    save_plot=False,
    score_label: str = 'Score',
):
    """Per-DIV, per-phase performance score for a culture, using a pluggable objective.

    For each recording the network bursts are grouped by phase and scored with ``objective_fn``; the
    tidy result (``chip, well, div, phase, condition, trained_side, score``) is cached as a CSV and
    plotted as a learning curve.

    Each group is handed to ``objective_fn`` carrying a ``trained_side`` column, derived from the
    recording's ``exp_condition`` via :func:`trained_side`, so the default objective can score toward
    the culture's own target. Objectives that ignore the column are unaffected.

    :param cpath: A :class:`~mxtreme.paths.CulturePaths`.
    :param analysis_dir: Analysis output root (typically ``config.analysis_dir``).
    :param objective_fn: ``callable(burst_df) -> float`` scoring one (DIV, phase) group of bursts.
    :param score_label: Y-axis label for the learning-curve plot.
    :returns: The per-DIV/phase summary DataFrame.
    """
    cid = cpath.culture_id
    columns = ['chip', 'well', 'div', 'phase', 'condition', 'trained_side', 'score']

    save_path, csv_path = _summary_paths(cpath, analysis_dir, "performance", "performance_summary")

    # A cache without `trained_side` predates condition-aware scoring, so its right-trained scores carry
    # the wrong sign. Recompute rather than load it back.
    cached = pd.read_csv(csv_path) if use_existing and csv_path.exists() else None
    if cached is not None and 'trained_side' not in cached.columns:
        print(f"Ignoring pre-condition summary at {csv_path} (no trained_side column); recomputing.")
        cached = None

    if cached is not None:
        print(f"Loading existing summary from {csv_path}")
        summary_df = cached
    else:
        rows = []
        for div in cpath.recordings:
            burst_stats = cpath.recordings[div].burst_stats
            burst_data = pd.read_csv(burst_stats)

            # np.load is lazy, so reading this one key never decompresses the recording's spike arrays.
            with np.load(cpath.recordings[div].npz, allow_pickle=True) as npz:
                condition = npz['exp_condition'] if 'exp_condition' in npz else None
            side = trained_side(condition)
            condition = None if condition is None else np.asarray(condition).ravel().tolist()
            if side is None:
                print(f"No trained side for condition {condition} (DIV {div}); scoring as nan.")

            # Score network bursts only (the analogue of the old "HAL_like" class).
            burst_data = burst_data[burst_data['kind'] == 'network']

            phases = [p for p in burst_data['phase'].dropna().unique()]
            if not phases:
                continue

            for phase in phases:
                phase_bursts = burst_data[burst_data['phase'] == phase]
                if len(phase_bursts) == 0:
                    print(f"No bursts detected in phase {phase!r} (DIV {div}).")
                    continue
                rows.append({
                    'chip':         cid.chip,
                    'well':         cid.well,
                    'div':          div,
                    'phase':        phase,
                    'condition':    condition,
                    'trained_side': side,
                    # assign() scores a copy -- the group is a slice, so writing to it would warn.
                    'score':        objective_fn(phase_bursts.assign(trained_side=side)),
                })

        summary_df = pd.DataFrame(rows, columns=columns)

        os.makedirs(save_path, exist_ok=True)
        summary_df.to_csv(csv_path, index=False)
        print(f"Saved summary to {save_path}")

    if show_plot or save_plot:
        _plot_performance_summary(
            summary_df, cid=cid, analysis_dir=save_path,
            show_plot=show_plot, save_plot=save_plot, score_label=score_label,
        )

    return summary_df


def _plot_performance_summary(df, cid, analysis_dir, show_plot, save_plot, score_label='Score'):
    """Line plot of score vs DIV, one series per phase.

    The y-axis is left to the data rather than pinned to ``[0, 1]``: the objective is pluggable, so
    only the shipped default happens to be a fraction.
    """
    if df.empty:
        print(f"No performance data for {cid}")
        return

    save_path = (Path(analysis_dir) / f"{cid}_performance_summary.png") if save_plot else None

    plot_metric_grid(
        df, [('score', None, score_label, 'Performance')],
        suptitle=f'Performance Summary — {cid}',
        save_path=save_path,
        show_plot=show_plot,
        ncols=1,
        figsize=(7, 5),
        dpi=150,
    )


def plot_population_performance_summary(sel_paths,
                                        analysis_dir: Path,
                                        phase: str = None,
                                        savename=None,
                                        score_label: str = 'Score'):
    """Learning curve pooled across cultures: score vs DIV (mean ± SEM), with faint per-culture lines.

    :param sel_paths: ``dict[exp_id, ExperimentPaths]`` from :func:`~mxtreme.paths.resolve_paths`.
    :param analysis_dir: Analysis output root (typically ``config.analysis_dir``).
    :param phase: If given, restrict to this phase; otherwise pool all phases.
    :param savename: Filename for the saved figure, written into ``<analysis_dir>/performance/``.
    :param score_label: Y-axis label (should match the objective's units).
    """
    pop_df = load_population_summaries(
        sel_paths, data_dir=analysis_dir / "performance", suffix='performance_summary'
    )

    if phase is not None:
        pop_df = pop_df[pop_df['phase'] == phase]

    if pop_df.empty:
        print(f"No performance data for phase={phase!r}")
        return

    stats = aggregate_by_div_phase(pop_df, value_cols=['score'])

    n_cultures = pop_df['culture_id'].nunique()
    save_path = (Path(analysis_dir) / "performance" / savename) if savename else None

    return plot_metric_grid(
        stats, [('score', 'sem_score', score_label, 'Performance')],
        overlay_df=pop_df,
        suptitle=f'Population Performance\nmean ± SEM, n = {n_cultures} cultures',
        save_path=save_path,
        show_plot=True,
        ncols=1,
        figsize=(8, 5),
        dpi=300,
    )
