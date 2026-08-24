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
import matplotlib.pyplot as plt
from pathlib import Path

from mxtreme.analysis._paths import _summary_paths, load_population_summaries


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
    """Line plot of score vs DIV, one series per phase."""

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    if len(df):
        for phase in df['phase'].unique():
            phase_df = df[df['phase'] == phase].sort_values(by='div')
            ax.plot(phase_df['div'], phase_df['score'], marker='o', label=phase)

    ax.set_xlabel('DIV')
    ax.set_ylabel(score_label)
    ax.set_title(f'{cid.chip}, well {cid.well}')
    ax.set_ylim([-0.01, 1.01])
    ax.legend()
    fig.tight_layout()

    if save_plot:
        os.makedirs(analysis_dir, exist_ok=True)
        fig.savefig(analysis_dir / f"{cid}_performance_summary.png", dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_population_performance_summary(sel_paths,
                                        analysis_dir: Path,
                                        phase: str = None,
                                        savename=None):
    """Learning curve pooled across cultures: score vs DIV (mean ± SEM), with light per-culture lines.

    :param sel_paths: ``dict[exp_id, ExperimentPaths]`` from :func:`~mxtreme.paths.resolve_paths`.
    :param analysis_dir: Analysis output root (typically ``config.analysis_dir``).
    :param phase: If given, restrict to this phase; otherwise pool all phases.
    """
    pop_df = load_population_summaries(
        sel_paths, data_dir=analysis_dir / "performance", suffix='performance_summary'
    )

    if phase is not None:
        pop_df = pop_df[pop_df['phase'] == phase]

    df = pop_df.copy()
    if df.empty:
        print(f"No performance data for phase={phase!r}")
        return

    df['culture_id'] = df['chip'].astype(str) + '_' + df['well'].astype(str)
    cultures = sorted(df['culture_id'].unique())

    stats = (
        df.groupby('div')['score']
        .agg(mean='mean', sem=lambda x: x.sem())
        .reset_index()
        .sort_values('div')
    )

    cmap = plt.get_cmap('tab10')
    culture_colors = {c: cmap(i % 10) for i, c in enumerate(cultures)}

    fig, ax = plt.subplots(figsize=(8, 5))

    for cid in cultures:
        cdf = df[df['culture_id'] == cid].sort_values('div')
        ax.plot(cdf['div'], cdf['score'], color=culture_colors[cid], alpha=0.35,
                linewidth=1.2, marker='o', markersize=3, label=cid, zorder=2)

    ax.plot(stats['div'], stats['mean'], color='steelblue', linewidth=2.5,
            marker='o', markersize=6, label='Mean ± SEM', zorder=4)
    ax.fill_between(stats['div'], stats['mean'] - stats['sem'], stats['mean'] + stats['sem'],
                    color='steelblue', alpha=0.18, zorder=3)

    ax.set_xlabel('DIV')
    ax.set_ylabel('Score')
    cond_str = f'phase={phase}' if phase is not None else 'all phases'
    ax.set_title(f'Performance vs DIV  |  {cond_str}  (n={len(cultures)} cultures)')
    ax.set_ylim(0, 1)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, ncol=max(1, len(cultures) // 6 + 1),
              loc='upper left', framealpha=0.7)

    fig.tight_layout()

    if savename:
        out = analysis_dir / "performance"
        os.makedirs(out, exist_ok=True)
        fig.savefig(out / savename, dpi=300, bbox_inches='tight')

    plt.show()
