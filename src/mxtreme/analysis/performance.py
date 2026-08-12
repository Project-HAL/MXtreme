"""Performance / learning-curve analysis.

A *performance* summary tracks a single user-chosen scalar per recording (per DIV, per phase) so it
can be plotted as a learning curve over development. The scalar is produced by a pluggable
``objective_fn(burst_df) -> float`` -- the package ships a general-purpose default (burst propagation
direction) and makes no assumptions about a particular stimulation paradigm. Users studying a
closed-loop task supply their own objective (e.g. a closure that scores bursts against a target side).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from mxtreme.analysis._paths import _summary_paths, load_population_summaries


def default_direction_objective(burst_df: pd.DataFrame) -> float:
    """Default performance objective: burst propagation direction.

    Returns the fraction of (network) bursts whose spatial origin is to the *left* of their peak
    (``origin_x < peak_x``) -- i.e. bursts that propagate rightward. This is a general, paradigm-free
    readout of directional structure; supply a different ``objective_fn`` for task-specific scoring.

    :param burst_df: Bursts for one (DIV, phase) group. Must contain ``origin_x`` and ``peak_x``.
    :returns: Fraction in ``[0, 1]``, or ``nan`` for an empty group.
    """
    if len(burst_df) == 0:
        return np.nan
    return float((burst_df['origin_x'] < burst_df['peak_x']).mean())


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
    tidy result (``chip, well, div, phase, score``) is cached as a CSV and plotted as a learning curve.

    :param cpath: A :class:`~mxtreme.paths.CulturePaths`.
    :param analysis_dir: Analysis output root (typically ``config.analysis_dir``).
    :param objective_fn: ``callable(burst_df) -> float`` scoring one (DIV, phase) group of bursts.
    :param score_label: Y-axis label for the learning-curve plot.
    :returns: The per-DIV/phase summary DataFrame.
    """
    cid = cpath.culture_id

    save_path, csv_path = _summary_paths(cpath, analysis_dir, "performance", "performance_summary")

    if use_existing and csv_path.exists():
        print(f"Loading existing summary from {csv_path}")
        summary_df = pd.read_csv(csv_path)
    else:
        rows = []
        for div in cpath.recordings:
            burst_stats = cpath.recordings[div].burst_stats
            burst_data = pd.read_csv(burst_stats)

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
                    'chip':  cid.chip,
                    'well':  cid.well,
                    'div':   div,
                    'phase': phase,
                    'score': objective_fn(phase_bursts),
                })

        summary_df = pd.DataFrame(rows, columns=['chip', 'well', 'div', 'phase', 'score'])

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
