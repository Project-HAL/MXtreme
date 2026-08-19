import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from mxtreme import io
from mxtreme.recording import Recording
from mxtreme.analysis._paths import _summary_paths, load_population_summaries
from mxtreme.analysis._stats import aggregate_by_div_phase


def _get_stim_info(rec: Recording):
    """Return the recording's stimulation events, with the pulse phase in a ``stim_phase`` column.

    Event messages are dicts (see :attr:`Recording.event_df <mxtreme.recording.Recording.event_df>`);
    a stimulation event is one carrying both a ``start_stimulation`` and a ``phase_us`` key. Messages
    that aren't dicts (older stores hold plain strings) are skipped rather than raising.

    .. note:: The ``start_stimulation`` / ``phase_us`` key names are maxlab closed-loop-stim specific.
    """
    is_stim = rec.event_df['eventmessage'].map(
        lambda d: isinstance(d, dict) and ('phase_us' in d) and ('start_stimulation' in d)
    )
    stim_rows = rec.event_df[is_stim].copy()
    stim_rows['stim_phase'] = stim_rows['eventmessage'].map(
        lambda d: d.get('phase_us', float('nan'))
    ).astype(float)

    return stim_rows


def _phase_duration_min(rec: Recording, name: str) -> float | None:
    """Return the duration (minutes) of ``rec``'s phase called ``name``, or ``None`` if absent."""
    for phase in rec.phases:
        if phase.name == name:
            return (phase.end_frame - phase.start_frame) / rec.samp_rate / 60

    return None


def stim_summary(cpath, analysis_dir: Path, use_existing=True, show_plot=True, save_plot=False,
                 train_phase: str = "train"):
    """Summarise stimulation delivered to one culture, one row per DIV.

    For each DIV: ``total_stim_ms`` is the summed pulse phase of the recording's stimulation events
    (maxlab reports these in **microseconds** as ``phase_us``, so they are divided by 1000 to give
    milliseconds), and ``total_train_min`` is the duration of the ``train_phase`` window taken from
    the recording's own phases.

    :param cpath: The culture's :class:`~mxtreme.paths.CulturePaths`.
    :param analysis_dir: Analysis output root (typically ``config.analysis_dir``).
    :param use_existing: Reuse the cached summary CSV when one already exists.
    :param train_phase: Name of the phase whose span counts as training time. When the recording has
        no such phase, the whole recording is used instead (and a notice is printed).
    :param show_plot: Show the per-culture summary figure.
    :param save_plot: Save that figure next to the summary CSV.
    :returns: One row per DIV with columns ``div``, ``total_stim_ms``, ``total_train_min``,
        ``culture_id``.
    :rtype: pandas.DataFrame
    """
    divs = []
    total_stim_times = []
    total_train_times = []

    cid = cpath.culture_id
    print(cid)

    save_path, csv_path = _summary_paths(cpath, analysis_dir, "stimulation", "stim_summary")

    if use_existing and csv_path.exists():
        print(f"Loading existing summary from {csv_path}")
        summary_df = pd.read_csv(csv_path)

    else:
        for div in cpath.recordings:

            npz = cpath.recordings[div].npz

            rec = Recording(0, io.load_preprocessed(npz))

            stim_df = _get_stim_info(rec)

            total_stim_time = stim_df['stim_phase'].sum() / 1000  # phase_us (µs) -> ms

            # Training time is the span of the recording's own `train_phase` window. Recordings with
            # no such phase (e.g. the default single "full" phase) fall back to the whole recording.
            total_train_time = _phase_duration_min(rec, train_phase)
            if total_train_time is None:
                print(f"  DIV{div}: no '{train_phase}' phase (found: {', '.join(rec.phases.names)}); "
                      f"using the full recording as train time")
                total_train_time = rec.rec_t_sec / 60

            divs.append(div)
            total_stim_times.append(total_stim_time)
            total_train_times.append(total_train_time)

        summary_df = pd.DataFrame({
            'div':              divs,
            'total_stim_ms':    total_stim_times,
            'total_train_min':  total_train_times,
            'culture_id':       cpath.culture_id,
        })

        os.makedirs(save_path, exist_ok=True)

        summary_df.to_csv(csv_path, index=False)
        print(f"Saved summary to {save_path}")

    if show_plot or save_plot:
        _plot_stim_summary(summary_df, cid=cpath.culture_id, analysis_dir=save_path, show_plot=show_plot, save_plot=save_plot)

    return summary_df


def _plot_stim_summary(df: pd.DataFrame, cid, analysis_dir, show_plot, save_plot, title_suffix: str = ''):
    """Reusable plotting helper — works on any df with columns: div, total_stim_ms, total_train_min."""

    x = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x, df['total_stim_ms'], color='steelblue', edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['div'], rotation=45, ha='center')
    ax1.set_xlabel('DIV')
    ax1.set_ylabel('Total Stimulation Time (ms)')
    ax1.set_title(f'Total Stimulation Time per DIV{" — " + title_suffix if title_suffix else ""}')

    ax2.bar(x, df['total_train_min'], color='darkorange', edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['div'], rotation=45, ha='center')
    ax2.set_xlabel('DIV')
    ax2.set_ylabel('Total Train Time (min)')
    ax2.set_title(f'Total Train Time per DIV{" — " + title_suffix if title_suffix else ""}')

    plt.tight_layout()
    if save_plot:
        plt.savefig(analysis_dir/f"{cid}_stim_summary.png", dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_population_stim_summary(sel_paths, analysis_dir: Path, savename=None):
    """Bar chart of mean ± SEM across cultures, one bar per DIV."""

    # TODO: update if sel_paths has more than one exp_id, decide whether to combine them or plot them separately

    pop_df = load_population_summaries(sel_paths, data_dir=analysis_dir/"stimulation", suffix='stim_summary')

    stats = aggregate_by_div_phase(
        pop_df, value_cols=['total_stim_ms', 'total_train_min'], group_cols=('div',)
    )

    n_cultures = pop_df['culture_id'].nunique()
    x = np.arange(len(stats))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x, stats['total_stim_ms'], yerr=stats['sem_total_stim_ms'],
            color='steelblue', edgecolor='black', capsize=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats['div'], rotation=45, ha='center')
    ax1.set_xlabel('DIV')
    ax1.set_ylabel('Total Stimulation Time (ms)')
    ax1.set_title(f'Total Stimulation Time per DIV (n={n_cultures} cultures)')

    ax2.bar(x, stats['total_train_min'], yerr=stats['sem_total_train_min'],
            color='darkorange', edgecolor='black', capsize=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stats['div'], rotation=45, ha='center')
    ax2.set_xlabel('DIV')
    ax2.set_ylabel('Total Train Time (min)')
    ax2.set_title(f'Total Train Time per DIV (n={n_cultures} cultures)')

    save_path = analysis_dir / "stimulation"
    os.makedirs(save_path, exist_ok=True)

    plt.tight_layout()
    if savename:
        plt.savefig(save_path / savename, dpi=300, bbox_inches='tight')
    plt.show()
