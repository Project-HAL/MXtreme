import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from mxtreme.utils import load_data
from mxtreme.recording import Recording
from mxtreme.analysis._paths import ANALYSIS_DIR, _summary_paths, load_population_summaries
from mxtreme.analysis._stats import aggregate_by_div_phase


def _get_stim_info(rec: Recording):

    # unique_messages = rec.event_df['eventmessage'].astype(str).unique()

    stim_rows = rec.event_df[rec.event_df['eventmessage'].map(lambda d: ('phase_us' in d)&('start_stimulation' in d) )].copy()
    stim_rows['stim_phase'] = rec.event_df['eventmessage'].map(lambda d: d.get('phase_us', float('nan'))).astype(float)

    return stim_rows


def stim_summary(cpath, analysis_dir: Path = ANALYSIS_DIR, use_existing=True, show_plot=True, save_plot=False):

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
            burst_stats = cpath.recordings[div].burst_stats

            rec = Recording(0, exp_data=load_data(npz), burst_csv=burst_stats)

            stim_df = _get_stim_info(rec)

            total_stim_time = stim_df['stim_phase'].sum() / 1000  # in ms
            total_train_time = (rec.rec_t_sec - (40 * 60)) / 60  # in min

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


def plot_population_stim_summary(sel_paths, analysis_dir: Path=ANALYSIS_DIR, savename=None):
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
