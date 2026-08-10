import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import ast

from mxtreme import constants
from mxtreme import device
from mxtreme.analysis._paths import ANALYSIS_DIR, _summary_paths, load_population_summaries


def compute_learning_score(df, target, metric='absolute'):

    mid_point = (device.CHIP_WIDTH / 2) * device.ELEC_SIZE # This is the x mid point

    if metric == 'absolute':

        if target == 'left':

            pct_correct = np.sum(df['origin_x']<mid_point)/len(df)

        elif target == 'right':

            pct_correct = np.sum(df['origin_x']>mid_point)/len(df)

    elif metric == 'relative':

        if target == 'left':

            pct_correct = np.sum(df['origin_x']<df['peak_x'])/len(df)

        elif target == 'right':

            pct_correct = np.sum(df['origin_x']>df['peak_x'])/len(df)

    return pct_correct

def learning_summary(cpath, analysis_dir: Path = ANALYSIS_DIR, use_existing=True, metric='relative', show_plot=True, save_plot=False):

    cid = cpath.culture_id
    print(cid)

    save_path, csv_path = _summary_paths(cpath, analysis_dir, "learning", "learning_summary")

    if use_existing and csv_path.exists():
        print(f"Loading existing summary from {csv_path}")
        learning_data = pd.read_csv(csv_path)
    else:
        EXP_COND_PATH = cpath.experimental_conditions
        exp_cond_df = pd.read_csv(EXP_COND_PATH, converters={'experimental condition':ast.literal_eval})

        chip_ids = []
        wells = []
        phase_col = []
        DIVs = []
        # dates = []
        pct_corrects = []
        exp_conditions = []
        targets = []

        for div in cpath.recordings:

            burst_stats = cpath.recordings[div].burst_stats
            burst_data = pd.read_csv(burst_stats)

            # Filter burst dataframe
            burst_data = burst_data[burst_data['type']=='HAL_like']

            # Extract the experimental conditions for the particular DIV
            left_stim = int(exp_cond_df[exp_cond_df['DIV']==div]['experimental condition'].iloc[0]['left_stim'])
            right_stim = int(exp_cond_df[exp_cond_df['DIV']==div]['experimental condition'].iloc[0]['right_stim'])

            # Determine the stimulation regime
            if np.min((left_stim, right_stim))==0: # no stim
                EXP_CONDITION = 'rand_nostim' # punish, reward
            elif np.min((left_stim, right_stim))==1:
                EXP_CONDITION = 'rand_reg'

            # print(left_stim, right_stim)
            # print(EXP_CONDITION)

            # Determine the target side
            if left_stim<right_stim: # target is left
                TARGET = 'left'
            elif right_stim<left_stim: # target is right
                TARGET = 'right'
            else:
                TARGET = 'left' # default for controls

            phases = np.unique([x for x in burst_data['phase'] if str(x)!='nan'])

            if len(phases) > 0:

                for phase in phases:

                    phase_burst_data = burst_data[burst_data['phase']==phase]

                    if len(phase_burst_data)==0: # no bursts detected in phase

                        print('No bursts detected in phase.')

                        continue

                        # print(left_stim, right_stim)
                        # exit()

                    pct_correct = compute_learning_score(phase_burst_data, TARGET, metric)

                    chip_ids.append(cid.chip)
                    wells.append(cid.well)
                    pct_corrects.append(pct_correct)
                    targets.append(TARGET)
                    DIVs.append(div)
                    exp_conditions.append(EXP_CONDITION)
                    phase_col.append(phase)

            else:
                if len(burst_data)==0: # no bursts detected in phase
                    print('No bursts detected.')
                    continue

                # compute percent left burst for controls
                pct_correct = compute_learning_score(burst_data, target='left', metric=metric)

                chip_ids.append(cid.chip)
                wells.append(cid.well)
                pct_corrects.append(pct_correct)
                targets.append(TARGET)
                DIVs.append(div)
                exp_conditions.append(EXP_CONDITION)
                phase_col.append('control')


        learning_data = pd.DataFrame({'chip':chip_ids, 'well':wells, 'div':DIVs, 'exp_cond':exp_conditions, 'target':targets, 'phase':phase_col, 'pct_correct_bursts':pct_corrects})

        os.makedirs(save_path, exist_ok=True)

        learning_data.to_csv(csv_path, index=False)
        print(f"Saved summary to {save_path}")

    if show_plot or save_plot:
        _plot_learning_summary(learning_data, cid=cpath.culture_id, analysis_dir=save_path, show_plot=show_plot, save_plot=save_plot,title_suffix=f"Target: {learning_data['target'].iloc[0]}")


def _plot_learning_summary(df, cid, analysis_dir, show_plot, save_plot, title_suffix=None):
    """Reusable plotting helper"""

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for phase in df['phase'].unique():

        phase_df = df[df['phase']==phase].sort_values(by='div')

        x=np.arange(len(phase_df))

        ax.plot(x, phase_df['pct_correct_bursts'], marker='o', label=phase)
        ax.set_xticks(x)
        ax.set_xticklabels(phase_df['div'])
        ax.set_xlabel('DIV')
        ax.set_ylabel('Proportion correct bursts')
        ax.set_title(f'{cid.chip}, well {cid.well} {" — " + title_suffix if title_suffix else ""}')

    plt.legend()
    plt.ylim([-0.01,1.01])
    plt.tight_layout()
    if save_plot:
        plt.savefig(analysis_dir/f"{cid}_learning_summary.png", dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_population_learning_summary(sel_paths,
                            analysis_dir: Path = ANALYSIS_DIR,
                            phase: str = 'pre',
                            exp_cond=None,
                            savename=None):
    """
    Line plot of pct_correct_bursts vs DIV.
    - One light curve per culture (chip × well pair), labeled.
    - Bold average line with SEM error bars.

    Parameters
    ----------
    phase : str
        Which phase to plot (default 'pre').
    exp_cond : str or None
        Filter to a single experimental condition. None = all conditions.
    """
    pop_df = load_population_summaries(
        sel_paths, data_dir=analysis_dir / "learning", suffix='learning_summary'
    )

    # Filter
    mask = pop_df['phase'] == phase
    if exp_cond is not None:
        mask &= pop_df['exp_cond'] == exp_cond
    df = pop_df[mask].copy()

    if df.empty:
        print(f"No data for phase={phase!r}, exp_cond={exp_cond!r}")
        return

    df['culture_id'] = df['chip'].astype(str) + '_' + df['well'].astype(str)
    cultures = sorted(df['culture_id'].unique())
    divs = sorted(df['div'].unique())

    # Average + SEM across cultures at each DIV
    stats = (
        df.groupby('div')['pct_correct_bursts']
        .agg(mean='mean', sem=lambda x: x.sem())
        .reset_index()
        .sort_values('div')
    )

    # Color palette for individual cultures
    cmap = plt.get_cmap('tab10')
    culture_colors = {c: cmap(i % 10) for i, c in enumerate(cultures)}

    fig, ax = plt.subplots(figsize=(8, 5))

    # Individual culture lines (light)
    for cid in cultures:
        cdf = df[df['culture_id'] == cid].sort_values('div')
        color = culture_colors[cid]
        ax.plot(
            cdf['div'], cdf['pct_correct_bursts'],
            color=color, alpha=0.35, linewidth=1.2,
            marker='o', markersize=3,
            label=cid, zorder=2,
        )

    # Mean ± SEM
    ax.plot(
        stats['div'], stats['mean'],
        color='steelblue', linewidth=2.5,
        marker='o', markersize=6,
        label='Mean ± SEM', zorder=4,
    )
    ax.fill_between(
        stats['div'],
        stats['mean'] - stats['sem'],
        stats['mean'] + stats['sem'],
        color='steelblue', alpha=0.18, zorder=3,
    )

    ax.set_xlabel('DIV')
    ax.set_ylabel('Proportion Correct Bursts')
    cond_str = exp_cond if exp_cond is not None else 'all conditions'
    ax.set_title(
        f'Proportion Correct Bursts vs DIV  |  phase={phase}  |  {cond_str}'
        f'  (n={len(cultures)} cultures)'
    )
    ax.set_ylim(0, 1)

    # Legend: cultures first, then mean
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        fontsize=8, ncol=max(1, len(cultures) // 6 + 1),
        loc='upper left', framealpha=0.7,
    )

    plt.tight_layout()

    if savename:
        save_path = analysis_dir / "learning"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path / savename, dpi=300, bbox_inches='tight')

    plt.show()
