"""
Shared plotting helpers for analysis/single_culture.py and analysis/population.py.

Keeps phase coloring and the repeated 4-panel metric-grid layout in one place so the
per-culture and population plotting code can't drift from each other.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PHASE_STYLE = {
    'pre':   {'color': '#4C72B0', 'marker': 'o', 'label': 'Pre'},
    'train': {'color': '#DD8452', 'marker': 's', 'label': 'Train'},
    'post':  {'color': '#55A868', 'marker': '^', 'label': 'Post'},
    'full':  {'color': '#8172B2', 'marker': 'D', 'label': 'Full'},
}

_PHASE_ORDER = ['full', 'pre', 'train', 'post']


def sort_phases(phases):
    """Order pre/train/post/full first (in that priority); unknown phases after, alphabetically."""
    known = [p for p in _PHASE_ORDER if p in phases]
    unknown = sorted(p for p in phases if p not in _PHASE_ORDER)
    return known + unknown


def plot_metric_grid(df, metrics, group_col='div', phase_col='phase',
                      suptitle=None, save_path=None, show_plot=True,
                      ncols=2, figsize=(11, 8), dpi=150):
    """
    Grid of (metric vs group_col) panels, one errorbar series per phase.

    :param df: DataFrame containing group_col, phase_col, and every y_col/err_col in metrics.
    :param metrics: list of (y_col, err_col_or_None, y_label, panel_title) tuples, one per panel.
    :param group_col: x-axis grouping column (usually 'div').
    :param phase_col: column identifying each series (usually 'phase').
    :param suptitle: figure-level title, or None to omit.
    :param save_path: full path (including filename) to save the figure to, or None to skip saving.
    :param show_plot: whether to call plt.show() (mirrors the show_plot arg used elsewhere).
    :return: the matplotlib Figure.
    """
    nrows = -(-len(metrics) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    groups = sorted(df[group_col].dropna().unique())
    phases = sort_phases(df[phase_col].dropna().unique())

    for ax, (y_col, err_col, y_label, title) in zip(axes, metrics):
        for phase in phases:
            style = PHASE_STYLE.get(phase, {'color': 'gray', 'marker': 'D', 'label': phase})
            sub = df[df[phase_col] == phase].set_index(group_col).reindex(groups)

            y = sub[y_col].values
            err = sub[err_col].values if err_col else None

            ax.errorbar(
                groups, y, yerr=err,
                label=style['label'], color=style['color'], marker=style['marker'],
                linewidth=1.8, markersize=6, capsize=4, elinewidth=1.2,
            )

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(group_col.upper(), fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=9, framealpha=0.8)
        ax.spines[['top', 'right']].set_visible(False)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight='bold')

    if save_path:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved plot → {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig
