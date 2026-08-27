"""
Shared plotting helpers for the analysis topic modules.

Keeps phase coloring, the repeated 4-panel metric-grid layout, and the ECDF grid in one place so the
per-culture and population plotting code can't drift from each other. Every population-level figure
in the package goes through :func:`plot_metric_grid`, which is what makes the report's sections look
alike: a bold mean +/- SEM series over a faint line per culture.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from mxtreme.utils import get_n_colors

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


def culture_label(row) -> str:
    """Human-readable label for one culture's series, e.g. ``'M07462 well2'``.

    Falls back to ``culture_id`` for frames that predate the chip/well stamping in
    :func:`~mxtreme.analysis._paths.load_population_summaries`.
    """
    chip, well = row.get('chip'), row.get('well')
    if chip is None or well is None:
        return str(row.get('culture_id', ''))
    return f"{chip} well{well}"


def _draw_culture_overlay(ax, overlay_df, y_col, group_col, overlay_col, colors):
    """Draw one faint line per culture behind the aggregate series (in-place on ``ax``)."""
    for cid, sub in overlay_df.groupby(overlay_col, sort=True):
        sub = sub.sort_values(group_col)
        ax.plot(sub[group_col], sub[y_col], color=colors[cid], alpha=0.35, linewidth=1.2,
                marker='o', markersize=3, label=culture_label(sub.iloc[0]), zorder=2)


def plot_metric_grid(df, metrics, group_col='div', phase_col='phase',
                      overlay_df=None, overlay_col='culture_id',
                      suptitle=None, save_path=None, show_plot=True,
                      ncols=2, figsize=(11, 8), dpi=150):
    """
    Grid of (metric vs group_col) panels, one errorbar series per phase.

    :param df: DataFrame containing group_col, phase_col, and every y_col/err_col in metrics.
    :param metrics: list of (y_col, err_col_or_None, y_label, panel_title) tuples, one per panel.
    :param group_col: x-axis grouping column (usually 'div').
    :param phase_col: column identifying each series (usually 'phase').
    :param overlay_df: Optional per-culture rows (one row per culture per group_col value, i.e. the
        un-aggregated frame `df` was built from). Each culture is drawn as a faint line behind the
        aggregate series, so an outlier or a dying culture stays visible under the mean.
    :param overlay_col: Column of ``overlay_df`` identifying each culture (usually 'culture_id').
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

    overlay_colors = {}
    if overlay_df is not None and not overlay_df.empty:
        # A qualitative map indexed discretely: sampling it continuously (as `get_n_colors` does for
        # the sequential maps) lands between its entries and washes the faint lines out.
        cmap = plt.get_cmap('tab10')
        cultures = sorted(overlay_df[overlay_col].dropna().unique())
        overlay_colors = {c: cmap(i % cmap.N) for i, c in enumerate(cultures)}

    for ax, (y_col, err_col, y_label, title) in zip(axes, metrics):
        if overlay_colors and y_col in overlay_df.columns:
            _draw_culture_overlay(ax, overlay_df, y_col, group_col, overlay_col, overlay_colors)

        for phase in phases:
            style = PHASE_STYLE.get(phase, {'color': 'gray', 'marker': 'D', 'label': phase})
            sub = df[df[phase_col] == phase].set_index(group_col).reindex(groups)

            y = sub[y_col].values
            err = sub[err_col].values if err_col else None

            label = style['label']
            if overlay_colors:
                label = f"{label} (mean ± SEM)" if err_col else f"{label} (mean)"

            ax.errorbar(
                groups, y, yerr=err,
                label=label, color=style['color'], marker=style['marker'],
                linewidth=1.8, markersize=6, capsize=4, elinewidth=1.2, zorder=4,
            )

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(group_col.upper(), fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        # A per-culture overlay adds one legend entry per culture, so give it columns and a smaller
        # font rather than letting it eat the panel.
        n_series = len(overlay_colors) + len(phases)
        ax.legend(fontsize=7 if overlay_colors else 9, framealpha=0.8,
                  ncol=max(1, n_series // 6 + 1))
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


def _ecdf(values, max_points: int = 20_000):
    """Return ``(x, y)`` of the empirical CDF of ``values``, thinned to at most ``max_points``.

    Non-finite entries are dropped. The thinning is what keeps a report readable in size: a single
    recording contributes ~1.5M inter-spike intervals, and a PDF holding one polyline per DIV per
    culture at full resolution runs to hundreds of megabytes while looking identical.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([]), np.array([])

    v.sort()
    y = np.arange(1, v.size + 1) / v.size
    if v.size > max_points:
        # Keep the last point so the curve always reaches 1.0.
        step = -(-v.size // max_points)  # ceil division
        keep = np.append(np.arange(0, v.size, step), v.size - 1)
        v, y = v[keep], y[keep]
    return v, y


def plot_cdf_grid(dists, metrics, suptitle=None, save_path=None, show_plot=True,
                  ncols=2, figsize=(11, 8), dpi=150, cmap_name='viridis'):
    """Grid of ECDF panels, one line per DIV.

    Where :func:`plot_metric_grid` shows a summary statistic moving over development, this shows the
    whole distribution behind it -- a culture whose mean firing rate holds steady while its
    distribution splits into a silent and a hyperactive population looks identical in one and obvious
    in the other.

    :param dists: ``{div: {metric_key: array_of_values}}``, e.g. from
        :func:`mxtreme.analysis.activity.culture_distributions`.
    :param metrics: list of ``(metric_key, x_label, panel_title, logx)`` tuples, one per panel.
    :param suptitle: figure-level title, or None to omit.
    :param save_path: full path (including filename) to save to, or None to skip saving.
    :param show_plot: whether to call ``plt.show()``.
    :param cmap_name: sequential colormap mapped over the DIVs, so color reads as developmental time.
    :return: the matplotlib Figure.
    """
    nrows = -(-len(metrics) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    divs = sorted(dists)
    colors = dict(zip(divs, get_n_colors(len(divs), cmap_name=cmap_name)))

    for ax, (key, x_label, title, logx) in zip(axes, metrics):
        for div in divs:
            x, y = _ecdf(dists[div].get(key, []))
            if x.size == 0:
                continue
            if logx:
                # A log axis can't show non-positive values; they are not meaningful for any of
                # these quantities (rates, intervals, sizes) anyway.
                positive = x > 0
                x, y = x[positive], y[positive]
                if x.size == 0:
                    continue
            ax.plot(x, y, color=colors[div], linewidth=1.5, label=f"DIV{div}")

        if logx:
            ax.set_xscale('log')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel('Cumulative fraction', fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=7, framealpha=0.8, ncol=max(1, len(divs) // 6 + 1), loc='lower right')
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
