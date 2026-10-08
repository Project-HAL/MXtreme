"""
Shared plotting helpers for the analysis topic modules.

Keeps phase coloring, the repeated 4-panel metric-grid layout, and the ECDF grid in one place so the
per-culture and population plotting code can't drift from each other. Every population-level figure
in the package goes through :func:`plot_metric_grid`, which is what makes the report's sections look
alike: a bold mean +/- SEM series over a faint line per culture.
"""

import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from mxtreme.utils import get_n_colors
from mxtreme.visualizations import finish_figure
from mxtreme.analysis._paths import recording_label

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


def is_unphased(phases) -> bool:
    """Whether ``phases`` holds nothing but ``"full"`` -- recordings with no user-supplied phases, where
    naming the phase in a label or title says nothing."""
    names = {p for p in phases if isinstance(p, str)}
    return names == {'full'}


def phase_tag(phase, unphased: bool, template: str = "  |  phase: {phase}") -> str:
    """``template`` filled with ``phase`` for titles, or ``""`` when the data is unphased."""
    return "" if unphased or phase is None else template.format(phase=phase)


def culture_label(row) -> str:
    """Human-readable label for one culture's series, e.g. ``'M07462 well2'``.

    Falls back to ``culture_id`` for frames that predate the chip/well stamping in
    :func:`~mxtreme.analysis._paths.load_population_summaries`.
    """
    chip, well = row.get('chip'), row.get('well')
    if chip is None or well is None:
        return str(row.get('culture_id', ''))
    return f"{chip} well{well}"


# Linestyles telling apart experiment labels within a phase's colour (first label is solid).
_EXPERIMENT_LINESTYLES = ['-', '--', ':', '-.']


def natural_key(text: str):
    """Sort key putting ``NS_ATP1hr`` before ``NS_ATP12hr``: digit runs compare as numbers."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def experiment_label(experiment) -> str:
    """Display name for an ``experiment`` label; an unlabelled recording's ``""`` reads 'unlabelled'."""
    return str(experiment) if experiment else 'unlabelled'


def _experiments(df, experiment_col):
    """Distinct experiment labels in ``df`` (``[None]`` when it has no such column), sorted."""
    if experiment_col not in df.columns:
        return [None]
    return sorted(df[experiment_col].fillna("").astype(str).unique(), key=natural_key)


def _with_experiment(df, experiment_col, experiment):
    """Rows of ``df`` carrying ``experiment`` (every row when ``experiment`` is ``None``)."""
    if experiment is None:
        return df
    return df[df[experiment_col].fillna("").astype(str) == experiment]


# Series styles when a column other than phase drives the series (e.g. comparing selections), and
# the markers telling experiment labels apart within a phase.
_SERIES_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


def _draw_culture_overlay(ax, overlay_df, y_col, group_col, overlay_col, colors,
                          experiment_col='experiment', phase_col='phase', experiment_index=None,
                          offsets=None):
    """Draw one faint line per culture (per experiment label and phase) behind the aggregate series.

    Splitting by label and phase keeps a culture's line from zig-zagging between its several
    recordings, or phases, on one DIV. Each culture keeps one colour and one legend entry. With
    several labels, each label's faint lines carry that label's linestyle and marker -- the same ones
    as its bold series (``experiment_index`` maps a label to its position in that series order) --
    and sit at the same x offset (``offsets``: ``{(experiment, phase): dx}``).
    """
    offsets = offsets or {}
    experiments = _experiments(overlay_df, experiment_col)
    multi_experiment = len(experiments) > 1
    if experiment_index is None:
        experiment_index = {e: i for i, e in enumerate(experiments)}
    phases = (sort_phases(overlay_df[phase_col].dropna().unique())
              if phase_col in overlay_df.columns else [None])
    for cid, culture_rows in overlay_df.groupby(overlay_col, sort=True):
        labelled = False
        for experiment in experiments:
            i = experiment_index.get(experiment, 0)
            for phase in phases:
                sub = _with_experiment(culture_rows, experiment_col, experiment)
                if phase is not None:
                    sub = sub[sub[phase_col] == phase]
                sub = sub.sort_values(group_col)
                if sub.empty:
                    continue
                ax.plot(sub[group_col] + offsets.get((experiment, phase), 0.0), sub[y_col], color=colors[cid], alpha=0.35,
                        linewidth=1.2,
                        linestyle=_EXPERIMENT_LINESTYLES[i % len(_EXPERIMENT_LINESTYLES)],
                        marker=_SERIES_MARKERS[i % len(_SERIES_MARKERS)] if multi_experiment else 'o',
                        markersize=3,
                        label='_nolegend_' if labelled else culture_label(sub.iloc[0]), zorder=2)
                labelled = True

# Horizontal offset between experiment labels sharing a DIV, when the caller set no dodge of its own.
_EXPERIMENT_DODGE = 0.15


def plot_metric_grid(df, metrics, group_col='div', phase_col='phase',
                      overlay_df=None, overlay_col='culture_id', experiment_col='experiment',
                      series_col=None, err_label='SEM', dodge=0.0,
                      suptitle=None, save_path=None, show_plot=True,
                      ncols=2, figsize=(11, 8), dpi=150):
    """
    Grid of (metric vs group_col) panels, one errorbar series per (experiment, phase).

    A phase keeps its colour across experiment labels; labels are told apart by marker and
    linestyle (and by colour too when there is only one phase), named in the legend, and dodged
    apart on the x axis. With a single label (the usual case) the figure is one series per phase,
    exactly as before labels existed. Data with no phases but ``"full"`` leaves the phase out of
    the legend.

    With ``series_col`` set, its values drive the series instead (one colour per value) and the
    phase/experiment split is skipped -- the caller restricts ``df`` to what it wants compared,
    e.g. one phase across several groups of cultures.

    :param df: DataFrame containing group_col, phase_col, and every y_col/err_col in metrics.
    :param metrics: list of (y_col, err_col_or_None, y_label, panel_title) tuples, one per panel.
    :param group_col: x-axis grouping column (usually 'div').
    :param phase_col: column identifying each series (usually 'phase').
    :param overlay_df: Optional per-culture rows (one row per culture per group_col value, i.e. the
        un-aggregated frame `df` was built from). Each culture is drawn as a faint line behind the
        aggregate series, so an outlier or a dying culture stays visible under the mean.
    :param overlay_col: Column of ``overlay_df`` identifying each culture (usually 'culture_id').
    :param experiment_col: Column holding each row's ``experiment`` label; ignored if absent.
    :param series_col: Column whose values each become one series, overriding the phase split.
    :param err_label: Name of the error measure in the legend when an overlay is drawn
        (``"SEM"``, ``"SD"``).
    :param dodge: Horizontal offset between series, in x units, so series sharing an x value
        (e.g. a single recording's phases) don't hide each other's error bars. Defaults to a small
        offset when several experiment labels are plotted.
    :param suptitle: figure-level title, or None to omit.
    :param save_path: full path (including filename) to save the figure to, or None to skip saving.
    :param show_plot: whether to call plt.show() (mirrors the show_plot arg used elsewhere).
    :return: the matplotlib Figure.
    """
    nrows = -(-len(metrics) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    phases = sort_phases(df[phase_col].dropna().unique()) if phase_col in df.columns else []
    unphased = is_unphased(phases)
    experiments = _experiments(df, experiment_col)
    multi_experiment = len(experiments) > 1
    # Experiment labels sharing a DIV would otherwise stack their error bars on one x.
    if series_col is None and multi_experiment and not dodge:
        dodge = _EXPERIMENT_DODGE
    series_cmap = plt.get_cmap('tab10')
    if series_col is not None:
        series_values = list(dict.fromkeys(df[series_col].dropna()))  # first-seen order
        n_total = len(series_values)
    else:
        n_total = len(experiments) * len(phases)

    def _offset(k):
        return (k - (n_total - 1) / 2) * dodge

    # Position of each (experiment, phase) series, shared by the bold series and the faint overlay
    # so a culture's line sits at the same x as the mean it belongs to.
    experiment_index = {e: i for i, e in enumerate(experiments)}
    offsets = {(e, p): _offset(i * len(phases) + j)
               for i, e in enumerate(experiments) for j, p in enumerate(phases)}

    overlay_colors = {}
    if overlay_df is not None and not overlay_df.empty:
        # A qualitative map indexed discretely: sampling it continuously (as `get_n_colors` does for
        # the sequential maps) lands between its entries and washes the faint lines out.
        cmap = plt.get_cmap('tab10')
        cultures = sorted(overlay_df[overlay_col].dropna().unique())
        overlay_colors = {c: cmap(i % cmap.N) for i, c in enumerate(cultures)}

    for ax, (y_col, err_col, y_label, title) in zip(axes, metrics):
        if overlay_colors and y_col in overlay_df.columns:
            _draw_culture_overlay(ax, overlay_df, y_col, group_col, overlay_col, overlay_colors,
                                  experiment_col, phase_col, experiment_index=experiment_index,
                                  offsets=offsets if series_col is None else None)

        n_series = 0
        if series_col is not None:
            for i, value in enumerate(series_values):
                sub = df[df[series_col] == value].dropna(subset=[group_col]).sort_values(group_col)
                ax.errorbar(
                    sub[group_col].values + _offset(i), sub[y_col].values,
                    yerr=sub[err_col].values if err_col else None,
                    label=str(value),
                    color=series_cmap(i % series_cmap.N),
                    marker=_SERIES_MARKERS[i % len(_SERIES_MARKERS)],
                    linewidth=1.8, markersize=6, capsize=4, elinewidth=1.2, zorder=4,
                )
                n_series += 1

        for i, experiment in enumerate(experiments if series_col is None else []):
            linestyle = _EXPERIMENT_LINESTYLES[i % len(_EXPERIMENT_LINESTYLES)]
            for phase in phases:
                style = PHASE_STYLE.get(phase, {'color': 'gray', 'marker': 'D', 'label': phase})
                sub = _with_experiment(df[df[phase_col] == phase], experiment_col, experiment)
                sub = sub.dropna(subset=[group_col]).sort_values(group_col)
                if sub.empty:
                    continue

                # Colour means phase and marker means experiment label -- except that with one phase
                # the colour is free, so it tells labels apart too.
                color, marker = style['color'], style['marker']
                if multi_experiment:
                    marker = _SERIES_MARKERS[i % len(_SERIES_MARKERS)]
                    if len(phases) == 1:
                        color = series_cmap(i % series_cmap.N)

                parts = [experiment_label(experiment)] if multi_experiment else []
                if not unphased:
                    parts.append(style['label'])
                label = " · ".join(parts)
                if overlay_colors:
                    stat = f"mean ± {err_label}" if err_col else "mean"
                    label = f"{label} ({stat})" if label else stat

                ax.errorbar(
                    sub[group_col].values + offsets[(experiment, phase)], sub[y_col].values,
                    yerr=sub[err_col].values if err_col else None,
                    label=label or '_nolegend_', color=color, marker=marker, linestyle=linestyle,
                    linewidth=1.8, markersize=6, capsize=4, elinewidth=1.2, zorder=4,
                )
                n_series += 1

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(group_col.upper(), fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        if dodge:
            # Offsets put points between whole DIVs; keep the ticks on the real values.
            xs = sorted(df[group_col].dropna().unique())
            ax.set_xticks(xs)
            if len(xs) == 1:
                ax.set_xlim(xs[0] - 0.5, xs[0] + 0.5)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        # A per-culture overlay adds one legend entry per culture, so give it columns and a smaller
        # font rather than letting it eat the panel.
        n_series += len(overlay_colors)
        # A lone unphased, unlabelled series has nothing to name.
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7 if overlay_colors else 9, framealpha=0.8,
                      ncol=max(1, n_series // 6 + 1))
        ax.spines[['top', 'right']].set_visible(False)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight='bold')

    return finish_figure(fig, save_path=save_path, show_plot=show_plot, dpi=dpi)


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
    """Grid of ECDF panels, one line per recording.

    Where :func:`plot_metric_grid` shows a summary statistic moving over development, this shows the
    whole distribution behind it -- a culture whose mean firing rate holds steady while its
    distribution splits into a silent and a hyperactive population looks identical in one and obvious
    in the other.

    :param dists: ``{(div, experiment): {metric_key: array_of_values}}``, e.g. from
        :func:`mxtreme.analysis.activity.culture_distributions`. Plain ``{div: ...}`` keys work too.
    :param metrics: list of ``(metric_key, x_label, panel_title, logx)`` tuples, one per panel.
    :param suptitle: figure-level title, or None to omit.
    :param save_path: full path (including filename) to save to, or None to skip saving.
    :param show_plot: whether to call ``plt.show()``.
    :param cmap_name: sequential colormap mapped over the recordings in DIV order, so color reads as
        developmental time.
    :return: the matplotlib Figure.
    """
    nrows = -(-len(metrics) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    rec_keys = sorted(dists)
    colors = dict(zip(rec_keys, get_n_colors(len(rec_keys), cmap_name=cmap_name)))

    for ax, (key, x_label, title, logx) in zip(axes, metrics):
        for rec_key in rec_keys:
            x, y = _ecdf(dists[rec_key].get(key, []))
            if x.size == 0:
                continue
            if logx:
                # A log axis can't show non-positive values; they are not meaningful for any of
                # these quantities (rates, intervals, sizes) anyway.
                positive = x > 0
                x, y = x[positive], y[positive]
                if x.size == 0:
                    continue
            label = recording_label(*rec_key) if isinstance(rec_key, tuple) else recording_label(rec_key, None)
            ax.plot(x, y, color=colors[rec_key], linewidth=1.5, label=label)

        if logx:
            ax.set_xscale('log')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel('Cumulative fraction', fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=7, framealpha=0.8, ncol=max(1, len(rec_keys) // 6 + 1), loc='lower right')
        ax.spines[['top', 'right']].set_visible(False)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight='bold')

    return finish_figure(fig, save_path=save_path, show_plot=show_plot, dpi=dpi)
