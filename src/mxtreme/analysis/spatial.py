import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from mxtreme import io
from mxtreme import device
from mxtreme import visualizations as viz
from mxtreme.recording import Recording
from mxtreme.analysis._paths import _summary_paths, load_population_summaries
from mxtreme.analysis._plotting import plot_metric_grid
from mxtreme.analysis._stats import aggregate_by_div_phase


def _load_channelmaps(cpath):
    """{div: (channelmap, stim_elecs)} for every recording in a culture. Not cached by
    use_existing since the plotting functions below always need the raw arrays, not a tabular
    summary of them."""
    channelmaps = {}
    for div in cpath.recordings:
        npz = cpath.recordings[div].npz
        rec = Recording(0, io.load_preprocessed(npz))
        channelmaps[div] = (rec.channelmap, rec.stim_elecs)
    return channelmaps


def compute_spatial_metrics(channelmap):
    """
    Spatial spread metrics for a recording's electrode configuration.

    :param channelmap: (n_electrodes, 5) array [index, channel, electrode, x_um, y_um]
    :return: dict of metrics
    """
    xy = channelmap[:, 3:5]
    n_electrodes = xy.shape[0]

    centroid_x, centroid_y = xy.mean(axis=0)

    if n_electrodes >= 3:
        hull_area_um2 = ConvexHull(xy).volume  # 2D points -> ConvexHull.volume is the area
    else:
        hull_area_um2 = np.nan

    chip_area_um2 = (device.CHIP_WIDTH * device.ELEC_SIZE) * (device.CHIP_HEIGHT * device.ELEC_SIZE)
    pct_chip_covered = 100 * hull_area_um2 / chip_area_um2 if not np.isnan(hull_area_um2) else np.nan
    electrode_density = n_electrodes / hull_area_um2 if hull_area_um2 else np.nan

    if n_electrodes >= 2:
        dists = cdist(xy, xy)
        np.fill_diagonal(dists, np.inf)
        nn_dists = dists.min(axis=1)
        mean_nn_distance_um = nn_dists.mean()
        median_nn_distance_um = np.median(nn_dists)
    else:
        mean_nn_distance_um = np.nan
        median_nn_distance_um = np.nan

    return {
        'n_electrodes':          n_electrodes,
        'centroid_x_um':         centroid_x,
        'centroid_y_um':         centroid_y,
        'hull_area_um2':         hull_area_um2,
        'pct_chip_covered':      pct_chip_covered,
        'mean_nn_distance_um':   mean_nn_distance_um,
        'median_nn_distance_um': median_nn_distance_um,
        'electrode_density':     electrode_density,
    }


def mea_layout_summary(cpath, analysis_dir: Path, use_existing=True, show_plot=True, save_plot=False):
    """
    Visualizes the MEA electrode configuration (core.visualizations.MEA) for a culture. Plots
    one panel per distinct electrode configuration found across the culture's DIVs -- a single
    panel if the configuration never changes, otherwise one per DIV (or DIV group) so any change
    over time is visible.
    """
    cid = cpath.culture_id

    save_path, csv_path = _summary_paths(cpath, analysis_dir, "spatial", "mea_layout_summary")

    channelmaps = None

    if use_existing and csv_path.exists():
        print(f"Loading existing summary from {csv_path}")
        summary_df = pd.read_csv(csv_path)
    else:
        channelmaps = _load_channelmaps(cpath)

        summary_df = pd.DataFrame([
            {'culture_id': str(cid), 'div': div, 'phase': 'full', 'n_electrodes': cmap.shape[0]}
            for div, (cmap, _) in channelmaps.items()
        ])

        os.makedirs(save_path, exist_ok=True)
        summary_df.to_csv(csv_path, index=False)
        print(f"Saved summary to {save_path}")

    if show_plot or save_plot:
        if channelmaps is None:
            channelmaps = _load_channelmaps(cpath)
        _plot_mea_layout_summary(channelmaps, cid, analysis_dir=save_path, show_plot=show_plot, save_plot=save_plot)

    return summary_df


def _plot_mea_layout_summary(channelmaps, cid, analysis_dir: Path, show_plot=True, save_plot=False):
    """One viz.MEA() panel per distinct electrode configuration across the culture's DIVs."""

    divs = sorted(channelmaps.keys())

    # Group DIVs that share an identical electrode configuration so unchanged layouts collapse
    # into one panel instead of a wall of duplicates.
    groups = []  # list of [divs], channelmap, stim_elecs
    for div in divs:
        cmap, stim_elecs = channelmaps[div]
        match = next((g for g in groups if np.array_equal(g[1], cmap)), None)
        if match:
            match[0].append(div)
        else:
            groups.append(([div], cmap, stim_elecs))

    n_panels = len(groups)
    ncols = min(3, n_panels)
    nrows = -(-n_panels // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, (div_group, cmap, stim_elecs) in zip(axes, groups):
        title = f"DIV {','.join(str(d) for d in div_group)}"
        viz.MEA(ax, cmap, stim_elecs, title=title)

    for ax in axes[len(groups):]:
        ax.set_visible(False)

    config_note = f' ({n_panels} configurations across DIVs)' if n_panels > 1 else ''
    fig.suptitle(f'MEA Layout — {cid}{config_note}', fontsize=13, fontweight='bold')
    fig.tight_layout()

    if save_plot:
        os.makedirs(analysis_dir, exist_ok=True)
        plot_path = analysis_dir / f"{cid}_mea_layout_summary.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot → {plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def spatial_summary(cpath, analysis_dir: Path, use_existing=True, show_plot=True, save_plot=False):
    """Electrode spread metrics (compute_spatial_metrics) per DIV for a culture."""

    cid = cpath.culture_id

    save_path, csv_path = _summary_paths(cpath, analysis_dir, "spatial", "spatial_summary")

    if use_existing and csv_path.exists():
        print(f"Loading existing summary from {csv_path}")
        summary_df = pd.read_csv(csv_path)
    else:
        rows = []
        for div in cpath.recordings:
            npz = cpath.recordings[div].npz
            rec = Recording(0, io.load_preprocessed(npz))

            rows.append({
                'culture_id': str(cid),
                'div': div,
                'phase': 'full',
                **compute_spatial_metrics(rec.channelmap),
            })

        summary_df = pd.DataFrame(rows)

        os.makedirs(save_path, exist_ok=True)
        summary_df.to_csv(csv_path, index=False)
        print(f"Saved summary to {save_path}")

    if show_plot or save_plot:
        _plot_spatial_summary(summary_df, cid, analysis_dir=save_path, show_plot=show_plot, save_plot=save_plot)

    return summary_df


def _plot_spatial_summary(df: pd.DataFrame, cid, analysis_dir: Path, show_plot=True, save_plot=False):
    """Four-panel grid of electrode-spread metrics vs DIV (see compute_spatial_metrics)."""

    metrics = [
        # (y_col,                  err_col,  y_label,                  panel_title)
        ('n_electrodes',           None,   'Electrode count',          'Electrode Count'),
        ('pct_chip_covered',       None,   'Chip covered (%)',         'Chip Coverage'),
        ('mean_nn_distance_um',    None,   'Mean NN distance (µm)',    'Electrode Spacing'),
        ('electrode_density',      None,   'Electrodes / µm²',         'Electrode Density'),
    ]

    save_path = (analysis_dir / f"{cid}_spatial_summary.png") if save_plot else None

    plot_metric_grid(
        df, metrics,
        suptitle=f'Spatial Summary — {cid}',
        save_path=save_path,
        show_plot=show_plot,
        figsize=(11, 8),
        dpi=150,
    )


def plot_population_spatial_summary(sel_paths, analysis_dir: Path, savename: str = None):
    """
    Aggregate per-culture spatial (electrode spread) summaries and plot population-level
    measures (mean ± SEM across cultures) vs DIV.

    No MEA-layout population counterpart -- overlaying absolute electrode positions across
    cultures on different physical chips isn't meaningful the way averaging a scalar metric is;
    mea_layout_summary is a per-culture-only diagnostic.
    """
    pop_df = load_population_summaries(
        sel_paths, data_dir=analysis_dir / "spatial", suffix='spatial_summary'
    )

    n_cultures = pop_df['culture_id'].nunique()
    n_exps     = len(sel_paths)
    exp_ids    = list(sel_paths.keys())

    stats = aggregate_by_div_phase(
        pop_df, value_cols=['n_electrodes', 'pct_chip_covered', 'mean_nn_distance_um', 'electrode_density']
    )

    metrics = [
        # (y_col,                err_col,                     y_label,                 panel_title)
        ('n_electrodes',        'sem_n_electrodes',          'Electrode count',       'Electrode Count'),
        ('pct_chip_covered',    'sem_pct_chip_covered',      'Chip covered (%)',      'Chip Coverage'),
        ('mean_nn_distance_um', 'sem_mean_nn_distance_um',   'Mean NN distance (µm)', 'Electrode Spacing'),
        ('electrode_density',   'sem_electrode_density',     'Electrodes / µm²',      'Electrode Density'),
    ]

    exp_label = f"{n_exps} experiments pooled ({', '.join(exp_ids)})" if n_exps > 1 else exp_ids[0]
    suptitle = f'Population Spatial Summary — {exp_label}\nmean ± SEM, n = {n_cultures} cultures'

    save_path = (Path(analysis_dir) / "spatial" / savename) if savename else None

    plot_metric_grid(stats, metrics, suptitle=suptitle, save_path=save_path,
                      show_plot=True, figsize=(11, 8), dpi=300)
