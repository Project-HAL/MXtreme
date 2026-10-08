"""Spatial layout of a recording: where on the array its electrodes are, and how they're spread.

Recording level: :func:`spatial_metrics` describes one recording's electrode configuration.

Group level: :func:`summarize_spatial` / :func:`plot_spatial_summary` take a ``RecordingID``,
``CultureID``, ``CultureSelector`` or several selectors, as in :mod:`~mxtreme.analysis.activity`.

Culture level: :func:`plot_mea_layouts` draws each distinct electrode configuration a culture was
recorded with. There is no population version: overlaying absolute electrode positions from cultures
on different physical chips isn't meaningful the way averaging a scalar is.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from mxtreme import device, io
from mxtreme import visualizations as viz
from mxtreme.config import Config
from mxtreme.identity import CultureID
from mxtreme.paths import CulturePaths, resolve_paths
from mxtreme.recording import Recording
from mxtreme.analysis._group import SummarySpec, culture_summary, fill_docs, plot_summary, summarize
from mxtreme.analysis._paths import recording_label

__all__ = ["spatial_metrics", "summarize_spatial", "plot_spatial_summary", "plot_mea_layouts"]


def _metrics(channelmap) -> dict:
    """Electrode-spread metrics of a ``(n, 5)`` channel map (``index, channel, electrode, x, y``)."""
    xy = np.asarray(channelmap)[:, 3:5]
    n_electrodes = xy.shape[0]
    centroid_x, centroid_y = xy.mean(axis=0) if n_electrodes else (np.nan, np.nan)

    hull_area_um2 = ConvexHull(xy).volume if n_electrodes >= 3 else np.nan  # 2-D: volume is the area
    chip_area_um2 = (device.CHIP_WIDTH * device.ELEC_SIZE) * (device.CHIP_HEIGHT * device.ELEC_SIZE)
    pct_chip_covered = 100 * hull_area_um2 / chip_area_um2 if not np.isnan(hull_area_um2) else np.nan
    electrode_density = n_electrodes / hull_area_um2 if hull_area_um2 else np.nan

    if n_electrodes >= 2:
        dists = cdist(xy, xy)
        np.fill_diagonal(dists, np.inf)
        nn_dists = dists.min(axis=1)
        mean_nn, median_nn = nn_dists.mean(), np.median(nn_dists)
    else:
        mean_nn = median_nn = np.nan

    return {
        "n_electrodes": n_electrodes,
        "centroid_x_um": centroid_x,
        "centroid_y_um": centroid_y,
        "hull_area_um2": hull_area_um2,
        "pct_chip_covered": pct_chip_covered,
        "mean_nn_distance_um": mean_nn,
        "median_nn_distance_um": median_nn,
        "electrode_density": electrode_density,
    }


# --- recording level ------------------------------------------------------------------------------


def spatial_metrics(rec: Recording, analysis_dir=None, *, save: bool = True) -> pd.DataFrame:
    """Spread of one recording's electrodes over the array.

    The configuration is fixed for a recording, so there is one row (phase ``"full"``).

    :param rec: The recording.
    :param analysis_dir: Analysis output root; saved there as ``..._spatial_metrics.csv`` when ``save``
        is on.
    :param save: Write the result (needs ``analysis_dir``).
    :returns: One row: ``phase``, ``n_electrodes``, ``centroid_x_um`` / ``centroid_y_um``,
        ``hull_area_um2`` (convex hull of the electrodes), ``pct_chip_covered`` (that hull as a % of the
        array), ``mean_nn_distance_um`` / ``median_nn_distance_um`` (distance from each electrode to
        its nearest recorded neighbour -- how tightly the electrodes are packed), and
        ``electrode_density`` (electrodes per µm² of hull).
    """
    df = pd.DataFrame([{"phase": "full", **_metrics(rec.channelmap)}])

    if save and analysis_dir is not None:
        out_dir = Path(analysis_dir) / "spatial" / str(rec.batch_id) / str(rec.chip) / f"well{rec.well}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / io.recording_file_name(rec.DIV, rec.plate_date, rec.chip, rec.batch_id,
                                                rec.well, "spatial_metrics.csv", rec.experiment or "")
        df.to_csv(path, index=False)
        print(f"Saved spatial metrics to {path}")
    return df


# --- group level ----------------------------------------------------------------------------------

_SPEC = SummarySpec(
    title="Electrode Layout",
    metrics=[
        ("n_electrodes",        None, "Electrode count",                 "Electrode Count"),
        ("pct_chip_covered",    None, "Chip covered (%)",                "Chip Coverage"),
        ("mean_nn_distance_um", None, "Mean nearest-neighbour dist. (µm)", "Electrode Spacing"),
        ("electrode_density",   None, "Electrodes / µm²",                "Electrode Density"),
    ],
)


def _spatial_rows(rp) -> list[dict]:
    rec = Recording(0, io.load_preprocessed(rp.npz))
    return [{"div": rp.recording_id.div, "experiment": rp.recording_id.experiment or "",
             "phase": "full", **_metrics(rec.channelmap)}]


def _culture_spatial_summary(cpath, analysis_dir: Path, use_existing: bool = True) -> pd.DataFrame:
    """Per-culture spatial metrics (one row per recording), cached."""
    return culture_summary(cpath, analysis_dir, "spatial", "spatial_summary", _spatial_rows, use_existing)


def summarize_spatial(target, config: Config, *, use_existing: bool = True) -> pd.DataFrame:
    """Table of electrode-spread metrics: one row per recording (columns as in :func:`spatial_metrics`).

    Identity columns lead: ``batch_id, culture_id, chip, well, div, experiment, phase``, plus ``group``
    when several selectors are given.

    {target}
    :param use_existing: Reuse cached per-culture summaries; ``False`` recomputes the target's
        recordings.
    :returns: The summary table.
    """
    return summarize(target, config, _culture_spatial_summary, use_existing=use_existing)[0]


def plot_spatial_summary(target, config: Config, *, split_by: str = "phase", error: str = "sem",
                         show_plot: bool = True, save_path=None):
    """Four-panel electrode-spread summary: electrode count, chip coverage, spacing, density, vs DIV.

    {plot_doc}
    """
    return plot_summary(target, config, _culture_spatial_summary, _SPEC, split_by=split_by,
                        error=error, show_plot=show_plot, save_path=save_path)


fill_docs(summarize_fns=(summarize_spatial,), plot_fns=((plot_spatial_summary, None),))
# One phase only, so the shared plot docstring's phase notes don't apply.
plot_spatial_summary.__doc__ = plot_spatial_summary.__doc__.replace(
    """    :param phase: The phase to plot. Required in effect when splitting by experiment or comparing
        groups (defaults to the first phase); otherwise ``None`` plots every phase.
""", "")


# --- culture level --------------------------------------------------------------------------------


def plot_mea_layouts(culture, config: Config, *, show_plot: bool = True, save_path=None):
    """The electrode configurations one culture was recorded with, drawn on the array.

    Recordings sharing an identical configuration collapse into one panel, so a culture whose layout
    never changed gets a single panel, and any change over time shows up as a new one.

    :param culture: A :class:`~mxtreme.identity.CultureID`.
    :param config: The :class:`~mxtreme.config.Config` describing the store.
    :param show_plot: Show the figure.
    :param save_path: Full path (including filename) to save the figure to.
    :returns: The matplotlib Figure (``None`` if the culture has no recordings).
    """
    if isinstance(culture, CulturePaths):
        cpath = culture
    elif isinstance(culture, CultureID):
        cpath = resolve_paths(culture, config)
    else:
        raise TypeError(f"Pass a CultureID, not {type(culture).__name__}.")

    groups = []  # ([(div, experiment), ...], channelmap, stim_elecs)
    for rp in cpath.recordings:
        rec = Recording(0, io.load_preprocessed(rp.npz))
        key = (rp.recording_id.div, rp.recording_id.experiment or "")
        match = next((g for g in groups if np.array_equal(g[1], rec.channelmap)), None)
        if match:
            match[0].append(key)
        else:
            groups.append(([key], rec.channelmap, rec.stim_elecs))
    if not groups:
        return None

    ncols = min(3, len(groups))
    nrows = -(-len(groups) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax, (keys, cmap, stim_elecs) in zip(axes, groups):
        viz.MEA(cmap, ax=ax, stim_elecs=stim_elecs, title=", ".join(recording_label(*k) for k in keys))
    for ax in axes[len(groups):]:
        ax.set_visible(False)

    note = f" ({len(groups)} configurations)" if len(groups) > 1 else ""
    fig.suptitle(f"MEA Layout — {cpath.culture_id}{note}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return viz.finish_figure(fig, save_path=save_path, show_plot=show_plot)
