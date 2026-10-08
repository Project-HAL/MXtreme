"""Basic, data-level visualizations of a preprocessed recording (no analysis).

Views of the cleaned/binned data:
- :func:`MEA` -- spatial map of the electrodes on the array.
- :func:`plot_asdr` -- array-wide spike detection rate (ASDR) time series.
- :func:`raster` -- raster of the binned spikes.
- :func:`plot_timeseries_heatmap` -- a (rows x time) matrix as a heat map, e.g. instantaneous firing rate.

Helpers:
- :func:`finish_figure` -- the save / show / close tail shared by figure-building functions.

Burst overlays (consume a burst DataFrame from :meth:`BurstSet.to_dataframe
<mxtreme.bursting.detection.BurstSet.to_dataframe>`):
- :func:`plot_bursts_on_asdr` -- ASDR with burst peaks marked.
- :func:`plot_origin_heatmap` -- burst-origin density over the MEA (``method="hist"``/``"kde"``).
- :func:`plot_burst_vectors` -- origin->peak arrows over the MEA.

Each accepts an optional ``ax`` so panels can be composed (e.g. a shared-x ASDR-over-raster figure); when
``ax`` is ``None`` the function makes its own figure and shows it.
"""

import ast

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

from mxtreme import device


def MEA(channelmap, ax=None, stim_elecs=None, title="MEA Channel Layout", marker_size=36, stim_fontsize=16):
    """Plot the electrode layout of the MEA, marking stimulation electrodes.

    :param ax: Matplotlib axes to draw on.
    :param channelmap: ``(n, 5)`` channel map (cols: index, channel, electrode, x, y; positions in µm).
    :param stim_elecs: Electrode IDs to mark with a ⚡ (may be ``None``/empty).
    :param title: Axes title.
    :param marker_size: Electrode marker area. The default suits a full-page axes; shrink it when
        drawing the array into a small panel, where ~1k default-sized markers merge into a blob.
    :param stim_fontsize: Size of the ⚡ marking stimulation electrodes; scale it with ``marker_size``.
    """
    owns_fig = ax is None
    if owns_fig:
        _, ax = plt.subplots(1, 1, figsize=(12, 6))

    mid_point = (device.CHIP_WIDTH / 2) * device.ELEC_SIZE  # x midpoint of the array
    chip_ht_um = device.CHIP_HEIGHT * device.ELEC_SIZE

    # channelmap[:, 3] are x locations, channelmap[:, 4] are y locations
    ax.scatter(channelmap[:, 3], chip_ht_um - channelmap[:, 4], marker=".", s=marker_size,
               facecolors="none", edgecolors="k")
    ax.plot([mid_point, mid_point], [0, chip_ht_um], "b--", label="Midpoint")
    ax.set_title(title)
    ax.set_xlabel("x direction (µm)")
    ax.set_ylabel("y direction (µm)")

    if stim_elecs is None:
        return

    # Mark stimulation electrodes (channelmap[:, 2] is the electrode ID) with lightning-bolt emojis
    stim_coords = channelmap[np.isin(channelmap[:, 2], stim_elecs), 3:5]
    for x, y in stim_coords:
        txt = ax.text(x, chip_ht_um - y, "⚡", fontsize=stim_fontsize, ha="center", va="center",
                      color="gold")
        txt.set_path_effects([path_effects.Stroke(linewidth=1, foreground="black"), path_effects.Normal()])


def plot_asdr(spike_bin, title=None, zoom=None, ax=None, save_path=None, savefilename=None):
    """Plot the array-wide spike detection rate (ASDR): mean binned activity across channels over time.

    To overlay detected bursts, use :func:`plot_bursts_on_asdr`.

    :param spike_bin: ``(n_channels, n_bins)`` binary spike matrix.
    :param title: Axes title.
    :param zoom: Optional ``(start_bin, end_bin)`` x-limits.
    :param ax: Axes to draw on. If ``None``, a new figure is created and shown/saved.
    :param save_path: If given (and ``ax`` is ``None``), directory to save into.
    :param savefilename: Filename used with ``save_path``.
    """
    asdr = spike_bin.mean(axis=0)

    owns_fig = ax is None
    if owns_fig:
        _, ax = plt.subplots(1, 1, figsize=(12, 3))

    ax.plot(asdr)
    ax.set_ylabel("ASDR", fontsize=10)
    ax.set_xlabel("Time (bins)", fontsize=10)
    ax.set_title(title, fontsize=15)

    if zoom:
        ax.set_xlim(zoom)

    if not owns_fig:
        return ax
    if save_path is not None:
        plt.savefig(save_path / savefilename, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def raster(spike_bin, zoom=None, ax=None, title=None):
    """Plot a raster (channels x time) of the binned spikes.

    :param spike_bin: ``(n_channels, n_bins)`` binary spike matrix.
    :param zoom: Optional ``(start_bin, end_bin)`` window of bins to display.
    :param ax: Axes to draw on. If ``None``, a new figure is created and shown.
    :param title: Axes title.
    """
    owns_fig = ax is None
    if owns_fig:
        _, ax = plt.subplots(1, 1, figsize=(12, 3))

    ax.imshow(spike_bin, cmap="binary", aspect="auto", interpolation="none")
    ax.set_ylabel("Channels", fontsize=10)
    ax.set_xlabel("Time (bins)", fontsize=10)
    ax.set_title(title, fontsize=15)

    if zoom:
        ax.set_xlim(zoom)

    if owns_fig:
        plt.show()
    return ax


def plot_timeseries_heatmap(matrix, time=None, rows=None, ax=None, cmap="magma", vmin=None, vmax=None,
                            log=False, colorbar=True, cbar_label=None, title=None,
                            xlabel="Time (s)", ylabel="Channel"):
    """Heat map of a (rows x time) matrix, e.g. per-channel firing rate over time.

    Built for the output of :func:`mxtreme.analysis.activity.instantaneous_firing_rate`, whose
    ``rate`` / ``time_sec`` / ``channels`` map straight onto ``matrix`` / ``time`` / ``rows``::

        ifr = activity.instantaneous_firing_rate(rec, save=False)
        viz.plot_timeseries_heatmap(ifr["rate"], ifr["time_sec"], cbar_label="Firing rate (Hz)")

    :param matrix: ``(n_rows, n_times)`` array.
    :param time: Optional length-``n_times`` array of bin start times, used for the x extent. Bins
        are assumed evenly spaced. ``None`` labels the x axis in bin indices.
    :param rows: Optional row labels (e.g. channel ids). Only the row count is used; with ~1k channels,
        per-row tick labels would be unreadable, so the y axis shows row positions.
    :param ax: Axes to draw on. If ``None``, a new figure is created and shown.
    :param cmap: Colormap.
    :param vmin: Lower colour limit; defaults to the data minimum.
    :param vmax: Upper colour limit. Defaults to the 99th percentile, so a few saturated
        bins don't flatten the rest of the map.
    :param log: Use a logarithmic colour scale (non-positive values are drawn as the lowest colour).
    :param colorbar: Attach a colour bar.
    :param cbar_label: Colour bar label.
    :param title: Axes title.
    :param xlabel: X-axis label.
    :param ylabel: Y-axis label.
    :returns: The ``AxesImage``.
    """
    from matplotlib.colors import LogNorm

    matrix = np.asarray(matrix, dtype=float)
    n_rows, n_times = matrix.shape

    owns_fig = ax is None
    if owns_fig:
        _, ax = plt.subplots(1, 1, figsize=(12, 4))

    if time is not None and len(time) > 0:
        time = np.asarray(time, dtype=float)
        step = time[1] - time[0] if len(time) > 1 else 1.0
        extent = (time[0], time[-1] + step, n_rows, 0)
    else:
        extent = (0, n_times, n_rows, 0)
        if xlabel == "Time (s)":
            xlabel = "Time (bins)"

    finite = matrix[np.isfinite(matrix)]
    if vmax is None:
        vmax = float(np.percentile(finite, 99)) if finite.size else None
    if log:
        positive = finite[finite > 0]
        lo = vmin if vmin is not None else (float(positive.min()) if positive.size else 1e-3)
        norm = LogNorm(vmin=lo, vmax=max(vmax or lo * 10, lo * 10))
        image = ax.imshow(np.where(matrix > 0, matrix, lo), cmap=cmap, norm=norm, aspect="auto",
                          interpolation="nearest", extent=extent)
    else:
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                          interpolation="nearest", extent=extent)
    # ~1k rows x thousands of bins is millions of vector quads in a PDF and looks identical as a raster.
    image.set_rasterized(True)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel if rows is None else f"{ylabel} ({len(rows)})", fontsize=10)
    if title:
        ax.set_title(title, fontsize=12)
    if colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.03, pad=0.01, label=cbar_label)

    if owns_fig:
        plt.show()
    return image


def finish_figure(fig, save_path=None, show_plot=True, dpi=150):
    """Save ``fig`` (creating parent directories) and/or show it; close it when not shown.

    The tail every figure-building function ends with, in one place.

    :param fig: The matplotlib Figure.
    :param save_path: Full path (including filename) to save to, or ``None`` to skip saving.
    :param show_plot: Call ``plt.show()``; otherwise the figure is closed to free its memory.
    :param dpi: Resolution for the saved file.
    :returns: ``fig``.
    """
    if save_path:
        from pathlib import Path

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved plot → {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig


# --- burst overlays -----------------------------------------------------------------------------


def _as_dict(value):
    """Parse a channel-counts cell that may be a dict (in-memory) or a str (loaded from CSV)."""
    if isinstance(value, dict):
        return value
    try:
        return dict(ast.literal_eval(value))
    except (ValueError, SyntaxError, TypeError):
        return {}


def _select(burst_df, kind, phase):
    """Return the burst rows matching ``kind`` (and ``phase`` if given)."""
    df = burst_df
    if kind is not None:
        df = df[df["kind"] == kind]
    if phase is not None:
        df = df[df["phase"] == phase]
    return df


def plot_bursts_on_asdr(recording, burst_df, ax=None, zoom=None, kind="network", title=None):
    """Plot a recording's ASDR with detected burst peaks marked.

    :param recording: The :class:`~mxtreme.recording.Recording` (for ``asdr``, ``samp_rate``, ``bin_size``).
    :param burst_df: Burst DataFrame (new-schema columns ``peak_frame`` / ``peak_amp`` / ``kind``).
    :param ax: Axes to draw on. If ``None``, a new figure is created and shown.
    :param zoom: Optional ``(start_bin, end_bin)`` x-limits.
    :param kind: Burst kind to mark (``"network"``, ``"mini"``, or ``None`` for all).
    :param title: Axes title.
    """
    owns_fig = ax is None
    if owns_fig:
        _, ax = plt.subplots(1, 1, figsize=(12, 3))

    asdr = recording.asdr
    ax.plot(asdr)
    ax.set_ylabel("ASDR", fontsize=10)
    ax.set_xlabel("Time (bins)", fontsize=10)
    ax.set_title(title, fontsize=15)

    df = _select(burst_df, kind, None)
    if len(df):
        peak_bins = (df["peak_frame"] / (recording.samp_rate * recording.bin_size)).astype(int)
        ax.scatter(peak_bins, df["peak_amp"] + 0.02, marker="v", color="red", s=12)

    if zoom:
        ax.set_xlim(zoom)

    if owns_fig:
        plt.show()
    return ax


def _origin_channel_density(channelmap, burst_df, chip_ht_um):
    """Pool every burst's origin channels into flat ``(xs, ys, weights)`` arrays.

    Each burst contributes one point per channel in its ``origin_chan_counts``, weighted by that
    channel's spike count, so a channel that fires heavily at burst onset counts for more than one
    that barely participates. ``ys`` are already inverted to plot coordinates.
    """
    xs, ys, weights = [], [], []
    for counts in burst_df["origin_chan_counts"]:
        counts = _as_dict(counts)
        if not counts:
            continue
        chanmap_slice = channelmap[np.isin(channelmap[:, 1], list(counts.keys()))]
        if not len(chanmap_slice):
            continue
        xs.append(chanmap_slice[:, 3])
        ys.append(chip_ht_um - chanmap_slice[:, 4])  # invert y
        weights.append([counts.get(c, 0) for c in chanmap_slice[:, 1]])

    if not xs:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(weights).astype(float)


def _is_degenerate(xs, ys) -> bool:
    """Whether a 2-D KDE cannot be fitted: fewer than three points, or all of them collinear."""
    if len(xs) < 3:
        return True
    centered = np.column_stack([xs - xs.mean(), ys - ys.mean()])
    return bool(np.linalg.matrix_rank(centered, tol=1e-8) < 2)


def _median_nn_distance(channelmap) -> float:
    """Median nearest-neighbour distance (µm) between the recording's routed electrodes.

    This is the array's *effective* electrode spacing: MaxOne/MaxTwo route ~1k channels out of a much
    denser grid, so neighbouring routed electrodes sit far further apart than ``device.ELEC_SIZE``.
    Matches ``median_nn_distance_um`` from :func:`mxtreme.analysis.spatial.compute_spatial_metrics`.
    """
    xy = np.asarray(channelmap)[:, 3:5]
    if len(xy) < 2:
        return float(device.ELEC_SIZE)
    dists = cdist(xy, xy)
    np.fill_diagonal(dists, np.inf)
    return float(np.median(dists.min(axis=1)))


def _smoothed_density(xs, ys, weights, *, bandwidth_um, chip_wd_um, chip_ht_um):
    """Bin the pooled origin channels onto the electrode grid and smooth with a **fixed** kernel.

    The bandwidth is an absolute distance in µm, so it does not shift with sample size or spatial
    spread the way Scott's / Silverman's rules do. That is what makes two of these plots comparable:
    the only thing that varies between them is the data.
    """
    density, x_edges, y_edges = np.histogram2d(
        xs, ys, bins=(device.CHIP_WIDTH, device.CHIP_HEIGHT),
        range=((0, chip_wd_um), (0, chip_ht_um)), weights=weights,
    )
    # Bins are one electrode pitch wide, so µm -> bins is a division by the pitch.
    return gaussian_filter(density, sigma=bandwidth_um / device.ELEC_SIZE), x_edges, y_edges


def plot_origin_heatmap(recording, burst_df, ax=None, phase=None, kind="network",
                        color="r", kde_cmap="Blues", title=None, method="hist",
                        bandwidth_um=None, vmax=None, colorbar=True, marker_size=100,
                        elec_marker_size=36):
    """Plot burst-origin locations and their spiking-channel density over the MEA.

    The density layer pools every burst's ``origin_chan_counts``, weighting each electrode by its
    spike count, and is reported as **spikes per burst per electrode bin** -- normalising by burst
    count so that cultures with different numbers of bursts sit on the same scale. Pin ``vmax`` to
    fix the color scale across a set of plots; otherwise each plot scales to its own peak and colors
    are not comparable between figures.

    :param recording: The :class:`~mxtreme.recording.Recording` (for the channel map / stim electrodes).
    :param burst_df: Burst DataFrame with ``origin_x`` / ``origin_y`` / ``origin_chan_counts`` / ``kind``.
    :param ax: Axes to draw on. If ``None``, a new figure is created and shown.
    :param phase: If given, restrict to bursts with this ``phase``.
    :param kind: Burst kind to include (default ``"network"``; ``None`` for all).
    :param color: Marker color for origin points.
    :param kde_cmap: Colormap for the origin-channel density.
    :param title: Axes title. The burst and electrode counts are appended to it either way, since the
        density is a mean over those bursts and is not interpretable without them.
    :param method: How to render the density layer:

        - ``"hist"`` (default) -- count-weighted 2-D histogram on the electrode grid, smoothed with a
          fixed-width Gaussian. The only method with a data-independent bandwidth and a pinnable
          color scale, and by far the cheapest.
        - ``"kde"`` -- one count-weighted :func:`seaborn.kdeplot` over the pooled channels.
        - ``"per_burst"`` -- one KDE per burst, overlaid translucently (the original behaviour).

        .. warning:: Both KDE methods pick their bandwidth by Scott's rule, which scales with each
           sample's size and spread. Differences between two such plots may therefore be a bandwidth
           artifact rather than a real difference in spatial distribution, and their filled contours
           are normalised per plot so ``vmax`` cannot pin them. Use ``"hist"`` to compare plots.
    :param bandwidth_um: Smoothing bandwidth in µm for ``method="hist"``. Defaults to the recording's
        median nearest-neighbour electrode distance -- i.e. tied to the array's routed geometry, not
        to the burst sample, which is what makes two plots comparable. (Note this is *not*
        ``device.ELEC_SIZE``: only ~1k of the array's electrodes are routed, so their effective
        spacing is several times the physical pitch.) Pass an explicit value to pin the bandwidth
        across recordings whose channel maps differ.
    :param vmax: Upper limit of the density color scale (``"hist"`` only). ``None`` scales each plot
        to its own peak; pass the same value across plots to preserve relative magnitude between them.
    :param colorbar: Draw a colorbar for the density (``"hist"`` only), so the scale is readable.
    :param marker_size: Area of the origin markers; shrink it when drawing into a small panel.
    :param elec_marker_size: Area of the electrode markers, passed to :func:`MEA`.
    """
    if method not in ("kde", "hist", "per_burst"):
        raise ValueError(f"Unknown method {method!r}; expected 'hist', 'kde' or 'per_burst'.")
    if bandwidth_um is None:
        bandwidth_um = _median_nn_distance(recording.channelmap)

    owns_fig = ax is None
    if owns_fig:
        _, ax = plt.subplots(1, 1, figsize=(9, 5))

    chip_wd_um = device.CHIP_WIDTH * device.ELEC_SIZE
    chip_ht_um = device.CHIP_HEIGHT * device.ELEC_SIZE
    channelmap = recording.channelmap
    bursts = _select(burst_df, kind, phase)

    xs, ys, weights = _origin_channel_density(channelmap, bursts, chip_ht_um)
    n_bursts = len(bursts)
    n_elec = len(np.unique(np.column_stack([xs, ys]), axis=0)) if len(xs) else 0

    if method == "per_burst":
        for _, burst in bursts.iterrows():
            channels = list(_as_dict(burst["origin_chan_counts"]).keys())
            if not channels:
                continue
            chanmap_slice = channelmap[np.isin(channelmap[:, 1], channels)]
            burst_xs = chanmap_slice[:, 3]
            burst_ys = chip_ht_um - np.array(chanmap_slice[:, 4])  # invert y
            sns.kdeplot(x=burst_xs, y=burst_ys, fill=True, alpha=0.2, cmap=kde_cmap,
                        warn_singular=False, ax=ax)
    # A 2-D KDE can't be fitted to a degenerate cloud (all origin channels on one row/column of the
    # array); fall back to the fixed-bandwidth histogram there rather than failing.
    elif len(xs) and method == "kde" and not _is_degenerate(xs, ys):
        sns.kdeplot(x=xs, y=ys, weights=weights, fill=True, alpha=0.6, cmap=kde_cmap,
                    warn_singular=False, ax=ax)
    elif len(xs):
        density, x_edges, y_edges = _smoothed_density(
            xs, ys, weights / max(n_bursts, 1),
            bandwidth_um=bandwidth_um, chip_wd_um=chip_wd_um, chip_ht_um=chip_ht_um,
        )
        mesh = ax.pcolormesh(x_edges, y_edges, density.T, cmap=kde_cmap, alpha=0.6,
                             shading="auto", vmin=0, vmax=vmax)
        if colorbar:
            ax.figure.colorbar(mesh, ax=ax, label="spikes per burst per electrode")

    x_origin = bursts["origin_x"].to_numpy()
    y_origin = bursts["origin_y"].to_numpy()

    MEA(channelmap, ax=ax, stim_elecs=recording.stim_elecs, title="", marker_size=elec_marker_size,
        stim_fontsize=max(6, 16 * elec_marker_size / 36))
    # Semi-transparent: with a few hundred bursts these markers otherwise merge into a solid blob and
    # hide both the density layer and their own concentration.
    ax.scatter(x_origin, chip_ht_um - y_origin, color=color, marker="x", s=marker_size,
               linewidths=3 * min(1, marker_size / 100), alpha=0.5)
    ax.set_xlim([0, chip_wd_um])
    ax.set_ylim([0, chip_ht_um])
    base = title if title is not None else "MEA electrode layout - burst origins"
    ax.set_title(f"{base}\n{n_bursts} bursts, {n_elec} electrodes")

    if owns_fig:
        plt.show()
    return ax


def plot_burst_vectors(recording, burst_df, ax=None, phase=None, kind="network", title=None):
    """Plot origin->peak arrows for each burst over the MEA.

    :param recording: The :class:`~mxtreme.recording.Recording` (for the channel map / stim electrodes).
    :param burst_df: Burst DataFrame with ``origin_x/y`` / ``peak_x/y`` / ``kind``.
    :param ax: Axes to draw on. If ``None``, a new figure is created and shown.
    :param phase: If given, restrict to bursts with this ``phase``.
    :param kind: Burst kind to include (default ``"network"``; ``None`` for all).
    :param title: Axes title.
    """
    owns_fig = ax is None
    if owns_fig:
        _, ax = plt.subplots(1, 1, figsize=(9, 5))

    chip_ht_um = device.CHIP_HEIGHT * device.ELEC_SIZE
    df = _select(burst_df, kind, phase)

    MEA(recording.channelmap, ax=ax, stim_elecs=recording.stim_elecs, title="")

    x_origin = df["origin_x"].to_numpy()
    y_origin = chip_ht_um - df["origin_y"].to_numpy()
    dx = df["peak_x"].to_numpy() - df["origin_x"].to_numpy()
    dy = (chip_ht_um - df["peak_y"].to_numpy()) - y_origin
    ax.quiver(x_origin, y_origin, dx, dy, angles="xy", scale_units="xy", scale=1, color="black",
              width=0.002, headwidth=3, headlength=4, headaxislength=3, alpha=0.6)

    ax.set_xlim([0, device.CHIP_WIDTH * device.ELEC_SIZE])
    ax.set_ylim([0, chip_ht_um])
    ax.set_title(title if title is not None else "MEA electrode layout - burst vectors")

    if owns_fig:
        plt.show()
    return ax
