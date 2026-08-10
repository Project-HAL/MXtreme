"""Basic, data-level visualizations of a preprocessed recording (no analysis).

Three views of the cleaned/binned data:
- :func:`MEA` -- spatial map of the electrodes on the array.
- :func:`plot_asdr` -- array-wide spike detection rate (ASDR) time series.
- :func:`raster` -- raster of the binned spikes.

Each accepts an optional ``ax`` so panels can be composed (e.g. a shared-x ASDR-over-raster figure); when
``ax`` is ``None`` the function makes its own figure and shows it.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from mxtreme import device


def MEA(ax, channelmap, stim_elecs, title="MEA Channel Layout"):
    """Plot the electrode layout of the MEA, marking stimulation electrodes.

    :param ax: Matplotlib axes to draw on.
    :param channelmap: ``(n, 5)`` channel map (cols: index, channel, electrode, x, y; positions in µm).
    :param stim_elecs: Electrode IDs to mark with a ⚡ (may be ``None``/empty).
    :param title: Axes title.
    """
    mid_point = (device.CHIP_WIDTH / 2) * device.ELEC_SIZE  # x midpoint of the array
    chip_ht_um = device.CHIP_HEIGHT * device.ELEC_SIZE

    # channelmap[:, 3] are x locations, channelmap[:, 4] are y locations
    num_chans_left = np.sum(channelmap[:, 3] < mid_point)
    num_chans_right = np.sum(channelmap[:, 3] > mid_point)
    print(f"Left: {num_chans_left}, Right: {num_chans_right}")

    ax.scatter(channelmap[:, 3], chip_ht_um - channelmap[:, 4], marker=".", facecolors="none", edgecolors="k")
    ax.plot([mid_point, mid_point], [0, chip_ht_um], "b--", label="Midpoint")
    ax.set_title(title)
    ax.set_xlabel("x direction (µm)")
    ax.set_ylabel("y direction (µm)")

    if stim_elecs is None:
        return

    # Mark stimulation electrodes (channelmap[:, 2] is the electrode ID) with lightning-bolt emojis
    stim_coords = channelmap[np.isin(channelmap[:, 2], stim_elecs), 3:5]
    for x, y in stim_coords:
        txt = ax.text(x, chip_ht_um - y, "⚡", fontsize=16, ha="center", va="center", color="gold")
        txt.set_path_effects([path_effects.Stroke(linewidth=1, foreground="black"), path_effects.Normal()])


def plot_asdr(spike_bin, title=None, zoom=None, ax=None, save_path=None, savefilename=None,
              annotate_bursts=False, bursts=None, burst_threshold=None):
    """Plot the array-wide spike detection rate (ASDR): mean binned activity across channels over time.

    :param spike_bin: ``(n_channels, n_bins)`` binary spike matrix.
    :param title: Axes title.
    :param zoom: Optional ``(start_bin, end_bin)`` x-limits.
    :param ax: Axes to draw on. If ``None``, a new figure is created and shown/saved.
    :param save_path: If given (and ``ax`` is ``None``), directory to save into.
    :param savefilename: Filename used with ``save_path``.
    :param annotate_bursts: If ``True`` and ``bursts`` is given, mark burst peaks.
    :param bursts: DataFrame with ``t_peak_bin`` and ``peak_bin_amp`` columns.
    :param burst_threshold: If given, draw a horizontal threshold line.
    """
    asdr = spike_bin.mean(axis=0)

    owns_fig = ax is None
    if owns_fig:
        _, ax = plt.subplots(1, 1, figsize=(12, 3))

    ax.plot(asdr)
    ax.set_ylabel("ASDR", fontsize=10)
    ax.set_xlabel("Time (bins)", fontsize=10)
    ax.set_title(title, fontsize=15)

    if annotate_bursts and bursts is not None:
        ax.scatter(bursts["t_peak_bin"], bursts["peak_bin_amp"] + 0.02, marker="v", color="red", s=10)

    if burst_threshold is not None:
        ax.hlines(burst_threshold, 0, len(asdr), colors="g", linestyles="dashed")

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
