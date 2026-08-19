"""Smoke tests for the data-level visualizations (``mxtreme.visualizations``).

Uses the non-interactive Agg backend; asserts the plots build without error, not their appearance.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mxtreme import visualizations as viz


def _spike_bin():
    rng = np.random.default_rng(0)
    return (rng.random((32, 500)) > 0.9).astype(float)


def _channelmap(n=32):
    cm = np.zeros((n, 5))
    cm[:, 0] = np.arange(n)
    cm[:, 1] = np.arange(n)
    cm[:, 2] = np.arange(n)
    # An 8x4 grid over the array -- a real channel map spans two dimensions.
    cm[:, 3] = (np.arange(n) % 8) * 400.0    # x (µm)
    cm[:, 4] = (np.arange(n) // 8) * 400.0   # y (µm)
    return cm


def test_plot_asdr_on_ax():
    fig, ax = plt.subplots()
    out = viz.plot_asdr(_spike_bin(), ax=ax)
    assert out is ax
    plt.close(fig)


def test_raster_on_ax():
    fig, ax = plt.subplots()
    out = viz.raster(_spike_bin(), zoom=(0, 200), ax=ax)
    assert out is ax
    plt.close(fig)


def test_mea_runs_with_and_without_stim():
    fig, ax = plt.subplots()
    viz.MEA(ax, _channelmap(), stim_elecs=None)
    viz.MEA(ax, _channelmap(), stim_elecs=np.array([0, 5]))
    plt.close(fig)


class _FakeRecording:
    """Minimal stand-in exposing just what the burst overlays read off a Recording."""

    def __init__(self, channelmap):
        self.channelmap = channelmap
        self.stim_elecs = np.array([])


def _burst_df(n=6, stringify=False):
    """A burst frame shaped like BurstSet.to_dataframe() output, with n network bursts."""
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        counts = {int(c): int(rng.integers(1, 20)) for c in rng.choice(32, size=5, replace=False)}
        rows.append({
            "kind": "network",
            "phase": "full",
            "origin_chan_counts": str(counts) if stringify else counts,
            "origin_x": float(rng.uniform(0, 3000)),
            "origin_y": float(rng.uniform(0, 1800)),
        })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("method", ["kde", "hist", "per_burst"])
def test_plot_origin_heatmap_methods_run(method):
    fig, ax = plt.subplots()
    out = viz.plot_origin_heatmap(_FakeRecording(_channelmap()), _burst_df(), ax=ax, method=method)
    assert out is ax
    plt.close(fig)


@pytest.mark.parametrize("method", ["kde", "hist", "per_burst"])
def test_plot_origin_heatmap_accepts_stringified_counts(method):
    """Counts loaded back from a burst CSV are strings, not dicts (the _as_dict path)."""
    fig, ax = plt.subplots()
    viz.plot_origin_heatmap(
        _FakeRecording(_channelmap()), _burst_df(stringify=True), ax=ax, method=method
    )
    plt.close(fig)


@pytest.mark.parametrize("method", ["kde", "hist", "per_burst"])
def test_plot_origin_heatmap_handles_no_matching_bursts(method):
    """No bursts of the requested kind means no density layer, but the MEA panel still renders."""
    df = _burst_df()
    df["kind"] = "mini"
    fig, ax = plt.subplots()
    viz.plot_origin_heatmap(_FakeRecording(_channelmap()), df, ax=ax, method=method)
    plt.close(fig)


def test_plot_origin_heatmap_rejects_unknown_method():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Unknown method"):
        viz.plot_origin_heatmap(_FakeRecording(_channelmap()), _burst_df(), ax=ax, method="nope")
    plt.close(fig)


def test_origin_heatmap_bandwidth_is_data_independent():
    """The default hist bandwidth comes from the channel map, not from the burst sample."""
    cm = _channelmap()
    rec = _FakeRecording(cm)
    expected = viz._median_nn_distance(cm)

    # Same array, very different burst samples -> same bandwidth.
    assert viz._median_nn_distance(rec.channelmap) == expected
    assert expected == pytest.approx(400.0)  # the 400 µm grid spacing in _channelmap


def test_origin_heatmap_density_is_normalised_per_burst():
    """Doubling the burst count with identical bursts leaves the per-burst density unchanged."""
    import pandas as pd

    cm = _channelmap()
    chip_ht = 120 * 17.5
    one = _burst_df(n=4)
    two = pd.concat([one, one], ignore_index=True)

    def peak(df):
        xs, ys, w = viz._origin_channel_density(cm, df, chip_ht)
        d, _, _ = viz._smoothed_density(
            xs, ys, w / len(df), bandwidth_um=400.0, chip_wd_um=220 * 17.5, chip_ht_um=chip_ht,
        )
        return d.max()

    assert peak(one) == pytest.approx(peak(two))


def test_origin_heatmap_vmax_pins_the_color_scale():
    fig, ax = plt.subplots()
    viz.plot_origin_heatmap(_FakeRecording(_channelmap()), _burst_df(), ax=ax,
                            method="hist", vmax=0.5, colorbar=False)
    mesh = [c for c in ax.collections if hasattr(c, "get_clim")][0]
    assert mesh.get_clim() == (0.0, 0.5)
    plt.close(fig)
