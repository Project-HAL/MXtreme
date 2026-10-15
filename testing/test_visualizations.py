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


@pytest.mark.parametrize("log", [False, True])
def test_timeseries_heatmap_spans_the_given_time_axis(log):
    rate = np.random.default_rng(0).poisson(3, size=(12, 50)).astype(float)
    time = np.arange(50) * 0.5

    fig, ax = plt.subplots()
    image = viz.plot_timeseries_heatmap(rate, time, rows=np.arange(12), ax=ax, log=log,
                                        cbar_label="Firing rate (Hz)")

    left, right, bottom, top = image.get_extent()
    assert (left, right) == (0.0, 25.0)  # last bin's start + its width
    assert (bottom, top) == (12, 0)      # first row at the top
    assert ax.get_xlabel() == "Time (s)"
    assert len(fig.axes) == 2            # the heat map and its colour bar
    plt.close(fig)


def test_timeseries_heatmap_without_time_labels_bins():
    fig, ax = plt.subplots()
    viz.plot_timeseries_heatmap(np.zeros((3, 4)), ax=ax, colorbar=False)
    assert ax.get_xlabel() == "Time (bins)"
    plt.close(fig)


def test_finish_figure_saves_and_closes(tmp_path):
    fig, _ax = plt.subplots()
    out = tmp_path / "nested" / "fig.png"
    assert viz.finish_figure(fig, save_path=out, show_plot=False) is fig
    assert out.exists()
    assert not plt.fignum_exists(fig.number)


def test_mea_runs_with_and_without_stim():
    fig, ax = plt.subplots()
    viz.MEA(_channelmap(), ax=ax, stim_elecs=None)
    viz.MEA(_channelmap(), ax=ax, stim_elecs=np.array([0, 5]))
    plt.close(fig)


class _FakeRecording:
    """Minimal stand-in exposing just what the burst overlays read off a Recording."""

    def __init__(self, channelmap=None, spike_bin=None):
        self.channelmap = _channelmap() if channelmap is None else channelmap
        self.stim_elecs = np.array([])
        # The burst-panel plots read the binned spikes and the identity; the overlay plots don't.
        self.spike_bin = _spike_bin() if spike_bin is None else spike_bin
        self.bin_size = 0.01
        self.samp_rate = 1000.0
        self.DIV = 27
        self.experiment = "NS_ATP1hr"


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


# --- burst panels ---------------------------------------------------------------------------------


def _span_burst_df(n=4, extracted=True):
    """Burst rows with ISI-N bounds and, when ``extracted``, tighter feature bounds inside them.

    Bursts sit 1 s apart from 0.5 s, well inside the 5 s the default ``_spike_bin`` covers.
    """
    import pandas as pd

    group_start = np.array([500 + 1000 * i for i in range(n)])
    group_stop = group_start + 300
    onset = group_start + 100   # strictly inside the group bounds, so the two spans differ
    offset = group_stop - 100
    if not extracted:
        onset = np.full(n, -1)
        offset = np.full(n, -1)
    return pd.DataFrame({
        "id": np.arange(1, n + 1),
        "kind": ["network"] * n,
        "phase": ["full"] * n,
        "peak_frame": group_start + 150,
        "group_start_frame": group_start,
        "group_stop_frame": group_stop,
        "onset_frame": onset,
        "offset_frame": offset,
    })


def _panel_axes(fig):
    """``(raster_axes, rate_axes)`` for each panel, in the order they were added."""
    return fig.axes[0::2], fig.axes[1::2]


def _span_extent(ax):
    """``(start, stop)`` in seconds of the shaded burst span on ``ax``."""
    patch = ax.patches[0]
    return patch.get_x(), patch.get_x() + patch.get_width()


def test_display_raster_sums_into_display_bins():
    spike_bin = np.arange(2 * 20, dtype=float).reshape(2, 20)
    counts, factor = viz._display_raster(spike_bin, np.arange(2), bin_size=0.01,
                                         display_bin_sec=0.05, first_bin=0, last_bin=18)
    assert factor == 5
    assert counts.shape == (2, 3)  # 18 bins // 5 = 3 whole display bins; the tail is dropped
    assert counts[0, 0] == pytest.approx(sum(range(5)))
    assert counts[1, 1] == pytest.approx(sum(range(25, 30)))


def test_burst_spans_frames_prefers_the_requested_definition():
    df = _span_burst_df(3)
    df.loc[1, ["onset_frame", "offset_frame"]] = -1

    start, stop, fell_back = viz._burst_spans_frames(df, "isi_n")
    assert not fell_back.any()
    np.testing.assert_array_equal(start, df["group_start_frame"])

    start, stop, fell_back = viz._burst_spans_frames(df, "features")
    # The contract is a 3-tuple: callers must be able to mark the fallback in the figure.
    np.testing.assert_array_equal(fell_back, [False, True, False])
    assert start[0] == df.loc[0, "onset_frame"] and start[1] == df.loc[1, "group_start_frame"]
    assert stop[2] == df.loc[2, "offset_frame"]


def test_plot_burst_panels_draws_a_raster_and_rate_axes_per_burst():
    fig = viz.plot_burst_panels(_FakeRecording(), _span_burst_df(), n=3, show_plot=False)
    rasters, rates = _panel_axes(fig)
    assert len(fig.axes) == 6
    assert all(len(ax.images) == 1 for ax in rasters)
    assert all(len(ax.lines) >= 1 for ax in rates)
    assert rates[0].get_ylabel() == "Spikes/s"
    plt.close(fig)


@pytest.mark.parametrize("span,start_col,stop_col",
                         [("isi_n", "group_start_frame", "group_stop_frame"),
                          ("features", "onset_frame", "offset_frame")])
def test_plot_burst_panels_span_selects_isi_n_or_feature_bounds(span, start_col, stop_col):
    rec, df = _FakeRecording(), _span_burst_df()
    fig = viz.plot_burst_panels(rec, df, burst_ids=[1], span=span, show_plot=False)

    start, stop = _span_extent(fig.axes[0])
    assert start == pytest.approx(df.loc[0, start_col] / rec.samp_rate)
    assert stop == pytest.approx(df.loc[0, stop_col] / rec.samp_rate)
    plt.close(fig)


def test_plot_burst_panels_falls_back_to_isi_n_bounds_for_unextracted_bursts():
    rec, df = _FakeRecording(), _span_burst_df()
    df.loc[1, ["onset_frame", "offset_frame"]] = -1
    fig = viz.plot_burst_panels(rec, df, burst_ids=[1, 2], span="features", show_plot=False)

    start, stop = _span_extent(fig.axes[2])
    assert start == pytest.approx(df.loc[1, "group_start_frame"] / rec.samp_rate)
    assert stop == pytest.approx(df.loc[1, "group_stop_frame"] / rec.samp_rate)
    # The other panel kept its (tighter) feature bounds.
    assert _span_extent(fig.axes[0])[0] == pytest.approx(df.loc[0, "onset_frame"] / rec.samp_rate)
    plt.close(fig)


def test_plot_burst_panels_marks_fallback_spans_unmistakably():
    """A fallback span is wider than the feature span a reader assumes it is: it must look
    different *and* say so, so losing either cue fails this test."""
    df = _span_burst_df()
    df.loc[1, ["onset_frame", "offset_frame"]] = -1
    fig = viz.plot_burst_panels(_FakeRecording(), df, burst_ids=[1, 2], span="features",
                                show_plot=False)
    normal, fallback = fig.axes[0], fig.axes[2]

    assert "ISI-N bounds (no features)" in fallback.get_title()
    assert "ISI-N" not in normal.get_title()
    assert fallback.patches[0].get_hatch() == "//" and not fallback.patches[0].get_fill()
    assert normal.patches[0].get_hatch() is None and normal.patches[0].get_fill()
    assert "1 of 2 use ISI-N bounds" in fig._suptitle.get_text()
    plt.close(fig)


def test_plot_burst_panels_rejects_features_when_nothing_was_extracted():
    rec = _FakeRecording()
    with pytest.raises(ValueError, match="extract_features"):
        viz.plot_burst_panels(rec, _span_burst_df(extracted=False), span="features", show_plot=False)

    without_columns = _span_burst_df().drop(columns=["onset_frame", "offset_frame"])
    with pytest.raises(ValueError, match="extract_features"):
        viz.plot_burst_panels(rec, without_columns, span="features", show_plot=False)


def test_plot_burst_panels_rejects_an_unknown_span():
    with pytest.raises(ValueError, match="span must be"):
        viz.plot_burst_panels(_FakeRecording(), _span_burst_df(), span="nope", show_plot=False)


def test_plot_burst_panels_uses_explicit_burst_ids_in_order():
    fig = viz.plot_burst_panels(_FakeRecording(), _span_burst_df(), burst_ids=[3, 1],
                                show_plot=False)
    rasters, _ = _panel_axes(fig)
    assert "burst 3" in rasters[0].get_title()
    assert "burst 1" in rasters[1].get_title()
    plt.close(fig)


def test_plot_burst_panels_rejects_unknown_ids_and_both_selectors():
    rec, df = _FakeRecording(), _span_burst_df()
    with pytest.raises(ValueError, match="not in burst_df"):
        viz.plot_burst_panels(rec, df, burst_ids=[99], show_plot=False)
    with pytest.raises(ValueError, match="not both"):
        viz.plot_burst_panels(rec, df, n=2, burst_ids=[1], show_plot=False)


def test_plot_burst_panels_sampling_is_seeded_and_time_ordered():
    rec, df = _FakeRecording(), _span_burst_df(4)
    titles = []
    for _ in range(2):
        fig = viz.plot_burst_panels(rec, df, n=3, seed=7, show_plot=False)
        titles.append([ax.get_title() for ax in _panel_axes(fig)[0]])
        plt.close(fig)
    assert titles[0] == titles[1]

    fig = viz.plot_burst_panels(rec, df, n=50, show_plot=False)   # clamps, doesn't raise
    rasters, _ = _panel_axes(fig)
    assert len(rasters) == 4
    starts = [float(t.split("(")[-1].split("-")[0]) for t in (ax.get_title() for ax in rasters)]
    assert starts == sorted(starts)
    plt.close(fig)


def test_plot_burst_panels_rejects_an_empty_burst_df():
    with pytest.raises(ValueError, match="empty"):
        viz.plot_burst_panels(_FakeRecording(), _span_burst_df().iloc[:0], show_plot=False)


def test_plot_burst_panels_saves_and_closes(tmp_path):
    out = tmp_path / "nested" / "panels.png"
    fig = viz.plot_burst_panels(_FakeRecording(), _span_burst_df(), n=2, save_path=out,
                                show_plot=False)
    assert out.exists()
    assert not plt.fignum_exists(fig.number)
