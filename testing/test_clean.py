"""Unit tests for the composable cleaning steps in ``mxtreme.clean``."""

import numpy as np
import pandas as pd
import pytest

from mxtreme import clean

SPIKE_DTYPE = np.dtype([("frameno", "<i8"), ("channel", "<i4"), ("amplitude", "<f8")])


def _old_refractory(data, refractory_frames):
    """Reference: the original per-channel pandas/while implementation of the refractory filter."""
    df = pd.DataFrame(data)
    groups = []
    for _, group in df.groupby("channel"):
        inds = np.where(np.diff(group["frameno"]) < refractory_frames)[0]
        while len(inds) != 0:
            mask = np.ones(shape=len(group["frameno"]), dtype=int)
            for ind in inds:
                if np.array(group["amplitude"])[ind] < np.array(group["amplitude"])[ind + 1]:
                    mask[ind + 1] = 0
                else:
                    mask[ind] = 0
            group = group[mask.astype(bool)]
            inds = np.where(np.diff(group["frameno"]) < refractory_frames)[0]
        groups.append(group)
    return pd.concat(groups).sort_index().to_records(index=False)


def test_normalize_time_offsets_by_raw_start(make_well):
    well = make_well(raw_start=50)
    first = well["data"]["frameno"][0]
    clean.normalize_time(well)
    assert well["data"]["frameno"][0] == first - 50


def test_dac_to_voltage_scales_by_lsb(make_well):
    well = make_well(lsb=np.array([2.0]))
    amps = well["data"]["amplitude"].copy()
    clean.dac_to_voltage(well)
    np.testing.assert_allclose(well["data"]["amplitude"], amps * 2.0)


def test_remove_positive_deflections_keeps_only_negative(make_well):
    well = make_well()
    clean.remove_positive_deflections(well)
    assert (well["data"]["amplitude"] < 0).all()


def test_spike_filter_removes_subthreshold_and_stamps(make_well):
    well = make_well()
    clean.remove_positive_deflections(well)
    clean.spike_filter(well, amp_thresh=2e-5)
    assert (np.abs(well["data"]["amplitude"]) >= 2e-5).all()
    assert well["preprocessing_params"]["amp_thresh"] == 2e-5


def test_spike_filter_raises_when_threshold_too_high(make_well):
    well = make_well()
    with pytest.raises(ValueError):
        clean.spike_filter(well, amp_thresh=1.0)


def test_remove_spurious_spikes_enforces_refractory(make_well):
    well = make_well()
    # ch0 has spikes at frames 100 and 120 (1 ms apart); a 2 ms refractory should drop one of them.
    n_ch0_before = np.sum(well["data"]["channel"] == 0)
    clean.remove_spurious_spikes(well, refractory_period=0.002)
    n_ch0_after = np.sum(well["data"]["channel"] == 0)
    assert n_ch0_after == n_ch0_before - 1
    assert well["preprocessing_params"]["refractory_period"] == 0.002


def test_remove_spurious_channels_drops_unmapped(make_well):
    well = make_well()
    clean.remove_spurious_channels(well)
    assert 9 not in set(well["data"]["channel"].tolist())
    assert set(well["data"]["channel"].tolist()) <= {0, 1, 2}


def test_build_channel_map_shape_and_columns(make_well):
    well = make_well()
    clean.build_channel_map(well)
    cm = well["channelmap"]
    assert cm.shape == (3, 5)
    np.testing.assert_array_equal(cm[:, 0], np.arange(3))          # index column
    np.testing.assert_array_equal(cm[:, 1], well["mapping"]["channel"])


def test_remove_stim_frames_drops_stim_window(make_well):
    # Two events: start at 480, end at 520 -> spike at ch1 frame 500 falls inside and is removed.
    well = make_well(
        eventtime=np.array([480, 520], dtype="<i8"),
        event_messages=[{"start_stimulation": {}}, {"end_stimulation": {}}],
    )
    assert 500 in well["data"]["frameno"].tolist()
    clean.remove_stim_frames(well, post_stim_period=0.0)
    assert 500 not in well["data"]["frameno"].tolist()
    assert len(well["stim_frames"]) > 0
    assert well["preprocessing_params"]["post_stim_period"] == 0.0


def test_bin_spikes_produces_matrix_and_rec_length(make_well):
    well = make_well()
    clean.remove_spurious_channels(well)   # ensure every spike channel is mapped
    clean.build_channel_map(well)
    clean.bin_spikes(well, bin_size=0.01)
    n_chan = well["channelmap"].shape[0]
    assert well["spike_bin"].shape[0] == n_chan
    assert well["spike_bin"].shape[1] >= 1
    assert well["spike_bin"].dtype == np.uint8
    assert set(np.unique(well["spike_bin"]).tolist()) <= {0, 1}
    assert well["bin_size"] == 0.01
    assert well["rec_t_sec"] > 0
    assert well["preprocessing_params"]["bin_size"] == 0.01


@pytest.mark.parametrize("seed", range(8))
def test_remove_spurious_spikes_matches_reference(make_well, seed):
    """Vectorized refractory filter must reproduce the original per-channel algorithm exactly."""
    rng = np.random.default_rng(seed)
    n = 400
    rows = np.empty(n, dtype=SPIKE_DTYPE)
    rows["channel"] = rng.integers(0, 6, size=n)
    rows["frameno"] = rng.integers(0, 2000, size=n)
    rows["amplitude"] = -rng.integers(1, 8, size=n).astype(float)   # small ints -> forces amplitude ties
    rows = np.sort(rows, order="frameno")                            # recording order (per-channel ascending)

    refractory_period = 0.002
    samp_rate = np.float64(20000.0)
    refractory_frames = refractory_period * samp_rate

    reference = _old_refractory(rows, refractory_frames)
    well = make_well(data=rows.copy(), samp_rate=samp_rate)
    clean.remove_spurious_spikes(well, refractory_period=refractory_period)

    for field in ("frameno", "channel", "amplitude"):
        np.testing.assert_array_equal(well["data"][field], reference[field])


def test_remove_spurious_spikes_all_isis_exceed_refractory(make_well):
    well = make_well()
    clean.remove_spurious_spikes(well, refractory_period=0.002)
    refractory_frames = 0.002 * well["samp_rate"]
    df = pd.DataFrame(well["data"]).sort_values(["channel", "frameno"])
    for _, g in df.groupby("channel"):
        assert (np.diff(g["frameno"]) >= refractory_frames).all()
