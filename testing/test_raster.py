"""``mxtreme.raster``: a raw recording's spike table, read and binned for a quick-look raster."""

import numpy as np
import pytest

from mxtreme.raster import SpikeRaster, load_spike_raster

MAPPING = np.dtype([("channel", "<i4"), ("electrode", "<i4"), ("x", "<f8"), ("y", "<f8")])
SPIKES = np.dtype([("frameno", "<i8"), ("channel", "<i4"), ("amplitude", "<f4")])
RATE = 20000.0


def _recording(group, channels, electrodes, spikes, *, start_ms=None, span_ms=None, first_raw_frame=None):
    """One ``recNNNN`` group: a mapping, a spike table of ``(frameno, channel)`` pairs, and whichever
    of MaxLab's time anchors the case wants."""
    mapping = np.zeros(len(channels), dtype=MAPPING)
    mapping["channel"], mapping["electrode"] = channels, electrodes
    group.create_dataset("settings/mapping", data=mapping)
    group.create_dataset("settings/sampling", data=np.array([RATE]))
    table = np.zeros(len(spikes), dtype=SPIKES)
    table["frameno"] = [s[0] for s in spikes]
    table["channel"] = [s[1] for s in spikes]
    table["amplitude"] = -20
    group.create_dataset("spikes", data=table)
    if start_ms is not None:
        group.create_dataset("start_time", data=np.array([start_ms]))
        group.create_dataset("stop_time", data=np.array([start_ms + span_ms]))
    if first_raw_frame is not None:
        group.create_dataset("groups/routed/frame_nos", data=np.arange(first_raw_frame, first_raw_frame + 10))


def _scan(path):
    """Two recordings of well 0 on different electrodes, the way an activity scan is laid out."""
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset("assay/inputs/record_time", data=np.array([b"30"]))
        _recording(
            f.create_group("wells/well000/rec0000"),
            channels=[0, 1, 2], electrodes=[30, 10, 20],  # deliberately out of electrode order
            spikes=[(1_000_000, 1), (1_000_000 + int(10 * RATE), 0), (1_000_000 + int(5 * RATE), 7)],
            start_ms=1_700_000_000_000, span_ms=30_000,
        )
        _recording(
            f.create_group("wells/well000/rec0001"),
            channels=[0, 1], electrodes=[50, 40],
            spikes=[(2_000_000 + int(2 * RATE), 1), (2_000_000, 0)],  # out of time order
            start_ms=1_700_000_040_000, span_ms=30_000,
        )
    return path


def test_each_recording_becomes_a_segment_with_rows_in_electrode_order(tmp_path):
    r = load_spike_raster(_scan(tmp_path / "scan.raw.h5"), 0)

    assert [s.name for s in r.segments] == ["rec0000", "rec0001"]
    first, second = r.segments
    assert first.electrodes.tolist() == [10, 20, 30] and first.channels.tolist() == [1, 2, 0]
    assert second.electrodes.tolist() == [40, 50]
    assert first.duration_sec == 30.0 and first.started_at_ms == 1_700_000_000_000
    # The spike on channel 7 has no electrode in the mapping: counted, not drawn.
    assert (first.n_spikes, first.n_unrouted) == (2, 1)
    assert r.n_rows == 3 and r.n_spikes == 4 and r.duration_sec == 60.0 and r.samp_rate == RATE


def test_spikes_are_timed_from_their_own_recording_and_sorted(tmp_path):
    r = load_spike_raster(_scan(tmp_path / "scan.raw.h5"), 0)

    assert r.segment.tolist() == [0, 0, 1, 1]
    np.testing.assert_allclose(r.time_sec, [0.0, 10.0, 0.0, 2.0])
    assert r.row.tolist() == [0, 2, 1, 0]  # electrodes 10, 30, then 50, 40
    assert r.offsets.tolist() == [0.0, 30.0]


def test_binned_lays_the_recordings_end_to_end(tmp_path):
    r = load_spike_raster(_scan(tmp_path / "scan.raw.h5"), 0)

    b = r.binned(n_time_bins=60)  # one bin a second

    assert b.counts.shape == (3, 60) and b.rows_per_bin == 1 and b.bin_sec == 1.0
    assert sorted(zip(*(a.tolist() for a in np.nonzero(b.counts)))) == [(0, 0), (0, 32), (1, 30), (2, 10)]
    assert b.counts.sum() == 4


def test_binned_window_and_row_folding(tmp_path):
    r = load_spike_raster(_scan(tmp_path / "scan.raw.h5"), 0)

    zoom = r.binned(n_time_bins=10, t0=5.0, t1=15.0)
    assert zoom.counts.sum() == 1 and zoom.counts[2, 5] == 1 and (zoom.t0, zoom.t1) == (5.0, 15.0)

    folded = r.binned(n_time_bins=60, max_rows=2)  # 3 rows into at most 2: two rows a bin
    assert folded.rows_per_bin == 2 and folded.counts.shape == (2, 60)
    assert folded.counts[0].sum() == 3 and folded.counts[1].sum() == 1

    inverted = r.binned(n_time_bins=10, t0=20.0, t1=20.0)  # never a zero-width window
    assert inverted.t1 > inverted.t0


def test_a_recording_with_raw_traces_is_timed_from_its_first_frame(tmp_path):
    import h5py

    path = tmp_path / "net.raw.h5"
    with h5py.File(path, "w") as f:
        _recording(
            f.create_group("wells/well003/rec0000"),
            channels=[4], electrodes=[99],
            spikes=[(500_000 + int(3 * RATE), 4)], first_raw_frame=500_000,
        )

    r = load_spike_raster(path, 3)

    np.testing.assert_allclose(r.time_sec, [3.0])
    # No stamps and no record_time: the duration is what the spikes span.
    assert r.segments[0].started_at_ms is None and r.duration_sec == pytest.approx(3.0, abs=1e-3)


def test_an_empty_recording_still_has_its_segment(tmp_path):
    import h5py

    path = tmp_path / "silent.raw.h5"
    with h5py.File(path, "w") as f:
        _recording(f.create_group("wells/well000/rec0000"), [0, 1], [5, 6], [], start_ms=1_700_000_000_000, span_ms=30_000)

    r = load_spike_raster(path, 0)

    assert r.n_spikes == 0 and r.duration_sec == 30.0
    assert r.binned(n_time_bins=30).counts.sum() == 0


def test_save_and_load_round_trip(tmp_path):
    r = load_spike_raster(_scan(tmp_path / "scan.raw.h5"), 0)

    back = SpikeRaster.load(r.save(tmp_path / "raster.cache"))

    assert back.well == 0 and back.path == r.path and back.samp_rate == r.samp_rate
    assert [(s.name, s.duration_sec, s.n_spikes, s.n_unrouted) for s in back.segments] == [
        (s.name, s.duration_sec, s.n_spikes, s.n_unrouted) for s in r.segments
    ]
    assert back.segments[0].electrodes.tolist() == [10, 20, 30]
    np.testing.assert_array_equal(back.binned(60).counts, r.binned(60).counts)


def test_a_missing_well_or_a_file_that_is_not_a_recording_says_so(tmp_path):
    import h5py

    with pytest.raises(KeyError, match=r"no well 4 \(has \[0\]\)"):
        load_spike_raster(_scan(tmp_path / "scan.raw.h5"), 4)

    flat = tmp_path / "flat.h5"
    with h5py.File(flat, "w") as f:
        f.create_dataset("something", data=np.arange(3))
    with pytest.raises(ValueError, match="no /wells group"):
        load_spike_raster(flat, 0)

    bare = tmp_path / "bare.h5"
    with h5py.File(bare, "w") as f:
        f.create_group("wells/well000/rec0000")
    with pytest.raises(ValueError, match="no recording with a spike table"):
        load_spike_raster(bare, 0)
