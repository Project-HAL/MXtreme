"""Tests for the Pipeline orchestrator (``mxtreme.pipeline``)."""

from functools import partial

from mxtreme import clean
from mxtreme.pipeline import Pipeline


def _full_pipeline():
    return Pipeline([
        clean.normalize_time,
        clean.dac_to_voltage,
        clean.remove_positive_deflections,
        partial(clean.spike_filter, amp_thresh=2e-5),
        partial(clean.remove_spurious_spikes, refractory_period=0.002),
        clean.remove_spurious_channels,
        clean.build_channel_map,
        partial(clean.remove_stim_frames, post_stim_period=0.0),
        partial(clean.bin_spikes, bin_size=0.01),
    ])


def test_transform_returns_data_and_writes_nothing(make_well, tmp_path):
    data = {0: make_well()}
    out = _full_pipeline().transform(data)
    assert out is data
    assert "spike_bin" in data[0]                     # steps ran
    assert list(tmp_path.iterdir()) == []             # transform performs no I/O


def test_run_saves_one_npz_per_nonempty_well(make_well, tmp_path):
    data = {0: make_well(well=0), 1: make_well(well=1)}
    paths = _full_pipeline().run(data, datastore=tmp_path)
    assert len(paths) == 2
    assert all(p.exists() and p.suffix == ".npz" for p in paths)


def test_run_skips_wells_left_empty(make_well, tmp_path):
    # Well 0 keeps only positive spikes, so remove_positive_deflections empties it (no exception) and
    # run() should skip it; well 1 is normal and saved.
    empty_well = make_well()
    empty_well["data"] = empty_well["data"][empty_well["data"]["amplitude"] > 0]

    data = {0: empty_well, 1: make_well(well=1)}
    paths = Pipeline([clean.remove_positive_deflections]).run(data, datastore=tmp_path)

    assert len(paths) == 1
    assert "well1" in paths[0].name


def test_transform_records_step_log_and_retains_wells(make_well):
    data = {0: make_well()}
    _full_pipeline().transform(data)
    assert 0 in data                                  # wells retained
    log = data[0]["step_log"]
    assert [r["step"] for r in log][:3] == ["normalize_time", "dac_to_voltage", "remove_positive_deflections"]
    assert set(log[0]) == {"step", "n_before", "n_after", "removed", "seconds"}
    # spike_filter should have removed at least one spike in the synthetic well
    assert any(r["step"] == "spike_filter" and r["removed"] > 0 for r in log)


def test_run_frees_data_by_default_and_retains_when_false(make_well, tmp_path):
    data = {0: make_well(well=0), 1: make_well(well=1)}
    _full_pipeline().run(data, datastore=tmp_path)                 # free=True default
    assert data == {}                                             # wells released

    data = {0: make_well(well=0)}
    _full_pipeline().run(data, datastore=tmp_path, free=False)
    assert 0 in data and "spike_bin" in data[0] and "step_log" in data[0]
