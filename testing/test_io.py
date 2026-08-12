"""Tests for reading/writing preprocessed data (``mxtreme.io``)."""

import numpy as np

from mxtreme import clean, io


def _clean_well(make_well):
    well = make_well()
    clean.remove_positive_deflections(well)
    clean.remove_spurious_channels(well)
    clean.build_channel_map(well)
    clean.bin_spikes(well, bin_size=0.01)
    return well


def test_save_load_roundtrip_preserves_arrays_and_params(make_well, tmp_path):
    well = _clean_well(make_well)
    path = io.save_preprocessed(tmp_path, well)
    assert path.exists()

    loaded = io.load_preprocessed(path)
    np.testing.assert_array_equal(loaded["spike_bin"], well["spike_bin"])
    np.testing.assert_array_equal(loaded["channelmap"], well["channelmap"])
    assert int(loaded["well"]) == well["well"]
    assert str(loaded["exp_id"]) == well["exp_id"]
    assert loaded["preprocessing_params"].item()["bin_size"] == 0.01


def test_step_log_roundtrips(make_well, tmp_path):
    well = _clean_well(make_well)
    well["step_log"] = [
        {"step": "spike_filter", "n_before": 10, "n_after": 8, "removed": 2, "seconds": 0.01},
    ]
    path = io.save_preprocessed(tmp_path, well)
    loaded = io.load_preprocessed(path)
    log = list(loaded["step_log"])
    assert log[0]["step"] == "spike_filter"
    assert log[0]["removed"] == 2


def test_save_path_layout(make_well, tmp_path):
    well = _clean_well(make_well)
    path = io.save_preprocessed(tmp_path, well)
    # <root>/<exp_id>/<chip>/well<well>/DIV<div>_..._exp_data.npz
    assert path.parent == tmp_path / well["exp_id"] / well["chip"] / f"well{well['well']}"
    assert path.name.startswith(f"DIV{well['DIV']}_") and path.name.endswith("_exp_data.npz")


def test_no_overwrite_suffixes(make_well, tmp_path):
    well = _clean_well(make_well)
    first = io.save_preprocessed(tmp_path, well, overwrite=True)
    second = io.save_preprocessed(tmp_path, well, overwrite=False)
    assert first != second
    assert first.exists() and second.exists()


def test_register_upserts_without_duplicates(make_well, tmp_path):
    import pandas as pd

    well = make_well()
    registry = tmp_path / "registry.csv"

    # Re-registering the same recording must not append a duplicate row.
    io.register({well["well"]: well}, registry)
    io.register({well["well"]: well}, registry)
    df = pd.read_csv(registry)
    assert len(df) == 1

    # Even when a prior row was written with a different `well` dtype (e.g. a string "0"
    # from build_registry_from_disk), the upsert should still collapse to one row.
    df_str = df.copy()
    df_str["well"] = df_str["well"].astype(str)
    df_str.to_csv(registry, index=False)
    io.register({well["well"]: well}, registry)
    df2 = pd.read_csv(registry)
    assert len(df2) == 1
