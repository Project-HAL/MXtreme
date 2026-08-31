"""Tests for reading/writing preprocessed data (``mxtreme.io``)."""

import numpy as np

from mxtreme import clean, io


def _clean_well(make_well, **overrides):
    well = make_well(**overrides)
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


def test_phase_spec_roundtrips(make_well, tmp_path):
    spec = {"starts": {"pre": 0, "train": 20, "post": 40}, "end": 60}
    well = _clean_well(make_well)
    well["phase_spec"] = spec
    path = io.save_preprocessed(tmp_path, well)
    loaded = io.load_preprocessed(path)
    assert loaded["phase_spec"].item() == spec


def test_phase_spec_absent_roundtrips_as_none(make_well, tmp_path):
    # A well with no phase spec (the common, unphased case) stores None and loads back as None.
    well = _clean_well(make_well)
    path = io.save_preprocessed(tmp_path, well)
    loaded = io.load_preprocessed(path)
    assert loaded["phase_spec"].item() is None


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


def test_save_preprocessed_registers_recording(make_well, tmp_path):
    import pandas as pd

    well = _clean_well(make_well)
    well["experimental_condition"] = [1, 0]
    registry = tmp_path / "registry.csv"

    io.save_preprocessed(tmp_path, well, registry_path=registry)

    df = pd.read_csv(registry)
    assert len(df) == 1
    row = df.iloc[0]
    assert str(row["exp_id"]) == well["exp_id"]
    assert int(row["well"]) == well["well"]
    assert int(row["div"]) == well["DIV"]
    # Conditions are pasted through; the legacy status column is gone; timestamp is populated.
    assert row["conditions"] == "[1, 0]"
    assert "status" not in df.columns
    assert isinstance(row["timestamp"], str) and row["timestamp"]


def test_register_drops_legacy_status_column(make_well, tmp_path):
    import pandas as pd

    well = make_well()
    registry = tmp_path / "registry.csv"
    # Simulate an older registry that still carries a `status` column.
    pd.DataFrame(
        [{"exp_id": "old", "chip": "C9", "well": 3, "div": 1, "status": "complete",
          "timestamp": "2020-01-01T00:00:00"}]
    ).to_csv(registry, index=False)

    io.register({well["well"]: well}, registry)
    df = pd.read_csv(registry)
    assert "status" not in df.columns


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
    # in an older registry), the upsert should still collapse to one row.
    df_str = df.copy()
    df_str["well"] = df_str["well"].astype(str)
    df_str.to_csv(registry, index=False)
    io.register({well["well"]: well}, registry)
    df2 = pd.read_csv(registry)
    assert len(df2) == 1


def test_rebuild_registry_matches_live_registration(make_well, tmp_path):
    """A rebuilt registry is schema-identical to the one save_preprocessed wrote as it went."""
    import pandas as pd
    from mxtreme.config import Config

    config = Config(data_root=tmp_path)
    for div in (7, 8):
        well = _clean_well(make_well, DIV=div)
        io.save_preprocessed(config.preprocessed_dir, well, registry_path=config.registry_path)

    live = pd.read_csv(config.registry_path).sort_values("div").reset_index(drop=True)
    assert len(live) == 2

    config.registry_path.unlink()
    assert io.rebuild_registry(config) == 2

    rebuilt = pd.read_csv(config.registry_path).sort_values("div").reset_index(drop=True)
    assert list(rebuilt.columns) == list(live.columns)
    # Everything but the timestamp (which becomes the npz's mtime) should round-trip.
    cols = [c for c in live.columns if c != "timestamp"]
    pd.testing.assert_frame_equal(rebuilt[cols], live[cols])
    assert (rebuilt["conditions"] == live["conditions"]).all()


def test_rebuild_registry_is_idempotent(make_well, tmp_path):
    import pandas as pd
    from mxtreme.config import Config

    config = Config(data_root=tmp_path)
    well = _clean_well(make_well)
    io.save_preprocessed(config.preprocessed_dir, well, registry_path=config.registry_path)

    # Rebuilding on top of an existing registry upserts rather than duplicating.
    io.rebuild_registry(config)
    io.rebuild_registry(config)
    assert len(pd.read_csv(config.registry_path)) == 1


def test_rebuild_registry_preserves_rows_without_npz(make_well, tmp_path):
    """Recordings whose npz is gone stay in the registry -- a rebuild upserts, it doesn't truncate."""
    import pandas as pd
    from mxtreme.config import Config

    config = Config(data_root=tmp_path)
    well = _clean_well(make_well)
    io.save_preprocessed(config.preprocessed_dir, well, registry_path=config.registry_path)
    io.register({0: {"exp_id": "gone", "chip": "C9", "DIV": 99}}, config.registry_path)

    io.rebuild_registry(config)
    df = pd.read_csv(config.registry_path)
    assert set(df["exp_id"]) == {"testExp", "gone"}


# --- activity scans -------------------------------------------------------------------------------


class _FakeScanParams:
    """The five fields ``register_scan`` reads, without importing the rig-side scan module."""

    def __init__(self, wells=(0, 1), conditions=(), exp_id="scanExp", chip="C1", div=3):
        self.exp_id, self.chip, self.div = exp_id, chip, div
        self.wells, self.conditions = list(wells), list(conditions)


def test_register_scan_writes_one_row_per_well(tmp_path):
    import pandas as pd

    registry = tmp_path / "registry.csv"
    io.register_scan(_FakeScanParams(wells=(0, 2), conditions=("ctrl", "drug")), registry)

    df = pd.read_csv(registry).sort_values("well").reset_index(drop=True)
    assert list(df["well"]) == [0, 2]
    assert set(df["kind"]) == {"activity_scan"}
    assert list(df["conditions"]) == ["ctrl", "drug"]
    assert (df["div"] == 3).all()


def test_scan_and_recording_coexist_for_the_same_culture(make_well, tmp_path):
    """`kind` is part of the key, so a scan does not overwrite the recording preprocessed from it."""
    import pandas as pd

    registry = tmp_path / "registry.csv"
    well = _clean_well(make_well)
    io.save_preprocessed(tmp_path, well, registry_path=registry)
    io.register_scan(
        _FakeScanParams(wells=(well["well"],), exp_id=well["exp_id"], chip=well["chip"],
                        div=well["DIV"]),
        registry,
    )

    df = pd.read_csv(registry)
    assert len(df) == 2
    assert set(df["kind"]) == {"preprocessed", "activity_scan"}


def test_register_backfills_kind_on_an_older_registry(tmp_path):
    """A registry written before scans were indexed holds recordings, and is labelled as such."""
    import pandas as pd

    registry = tmp_path / "registry.csv"
    pd.DataFrame(
        [{"exp_id": "old", "chip": "C9", "well": 3, "div": 1, "conditions": "",
          "timestamp": "2020-01-01T00:00:00"}]
    ).to_csv(registry, index=False)

    io.register_scan(_FakeScanParams(wells=(0,)), registry)

    df = pd.read_csv(registry)
    assert len(df) == 2
    assert df.loc[df["exp_id"] == "old", "kind"].item() == "preprocessed"
    assert df.loc[df["exp_id"] == "scanExp", "kind"].item() == "activity_scan"


def _store_with_scan(tmp_path, wells=(0, 1), metadata=True):
    """Write a scan .h5 into a managed store, carrying the /assay/metadata blob MaxLab writes."""
    import h5py

    from mxtreme.config import Config

    config = Config(data_root=tmp_path)
    params = _FakeScanParams(wells=wells, conditions=("ctrl", "drug")[: len(wells)])
    h5_path = (
        config.scans_dir / params.exp_id / params.chip
        / f"DIV{params.div}_250512_{params.chip}_{params.exp_id}_activity_scan.raw.h5"
    )
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "w") as f:
        if metadata:
            blob = str({
                "Exp ID": params.exp_id, "Chip ID": params.chip, "Plate date": 250512,
                "DIV": params.div, "Well IDs": list(params.wells),
                "Conditions": list(params.conditions),
            })
            f.create_dataset("/assay/metadata", data=np.array([blob.encode("utf-8")]))
    return config, h5_path


def test_rebuild_registry_recovers_scans_from_their_embedded_metadata(tmp_path):
    import pandas as pd

    config, _ = _store_with_scan(tmp_path)
    assert io.rebuild_registry(config) == 1  # one scan, no recordings

    df = pd.read_csv(config.registry_path)
    assert list(df["well"]) == [0, 1]
    assert set(df["kind"]) == {"activity_scan"}
    assert list(df["conditions"]) == ["ctrl", "drug"]


def test_rebuild_registry_skips_a_scan_with_no_embedded_metadata(tmp_path, capsys):
    """Wells and DIV cannot be recovered from the file name alone, so the scan is reported, not guessed."""
    config, _ = _store_with_scan(tmp_path, metadata=False)

    assert io.rebuild_registry(config) == 0
    assert "1 scan file(s) skipped" in capsys.readouterr().out


def test_rebuild_registry_counts_recordings_and_scans_together(make_well, tmp_path):
    config, _ = _store_with_scan(tmp_path, wells=(0,))
    well = _clean_well(make_well)
    io.save_preprocessed(config.preprocessed_dir, well, registry_path=config.registry_path)

    config.registry_path.unlink()
    assert io.rebuild_registry(config) == 2  # one recording + one scan


# --- spike-order repair -------------------------------------------------------------------------


def _saved_store(make_well, tmp_path):
    """Save one fully-featured well into a managed store and return ``(config, path)``."""
    from mxtreme.config import Config

    config = Config(data_root=tmp_path)
    well = _clean_well(make_well)
    well["phase_spec"] = {"starts": {"pre": 0, "post": 20}, "end": 40}
    well["step_log"] = [{"step": "bin_spikes", "n_before": 4, "n_after": 4, "removed": 0, "seconds": 0.0}]
    path = io.save_preprocessed(config.preprocessed_dir, well, registry_path=config.registry_path)
    return config, path


def _scramble_stored_spikes(path):
    """Rewrite a saved npz with its spike rows reversed (all other fields untouched)."""
    contents = dict(np.load(path, allow_pickle=True))
    contents["spike_data"] = contents["spike_data"][::-1]
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **contents)


def test_repair_spike_order_sorts_and_preserves_other_fields(make_well, tmp_path):
    config, path = _saved_store(make_well, tmp_path)
    before = io.load_preprocessed(path)
    _scramble_stored_spikes(path)

    assert io.repair_spike_order(config) == [path]

    after = io.load_preprocessed(path)
    frameno = after["spike_data"]["frameno"]
    assert np.all(frameno[:-1] <= frameno[1:])
    np.testing.assert_array_equal(np.sort(after["spike_data"]["channel"]),
                                  np.sort(before["spike_data"]["channel"]))
    # Everything except the row order must survive the rewrite untouched.
    np.testing.assert_array_equal(after["channelmap"], before["channelmap"])
    np.testing.assert_array_equal(after["spike_bin"], before["spike_bin"])
    assert after["preprocessing_params"].item() == before["preprocessing_params"].item()
    assert after["phase_spec"].item() == before["phase_spec"].item()
    assert list(after["step_log"]) == list(before["step_log"])
    assert str(after["exp_id"]) == str(before["exp_id"])


def test_repair_spike_order_is_idempotent(make_well, tmp_path):
    config, path = _saved_store(make_well, tmp_path)
    _scramble_stored_spikes(path)

    io.repair_spike_order(config)
    assert io.repair_spike_order(config) == []


def test_repair_spike_order_dry_run_writes_nothing(make_well, tmp_path):
    config, path = _saved_store(make_well, tmp_path)
    _scramble_stored_spikes(path)
    mtime = path.stat().st_mtime_ns

    assert io.repair_spike_order(config, dry_run=True) == [path]
    assert path.stat().st_mtime_ns == mtime

    frameno = io.load_preprocessed(path)["spike_data"]["frameno"]
    assert not np.all(frameno[:-1] <= frameno[1:])  # still unsorted -- nothing was written
