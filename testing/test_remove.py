"""Tests for taking a recording out of the managed store cleanly."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mxtreme import io, store
from mxtreme import transactions as tx
from mxtreme.config import Config

BATCH = "fall2026_batch1_E18_M1"
PLATE = 260813
CHIP = "P1"


class _Params:
    exp_id, batch_id, plate_date, chip, div = "", BATCH, PLATE, CHIP, 26
    wells, conditions = [0], []


class _Recording:
    exp_id, chip, well, DIV, plate_date = BATCH, CHIP, 0, 26, PLATE


class _Bursts:
    def to_dataframe(self):
        return pd.DataFrame({"peak_frame": [1, 2, 3]})

    def __len__(self):
        return 3


@pytest.fixture
def populated(tmp_path):
    """A store with one culture on DIV 26: activity scan + selection outputs, network scan +
    preprocessed npz + burst CSV + burst-log row, all registered and journaled."""
    config = Config(data_root=tmp_path / "store")
    div_dir = store.recording_dir(config, BATCH, PLATE, CHIP, 0, 26)
    div_dir.mkdir(parents=True)
    stem = store.recording_stem(BATCH, PLATE, CHIP, 0, 26)
    activity = div_dir / f"{stem}_activity_scan.raw.h5"
    network = div_dir / f"{stem}_network_scan.raw.h5"
    activity.write_bytes(b"AS")
    network.write_bytes(b"NS")
    sel = div_dir / "electrode_selection"
    sel.mkdir()
    (sel / f"{stem}_AS_well0.png").write_bytes(b"png")
    (sel / f"{stem}_network_well0.npz").write_bytes(b"npz")
    io.register_scan(_Params(), config.registry_path, kind="activity_scan", well_files={0: activity})
    io.register_scan(_Params(), config.registry_path, kind="network_scan", well_files={0: network})
    pre = config.preprocessed_dir / BATCH / CHIP / "well0"
    pre.mkdir(parents=True)
    npz = pre / f"DIV26_{PLATE}_{CHIP}_{BATCH}_well0_exp_data.npz"
    np.savez(npz, x=np.arange(3))
    io.register({0: {"exp_id": BATCH, "chip": CHIP, "DIV": 26, "plate_date": PLATE}}, config.registry_path)
    csv = io.save_burst_data(_Bursts(), config.burst_data_dir, _Recording())
    log = config.burst_data_dir / f"{BATCH}_burst_log.csv"
    pd.DataFrame(
        [
            {"exp_id": BATCH, "chip": CHIP, "well": 0, "DIV": 26, "n_bursts": 3},
            {"exp_id": BATCH, "chip": CHIP, "well": 0, "DIV": 19, "n_bursts": 5},
        ]
    ).to_csv(log, index=False)
    return config, activity, network, npz, csv, log


def _rows(config):
    return pd.read_csv(config.registry_path).fillna("")


def test_derived_files_of_each_kind(populated):
    config, activity, network, npz, csv, _ = populated
    assert [p.name for p in store.derived_files(config, activity)] == [
        f"{activity.name.removesuffix('_activity_scan.raw.h5')}_AS_well0.png",
        f"{activity.name.removesuffix('_activity_scan.raw.h5')}_network_well0.npz",
    ]
    assert store.derived_files(config, network) == [npz, csv]
    assert store.derived_files(config, Path("/elsewhere/x.h5")) == []


def test_remove_network_scan_takes_its_derivatives_rows_and_journal(populated):
    config, activity, network, npz, csv, log = populated
    before = len(_rows(config))
    removal = store.remove_recording(
        network, config, reason="empty recording", actor="kam", on_progress=lambda _: None
    )

    assert removal.files == (network, npz, csv)
    assert not network.exists() and not npz.exists() and not csv.exists()
    assert activity.exists()  # the other recording on that DIV is untouched
    # moved to the trash at their store-relative paths, not deleted
    assert removal.trashed_to is not None and removal.trashed_to.parent == config.trash_dir
    assert (removal.trashed_to / network.relative_to(config.data_root)).read_bytes() == b"NS"
    assert (removal.trashed_to / npz.relative_to(config.data_root)).is_file()
    # registry: the network_scan row and the preprocessed row went, the activity_scan row stayed
    assert removal.registry_rows == 2 and len(_rows(config)) == before - 2
    assert list(_rows(config)["kind"]) == ["activity_scan"]
    # burst log: only the DIV 26 row went
    assert removal.burst_log_rows == 1
    assert list(pd.read_csv(log)["DIV"]) == [19]
    # journaled
    last = tx.read(config)[-1]
    assert (
        last.op == "recording.removed"
        and last.div == 26
        and last.actor == "kam"
        and last.note == "empty recording"
    )
    assert last.data["kind"] == "network_scan" and last.data["registry_rows"] == 2
    assert last.data["derived"] == [str(npz), str(csv)]
    assert removal.pruned_dirs == ()  # the DIV directory still holds the activity scan


def test_remove_activity_scan_takes_its_selection_and_prunes_empty_dirs(populated):
    config, activity, network, *_ = populated
    store.remove_recording(network, config, on_progress=lambda _: None)
    removal = store.remove_recording(activity, config, on_progress=lambda _: None)
    assert len(removal.files) == 3  # the scan and its two selection outputs
    assert not activity.parent.exists()  # DIV_26 emptied and pruned, electrode_selection with it
    assert set(removal.pruned_dirs) == {activity.parent / "electrode_selection", activity.parent}
    assert activity.parent.parent.is_dir()  # the well directory stays
    assert _rows(config).empty


def test_dry_run_touches_nothing(populated):
    config, _, network, npz, csv, log = populated
    before = _rows(config)
    n_tx = len(tx.read(config))
    removal = store.remove_recording(network, config, dry_run=True, on_progress=lambda _: None)
    assert removal.files == (network, npz, csv) and removal.trashed_to is None
    assert network.exists() and npz.exists() and csv.exists()
    assert _rows(config).equals(before) and len(pd.read_csv(log)) == 2
    assert len(tx.read(config)) == n_tx


def test_purge_deletes_instead_of_trashing(populated):
    config, _, network, *_ = populated
    removal = store.remove_recording(network, config, purge=True, on_progress=lambda _: None)
    assert removal.trashed_to is None and not network.exists() and not config.trash_dir.exists()
    assert tx.read(config)[-1].data["purged"] is True


def test_derived_false_keeps_the_derivatives(populated):
    config, _, network, npz, csv, log = populated
    removal = store.remove_recording(network, config, derived=False, on_progress=lambda _: None)
    assert removal.files == (network,) and npz.exists() and csv.exists()
    assert removal.registry_rows == 1 and "preprocessed" in set(_rows(config)["kind"])
    assert removal.burst_log_rows == 0 and len(pd.read_csv(log)) == 2


def test_refuses_files_outside_the_tree(populated, tmp_path):
    config, *_ = populated
    outside = tmp_path / "x.raw.h5"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError):
        store.remove_recording(outside, config)
    stray = config.recordings_dir / "not_a_plating" / "x.raw.h5"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"x")
    with pytest.raises(ValueError):
        store.remove_recording(stray, config)
    assert stray.exists()


def test_unregister_matches_on_the_given_fields_only(tmp_path):
    registry = tmp_path / "registry.csv"
    io.register(
        {0: {"exp_id": "", "batch_id": BATCH, "chip": CHIP, "DIV": 7, "plate_date": PLATE}},
        registry,
        kind="activity_scan",
    )
    io.register({0: {"exp_id": BATCH, "chip": CHIP, "DIV": 7, "plate_date": PLATE}}, registry)
    assert (
        io.unregister(registry, chip=CHIP, well=0, div=7, kind="preprocessed", batch_id=BATCH) == 0
    )  # blank batch_id
    assert io.unregister(registry, chip=CHIP, well=0, div=7, kind="preprocessed", exp_id=BATCH) == 1
    assert io.unregister(registry, chip=CHIP, well=0, div=7, kind="activity_scan") == 1
    assert io.unregister(registry, chip=CHIP, well=0, div=7, kind="activity_scan") == 0
