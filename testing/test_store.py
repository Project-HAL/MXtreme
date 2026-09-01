"""The recordings tree (``mxtreme.store``): batch ids, layout, splitting, and ingest."""

import numpy as np
import pandas as pd
import pytest

from mxtreme import store
from mxtreme.config import Config
from mxtreme.store import Batch

BATCH = "fall2026_batch1_DRG_M1"


# --- batch ids ------------------------------------------------------------------------------------


def test_batch_id_round_trips():
    batch = Batch(semester="fall", year=2026, number=1, cell_type="DRG", system="M1")
    assert batch.id == BATCH
    assert Batch.parse(BATCH) == batch
    assert Batch.parse(batch) is batch  # a Batch passes through unchanged


def test_batch_cell_type_may_contain_underscores():
    """Only the cell type is free-form, so its underscores must parse unambiguously."""
    batch = Batch.parse("spring2027_batch12_rat_cortical_M2")
    assert batch.cell_type == "rat_cortical"
    assert batch.number == 12
    assert batch.system == "M2"


@pytest.mark.parametrize("bad", [
    "autumn2026_batch1_DRG_M1",   # not a semester
    "fall26_batch1_DRG_M1",       # two-digit year
    "fall2026_batch0_DRG_M1",     # batches start at 1
    "fall2026_batch1_DRG_M3",     # no such system
    "fall2026_batch1_DRG",        # system missing
    "fall2026_1_DRG_M1",          # 'batch' marker missing
])
def test_malformed_batch_ids_are_rejected(bad):
    with pytest.raises(ValueError, match="batch"):
        Batch.parse(bad)


def test_batch_constructor_validates_fields():
    with pytest.raises(ValueError, match="semester"):
        Batch(semester="autumn", year=2026, number=1, cell_type="DRG", system="M1")
    with pytest.raises(ValueError, match="cell_type"):
        Batch(semester="fall", year=2026, number=1, cell_type="D/RG", system="M1")


# --- layout ---------------------------------------------------------------------------------------


def test_layout_and_parse_are_inverses(tmp_path):
    """What recording_dir/recording_stem write down, parse_recording_path must read back."""
    config = Config(data_root=tmp_path)
    stem = store.recording_stem(BATCH, 260810, "M07460", 4, 21)
    directory = store.recording_dir(config, BATCH, 260810, "M07460", 4, 21)

    for tail, kind, exp_id in [
        ("activity_scan", "activity_scan", ""),
        ("network_scan", "network_scan", ""),
        ("network_scan_1", "network_scan", ""),  # MaxLab's collision rename
        ("stim_trial_3", "experiment", "stim_trial_3"),
    ]:
        location = store.parse_recording_path(
            directory / f"{stem}_{tail}.raw.h5", config.recordings_dir
        )
        assert location is not None, tail
        assert (location.batch.id, location.plate_date) == (BATCH, 260810)
        assert (location.chip, location.well, location.div) == ("M07460", 4, 21)
        assert (location.kind, location.exp_id) == (kind, exp_id)


def test_parse_rejects_paths_outside_the_layout(tmp_path):
    config = Config(data_root=tmp_path)
    bad = config.recordings_dir / "misc" / "chip_M1_C1" / "well_0" / "DIV_1" / "x.raw.h5"
    assert store.parse_recording_path(bad, config.recordings_dir) is None


def test_plate_date_is_validated():
    with pytest.raises(ValueError, match="YYMMDD"):
        store.recording_stem(BATCH, 26081, "M07460", 0, 1)


# --- splitting a multi-well file ------------------------------------------------------------------


def _multiwell_h5(path, wells=(0, 3), conditions=("ctrl", "drug")):
    """A minimal MaxWell-shaped file: /assay metadata, /wells/wellNNN, and the /recordings mirror."""
    import h5py

    blob = str({
        "Exp ID": BATCH, "Chip ID": "C1", "Plate date": 250512, "DIV": 7,
        "Well IDs": list(wells), "Conditions": list(conditions),
    })
    with h5py.File(path, "w") as f:
        f.attrs["version"] = "20190530"
        f.create_dataset("/assay/metadata", data=np.array([blob.encode("utf-8")]))
        f.create_dataset("/assay/inputs/record_time", data=np.array([60]))
        for well in wells:
            f.create_dataset(f"/wells/well{well:03d}/rec0000/spikes", data=np.arange(well + 2))
        for well in wells:
            f[f"/recordings/rec0000/well{well:03d}"] = f[f"/wells/well{well:03d}/rec0000"]
    return path


def test_split_by_well_gives_each_culture_its_own_file(tmp_path):
    import h5py

    src = _multiwell_h5(tmp_path / "combined.raw.h5")
    written = store.split_by_well(
        src, lambda w: tmp_path / f"well_{w}.raw.h5", on_progress=lambda _: None
    )

    assert sorted(written) == [0, 3]
    for well, path in written.items():
        with h5py.File(path, "r") as f:
            # Only this well's data came along, under both of MaxWell's views.
            assert list(f["wells"]) == [f"well{well:03d}"]
            assert list(f["recordings/rec0000"]) == [f"well{well:03d}"]
            np.testing.assert_array_equal(
                f[f"wells/well{well:03d}/rec0000/spikes"][:], np.arange(well + 2)
            )
            # Shared pieces are copied: root attrs and the whole /assay group.
            assert f.attrs["version"] == "20190530"
            assert int(f["assay/inputs/record_time"][0]) == 60
            # The metadata blob is narrowed to the one well, keeping its own condition.
            blob = eval(f["assay/metadata"][:][0].decode())
            assert blob["Well IDs"] == [well]
            assert blob["Conditions"] == ["ctrl" if well == 0 else "drug"]
    assert src.exists()  # delete_original defaults to False


def test_split_by_well_removes_the_original_only_on_success(tmp_path):
    src = _multiwell_h5(tmp_path / "combined.raw.h5")

    # A destination that already exists fails the split -- and the original must survive.
    (tmp_path / "well_3.raw.h5").touch()
    with pytest.raises(FileExistsError):
        store.split_by_well(
            src, lambda w: tmp_path / f"well_{w}.raw.h5",
            on_progress=lambda _: None, delete_original=True,
        )
    assert src.exists()

    (tmp_path / "well_3.raw.h5").unlink()
    (tmp_path / "well_0.raw.h5").unlink()
    store.split_by_well(
        src, lambda w: tmp_path / f"well_{w}.raw.h5",
        on_progress=lambda _: None, delete_original=True,
    )
    assert not src.exists()


# --- ingest ---------------------------------------------------------------------------------------


def _single_well_h5(path, well=0):
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset(f"/wells/well{well:03d}/rec0000/spikes", data=np.arange(5))
    return path


def test_ingest_places_names_and_registers_a_single_well_file(tmp_path):
    config = Config(data_root=tmp_path / "ms")
    src = _single_well_h5(tmp_path / "export.raw.h5")

    written = store.ingest_recording(
        src, config, batch=BATCH, plate_date=260810, chip="M07460", div=21,
        exp_id="stim_trial_3", conditions={0: "ctrl"}, on_progress=lambda _: None,
    )

    dest = written[0]
    stem = store.recording_stem(BATCH, 260810, "M07460", 0, 21)
    assert dest == store.recording_dir(config, BATCH, 260810, "M07460", 0, 21) / f"{stem}_stim_trial_3.raw.h5"
    assert dest.exists() and src.exists()  # copied, not moved, by default

    row = pd.read_csv(config.registry_path).iloc[0]
    assert (row["exp_id"], row["kind"]) == ("stim_trial_3", "experiment")
    assert (row["batch_id"], row["plate_date"]) == (BATCH, 260810)
    assert (int(row["well"]), int(row["div"])) == (0, 21)
    assert row["conditions"] == "ctrl"

    # And the ingested file resolves back to exactly the identity it went in under.
    location = store.parse_recording_path(dest, config.recordings_dir)
    assert (location.kind, location.exp_id, location.well) == ("experiment", "stim_trial_3", 0)


def test_ingest_move_removes_the_source(tmp_path):
    config = Config(data_root=tmp_path / "ms")
    src = _single_well_h5(tmp_path / "export.raw.h5")

    written = store.ingest_recording(
        src, config, batch=BATCH, plate_date=260810, chip="M07460", div=21,
        exp_id="trial", move=True, on_progress=lambda _: None,
    )
    assert written[0].exists() and not src.exists()


def test_ingest_splits_a_multiwell_file(tmp_path):
    config = Config(data_root=tmp_path / "ms")
    src = _multiwell_h5(tmp_path / "export.raw.h5")

    written = store.ingest_recording(
        src, config, batch=BATCH, plate_date=250512, chip="C1", div=7,
        exp_id="trial", conditions={0: "ctrl", 3: "drug"}, on_progress=lambda _: None,
    )

    assert sorted(written) == [0, 3]
    for well, path in written.items():
        assert path.parent == store.recording_dir(config, BATCH, 250512, "C1", well, 7)
    df = pd.read_csv(config.registry_path).sort_values("well")
    assert list(df["well"]) == [0, 3]
    assert set(df["kind"]) == {"experiment"}
    assert list(df["conditions"]) == ["ctrl", "drug"]
    assert src.exists()  # copy semantics: the source survives a split ingest too


def test_ingest_stamps_identity_into_a_blob_less_file(tmp_path):
    """The ingested copy becomes self-describing: extract and a registry rebuild can read it back."""
    import h5py

    from mxtreme import io

    config = Config(data_root=tmp_path / "ms")
    src = _single_well_h5(tmp_path / "export.raw.h5")

    written = store.ingest_recording(
        src, config, batch=BATCH, plate_date=260810, chip="M07460", div=21,
        exp_id="trial", conditions={0: "ctrl"}, on_progress=lambda _: None,
    )

    with h5py.File(written[0], "r") as f:
        blob = eval(f["assay/metadata"][:][0].decode())
    assert blob["Exp ID"] == "trial" and blob["Batch ID"] == BATCH
    assert blob["Well IDs"] == [0] and blob["Conditions"] == ["ctrl"]
    with h5py.File(src, "r") as f:  # the source is never touched
        assert "assay" not in f

    # A rebuild from disk recovers the condition from that blob.
    config.registry_path.unlink()
    io.rebuild_registry(config)
    row = pd.read_csv(config.registry_path).iloc[0]
    assert (row["exp_id"], row["conditions"]) == ("trial", "ctrl")


def test_ingest_leaves_an_existing_blob_alone(tmp_path):
    """First-hand metadata written at record time outranks the identity supplied at ingest."""
    import h5py

    config = Config(data_root=tmp_path / "ms")
    src = _single_well_h5(tmp_path / "export.raw.h5")
    with h5py.File(src, "r+") as f:
        f.create_dataset("/assay/metadata", data=np.array([b"{'Exp ID': 'original'}"]))

    written = store.ingest_recording(
        src, config, batch=BATCH, plate_date=260810, chip="M07460", div=21,
        exp_id="trial", on_progress=lambda _: None,
    )
    with h5py.File(written[0], "r") as f:
        assert f["assay/metadata"][:][0] == b"{'Exp ID': 'original'}"


def test_ingest_refuses_to_overwrite(tmp_path):
    config = Config(data_root=tmp_path / "ms")
    src = _single_well_h5(tmp_path / "export.raw.h5")
    kwargs = dict(batch=BATCH, plate_date=260810, chip="M07460", div=21, exp_id="trial")

    store.ingest_recording(src, config, **kwargs, on_progress=lambda _: None)
    with pytest.raises(FileExistsError):
        store.ingest_recording(src, config, **kwargs, on_progress=lambda _: None)


@pytest.mark.parametrize("bad", ["", "has space", "a/b", "activity_scan", "network_scan_2"])
def test_ingest_rejects_unusable_exp_ids(tmp_path, bad):
    config = Config(data_root=tmp_path / "ms")
    src = _single_well_h5(tmp_path / "export.raw.h5")

    with pytest.raises(ValueError, match="exp_id"):
        store.ingest_recording(
            src, config, batch=BATCH, plate_date=260810, chip="M07460", div=21,
            exp_id=bad, on_progress=lambda _: None,
        )


def test_ingest_requires_a_wells_format_file(tmp_path):
    import h5py

    config = Config(data_root=tmp_path / "ms")
    src = tmp_path / "flat.raw.h5"
    with h5py.File(src, "w") as f:
        f.create_dataset("sig", data=np.arange(3))

    with pytest.raises(ValueError, match="/wells"):
        store.ingest_recording(
            src, config, batch=BATCH, plate_date=260810, chip="M07460", div=21,
            exp_id="trial", on_progress=lambda _: None,
        )
