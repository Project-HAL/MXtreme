"""The recordings tree (``mxtreme.store``): batch ids, layout, splitting, and ingest."""

import numpy as np
import pandas as pd
import pytest

from mxtreme import store, transactions
from mxtreme.config import Config
from mxtreme.store import Batch

BATCH = "fall2026_batch1_DRG"


# --- batch ids ------------------------------------------------------------------------------------


def test_batch_id_round_trips():
    batch = Batch(semester="fall", year=2026, number=1, cell_type="DRG")
    assert batch.id == BATCH
    assert Batch.parse(BATCH) == batch
    assert Batch.parse(batch) is batch  # a Batch passes through unchanged


def test_batch_cell_type_may_contain_underscores():
    """Only the cell type is free-form, so its underscores must parse unambiguously."""
    batch = Batch.parse("spring2027_batch12_rat_cortical")
    assert batch.cell_type == "rat_cortical"
    assert batch.number == 12


def test_batch_id_carries_no_system():
    """The system is the chip directory's, not the batch's: one plating can span MaxOne and MaxTwo.

    An id written before the suffix was dropped still parses, the suffix landing in the cell type,
    so ``rename_batch`` can take it off.
    """
    legacy = Batch.parse("fall2026_batch1_E18_M1")
    assert legacy.cell_type == "E18_M1"
    assert legacy.id == "fall2026_batch1_E18_M1"


@pytest.mark.parametrize("bad", [
    "autumn2026_batch1_DRG",   # not a semester
    "fall26_batch1_DRG",       # two-digit year
    "fall2026_batch0_DRG",     # batches start at 1
    "fall2026_batch1",         # cell type missing
    "fall2026_1_DRG",          # 'batch' marker missing
])
def test_malformed_batch_ids_are_rejected(bad):
    with pytest.raises(ValueError, match="batch"):
        Batch.parse(bad)


def test_batch_constructor_validates_fields():
    with pytest.raises(ValueError, match="semester"):
        Batch(semester="autumn", year=2026, number=1, cell_type="DRG")
    with pytest.raises(ValueError, match="cell_type"):
        Batch(semester="fall", year=2026, number=1, cell_type="D/RG")


# --- layout ---------------------------------------------------------------------------------------


def test_layout_and_parse_are_inverses(tmp_path):
    """What recording_dir/recording_stem write down, parse_recording_path must read back."""
    config = Config(data_root=tmp_path)
    stem = store.recording_stem(BATCH, 260810, "M07460", 4, 21)
    directory = store.recording_dir(config, BATCH, 260810, "M07460", 4, 21, system="M1")

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


# --- listing platings -----------------------------------------------------------------------------


def test_list_platings_reads_the_tree_back_newest_first(tmp_path):
    config = Config(data_root=tmp_path)
    for plate_date, batch in [(260810, BATCH), (270115, "spring2027_batch2_iPSC")]:
        (config.recordings_dir / store.plating_dirname(batch, plate_date)).mkdir(parents=True)

    platings = store.list_platings(config)
    assert [(p.batch.id, p.plate_date) for p in platings] == [
        ("spring2027_batch2_iPSC", 270115),
        (BATCH, 260810),
    ]
    assert all(p.path.is_dir() for p in platings)


def test_list_platings_skips_what_does_not_follow_the_convention(tmp_path):
    config = Config(data_root=tmp_path)
    config.recordings_dir.mkdir(parents=True)
    (config.recordings_dir / "misc").mkdir()  # a stray folder
    (config.recordings_dir / "plating_260810_notabatchid").mkdir()  # plating-shaped, bad batch id
    (config.recordings_dir / f"plating_260810_{BATCH}").touch()  # right name, not a directory

    assert store.list_platings(config) == []


def test_list_platings_survives_a_missing_recordings_dir(tmp_path):
    assert store.list_platings(Config(data_root=tmp_path / "nowhere")) == []


def test_plating_chips_lists_the_chip_directories(tmp_path):
    config = Config(data_root=tmp_path)
    plating_dir = config.recordings_dir / store.plating_dirname(BATCH, 260810)
    for name in ["chip_M1_M07460", "chip_M1_M07123"]:
        (plating_dir / name).mkdir(parents=True)
    (plating_dir / "notes").mkdir()  # a stray folder
    (plating_dir / "chip_M1_M99999").touch()  # chip-shaped, not a directory

    [plating] = store.list_platings(config)
    assert plating.chips() == ["M07123", "M07460"]


# --- splitting a multi-well file ------------------------------------------------------------------


def _multiwell_h5(path, wells=(0, 3), conditions=("ctrl", "drug")):
    """A minimal MaxWell-shaped file: /assay metadata, /wells/wellNNN, and the /recordings mirror."""
    import h5py

    blob = str({
        "Exp ID": BATCH, "Chip ID": "C1", "Plate date": 250512, "DIV": 7,
        "Well IDs": list(wells), "Conditions": list(conditions),
    })
    with h5py.File(path, "w") as f:
        f.create_dataset("wellplate/version", data=np.array([b"MaxTwo 6 multi-well MEA"]))
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


def _single_well_h5(path, well=0, plate="MaxOne"):
    """A minimal wells-format file. ``plate`` is what ``/wellplate/version`` says (``None`` for a
    file that says nothing), which is where ingest reads the chip's system from."""
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset(f"/wells/well{well:03d}/rec0000/spikes", data=np.arange(5))
        if plate is not None:
            f.create_dataset("wellplate/version", data=np.array([plate.encode()]))
    return path


def test_ingest_files_the_chip_by_the_system_the_file_says(tmp_path):
    """The chip directory's M1/M2 comes from the file's plate description, or from ``system=``;
    a file that says nothing and is given nothing is refused before anything is copied."""
    config = Config(data_root=tmp_path / "ms")
    kwargs = dict(batch=BATCH, plate_date=260810, chip="M07460", div=21, exp_id="t", on_progress=lambda _: None)

    [dest] = store.ingest_recording(_single_well_h5(tmp_path / "two.raw.h5", plate="MaxTwo 6 multi-well MEA"), config, **kwargs).values()
    assert dest.parent.parent.parent.name == "chip_M2_M07460"
    assert store.parse_recording_path(dest, config.recordings_dir).system == "M2"
    assert store.find_chip_dir(config, BATCH, 260810, "M07460") == dest.parent.parent.parent
    [plating] = store.list_platings(config)
    assert plating.system_of("M07460") == "M2" and plating.chip_dir("M07460") == dest.parent.parent.parent

    mute = _single_well_h5(tmp_path / "mute.raw.h5", plate=None)
    with pytest.raises(ValueError, match="MaxOne or a MaxTwo"):
        store.ingest_recording(mute, config, **{**kwargs, "exp_id": "u"})
    assert not list((config.recordings_dir).rglob("*_u.raw.h5"))
    [given] = store.ingest_recording(mute, config, **{**kwargs, "exp_id": "u"}, system="M1").values()
    assert "chip_M1_M07460" in str(given)  # the same chip serial cannot be on both, but the store does not police that
    with pytest.raises(ValueError, match="system"):
        store.ingest_recording(mute, config, **{**kwargs, "exp_id": "v"}, system="M3")


def test_ingest_places_names_and_registers_a_single_well_file(tmp_path):
    config = Config(data_root=tmp_path / "ms")
    src = _single_well_h5(tmp_path / "export.raw.h5")

    written = store.ingest_recording(
        src, config, batch=BATCH, plate_date=260810, chip="M07460", div=21,
        exp_id="stim_trial_3", conditions={0: "ctrl"}, on_progress=lambda _: None,
    )

    dest = written[0]
    stem = store.recording_stem(BATCH, 260810, "M07460", 0, 21)
    assert dest == store.recording_dir(config, BATCH, 260810, "M07460", 0, 21, system="M1") / f"{stem}_stim_trial_3.raw.h5"
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
        assert path.parent == store.recording_dir(config, BATCH, 250512, "C1", well, 7, system="M2")
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


def test_network_scan_slot_numbers_a_divs_scans_in_arrival_order(tmp_path):
    stem = "plating_260810_fall2026_batch1_DRG_chip_M07460_well_0_DIV_21"
    div_dir = tmp_path / "DIV_21"
    assert store.network_scan_slot(div_dir, stem) == ("network_scan", [])  # no directory yet
    div_dir.mkdir()
    plain = div_dir / f"{stem}_network_scan.raw.h5"
    plain.touch()
    tail, renames = store.network_scan_slot(div_dir, stem)
    assert tail == "network_scan_1" and renames == [(plain, div_dir / f"{stem}_network_scan_0.raw.h5")]
    # Already numbered: the next free index, nothing to rename.
    plain.rename(renames[0][1])
    (div_dir / f"{stem}_network_scan_1.raw.h5").touch()
    assert store.network_scan_slot(div_dir, stem) == ("network_scan_2", [])
    # MaxLab's own collision rename (plain beside _1): the plain one takes the gap at 0.
    (div_dir / f"{stem}_network_scan_0.raw.h5").unlink()
    plain.touch()
    tail, renames = store.network_scan_slot(div_dir, stem)
    assert tail == "network_scan_2" and renames == [(plain, div_dir / f"{stem}_network_scan_0.raw.h5")]
    # Another well's files on the same DIV are not this well's.
    assert store.network_scan_slot(div_dir, stem.replace("well_0", "well_1")) == ("network_scan", [])
    assert [store.network_scan_index(n) for n in (f"{stem}_network_scan.raw.h5", f"{stem}_network_scan_3.raw.h5", f"{stem}_trial.raw.h5")] == [0, 3, None]


def test_ingest_numbers_several_network_scans_of_one_div(tmp_path):
    """A second network scan on the same well and DIV is not a collision: the pair is numbered
    from zero in arrival order, both read back as network scans, the registry keeps one row, and
    the rename is journaled with the file that caused it."""
    config = Config(data_root=tmp_path / "ms")
    kwargs = {"batch": BATCH, "plate_date": 260810, "chip": "M07460", "div": 21, "kind": "network_scan", "on_progress": lambda _: None}
    first = store.ingest_recording(_single_well_h5(tmp_path / "a.raw.h5"), config, **kwargs)[0]
    assert first.name.endswith("_DIV_21_network_scan.raw.h5")

    second = store.ingest_recording(_single_well_h5(tmp_path / "b.raw.h5"), config, **kwargs)[0]
    third = store.ingest_recording(_single_well_h5(tmp_path / "c.raw.h5"), config, **kwargs)[0]
    names = sorted(p.name for p in first.parent.glob("*.h5"))
    assert [n.rsplit("_DIV_21_", 1)[1] for n in names] == ["network_scan_0.raw.h5", "network_scan_1.raw.h5", "network_scan_2.raw.h5"]
    assert not first.exists() and second.name.endswith("_network_scan_1.raw.h5") and third.name.endswith("_network_scan_2.raw.h5")
    for p in first.parent.glob("*.h5"):
        assert store.recording_kind(p) == "network_scan"
        location = store.parse_recording_path(p, config.recordings_dir)
        assert location.kind == "network_scan" and location.exp_id == ""
    assert [store.network_scan_index(n) for n in names] == [0, 1, 2]

    df = pd.read_csv(config.registry_path, keep_default_na=False)
    assert len(df) == 1 and df.iloc[0]["kind"] == "network_scan" and df.iloc[0]["exp_id"] == ""

    log = [t for t in transactions.read(config) if t.op == "recording.ingested"]
    assert len(log) == 3 and all(t.data["kind"] == "network_scan" for t in log)
    assert "renamed" not in log[0].data and "renamed" not in log[2].data
    assert log[1].data["renamed"] == {str(first): str(first.parent / first.name.replace("_network_scan.raw", "_network_scan_0.raw"))}

    # An activity scan still refuses to collide: numbering is a network-scan convention.
    src = _single_well_h5(tmp_path / "as.raw.h5")
    store.ingest_recording(src, config, **{**kwargs, "kind": "activity_scan"})
    with pytest.raises(FileExistsError):
        store.ingest_recording(src, config, **{**kwargs, "kind": "activity_scan"})


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


# --- renaming a batch -----------------------------------------------------------------------------

OLD = "summer2026_batch1_E18"
NEW = "fall2026_batch1_E18"
PLATE_DATE = 260813


def _seed_batch(config, batch, chip="P1", div=19):
    """Lay down one well of one batch everywhere the store names it, the way a run leaves it."""
    import h5py

    from mxtreme import io

    stem = store.recording_stem(batch, PLATE_DATE, chip, 0, div)
    rec_dir = store.recording_dir(config, batch, PLATE_DATE, chip, 0, div, system="M1")
    (rec_dir / "electrode_selection").mkdir(parents=True)
    h5_path = rec_dir / f"{stem}_activity_scan.raw.h5"
    blob = str({"Exp ID": batch, "Batch ID": batch, "Chip ID": chip, "Plate date": PLATE_DATE,
                "DIV": div, "Well IDs": [0], "Conditions": []})
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("/assay/metadata", data=np.array([blob.encode("utf-8")]))
        f.create_dataset("/wells/well000/rec0000/spikes", data=np.arange(5))
    (rec_dir / "electrode_selection" / f"{stem}_activity_scan_AS_well0.png").write_bytes(b"png")
    np.savez_compressed(
        rec_dir / "electrode_selection" / f"{stem}_activity_scan_network_well0.npz",
        recording_electrodes=np.arange(3),
    )
    io.register(
        {0: {"chip": chip, "DIV": div, "batch_id": batch, "plate_date": PLATE_DATE}},
        config.registry_path, kind="activity_scan",
    )

    npz_dir = config.preprocessed_dir / batch / chip / "well0"
    npz_dir.mkdir(parents=True)
    npz_path = npz_dir / f"DIV{div}_{PLATE_DATE}_{chip}_{batch}_well0_exp_data.npz"
    np.savez_compressed(
        npz_path, spike_data=np.arange(4), exp_id=np.array(batch), chip=np.array(chip),
        well=np.array(0), DIV=np.array(div), plate_date=np.array(PLATE_DATE),
        exp_condition=np.asarray(None), path_to_h5=np.array(str(h5_path)),
        step_log=np.asarray([{"step": "normalize_time"}], dtype=object),
    )
    io.register(
        {0: {"exp_id": batch, "chip": chip, "DIV": div, "plate_date": PLATE_DATE}},
        config.registry_path,
    )

    burst_dir = config.burst_data_dir / batch / chip / "well0"
    burst_dir.mkdir(parents=True)
    (burst_dir / f"DIV{div}_{PLATE_DATE}_{chip}_{batch}_well0_burst_data.csv").write_text("a,b\n1,2\n")
    (config.burst_data_dir / f"{batch}_burst_log.csv").write_text(
        f"exp_id,chip,well,DIV\n{batch},{chip},0,{div}\n"
    )

    summary_dir = config.analysis_dir / "activity" / batch / chip / "well0"
    summary_dir.mkdir(parents=True)
    (summary_dir / f"{batch}_{chip}_well0_burst_activity_summary.csv").write_text(
        f"culture_id,div,n_bursts\n{batch}_{chip}_well0,{div},17\n"
    )
    (config.analysis_dir / "reports").mkdir(exist_ok=True)
    (config.analysis_dir / "reports" / f"{batch}_{chip}_well0_report.pdf").write_bytes(b"%PDF")
    return h5_path, npz_path


def _mentions(root, batch_id):
    """Every path under ``root`` whose name, or whose text (CSVs), carries ``batch_id``."""
    hits = [p for p in root.rglob("*") if batch_id in p.name]
    hits += [p for p in root.rglob("*.csv") if batch_id in p.read_text()]
    return hits


def test_rename_batch_renames_everything_the_store_names(tmp_path):
    import h5py

    from mxtreme import io

    config = Config(data_root=tmp_path)
    _seed_batch(config, OLD)
    _seed_batch(config, "fall2026_batch12_E18", chip="P2")  # a sibling whose id contains no token of OLD
    npz_before = next(p for p in config.preprocessed_dir.rglob("*.npz") if OLD in p.name).stat().st_mtime
    sibling_before = sorted(_mentions(tmp_path, "fall2026_batch12_E18"))
    transactions.record(config, "culture.mark_dead", batch_id=OLD, plate_date=PLATE_DATE, chip="P1", well=0)

    lines = []
    result = store.rename_batch(config, OLD, NEW, reason="wrong season", actor="kam", on_progress=lines.append)

    assert _mentions(tmp_path, OLD) == []
    assert OLD not in config.registry_path.read_text()
    assert {p.batch.id for p in store.list_platings(config)} == {NEW, "fall2026_batch12_E18"}
    assert (result.old.id, result.new.id) == (OLD, NEW)
    assert (len(result.h5_files), len(result.npz_files), len(result.csv_files)) == (1, 1, 2)
    assert result.registry_rows == 2
    assert result.plate_dates == (PLATE_DATE,)
    assert lines and lines[-1].startswith("Journal:")

    # The rename is journaled against the new id, and what was journaled under the old one follows.
    log = transactions.read(config)
    last = log[-1]
    assert last.op == "batch.renamed" and (last.batch_id, last.plate_date) == (NEW, PLATE_DATE)
    assert (last.actor, last.note, last.data["old"], last.data["new"]) == ("kam", "wrong season", OLD, NEW)
    assert last.data["paths"] == len(result.paths) and last.data["registry_rows"] == 2
    mark = transactions.is_dead(
        transactions.batch_states(config), transactions.culture_states(config), NEW, PLATE_DATE, "P1", 0
    )
    assert mark is not None and mark.dead
    assert OLD in config.transactions_path.read_text().splitlines()[0]  # the file is not rewritten

    # The raw file's blob, and the preprocessed file's identity and pointer back to the raw file.
    h5_path = next(p for p in config.recordings_dir.rglob("*.h5") if NEW in p.name)
    with h5py.File(h5_path, "r") as f:
        assert io._embedded_metadata(h5_path)["Batch ID"] == NEW
        assert f["/wells/well000/rec0000/spikes"][()].tolist() == [0, 1, 2, 3, 4]
    npz_path = next(p for p in config.preprocessed_dir.rglob("*.npz") if NEW in p.name)
    with np.load(npz_path, allow_pickle=True) as npz:
        assert npz["exp_id"].item() == NEW
        assert npz["path_to_h5"].item() == str(h5_path)
        assert npz["spike_data"].tolist() == [0, 1, 2, 3]
    assert npz_path.stat().st_mtime == npz_before  # a rebuilt registry keeps its timestamps

    # The sibling batch was not touched.
    assert sorted(_mentions(tmp_path, "fall2026_batch12_E18")) == sibling_before

    # And a registry rebuilt from disk agrees with the one rewritten in place.
    rebuilt = tmp_path / "rebuilt.csv"
    io.rebuild_registry(config, registry_path=rebuilt)
    key = ["exp_id", "batch_id", "chip", "well", "div", "kind"]
    live = pd.read_csv(config.registry_path).fillna("")[key].astype(str)
    fresh = pd.read_csv(rebuilt).fillna("")[key].astype(str)
    assert set(map(tuple, live.values)) == set(map(tuple, fresh.values))


def test_rename_batch_dry_run_reports_the_plan_and_touches_nothing(tmp_path):
    config = Config(data_root=tmp_path)
    _seed_batch(config, OLD)
    before = sorted(str(p) for p in tmp_path.rglob("*"))

    result = store.rename_batch(config, OLD, NEW, dry_run=True)

    assert sorted(str(p) for p in tmp_path.rglob("*")) == before
    assert OLD in config.registry_path.read_text()
    assert "batch.renamed" not in {t.op for t in transactions.read(config)}
    assert result.registry_rows == 2 and len(result.paths) == 12
    assert all(NEW in dest.name and OLD not in dest.name for _, dest in result.paths)


@pytest.mark.parametrize("new, error", [
    (OLD, ValueError),                          # nothing to do
    ("batch-one", ValueError),                  # not a batch id
    ("fall2026_batch2_E18", FileExistsError),  # already in the store
])
def test_rename_batch_refuses_bad_targets(tmp_path, new, error):
    config = Config(data_root=tmp_path)
    _seed_batch(config, OLD)
    _seed_batch(config, "fall2026_batch2_E18", chip="P2")
    before = sorted(str(p) for p in tmp_path.rglob("*"))

    with pytest.raises(error):
        store.rename_batch(config, OLD, new)
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_rename_batch_can_take_the_old_system_suffix_off(tmp_path):
    """The 2026-09-21 migration: ``..._E18_M1`` -> ``..._E18``. The new id stands as a whole token
    inside every old name, which must read as the rename, not as the target already existing."""
    config = Config(data_root=tmp_path)
    _seed_batch(config, "fall2026_batch1_E18_M1")
    _seed_batch(config, "fall2026_batch2_E18_M1", chip="P2")  # a sibling, left alone

    result = store.rename_batch(config, "fall2026_batch1_E18_M1", "fall2026_batch1_E18")

    assert result.registry_rows == 2 and len(result.paths) == 12
    assert not _mentions(tmp_path, "fall2026_batch1_E18_M1")
    assert {p.batch.id for p in store.list_platings(config)} == {"fall2026_batch1_E18", "fall2026_batch2_E18_M1"}
    [plating] = [p for p in store.list_platings(config) if p.batch.id == "fall2026_batch1_E18"]
    assert plating.systems() == {"P1": "M1"}  # the chip directory still says which system
    with pytest.raises(FileExistsError):  # and a real collision is still one
        store.rename_batch(config, "fall2026_batch2_E18_M1", "fall2026_batch1_E18")


def test_rename_batch_can_leave_the_blobs_alone(tmp_path):
    """An archive copy on a share that refuses read-write HDF5 opens: paths, registry, CSVs and
    npz follow the rename; the raw files keep their blob until the originals are synced over."""
    import h5py

    from mxtreme import io

    config = Config(data_root=tmp_path)
    _seed_batch(config, OLD)

    result = store.rename_batch(config, OLD, NEW, h5_blobs=False)

    assert result.h5_files == [] and result.npz_files and result.csv_files and result.registry_rows == 2
    assert not _mentions(tmp_path, OLD)  # every name and every CSV cell moved on
    h5_path = next(config.recordings_dir.rglob("*.h5"))
    assert io._embedded_metadata(h5_path)["Batch ID"] == OLD  # the blob is the one thing left
    assert store.parse_recording_path(h5_path, config.recordings_dir).batch.id == NEW


def test_rename_batch_refuses_a_batch_it_cannot_find(tmp_path):
    config = Config(data_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        store.rename_batch(config, OLD, NEW)


def test_rename_batch_refuses_an_id_nested_inside_another(tmp_path):
    config = Config(data_root=tmp_path)
    _seed_batch(config, OLD)
    _seed_batch(config, f"{OLD}_E18", chip="P2")  # cell type "E18_E18": contains OLD whole

    with pytest.raises(ValueError, match="inside other batch ids"):
        store.rename_batch(config, OLD, NEW)


# --- ingesting scans, choosing wells, describing a file -------------------------------------------


def _scope_activity_scan_h5(path, wells=(0, 1), n_recordings=3):
    """A file shaped the way MaxLab Live's Activity Scan assay writes one: no /assay/metadata,
    the chip in /wellplate/id, a Plating Date per well, several recordings per well each on a
    different electrode set, with MaxLab's per-recording start/stop stamps."""
    import h5py

    mapping_dtype = np.dtype([("channel", "<i4"), ("electrode", "<i4"), ("x", "<f8"), ("y", "<f8")])
    spike_dtype = np.dtype([("frameno", "<i8"), ("channel", "<i4"), ("amplitude", "<f4")])
    with h5py.File(path, "w") as f:
        f.create_dataset("version", data=np.array([b"20190530"]))
        f.create_dataset("mxw_version", data=np.array([b"25.1.8.2"]))
        f.create_dataset("assay/script_id", data=np.array([b"ActivityScan_v1.0"]))
        f.create_dataset("assay/inputs/record_time", data=np.array([b"30"]))
        f.create_dataset("wellplate/id", data=np.array([b"M07474"]))
        f.create_dataset("wellplate/version", data=np.array([b"MaxTwo 6 multi-well MEA"]))
        for well in wells:
            f.create_dataset(f"wellplate/well{well:03d}/Plating Date", data=np.array([b"28.3.2024"]))
            f.create_dataset(f"wellplate/well{well:03d}/name", data=np.array([str(well + 1).encode()]))
            for rec in range(n_recordings):
                g = f.create_group(f"wells/well{well:03d}/rec{rec:04d}")
                mapping = np.zeros(4, dtype=mapping_dtype)
                mapping["channel"] = np.arange(4)
                mapping["electrode"] = np.arange(4) + 4 * rec  # a different subset every recording
                g.create_dataset("settings/mapping", data=mapping)
                g.create_dataset("settings/sampling", data=np.array([20000.0]))
                g.create_dataset("settings/lsb", data=np.array([6.3e-6]))
                spikes = np.zeros(5, dtype=spike_dtype)
                spikes["channel"] = np.arange(5) % 4
                spikes["amplitude"] = -20
                g.create_dataset("spikes", data=spikes)
                start = 1713555896329 + rec * 40_000
                g.create_dataset("start_time", data=np.array([start]))
                g.create_dataset("stop_time", data=np.array([start + 30_000]))
                f[f"recordings/rec{rec:04d}/well{well:03d}"] = g
    return path


def test_describe_reads_what_a_scope_activity_scan_says_about_itself(tmp_path):
    src = _scope_activity_scan_h5(tmp_path / "M07474_240419.h5")

    d = store.describe_recording(src)

    assert d.ok and d.problems == [] and d.warnings == []
    assert d.chip == "M07474" and d.system == "M2"
    assert d.script_id == "ActivityScan_v1.0" and d.kind_guess == "activity_scan"
    assert d.record_time == 30 and d.recorded_date == 240419 and d.recorded_at.startswith("2024-04-19")
    assert d.plating_date == 240328 and d.metadata is None
    assert [w.well for w in d.wells] == [0, 1]
    w = d.wells[0]
    assert (w.n_recordings, w.n_spikes, w.n_channels, w.n_electrodes) == (3, 15, 4, 12)
    assert w.sampling_hz == 20000.0 and abs(w.seconds - 90) < 1e-6 and not w.has_raw
    assert w.plating_date == 240328 and w.wellplate["name"] == "1"


def test_describe_guesses_a_network_scan_from_one_raw_recording(tmp_path):
    src = _scope_activity_scan_h5(tmp_path / "net.h5", wells=(0,), n_recordings=1)
    import h5py

    with h5py.File(src, "r+") as f:
        del f["assay/script_id"]
        f.create_dataset("wells/well000/rec0000/groups/routed/raw", data=np.zeros((4, 10), dtype="<i2"))

    d = store.describe_recording(src)
    assert d.ok and d.kind_guess == "network_scan" and d.wells[0].has_raw


def test_describe_reports_instead_of_raising(tmp_path):
    import h5py

    not_h5 = tmp_path / "notes.h5"
    not_h5.write_bytes(b"just text")
    assert store.describe_recording(not_h5).problems[0].startswith("Not an HDF5 file")
    assert store.describe_recording(tmp_path / "missing.h5").problems[0].startswith("Cannot read")

    flat = tmp_path / "flat.h5"
    with h5py.File(flat, "w") as f:
        f.create_dataset("sig", data=np.arange(3))
    d = store.describe_recording(flat)
    assert not d.ok and "/wells" in d.problems[0] and d.kind_guess == "experiment"

    empty_well = tmp_path / "empty.h5"
    with h5py.File(empty_well, "w") as f:
        f.create_dataset("wells/well000/rec0000/events", data=np.arange(0))
    d = store.describe_recording(empty_well)
    assert not d.ok and "lack spikes" in d.problems[0]


def test_describe_keeps_mxtreme_scan_metadata(tmp_path):
    src = _multiwell_h5(tmp_path / "combined.raw.h5")
    d = store.describe_recording(src)
    assert d.metadata["Chip ID"] == "C1" and d.metadata["Well IDs"] == [0, 3]
    assert d.chip is None  # MXtreme's own scans carry no /wellplate group in this fixture


def test_ingest_files_a_scope_activity_scan_as_an_activity_scan(tmp_path):
    """An outside scan lands under the scan tail and registers as a scan, so it reads back exactly
    like one MXtreme ran: kind from the name, blank exp_id, blob carrying the batch as exp id."""
    import h5py

    from mxtreme.scans.electrode_selection import load_activity_scan

    config = Config(data_root=tmp_path / "ms")
    src = _scope_activity_scan_h5(tmp_path / "M07474_240419.h5")

    written = store.ingest_recording(
        src, config, batch="spring2024_batch1_DRG", plate_date=240328, chip="M07474", div=22,
        kind="activity_scan", wells=[1], actor="kam", note="from the Scope archive",
        on_progress=lambda _: None,
    )

    assert sorted(written) == [1]
    dest = written[1]
    assert dest.name.endswith("_well_1_DIV_22_activity_scan.raw.h5")
    assert store.recording_kind(dest) == "activity_scan"
    location = store.parse_recording_path(dest, config.recordings_dir)
    assert (location.kind, location.well, location.div) == ("activity_scan", 1, 22) and not location.exp_id

    df = pd.read_csv(config.registry_path, keep_default_na=False)
    assert len(df) == 1
    row = df.iloc[0]
    assert (row["kind"], row["exp_id"], int(row["well"]), int(row["div"])) == ("activity_scan", "", 1, 22)

    with h5py.File(dest, "r") as f:
        blob = eval(f["assay/metadata"][:][0].decode())
        assert list(f["wells"]) == ["well001"]
    assert blob["Exp ID"] == "spring2024_batch1_DRG" and blob["Well IDs"] == [1]
    assert sorted(load_activity_scan(str(dest))) == [1]  # the selection pipeline reads it

    log = transactions.read(config)
    assert [t.op for t in log] == ["recording.ingested"]
    assert (log[0].actor, log[0].note, log[0].data["kind"]) == ("kam", "from the Scope archive", "activity_scan")
    assert log[0].exp_id == "spring2024_batch1_DRG"
    assert src.exists()  # copied; only the chosen well was split out


def test_ingest_derives_record_time_for_a_scan_that_lacks_it(tmp_path):
    import h5py

    config = Config(data_root=tmp_path / "ms")
    src = _scope_activity_scan_h5(tmp_path / "scan.h5", wells=(0,))
    with h5py.File(src, "r+") as f:
        del f["assay/inputs/record_time"]

    written = store.ingest_recording(
        src, config, batch=BATCH, plate_date=240328, chip="M07474", div=22,
        kind="activity_scan", on_progress=lambda _: None,
    )
    with h5py.File(written[0], "r") as f:
        assert int(f["assay/inputs/record_time"][0]) == 30  # from the start/stop stamps


def test_ingest_wells_must_exist_and_a_scan_takes_no_exp_id(tmp_path):
    config = Config(data_root=tmp_path / "ms")
    src = _scope_activity_scan_h5(tmp_path / "scan.h5")
    common = {"batch": BATCH, "plate_date": 240328, "chip": "M07474", "div": 22, "on_progress": lambda _: None}

    with pytest.raises(ValueError, match="holds no well"):
        store.ingest_recording(src, config, kind="activity_scan", wells=[5], **common)
    with pytest.raises(ValueError, match="exp_id"):
        store.ingest_recording(src, config, kind="network_scan", exp_id="x", **common)
    with pytest.raises(ValueError, match="exp_id"):
        store.ingest_recording(src, config, **common)  # an experiment with no name
    with pytest.raises(ValueError, match="kind"):
        store.ingest_recording(src, config, kind="stimulation", exp_id="x", **common)
    assert not (config.data_root / "recordings").exists()


def test_split_by_well_can_pick_wells(tmp_path):
    src = _multiwell_h5(tmp_path / "combined.raw.h5")
    written = store.split_by_well(src, lambda w: tmp_path / f"w{w}.h5", wells=[3], on_progress=lambda _: None)
    assert sorted(written) == [3]
    with pytest.raises(ValueError, match="holds no well"):
        store.split_by_well(src, lambda w: tmp_path / f"x{w}.h5", wells=[1], on_progress=lambda _: None)
