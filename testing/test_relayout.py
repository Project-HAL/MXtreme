"""Bringing a store from the per-well layout (``chip_*/well_<N>/DIV_<d>/``) into the per-DIV one."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mxtreme import io, store
from mxtreme import transactions as tx
from mxtreme.config import Config

BATCH = "fall2026_batch2_E18"
PLATE = 260915


def _old_stem(chip, well, div):
    return f"plating_{PLATE}_{BATCH}_chip_{chip}_well_{well}_DIV_{div}"


def _piece(path, well, *, start, n_recordings=1, wells_in_blob=None, blob=True):
    """One well's file as the old split left it (or as a MaxOne scan was recorded): the well's
    recordings under both of MaxWell's views, the shared parts, and a one-well metadata blob."""
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["version"] = "20190530"
        f.create_dataset("/assay/inputs/record_time", data=np.array([60]))
        if blob:
            meta = str({"Batch ID": BATCH, "Chip ID": path.name.split("_chip_")[1].split("_well_")[0],
                        "Plate date": PLATE, "DIV": int(path.parent.name.removeprefix("DIV_")),
                        "Well IDs": [well], "Conditions": []})
            f.create_dataset("/assay/metadata", data=np.array([meta.encode("utf-8")]))
        for rec in range(n_recordings):
            g = f.create_group(f"/wells/well{well:03d}/rec{rec:04d}")
            g.create_dataset("spikes", data=np.arange(3) + well)
            g.create_dataset("start_time", data=np.array([start + well + 130_000 * rec]))
            g.create_dataset("stop_time", data=np.array([start + 60_000 + 130_000 * rec]))
            f[f"/recordings/rec{rec:04d}/well{well:03d}"] = g
    return path


def _npz(config, chip, well, div, h5_path):
    d = config.preprocessed_dir / BATCH / chip / f"well{well}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"DIV{div}_{PLATE}_{chip}_{BATCH}_well{well}_exp_data.npz"
    np.savez_compressed(path, spike_data=np.arange(4), batch_id=np.array(BATCH), well=np.array(well),
                        path_to_h5=np.array(str(h5_path)))
    return path


@pytest.fixture
def old_store(tmp_path):
    """A store as the rigs left it on 2026-10-28.

    A MaxOne chip ``P1``: single-well files on DIV 5 (activity scan with selection outputs) and
    DIV 9 (activity + network scan). A MaxTwo chip ``M1``: on DIV 9 the six pieces of one split
    activity scan, a network scan that only ever covered well 0, selection outputs for well 0
    under the old prefix, and a stray ``chip_*/electrode_selection/`` file from a selection run on
    the unsplit file; on DIV 11 two pieces of a split network scan. Preprocessed files point at a
    moved piece (M1 DIV 9 well 0) and at a stitched one (M1 DIV 11 well 1).
    """
    config = Config(data_root=tmp_path / "store")
    rec = config.recordings_dir / f"plating_{PLATE}_{BATCH}"
    p1, m1 = rec / "chip_M1_P1", rec / "chip_M2_M1"
    files = {}

    files["p1_as5"] = _piece(p1 / "well_0" / "DIV_5" / f"{_old_stem('P1', 0, 5)}_activity_scan.raw.h5", 0, start=1_000_000, n_recordings=2)
    sel = p1 / "well_0" / "DIV_5" / "electrode_selection"
    sel.mkdir()
    (sel / f"{_old_stem('P1', 0, 5)}_activity_scan_AS_well0.png").write_bytes(b"png")
    (sel / f"{_old_stem('P1', 0, 5)}_activity_scan_network_well0.npz").write_bytes(b"npz")
    files["p1_as9"] = _piece(p1 / "well_0" / "DIV_9" / f"{_old_stem('P1', 0, 9)}_activity_scan.raw.h5", 0, start=2_000_000, n_recordings=2)
    files["p1_ns9"] = _piece(p1 / "well_0" / "DIV_9" / f"{_old_stem('P1', 0, 9)}_network_scan.raw.h5", 0, start=2_500_000)

    for well in range(6):
        files[f"m1_as9_{well}"] = _piece(m1 / f"well_{well}" / "DIV_9" / f"{_old_stem('M1', well, 9)}_activity_scan.raw.h5", well, start=3_000_000, n_recordings=2)
    files["m1_ns9"] = _piece(m1 / "well_0" / "DIV_9" / f"{_old_stem('M1', 0, 9)}_network_scan.raw.h5", 0, start=3_500_000)
    sel = m1 / "well_0" / "DIV_9" / "electrode_selection"
    sel.mkdir()
    (sel / f"{_old_stem('M1', 0, 9)}_activity_scan_AS_well0.png").write_bytes(b"as0")
    (sel / f"{_old_stem('M1', 0, 9)}_activity_scan_network_well0.npz").write_bytes(b"net0")
    stray = m1 / "electrode_selection"
    stray.mkdir()
    (stray / f"{_old_stem('M1', '0-1-2-3-4-5', 9)}_activity_scan_AS_well3.png").write_bytes(b"as3")
    (stray / "unrelated_AS_well0.png").write_bytes(b"?")
    for well in (0, 1):
        files[f"m1_ns11_{well}"] = _piece(m1 / f"well_{well}" / "DIV_11" / f"{_old_stem('M1', well, 11)}_network_scan.raw.h5", well, start=4_000_000)

    npz_moved = _npz(config, "M1", 0, 9, files["m1_ns9"])
    npz_stitched = _npz(config, "M1", 1, 11, files["m1_ns11_1"])

    for well in range(6):
        io.register({well: {"batch_id": BATCH, "chip": "M1", "DIV": 9, "plate_date": PLATE}}, config.registry_path, kind="activity_scan")
    io.register({0: {"batch_id": BATCH, "chip": "P1", "DIV": 5, "plate_date": PLATE}}, config.registry_path, kind="activity_scan")
    tx.record(config, "note", batch_id=BATCH, plate_date=PLATE, note="before")
    return config, files, npz_moved, npz_stitched


def _snapshot(root):
    return sorted(str(p) for p in Path(root).rglob("*"))


def test_dry_run_reports_the_plan_and_touches_nothing(old_store):
    config, _files, npz_moved, npz_stitched = old_store
    before = _snapshot(config.data_root)
    registry_before = config.registry_path.read_text()
    lines = []

    result = store.relayout_by_div(config, dry_run=True, on_progress=lines.append)

    assert _snapshot(config.data_root) == before
    assert config.registry_path.read_text() == registry_before
    assert [t.op for t in tx.read(config)] == ["note"]
    assert result.trashed_to is None and result.pruned_dirs == () and result.skipped == ()
    assert len(result.moved) == 4  # P1's three files, M1's well-0 network scan
    assert len(result.stitched) == 2  # M1 DIV 9 activity scan (6 pieces), M1 DIV 11 network scan (2)
    assert sorted(len(parts) for parts, _ in result.stitched) == [2, 6]
    assert set(result.npz_files) == {npz_moved, npz_stitched}
    assert any(line.startswith("Would stitch 6 files") for line in lines)
    assert any(line.startswith("Would move ") for line in lines)


def test_relayout_moves_stitches_and_journals(old_store):
    config, files, npz_moved, npz_stitched = old_store
    import h5py

    registry_before = config.registry_path.read_text()
    npz_mtime = npz_stitched.stat().st_mtime
    rec = config.recordings_dir / f"plating_{PLATE}_{BATCH}"
    p1, m1 = rec / "chip_M1_P1", rec / "chip_M2_M1"

    result = store.relayout_by_div(config, reason="per-DIV layout", actor="kam", on_progress=lambda _: None)

    # -- where everything landed, under its new name ---------------------------------------------
    as9 = m1 / "DIV_9" / f"{_old_stem('M1', '0-1-2-3-4-5', 9)}_activity_scan.raw.h5"
    ns11 = m1 / "DIV_11" / f"{_old_stem('M1', '0-1', 11)}_network_scan.raw.h5"
    expected = {
        p1 / "DIV_5" / files["p1_as5"].name,
        p1 / "DIV_9" / files["p1_as9"].name,
        p1 / "DIV_9" / files["p1_ns9"].name,
        m1 / "DIV_9" / files["m1_ns9"].name,
        as9,
        ns11,
    }
    assert set(config.recordings_dir.rglob("*.h5")) == expected
    assert not any(d.name.startswith("well_") for d in config.recordings_dir.rglob("*") if d.is_dir())
    assert dict(result.moved) == {
        files["p1_as5"]: p1 / "DIV_5" / files["p1_as5"].name,
        files["p1_as9"]: p1 / "DIV_9" / files["p1_as9"].name,
        files["p1_ns9"]: p1 / "DIV_9" / files["p1_ns9"].name,
        files["m1_ns9"]: m1 / "DIV_9" / files["m1_ns9"].name,
    }
    assert {dest: set(parts) for parts, dest in result.stitched} == {
        as9: {files[f"m1_as9_{w}"] for w in range(6)},
        ns11: {files["m1_ns11_0"], files["m1_ns11_1"]},
    }
    assert result.skipped == ()

    # -- the stitched files hold every well, readable both ways -----------------------------------
    assert store.wells_in_file(as9) == list(range(6)) and store.wells_in_file(ns11) == [0, 1]
    with h5py.File(as9, "r") as f:
        assert list(f["recordings/rec0001"]) == [f"well{w:03d}" for w in range(6)]
        assert eval(f["assay/metadata"][:][0].decode())["Well IDs"] == list(range(6))
        np.testing.assert_array_equal(f["recordings/rec0000/well004/spikes"][:], np.arange(3) + 4)
    for dest in expected:
        location = store.parse_recording_path(dest, config.recordings_dir)
        assert location is not None, dest
    assert store.parse_recording_path(as9, config.recordings_dir).wells == tuple(range(6))
    assert store.parse_recording_path(ns11, config.recordings_dir).wells == (0, 1)

    # -- the pieces went to the trash at their store-relative paths; moved files were not copied ---
    assert result.trashed_to is not None and result.trashed_to.parent == config.trash_dir
    for key in [f"m1_as9_{w}" for w in range(6)] + ["m1_ns11_0", "m1_ns11_1"]:
        assert (result.trashed_to / files[key].relative_to(config.data_root)).is_file()
    assert not (result.trashed_to / files["p1_as5"].relative_to(config.data_root)).exists()
    assert len(list(result.trashed_to.rglob("*.h5"))) == 8

    # -- selection outputs followed their scan, renamed to the new prefix --------------------------
    sel5 = p1 / "DIV_5" / "electrode_selection"
    assert sorted(p.name for p in sel5.iterdir()) == [
        f"{_old_stem('P1', 0, 5)}_activity_scan_AS_well0.png",
        f"{_old_stem('P1', 0, 5)}_activity_scan_network_well0.npz",
    ]
    sel9 = m1 / "DIV_9" / "electrode_selection"
    new_prefix = store.selection_prefix(as9)
    assert sorted(p.name for p in sel9.iterdir()) == [
        f"{new_prefix}_AS_well0.png", f"{new_prefix}_AS_well3.png", f"{new_prefix}_network_well0.npz",
    ]
    assert (sel9 / f"{new_prefix}_AS_well0.png").read_bytes() == b"as0"
    assert (sel9 / f"{new_prefix}_AS_well3.png").read_bytes() == b"as3"  # the stray one, from the chip dir
    assert (m1 / "electrode_selection" / "unrelated_AS_well0.png").is_file()  # not ours; left alone
    assert len(result.selection_files) == 5
    assert store.derived_files(config, as9) == sorted(sel9.iterdir())

    # -- preprocessed files point at the new raw file; their timestamps are kept ------------------
    assert set(result.npz_files) == {npz_moved, npz_stitched}
    with np.load(npz_moved, allow_pickle=True) as npz:
        assert npz["path_to_h5"].item() == str(m1 / "DIV_9" / files["m1_ns9"].name)
    with np.load(npz_stitched, allow_pickle=True) as npz:
        assert npz["path_to_h5"].item() == str(ns11) and npz["spike_data"].tolist() == [0, 1, 2, 3]
    assert npz_stitched.stat().st_mtime == npz_mtime

    # -- the registry carries no paths and is left alone; a rebuild agrees with the tree ----------
    assert config.registry_path.read_text() == registry_before
    rebuilt = config.data_root / "rebuilt.csv"
    io.rebuild_registry(config, registry_path=rebuilt)
    df = pd.read_csv(rebuilt)
    assert len(df[df["kind"] == "preprocessed"]) == 2  # the two .npz files, as before
    df = df[df["kind"] != "preprocessed"]
    assert len(df) == 1 + 2 + 6 + 1 + 2  # P1: DIV5 AS, DIV9 AS+NS; M1: six AS wells, NS well 0, NS wells 0-1
    assert sorted(df[(df["chip"] == "M1") & (df["div"] == 9) & (df["kind"] == "activity_scan")]["well"]) == list(range(6))
    assert sorted(df[(df["chip"] == "M1") & (df["div"] == 11)]["well"]) == [0, 1]

    # -- journaled once per well of each file, after the note that was already there --------------
    log = [t for t in tx.read(config) if t.op == "recording.moved"]
    assert len(log) == 1 + 2 + 6 + 1 + 2
    assert all(t.actor == "kam" and t.note == "per-DIV layout" for t in log)
    stitched = [t for t in log if t.data["path"] == str(as9)]
    assert [t.well for t in stitched] == list(range(6))
    assert all(t.data["stitched"] and len(t.data["from"]) == 6 for t in stitched)
    [moved] = [t for t in log if t.data["path"] == str(p1 / "DIV_9" / files["p1_ns9"].name)]
    assert moved.data == {"from": [str(files["p1_ns9"])], "path": str(p1 / "DIV_9" / files["p1_ns9"].name), "stitched": False}

    # -- emptied directories were pruned; a second run finds nothing to do ------------------------
    assert {d.name for d in result.pruned_dirs} >= {"well_0", "well_1", "well_5", "DIV_9", "DIV_11", "electrode_selection"}
    again = store.relayout_by_div(config, on_progress=lambda _: None)
    assert again.moved == () and again.stitched == () and again.skipped == ()


def test_relayout_numbers_a_file_whose_destination_is_taken(old_store):
    """A file already at the destination is another recording of the same wells, kind and DIV
    (whoever put it there), so the old one lands beside it as ``_1`` -- MaxLab's own collision
    name -- with its selection outputs renamed to match, and nothing is skipped."""
    config, files, *_ = old_store
    rec = config.recordings_dir / f"plating_{PLATE}_{BATCH}"
    taken = rec / "chip_M1_P1" / "DIV_5" / files["p1_as5"].name
    taken.parent.mkdir(parents=True)
    taken.write_bytes(b"already here")

    result = store.relayout_by_div(config, on_progress=lambda _: None)

    numbered = taken.with_name(taken.name.replace("_activity_scan.raw.h5", "_activity_scan_1.raw.h5"))
    assert result.skipped == ()
    assert dict(result.moved)[files["p1_as5"]] == numbered
    assert numbered.is_file() and taken.read_bytes() == b"already here"
    location = store.parse_recording_path(numbered, config.recordings_dir)
    assert (location.kind, location.wells, location.div) == ("activity_scan", (0,), 5)
    assert sorted(p.name for p in (taken.parent / "electrode_selection").iterdir()) == [
        f"{_old_stem('P1', 0, 5)}_activity_scan_1_AS_well0.png",
        f"{_old_stem('P1', 0, 5)}_activity_scan_1_network_well0.npz",
    ]
    assert len(result.moved) == 4 and len(result.stitched) == 2
    assert [t.div for t in tx.read(config) if t.op == "recording.moved" and t.chip == "P1"] == [5, 9, 9]


def test_relayout_keeps_two_runs_of_one_day_apart(tmp_path):
    """mx2's M10331 on 2026-09-24: a combined run at 14:34 whose activity scan split into six
    pieces, then a second run at 15:10 whose split failed, leaving its whole activity scan (and
    the selection outputs made from it) in the chip directory. Different recordings, so both are
    kept: the earlier, stitched, under the plain name; the later as ``_activity_scan_1``, its
    stray outputs following it under the numbered prefix rather than being claimed by the
    stitched scan whose base name they happen to share."""
    config = Config(data_root=tmp_path / "store")
    m1 = config.recordings_dir / f"plating_{PLATE}_{BATCH}" / "chip_M2_M1"
    pieces = [_piece(m1 / f"well_{w}" / "DIV_9" / f"{_old_stem('M1', w, 9)}_activity_scan.raw.h5", w, start=1_000_000, n_recordings=2) for w in range(6)]
    whole_stem = _old_stem("M1", "0-1-2-3-4-5", 9)
    import h5py

    whole = m1 / f"{whole_stem}_activity_scan.raw.h5"
    with h5py.File(whole, "w") as f:
        for well in range(6):
            for rec in range(2):
                g = f.create_group(f"/wells/well{well:03d}/rec{rec:04d}")
                g.create_dataset("spikes", data=np.arange(2))
                g.create_dataset("start_time", data=np.array([3_000_000 + well + 130_000 * rec]))
                g.create_dataset("stop_time", data=np.array([3_060_000 + 130_000 * rec]))
                f[f"/recordings/rec{rec:04d}/well{well:03d}"] = g
    stray = m1 / "electrode_selection"
    stray.mkdir()
    (stray / f"{whole_stem}_activity_scan_AS_well4.png").write_bytes(b"later run")

    result = store.relayout_by_div(config, dry_run=False, on_progress=lambda _: None)

    div9 = m1 / "DIV_9"
    plain, numbered = div9 / f"{whole_stem}_activity_scan.raw.h5", div9 / f"{whole_stem}_activity_scan_1.raw.h5"
    assert {dest: set(parts) for parts, dest in result.stitched} == {plain: set(pieces)}
    assert dict(result.moved) == {whole: numbered}
    assert result.redundant == () and result.skipped == ()
    assert store.wells_in_file(plain) == list(range(6)) and store.wells_in_file(numbered) == list(range(6))
    assert [p.name for p in (div9 / "electrode_selection").iterdir()] == [f"{whole_stem}_activity_scan_1_AS_well4.png"]
    assert store.parse_recording_path(numbered, config.recordings_dir).wells == tuple(range(6))


def test_relayout_on_a_store_already_in_the_new_layout_does_nothing(tmp_path):
    config = Config(data_root=tmp_path / "store")
    div_dir = store.recording_dir(config, BATCH, PLATE, "P1", 5, system="M1")
    _piece(div_dir / f"{store.recording_stem(BATCH, PLATE, 'P1', 0, 5)}_activity_scan.raw.h5", 0, start=1)
    before = _snapshot(config.data_root)

    result = store.relayout_by_div(config, on_progress=lambda _: None)

    assert result == store.Relayout((), (), (), (), (), (), None, ())
    assert _snapshot(config.data_root) == before
    assert store.relayout_by_div(Config(data_root=tmp_path / "nowhere"), on_progress=lambda _: None).moved == ()


def test_relayout_keeps_a_whole_file_a_failed_split_left_and_trashes_its_pieces(tmp_path):
    """What mx2 actually had on 2026-10-28 for one chip: the split of a six-well scan failed
    part-way, so the whole file stayed in the chip directory *and* the pieces it had written sat
    in their well directories (all six for the activity scan, well 0 alone for the network scan),
    with the selection run on the whole file leaving its outputs in the chip directory and a
    later one on the well-0 piece leaving the same figure under the old prefix.

    The whole file is the recording as the rig wrote it, so it is the one kept: moved into the DIV
    directory unchanged, nothing stitched, every piece and the piece-derived selection output
    trashed, the preprocessed files that named the whole file re-pointed at its new home.
    """
    config = Config(data_root=tmp_path / "store")
    m1 = config.recordings_dir / f"plating_{PLATE}_{BATCH}" / "chip_M2_M1"
    whole_stem = _old_stem("M1", "0-1-2-3-4-5", 9)
    import h5py

    def _whole(path, start, n_recordings):
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            meta = {"Batch ID": BATCH, "Chip ID": "M1", "Plate date": PLATE, "DIV": 9, "Well IDs": list(range(6)), "Conditions": []}
            f.create_dataset("/assay/metadata", data=np.array([str(meta).encode()]))
            for well in range(6):
                for rec in range(n_recordings):
                    g = f.create_group(f"/wells/well{well:03d}/rec{rec:04d}")
                    g.create_dataset("spikes", data=np.arange(3) + well)
                    g.create_dataset("start_time", data=np.array([start + well + 130_000 * rec]))
                    g.create_dataset("stop_time", data=np.array([start + 60_000 + 130_000 * rec]))
                    f[f"/recordings/rec{rec:04d}/well{well:03d}"] = g
        return path

    whole_as = _whole(m1 / f"{whole_stem}_activity_scan.raw.h5", 3_000_000, 2)
    whole_ns = _whole(m1 / f"{whole_stem}_network_scan.raw.h5", 3_500_000, 1)
    pieces = [_piece(m1 / f"well_{w}" / "DIV_9" / f"{_old_stem('M1', w, 9)}_activity_scan.raw.h5", w, start=3_000_000, n_recordings=2) for w in range(6)]
    pieces.append(_piece(m1 / "well_0" / "DIV_9" / f"{_old_stem('M1', 0, 9)}_network_scan.raw.h5", 0, start=3_500_000))
    stray = m1 / "electrode_selection"
    stray.mkdir()
    (stray / f"{whole_stem}_activity_scan_AS_well0.png").write_bytes(b"whole")
    (stray / f"{whole_stem}_activity_scan_network_well2.npz").write_bytes(b"net2")
    sel0 = m1 / "well_0" / "DIV_9" / "electrode_selection"
    sel0.mkdir()
    (sel0 / f"{_old_stem('M1', 0, 9)}_activity_scan_AS_well0.png").write_bytes(b"piece")
    npzs = [_npz(config, "M1", w, 9, whole_ns) for w in range(6)]

    result = store.relayout_by_div(config, dry_run=False, on_progress=lambda _: None)

    div9 = m1 / "DIV_9"
    assert result.stitched == () and result.skipped == ()
    assert dict(result.moved) == {whole_as: div9 / whole_as.name, whole_ns: div9 / whole_ns.name}
    assert set(result.redundant) == set(pieces) | {sel0 / f"{_old_stem('M1', 0, 9)}_activity_scan_AS_well0.png"}
    assert set(config.recordings_dir.rglob("*.h5")) == {div9 / whole_as.name, div9 / whole_ns.name}
    assert store.wells_in_file(div9 / whole_as.name) == list(range(6))
    assert sorted(p.name for p in (div9 / "electrode_selection").iterdir()) == [
        f"{whole_stem}_activity_scan_AS_well0.png",
        f"{whole_stem}_activity_scan_network_well2.npz",
    ]
    assert (div9 / "electrode_selection" / f"{whole_stem}_activity_scan_AS_well0.png").read_bytes() == b"whole"
    assert not any(d.name.startswith("well_") for d in m1.iterdir())
    assert not stray.exists()
    assert len(list(result.trashed_to.rglob("*"))) >= len(pieces) + 1
    for npz in npzs:
        with np.load(npz, allow_pickle=True) as z:
            assert str(z["path_to_h5"]) == str(div9 / whole_ns.name)
    assert result.npz_files == tuple(npzs)
    moves = [t for t in tx.read(config) if t.op == "recording.moved"]
    assert len(moves) == 12 and all(not t.data["stitched"] for t in moves)


def test_relayout_moves_an_empty_scan_by_its_name(tmp_path):
    """An aborted scan's file has no wells inside (MaxLab finalized an empty file); it cannot be
    grouped by its recordings, but its name says which well and DIV it was, so it moves alone."""
    import h5py

    config = Config(data_root=tmp_path / "store")
    p1 = config.recordings_dir / f"plating_{PLATE}_{BATCH}" / "chip_M1_P1"
    empty = p1 / "well_0" / "DIV_7" / f"{_old_stem('P1', 0, 7)}_activity_scan_0.raw.h5"
    empty.parent.mkdir(parents=True)
    with h5py.File(empty, "w") as f:
        f.create_dataset("version", data=np.array([b"x"]))

    result = store.relayout_by_div(config, dry_run=False, on_progress=lambda _: None)

    assert result.skipped == () and dict(result.moved) == {empty: p1 / "DIV_7" / empty.name}
    assert store.parse_recording_path(p1 / "DIV_7" / empty.name, config.recordings_dir).wells == (0,)
