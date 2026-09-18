"""Tests for the store's transaction log."""

import json

import pytest

from mxtreme import transactions as tx
from mxtreme.config import Config


@pytest.fixture
def config(tmp_path):
    return Config(data_root=tmp_path / "store")


BATCH = "fall2026_batch1_DRG_M1"


def test_record_appends_one_json_line_per_call(config):
    tx.record(config, "culture.mark_dead", batch_id=BATCH, plate_date=260813, chip="P1", well=0, actor="kam")
    tx.record(config, "batch.mark_dead", batch_id=BATCH, plate_date=260813, note="contamination")
    lines = config.transactions_path.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["op"] == "culture.mark_dead" and first["chip"] == "P1" and first["well"] == 0
    assert json.loads(lines[1])["note"] == "contamination"


def test_read_is_oldest_first_and_skips_bad_lines(config):
    tx.record(
        config,
        "culture.mark_dead",
        batch_id=BATCH,
        plate_date=260813,
        chip="P1",
        well=0,
        time="2026-09-01T10:00:00",
    )
    with config.transactions_path.open("a") as f:
        f.write("{not json\n")
        f.write(json.dumps({"op": "bogus.op", "batch_id": BATCH, "plate_date": 260813, "time": "x"}) + "\n")
    tx.record(
        config,
        "culture.mark_alive",
        batch_id=BATCH,
        plate_date=260813,
        chip="P1",
        well=0,
        time="2026-09-02T10:00:00",
    )
    ops = [t.op for t in tx.read(config)]
    assert ops == ["culture.mark_dead", "culture.mark_alive"]


def test_missing_log_reads_empty(config):
    assert tx.read(config) == []
    assert tx.culture_states(config) == {}


def test_validation(config):
    with pytest.raises(ValueError):
        tx.record(config, "culture.mark_dead", batch_id=BATCH, plate_date=260813)  # no chip/well
    with pytest.raises(ValueError):
        tx.record(
            config,
            "chip.set_device",
            batch_id=BATCH,
            plate_date=260813,
            chip="P1",
            data={"device": "MaxThree"},
        )
    with pytest.raises(ValueError):
        tx.record(config, "batch.mark_dead", batch_id="not_a_batch", plate_date=260813)
    with pytest.raises(ValueError):
        tx.record(config, "batch.mark_dead", batch_id=BATCH, plate_date="26-08-13")
    with pytest.raises(ValueError):
        tx.record(config, "nope", batch_id=BATCH, plate_date=260813)
    assert not config.transactions_path.exists()


def test_fold_culture_and_batch_marks(config):
    key = (BATCH, 260813, "P1", 0)
    tx.record(config, "culture.mark_dead", batch_id=BATCH, plate_date=260813, chip="P1", well=0)
    tx.record(config, "batch.mark_dead", batch_id=BATCH, plate_date=260813)
    log = tx.read(config)
    batches, cultures = tx.batch_states(config, log), tx.culture_states(config, log)
    assert cultures[key].dead and batches[(BATCH, 260813)].dead
    # A batch marked dead makes an unmarked culture dead too.
    assert tx.is_dead(batches, cultures, BATCH, 260813, "P1", 1) is batches[(BATCH, 260813)]
    # Undoing the batch mark restores each culture to its own mark.
    tx.record(config, "batch.mark_alive", batch_id=BATCH, plate_date=260813)
    log = tx.read(config)
    batches, cultures = tx.batch_states(config, log), tx.culture_states(config, log)
    assert tx.is_dead(batches, cultures, BATCH, 260813, "P1", 1) is None
    assert tx.is_dead(batches, cultures, BATCH, 260813, "P1", 0) is cultures[key]
    tx.record(config, "culture.mark_alive", batch_id=BATCH, plate_date=260813, chip="P1", well=0)
    log = tx.read(config)
    assert (
        tx.is_dead(tx.batch_states(config, log), tx.culture_states(config, log), BATCH, 260813, "P1", 0)
        is None
    )


def test_treatment_applied_validates_and_derives_div(config):
    with pytest.raises(ValueError, match="applied_at"):
        tx.record(config, "treatment.applied", batch_id=BATCH, plate_date=260813, chip="P1", well=0)
    with pytest.raises(ValueError, match="ISO 8601"):
        tx.record(
            config,
            "treatment.applied",
            batch_id=BATCH,
            plate_date=260813,
            chip="P1",
            well=0,
            data={"applied_at": "yesterday-ish"},
        )
    t = tx.record(
        config,
        "treatment.applied",
        batch_id=BATCH,
        plate_date=260813,
        chip="P1",
        well=0,
        actor="kam",
        note="bath applied after the DIV 21 scan",
        data={"applied_at": "2026-09-03T14:30", "name": " ATP ", "dose": 100, "units": "µM", "group": "g1"},
    )
    assert t.div == 21  # 2026-08-13 is DIV 0
    assert t.data["name"] == "ATP" and t.data["dose"] == "100" and t.data["units"] == "µM"
    assert t.data["applied_at"].startswith("2026-09-03T14:30:00")  # local offset appended
    assert "+" in t.data["applied_at"] or "-" in t.data["applied_at"][10:]
    # An explicit DIV is kept; a missing optional field reads back empty.
    t2 = tx.record(
        config,
        "treatment.applied",
        batch_id=BATCH,
        plate_date=260813,
        chip="P1",
        well=0,
        div=5,
        data={"applied_at": "2026-09-03T14:30:00+02:00"},
    )
    assert t2.div == 5 and t2.data["name"] == "" and t2.data["group"] == ""


def test_div_on():
    from datetime import date, datetime

    assert tx.div_on(260813, date(2026, 8, 13)) == 0
    assert tx.div_on(260813, datetime(2026, 9, 3, 23, 59)) == 21  # noqa: DTZ001 -- naive means local, by design
    assert tx.div_on(260813, date(2026, 8, 1)) == 0  # before plating clamps rather than going negative


def test_treatments_fold(config):
    common = {"batch_id": BATCH, "plate_date": 260813, "chip": "P1"}
    tx.record(
        config, "treatment.applied", well=0, data={"applied_at": "2026-09-05T10:00", "name": "ATP"}, **common
    )
    tx.record(
        config,
        "treatment.applied",
        well=0,
        data={"applied_at": "2026-09-01T10:00", "name": "TTX", "dose": "1", "units": "µM"},
        note="pilot",
        **common,
    )
    tx.record(
        config, "treatment.applied", well=1, data={"applied_at": "2026-09-05T10:00", "name": "ATP"}, **common
    )
    tx.record(config, "culture.mark_dead", well=0, **common)  # other ops are not treatments
    folded = tx.treatments(config)
    assert set(folded) == {(BATCH, 260813, "P1", 0), (BATCH, 260813, "P1", 1)}
    well0 = folded[(BATCH, 260813, "P1", 0)]
    assert [e.name for e in well0] == ["TTX", "ATP"]  # by application time, not record order
    assert well0[0].label == "TTX 1 µM" and well0[0].div == 19 and well0[0].note == "pilot"
    assert well0[1].label == "ATP"
    assert tx.Treatment(applied_at="x", div=None, note="media change").label == "media change"
    assert tx.treatments(config, []) == {}


def test_treatments_follow_a_rename(config):
    tx.record(
        config,
        "treatment.applied",
        batch_id=BATCH,
        plate_date=260813,
        chip="P1",
        well=0,
        data={"applied_at": "2026-09-05T10:00", "name": "ATP"},
    )
    new = "fall2026_batch2_DRG_M1"
    tx.record(config, "batch.renamed", batch_id=new, plate_date=260813, data={"old": BATCH, "new": new})
    assert list(tx.treatments(config)) == [(new, 260813, "P1", 0)]


def test_chip_devices(config):
    assert tx.default_device(BATCH) == "MaxOne"
    assert tx.default_device("fall2026_batch2_DRG_M2") == "MaxTwo"
    tx.record(
        config, "chip.set_device", batch_id=BATCH, plate_date=260813, chip="P1", data={"device": "MaxOne+"}
    )
    assert tx.chip_devices(config) == {(BATCH, 260813, "P1"): "MaxOne+"}


# --- MXtreme's own journal entries ----------------------------------------------------------------


class _ScanParams:
    def __init__(self, wells=(0,), conditions=()):
        self.wells, self.conditions = list(wells), list(conditions)
        self.exp_id, self.batch_id, self.plate_date = "", BATCH, 260813
        self.chip, self.div = "P1", 7


def test_register_scan_journals_one_record_per_well(config):
    from mxtreme import io

    io.register_scan(
        _ScanParams(wells=(0, 2)),
        config.registry_path,
        kind="network_scan",
        well_files={0: "/store/w0.h5", 2: "/store/w2.h5"},
    )
    log = tx.read(config)
    assert [t.op for t in log] == ["network_scan.registered"] * 2
    assert [(t.well, t.div, t.data["path"]) for t in log] == [(0, 7, "/store/w0.h5"), (2, 7, "/store/w2.h5")]
    assert log[0].batch_id == BATCH and log[0].plate_date == 260813 and log[0].chip == "P1"
    assert log[0].actor  # MXtreme's own entries name the OS user


def test_save_preprocessed_journals_the_npz(config, make_well):
    from mxtreme import io

    well = make_well()
    out = io.save_preprocessed(config.preprocessed_dir, well, registry_path=config.registry_path)
    log = tx.read(config)
    assert len(log) == 1 and log[0].op == "preprocessed.saved"
    assert log[0].exp_id == "testExp" and log[0].batch_id == "" and log[0].chip == "C0001"
    assert log[0].well == 0 and log[0].div == 7 and log[0].plate_date == 250101
    assert log[0].data == {"path": str(out), "source": "/tmp/fake.raw.h5"}


def test_save_burst_data_journals_the_csv(config):
    from types import SimpleNamespace

    import pandas as pd

    from mxtreme import io

    class FakeBurstSet:
        def to_dataframe(self):
            return pd.DataFrame({"peak_frame": [1, 2]})

        def __len__(self):
            return 2

    recording = SimpleNamespace(exp_id=BATCH, chip="P1", well=0, DIV=7, plate_date=260813)
    out = io.save_burst_data(FakeBurstSet(), config.burst_data_dir, recording)
    log = tx.read(config)
    assert [t.op for t in log] == ["bursts.saved"]
    assert log[0].exp_id == BATCH and log[0].chip == "P1" and log[0].div == 7
    assert log[0].data == {"path": str(out), "n_bursts": 2}


def test_ingest_journals_each_well(config, tmp_path):
    import h5py
    import numpy as np

    from mxtreme import store

    src = tmp_path / "outside.h5"
    with h5py.File(src, "w") as f:
        f.create_dataset("/wells/well000/rec0000/spikes", data=np.arange(5))
    written = store.ingest_recording(
        src,
        config,
        batch=BATCH,
        plate_date=260813,
        chip="P1",
        div=9,
        exp_id="stim1",
        on_progress=lambda _: None,
    )
    log = tx.read(config)
    assert [t.op for t in log] == ["recording.ingested"]
    assert log[0].exp_id == "stim1" and log[0].batch_id == BATCH and log[0].div == 9
    assert log[0].data == {"source": str(src), "path": str(written[0]), "moved": False, "kind": "experiment"}


def test_rebuild_registry_journals_once(config, make_well):
    from mxtreme import io

    io.save_preprocessed(config.preprocessed_dir, make_well(), registry_path=config.registry_path)
    io.rebuild_registry(config)
    ops = [t.op for t in tx.read(config)]
    assert ops == ["preprocessed.saved", "registry.rebuilt"]
    assert tx.read(config)[-1].data["n_recordings"] == 1


def test_for_culture_matches_by_batch_or_exp_id_and_includes_batch_level(config):
    tx.record(config, "activity_scan.registered", batch_id=BATCH, plate_date=260813, chip="P1", well=0, div=7)
    tx.record(config, "preprocessed.saved", exp_id=BATCH, chip="P1", well=0, div=7)  # older flow: exp id only
    tx.record(config, "preprocessed.saved", exp_id=BATCH, chip="P1", well=1, div=7)  # another well
    tx.record(
        config,
        "activity_scan.registered",
        batch_id="fall2026_batch2_DRG_M1",
        plate_date=260901,
        chip="P1",
        well=0,
        div=7,
    )
    tx.record(config, "batch.mark_dead", batch_id=BATCH, plate_date=260813)
    mine = tx.for_culture(tx.read(config), BATCH, 260813, "P1", 0)
    assert [t.op for t in mine] == ["activity_scan.registered", "preprocessed.saved", "batch.mark_dead"]
    assert len(tx.for_batch(tx.read(config), BATCH, 260813)) == 4


def test_reading_follows_a_batch_rename(config):
    old, new = "summer2026_batch1_DRG_M1", BATCH
    tx.record(config, "activity_scan.registered", batch_id=old, plate_date=260813, chip="P1", well=0, div=7)
    tx.record(config, "preprocessed.saved", exp_id=old, chip="P1", well=0, div=7)  # older flow: exp id only
    tx.record(config, "culture.mark_dead", batch_id=old, plate_date=260813, chip="P1", well=0)
    tx.record(
        config, "chip.set_device", batch_id=old, plate_date=260813, chip="P1", data={"device": "MaxOne+"}
    )
    tx.record(config, "batch.renamed", batch_id=new, plate_date=260813, data={"old": old, "new": new})
    tx.record(config, "bursts.saved", batch_id=new, plate_date=260813, chip="P1", well=0, div=7)
    tx.record(config, "note", batch_id=old, plate_date=270101, note="a later batch reusing the old id")

    log = tx.read(config)
    assert [t.batch_id or t.exp_id for t in log[:6]] == [new] * 6
    assert (log[1].batch_id, log[1].exp_id) == ("", new)  # the exp-id-only record follows too
    assert log[4].data == {"old": old, "new": new}  # the rename record remembers what it was
    assert log[6].batch_id == old  # a record after the rename is left as written
    assert [t.op for t in tx.for_culture(log, new, 260813, "P1", 0)] == [
        "activity_scan.registered",
        "preprocessed.saved",
        "culture.mark_dead",
        "chip.set_device",
        "batch.renamed",
        "bursts.saved",
    ]
    assert tx.for_batch(log, old, 260813) == []
    assert tx.is_dead(tx.batch_states(config), tx.culture_states(config), new, 260813, "P1", 0).dead
    assert tx.chip_devices(config) == {(new, 260813, "P1"): "MaxOne+"}
    # The file is untouched: the first line still names the old batch.
    assert json.loads(config.transactions_path.read_text().splitlines()[0])["batch_id"] == old


def test_renames_chain(config):
    a, b, c = "summer2026_batch1_DRG_M1", "fall2026_batch1_DRG_M1", "fall2026_batch9_DRG_M1"
    tx.record(config, "batch.mark_dead", batch_id=a, plate_date=260813)
    tx.record(config, "batch.renamed", batch_id=b, plate_date=260813, data={"old": a, "new": b})
    tx.record(config, "note", batch_id=b, plate_date=260813, note="under the middle name")
    tx.record(config, "batch.renamed", batch_id=c, plate_date=260813, data={"old": b, "new": c})

    log = tx.read(config)
    assert [t.batch_id for t in log] == [c] * 4
    assert tx.batch_states(config) == {(c, 260813): tx.Mark(True, log[0].time, log[0].actor, "")}


def test_journal_ops_need_their_identity(config):
    with pytest.raises(ValueError):
        tx.record(config, "preprocessed.saved", exp_id="x", chip="P1", well=0)  # no DIV
    with pytest.raises(ValueError):
        tx.record(config, "bursts.saved", chip="P1", well=0, div=1)  # neither batch nor exp id
    tx.record(config, "registry.rebuilt", data={"n_recordings": 0})  # store-level needs nothing
    assert tx.read(config)[0].scope == "store"


def test_backfill_from_registry_is_idempotent(config, make_well):
    from mxtreme import io

    io.save_preprocessed(config.preprocessed_dir, make_well(), registry_path=config.registry_path)
    io.register_scan(_ScanParams(wells=(0,)), config.registry_path)
    config.transactions_path.unlink()  # a store that predates the log
    assert tx.backfill(config, on_progress=lambda _: None) == 2
    ops = sorted(t.op for t in tx.read(config))
    assert ops == ["activity_scan.registered", "preprocessed.saved"]
    assert all(t.note == tx.BACKFILL_NOTE and t.actor == "" for t in tx.read(config))
    assert tx.backfill(config, on_progress=lambda _: None) == 0
