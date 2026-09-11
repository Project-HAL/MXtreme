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


def test_chip_devices(config):
    assert tx.default_device(BATCH) == "MaxOne"
    assert tx.default_device("fall2026_batch2_DRG_M2") == "MaxTwo"
    tx.record(
        config, "chip.set_device", batch_id=BATCH, plate_date=260813, chip="P1", data={"device": "MaxOne+"}
    )
    assert tx.chip_devices(config) == {(BATCH, 260813, "P1"): "MaxOne+"}
