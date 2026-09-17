"""The associative experiment: parameters, geometry and schedule, all without a rig."""

import json
import os
import sys
from itertools import pairwise
from pathlib import Path

import pytest

from mxtreme.experiments.associative import protocol
from mxtreme.experiments.associative.params import AssociativeParams

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.join(HERE, "..", "src", "mxtreme", "experiments", "associative")


def _params(**overrides):
    base = {
        "batch": "fall2026_batch1_iPSC_M1",
        "chip": "M07140",
        "plate_date": 260122,
        "div": 30,
        "well": 0,
        "regions": {"US": [800, 1000], "CS": [2000, 500], "NS": [3000, 1500]},
        "rec_electrodes": list(range(0, 26400, 7)),
        "amplitudes_source": "test",
    }
    base.update(overrides)
    return AssociativeParams(**base)


def test_a_run_refuses_uncalibrated_amplitudes_except_for_calibration_itself():
    for mode in ("connectivity", "conditioning"):
        with pytest.raises(ValueError, match="uncalibrated"):
            _params(mode=mode, amplitudes_source="default").validate(for_run=True)
    _params(mode="calibration", amplitudes_source="default").validate(for_run=True)
    _params(mode="conditioning", amplitudes_source="saline").validate(for_run=True)


def test_phases_together_deliver_exactly_the_one_recording_schedule():
    whole = _params(
        encode_cycles=3, num_retrievals=2, pre_min=1, post_min=1, decay_min=1, retrieval_interval_min=1
    )
    blocks, stimuli = protocol.build_schedule(whole)
    one = [(b.label, p.token, p.roles) for b in blocks for p in b.presentations]

    phases = ["baseline"] + [f"encode_{k}" for k in (1, 2, 3)] + ["retrieval"]
    split, used = [], set()
    for phase in phases:
        p = _params(**{**vars(whole), "phase": phase})
        pb, ps = protocol.build_schedule(p)
        split += [(b.label, x.token, x.roles) for b in pb for x in b.presentations]
        used |= set(ps)
        # A phase builds only the sequences it fires.
        assert set(ps) == {x.token for b in pb for x in b.presentations}
        assert p.file_name.endswith(f"_assoc_{phase}") and p.run_id == f"assoc_{phase}"
    assert split == one
    assert used == set(stimuli)

    # Within a cycle the timings are the whole schedule's, rebased to the cycle's start.
    enc2 = protocol.build_schedule(_params(**{**vars(whole), "phase": "encode_2"}))[0]
    assert [b.label for b in enc2] == ["lead", "encode"]
    cycle_len = sum(
        b.duration_sec
        for b in protocol.build_schedule(_params(**{**vars(whole), "phase": "encode_1"}))[0][1:]
    )
    all_encode = next(b for b in blocks if b.label == "encode")
    second = [p for p in all_encode.presentations if cycle_len <= p.t_sec < 2 * cycle_len]
    assert [round(p.t_sec - cycle_len, 3) for p in second] == [
        round(p.t_sec, 3) for p in enc2[1].presentations
    ]

    with pytest.raises(ValueError, match="phase must be"):
        _params(encode_cycles=3, phase="encode_4").validate()
    with pytest.raises(ValueError, match="only applies to conditioning"):
        _params(mode="calibration", phase="baseline").validate()


def test_package_imports_without_maxlab():
    from mxtreme.experiments.associative import display, params, preview, protocol  # noqa: F401

    assert "maxlab" not in sys.modules


def test_shipped_parameter_files_load_and_schedule():
    for name in ("params_default.json", "params_dryrun.json"):
        params = AssociativeParams.from_json(os.path.join(PACKAGE, name))
        params.validate()
        blocks, _stimuli = protocol.build_schedule(params)
        assert [b.label for b in blocks][:3] == ["pre", "baseline", "encode"]
        assert blocks[-1].label == "post"
    default = AssociativeParams.from_json(os.path.join(PACKAGE, "params_default.json"))
    # A session that fits in an afternoon: three hours at most, on purpose.
    blocks, stimuli = protocol.build_schedule(default)
    assert 150 <= protocol.total_minutes(blocks) <= 180
    # CS and NS get the same dose, which is what makes NS a control rather than a bystander.
    budget = protocol.stimulation_budget(blocks, stimuli)
    assert budget["CS"]["pulses"] == budget["NS"]["pulses"]


def test_unknown_parameter_is_refused(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"T_STIM": 300, "_note": "old style"}))
    with pytest.raises(ValueError, match="unknown parameter"):
        AssociativeParams.from_json(path)


def test_json_round_trip(tmp_path):
    params = _params(dt_cs_us=-5)
    params.to_json(tmp_path / "p.json", {"regions": "a note"})
    back = AssociativeParams.from_json(tmp_path / "p.json")
    assert back == params


def test_file_name_follows_the_store():
    params = _params()
    assert params.file_name == "plating_260122_fall2026_batch1_iPSC_M1_chip_M07140_well_0_DIV_30_assoc"
    assert params.as_metadata()["Well IDs"] == [0]
    with pytest.raises(ValueError):
        _ = params.h5_path


def test_validate_for_run_needs_regions():
    with pytest.raises(ValueError, match="rec_electrodes"):
        _params(rec_electrodes=None).validate(for_run=True)
    with pytest.raises(ValueError, match="centre"):
        _params(regions={}).validate(for_run=True)
    _params().validate(for_run=True)


def test_focal_site_geometry():
    drive, ring = protocol.stim_site((1000, 1000), {"shape": "focal", "inner": 2, "return_radius": 3})
    assert len(drive) == 4 and len(ring) == 4
    centre = protocol.nearest_electrode(1000, 1000)
    col, row = centre % protocol.COLS, centre // protocol.COLS
    assert set(drive) == {protocol.electrode_at(col + dx, row + dy) for dx in (-1, 0) for dy in (-1, 0)}
    # The ring surrounds the block, not the centre electrode: three electrodes beyond each edge.
    assert set(ring) == {protocol.electrode_at(col + dx, row + dy) for dx in (-4, 3) for dy in (-4, 3)}
    drive, ring = protocol.stim_site((1000, 1000), {"shape": "grid", "size": 3, "gap": 2})
    assert len(drive) == 9 and ring == []
    with pytest.raises(ValueError):
        protocol.stim_site((10, 10), {"shape": "focal", "inner": 2, "return_radius": 3})
    with pytest.raises(ValueError, match="does not reach outside"):
        protocol.stim_site((1000, 1000), {"shape": "focal", "inner": 2, "inner_gap": 4, "return_radius": 2})


@pytest.mark.parametrize("inner", [2, 3])
@pytest.mark.parametrize("gap", [0, 1, 2, 3, 4, 7])
def test_return_ring_is_centred_on_the_driven_block(inner, gap):
    """The ring's centre and the driven block's centre coincide for every layout, so no return
    electrode sits closer to one driven electrode than its opposite does."""
    radius = (inner // 2) * (gap + 1) + 2
    drive, ring = protocol.stim_site(
        (1750, 1000), {"shape": "focal", "inner": inner, "inner_gap": gap, "return_radius": radius}
    )
    dx = [protocol.electrode_xy(e) for e in drive]
    rx = [protocol.electrode_xy(e) for e in ring]
    block = (sum(x for x, _ in dx) / len(dx), sum(y for _, y in dx) / len(dx))
    hoop = (sum(x for x, _ in rx) / len(rx), sum(y for _, y in rx) / len(rx))
    assert block == pytest.approx(hoop)
    # and the driven spacing is exactly the one asked for
    cols = sorted({round(x / protocol.PITCH_UM) for x, _ in dx})
    assert all(b - a == gap + 1 for a, b in pairwise(cols))


def test_regions_take_focal_dacs_and_refuse_close_centres():
    regions = protocol.regions_for(
        _params(
            stim_site={
                "shape": "focal",
                "inner": 2,
                "inner_gap": 4,
                "return_radius": 6,
                "return_points": "corners",
            }
        )
    )
    assert regions["US"].drive_dac == 0 and regions["US"].return_dac == 1
    assert regions["CS"].return_dac == 1
    assert len(regions["US"].rec_electrodes) > 0
    with pytest.raises(ValueError, match="apart"):
        protocol.regions_for(_params(regions={"US": [800, 1000], "CS": [900, 1000], "NS": [3000, 1500]}))


def test_schedule_shape():
    params = _params(
        pre_min=1,
        post_min=1,
        decay_min=1,
        retrieval_interval_min=1,
        t_stim=20,
        t_probe=10,
        pulse_hz=1.0,
        probe_iti=20,
        encode_iti=30,
        encode_cycles=2,
        encode_cycle_rest=5,
        encode_pattern=["PAIR", "NS"],
        num_retrievals=2,
        probe_reps=2,
        encode_probe_roles=[],
    )
    blocks, stimuli = protocol.build_schedule(params)
    labels = [b.label for b in blocks]
    assert labels == ["pre", "baseline", "encode", "decay", "retrieval_1", "rest_1", "retrieval_2", "post"]
    baseline = blocks[1]
    assert len(baseline.presentations) == 3 * 2
    assert all(p.token.startswith("probe_") for p in baseline.presentations)
    encode = blocks[2]
    assert [p.token for p in encode.presentations] == ["stim_cs_us", "stim_ns"] * 2
    assert encode.presentations[2].t_sec == pytest.approx(2 * 30 + 5)
    assert stimuli["stim_cs_us"].roles == ["CS", "US"] and stimuli["stim_cs_us"].num_events == 20
    assert stimuli["probe_us"].num_events == 10
    assert all(
        stimuli[p.token].duration_sec + params.pulse_window_ms[1] / 1000 <= params.encode_iti
        for p in encode.presentations
    )


def test_encode_checkpoints_probe_alone_between_cycles_and_never_after_the_last():
    params = _params(
        t_stim=20,
        t_probe=10,
        pulse_hz=1.0,
        probe_iti=20,
        encode_iti=30,
        encode_cycles=3,
        encode_cycle_rest=5,
        encode_pattern=["PAIR", "NS"],
        encode_probe_roles=["CS", "NS"],
        encode_probe_reps=1,
        encode_probe_sec=5,
    )
    blocks, stimuli = protocol.build_schedule(params)
    encode = blocks[2]
    tokens = [p.token for p in encode.presentations]
    assert tokens.count("checkpoint_cs") == tokens.count("checkpoint_ns") == 2  # cycles 1, 2 not 3
    assert not tokens[-1].startswith("checkpoint_")
    # Each checkpoint sits after its cycle's rest, and its probes are probe_iti apart.
    checks = [(p.t_sec, p.token) for p in encode.presentations if p.token.startswith("checkpoint_")]
    assert checks[0][0] == pytest.approx(2 * 30 + 5)
    assert checks[1][0] - checks[0][0] == pytest.approx(20)
    assert {t for _, t in checks} == {"checkpoint_cs", "checkpoint_ns"}
    # Shorter than a probe block's probe, because an unpaired CS here is a small extinction trial.
    assert stimuli["checkpoint_cs"].num_events == 5 < stimuli["probe_cs"].num_events

    none = protocol.build_schedule(_params(**{**vars(params), "encode_probe_roles": []}))[0][2]
    assert not any(p.token.startswith("checkpoint") for p in none.presentations)


def test_schedule_refuses_an_iti_that_cannot_hold_the_presentation():
    with pytest.raises(ValueError, match="encode_iti"):
        protocol.build_schedule(_params(t_stim=300, encode_iti=200))
    with pytest.raises(ValueError, match="probe_iti"):
        protocol.build_schedule(_params(t_probe=60, probe_iti=30))


def test_pairing_offset_and_negative_offset():
    params = _params(dt_cs_us=20)
    stim = protocol.stimulus_for(params, ["CS", "US"])
    assert stim.delays_sec == {"CS": 0.0, "US": 0.02}
    stim = protocol.stimulus_for(_params(dt_cs_us=-20), ["CS", "US"])
    assert stim.delays_sec == {"CS": 0.02, "US": 0.0}


def test_calibration_interleaves_regions_and_shuffles_amplitudes():
    params = _params(
        mode="calibration",
        calibration_amplitudes_mv=[50, 100],
        calibration_polarities=["anodic-first", "cathodic-first"],
        calibration_reps=2,
        calibration_iti=3,
    )
    blocks, stimuli = protocol.build_schedule(params)
    cal = [b for b in blocks if b.label.startswith("calibrat")]
    assert [b.label for b in cal] == ["calibrate"]
    tokens = {p.token for p in cal[0].presentations}
    assert tokens == {
        f"stim_{r}_a{a}_{p}" for r in ("us", "cs", "ns") for a in (50, 100) for p in ("ac", "ca")
    }
    assert len(cal[0].presentations) == 3 * 2 * 2 * 2
    assert all(stimuli[p.token].num_events == 1 for p in cal[0].presentations)
    # Within each repeat every region appears, and the order mixes them rather than blocking by region.
    per_rep = len(tokens)
    first = [p.roles[0] for p in cal[0].presentations[:per_rep]]
    assert set(first) == {"US", "CS", "NS"} and first != sorted(first, key=["US", "CS", "NS"].index)
    # One polarity: tokens carry no polarity tag, and the pulse count halves.
    one, _ = protocol.build_schedule(_params(**{**vars(params), "calibration_polarities": ["anodic-first"]}))
    assert len(one[1].presentations) == 3 * 2 * 2 and all(
        "_a" in p.token and not p.token.endswith(("_ac", "_ca")) for p in one[1].presentations
    )


def test_summary_reports_duty():
    params = _params(
        pre_min=1,
        post_min=1,
        decay_min=1,
        retrieval_interval_min=1,
        t_stim=20,
        t_probe=10,
        pulse_hz=1.0,
        probe_iti=20,
        encode_iti=30,
        encode_cycles=1,
        encode_pattern=["PAIR", "NS"],
    )
    text = protocol.summary(*protocol.build_schedule(params))
    assert "encode: stimulating" in text and "paired" in text


def test_preview_draws_to_png(tmp_path):
    from mxtreme.experiments.associative.preview import preview

    params = _params(
        pre_min=1,
        post_min=1,
        decay_min=1,
        retrieval_interval_min=1,
        t_stim=20,
        t_probe=10,
        pulse_hz=1.0,
        probe_iti=20,
        encode_iti=30,
        encode_cycles=1,
    )
    out = tmp_path / "preview.png"
    preview(params, str(out))
    assert out.exists() and out.stat().st_size > 10000


def test_execute_schedule_fires_on_time_with_a_fake_clock():
    from mxtreme.experiments.associative.run import execute_schedule

    params = _params(
        pre_min=0.1,
        post_min=0.1,
        decay_min=0.1,
        retrieval_interval_min=0.1,
        t_stim=2,
        t_probe=1,
        pulse_hz=1.0,
        probe_iti=3,
        encode_iti=4,
        encode_cycles=1,
        encode_pattern=["PAIR"],
        num_retrievals=1,
        probe_reps=1,
    )
    blocks, _ = protocol.build_schedule(params)
    now = [100.0]
    fired_tokens, marks = [], []
    fired, stopped = execute_schedule(
        blocks,
        fired_tokens.append,
        marks.append,
        clock=lambda: now[0],
        sleep=lambda dt: now.__setitem__(0, now[0] + dt),
        on_progress=lambda _s: None,
    )
    assert not stopped
    assert marks == [b.label for b in blocks] + ["end_experiment"]
    assert fired_tokens == [p.token for b in blocks for p in b.presentations]
    assert all(abs(actual - planned) < 1e-9 for _, _, planned, actual in fired)
    assert now[0] - 100.0 == pytest.approx(sum(b.duration_sec for b in blocks))

    # should_stop ends it at the next poll, with what fired so far kept.
    now[0] = 0.0
    calls = [0]

    def stop():
        calls[0] += 1
        return calls[0] > 3

    fired, stopped = execute_schedule(
        blocks,
        lambda _t: None,
        lambda _l: None,
        clock=lambda: now[0],
        sleep=lambda dt: now.__setitem__(0, now[0] + dt),
        should_stop=stop,
        on_progress=lambda _s: None,
    )
    assert stopped and len(fired) < 5


def test_connectivity_mode_is_single_pulses_interleaved():
    params = _params(mode="connectivity", pre_min=1, post_min=1, check_reps=3, check_iti=4)
    blocks, stimuli = protocol.build_schedule(params)
    assert [b.label for b in blocks] == ["pre", "check", "post"]
    check = blocks[1]
    assert len(check.presentations) == 3 * 3
    assert {p.token for p in check.presentations} == {"check_us", "check_cs", "check_ns"}
    assert all(stimuli[p.token].num_events == 1 for p in check.presentations)
    # Interleaved: each rep of three has all three roles.
    for k in range(3):
        rep = [p.token for p in check.presentations[3 * k : 3 * k + 3]]
        assert sorted(rep) == ["check_cs", "check_ns", "check_us"]
    assert check.duration_sec == pytest.approx(9 * 4)


def test_overrides_apply_json_values_and_reject_unknown_keys():
    params = _params(t_stim=300.0)
    out = params.with_overrides(
        ["dt_cs_us=20", 'encode_pattern=["PAIR"]', "encode_cycles=1", "pulse_polarity=cathodic-first"]
    )
    assert out.dt_cs_us == 20 and out.encode_pattern == ["PAIR"] and out.encode_cycles == 1
    assert out.pulse_polarity == "cathodic-first"
    assert params.dt_cs_us == 0.0  # the original is untouched
    with pytest.raises(ValueError, match="not a parameter"):
        params.with_overrides(["T_STIM=600"])


def test_the_two_dry_run_variants_schedule():
    base = AssociativeParams.from_json(os.path.join(PACKAGE, "params_dryrun.json"))
    base.regions = {"US": [1750, 1600], "CS": [2500, 500], "NS": [1000, 500]}
    offset = base.with_overrides(["dt_cs_us=20"])
    stim = protocol.stimulus_for(offset, ["CS", "US"])
    assert stim.delays_sec == {"CS": 0.0, "US": 0.02}
    long = base.with_overrides(["t_stim=600", "encode_iti=660", 'encode_pattern=["PAIR"]', "encode_cycles=1"])
    _blocks, stimuli = protocol.build_schedule(long)
    assert stimuli["stim_cs_us"].num_events == round(600 * long.pulse_hz)


def test_site_footprint_and_the_unit_budget():
    from mxtreme.experiments.associative.protocol import STIM_UNITS, site_footprint

    default = AssociativeParams.from_json(os.path.join(PACKAGE, "params_default.json"))
    f = site_footprint(default.stim_site)
    # A plain 2x2 block, no return electrodes: four units per site. The chip hands units out per
    # area of the array (about seven each, measured on a MaxOne), and sites of eight and six
    # electrodes both failed to connect on the rig.
    assert f["units"] == 4 and f["return"] == 0 and f["units"] * 3 <= STIM_UNITS
    # The stimulation site fills the region whose response is measured, rather than being a point
    # inside it.
    assert f["driven_span_um"] == pytest.approx(122.5)

    # A 3x3 focal site needs 13 units per region, 39 for three, and cannot route.
    with pytest.raises(ValueError, match="stimulation units"):
        _params(stim_site={"shape": "focal", "inner": 3, "inner_gap": 2, "return_radius": 5}).validate()

    # A 3x3 grid has no return ring, so it does fit.
    grid = _params(stim_site={"shape": "grid", "size": 3, "gap": 2})
    grid.validate()
    assert site_footprint(grid.stim_site)["units"] == 9


def test_destination_follows_the_store_layout(tmp_path):
    from mxtreme.config import Config
    from mxtreme.experiments.associative.preview import destination

    config = Config(data_root=tmp_path / "mxtreme-data")
    params = _params(
        batch="fall2026_batch1_iPSC_M2", chip="M07459", plate_date=260401, div=27, well=3, exp_id="assoc"
    )
    where = destination(params, config)
    expected_dir = (
        tmp_path
        / "mxtreme-data"
        / "recordings"
        / "plating_260401_fall2026_batch1_iPSC_M2"
        / "chip_M2_M07459"
        / "well_3"
        / "DIV_27"
    )
    assert where["directory"] == str(expected_dir)
    assert where["recording"].endswith(
        "plating_260401_fall2026_batch1_iPSC_M2_chip_M07459_well_3_DIV_27_assoc.raw.h5"
    )
    assert "registry.csv" in where["registered"]

    scratch = destination(params.with_overrides([f"save_path={tmp_path / 'dry'}"]))
    assert scratch["directory"] == str(tmp_path / "dry")
    assert scratch["registered"].startswith("not registered")


def test_recordings_tree_walk_ignores_the_sidecars(tmp_path):
    # rebuild_registry globs recordings/*/*/*/*/*.h5; the files select and run leave beside a
    # recording must not look like recordings.
    from mxtreme.config import Config
    from mxtreme.experiments.associative.preview import destination

    config = Config(data_root=tmp_path / "store")
    where = destination(_params(), config)
    d = Path(where["directory"])
    d.mkdir(parents=True)
    for name in ("x_protocol.json", "x_fired.csv", "x.cfg", "regions.png", "params_M07140_well0.json"):
        (d / name).write_text("")
    assert list(config.recordings_dir.glob("*/*/*/*/*.h5")) == []


def test_check_device_refuses_a_batch_for_the_other_system():
    from mxtreme.experiments.associative import run

    class FakeApi:
        def __init__(self, reply):
            self.reply = reply

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def send(self, _):
            return self.reply

    class FakeMx:
        def __init__(self, reply):
            self.comm = type("C", (), {"api_context": lambda _s=None, r=reply: FakeApi(r)})()

    assert run.check_device("fall2026_batch1_iPSC_M1", FakeMx("0")) == "MaxOne"
    assert run.check_device("fall2026_batch1_iPSC_M2", FakeMx("1")) == "MaxTwo"
    with pytest.raises(RuntimeError, match="says M1"):
        run.check_device("fall2026_batch1_iPSC_M1", FakeMx("1"))


def test_expand_phases_knows_the_session_and_encode_shorthands():
    from mxtreme.experiments.associative.run import expand_phases

    p = _params(encode_cycles=3)
    assert expand_phases(p, None) == ["all"]
    assert expand_phases(p, "session") == ["baseline", "encode_1", "encode_2", "encode_3", "retrieval"]
    assert expand_phases(p, "encode") == ["encode_1", "encode_2", "encode_3"]
    assert expand_phases(p, "baseline, encode_2") == ["baseline", "encode_2"]
    assert expand_phases(p, ["retrieval"]) == ["retrieval"]
    with pytest.raises(ValueError, match="cannot be combined"):
        expand_phases(p, "all,retrieval")


def test_a_keyboard_interrupt_mid_schedule_is_a_clean_stop():
    from mxtreme.experiments.associative.run import execute_schedule

    blocks = [
        protocol.Block(
            "a", 2.0, [protocol.Presentation(0.0, "x", ["US"]), protocol.Presentation(1.0, "y", ["CS"])]
        )
    ]
    fired_tokens = []

    def fire(token):
        fired_tokens.append(token)
        if token == "x":
            raise KeyboardInterrupt

    now = [0.0]
    fired, stopped = execute_schedule(
        blocks,
        fire,
        lambda _label: None,
        clock=lambda: now[0],
        sleep=lambda dt: now.__setitem__(0, now[0] + dt),
        on_progress=lambda _m: None,
    )
    assert stopped and fired == [] and fired_tokens == ["x"]


def test_a_session_run_sets_the_rig_up_once_and_opens_one_recording_per_phase(monkeypatch, tmp_path):
    """Walks ``run`` against a fake maxlab: no hardware, but the real setup/record/close order."""
    import types
    from unittest import mock

    before = set(sys.modules)
    for name in (
        "maxlab",
        "maxlab.saving",
        "maxlab.system",
        "maxlab.chip",
        "maxlab.util",
        "maxlab.characterize",
    ):
        monkeypatch.setitem(sys.modules, name, mock.MagicMock())
    try:
        from mxtreme.experiments.associative import run as R

        p = _params(
            encode_cycles=2,
            save_path=str(tmp_path),
            exp_id="t",
            stim_site={
                "shape": "focal",
                "inner": 2,
                "inner_gap": 4,
                "return_radius": 6,
                "return_points": "corners",
            },
        )
        opened, inits, closed = [], [], []

        class Saving:
            def open_directory(self, d):
                pass

            def start_file(self, name):
                opened.append(name)

            def group_delete_all(self):
                pass

            def group_define(self, *a):
                pass

            def write_assay_property(self, k, v):
                return "Ok"

            def start_recording(self, w):
                pass

            def stop_recording(self):
                pass

            def stop_file(self):
                closed.append(len(opened))

        class Timing:
            waitInit = waitInMX2Offset = waitAfterRecording = 0

        class Core:
            def enable_stimulation_power(self, b):
                return "cmd"

        deleted = []

        class ServerObject:
            def __init__(self, token, persistent=False, initial_delay=100):
                self.token, self.persistent = token, persistent

            def close(self):  # maxlab.Array
                if not self.persistent:
                    deleted.append(self.token)

            shutdown = close  # maxlab.Sequence

        mx = types.SimpleNamespace(
            Saving=Saving,
            Timing=Timing,
            Core=Core,
            Array=ServerObject,
            Sequence=ServerObject,
            initialize=lambda: inits.append(1),
            send=lambda c: "Ok",
            clear_events=lambda: None,
            activate=lambda w: None,
            offset=lambda: None,
        )
        cfg = types.SimpleNamespace(
            get_channels=lambda: [1], get_channels_for_electrodes=lambda e: [0] * len(e)
        )
        units = {
            f"{r}_{k}": [i]
            for i, (r, k) in enumerate((r, k) for r in ("US", "CS", "NS") for k in ("drive", "return"))
        }
        with (
            mock.patch("mxtreme.scans.activity_scan._require_maxlab", return_value=mx),
            mock.patch("mxtreme.experiments.associative.run.check_device", return_value="MaxOne"),
            mock.patch(
                "mxtreme.stimulation.sequences.init_well_regions",
                return_value=(
                    types.SimpleNamespace(get_config=lambda: cfg),
                    units,
                    {
                        k: [
                            protocol.stim_site(p.regions[k.split("_")[0]], p.stim_site)[
                                0 if k.endswith("drive") else 1
                            ][0]
                        ]
                        for k in units
                    },
                ),
            ),
            mock.patch(
                "mxtreme.stimulation.sequences.build_region_sequence",
                return_value=(types.SimpleNamespace(send=lambda: None), 1.0),
            ),
            mock.patch("mxtreme.stimulation.sequences.mark"),
            mock.patch("mxtreme.scans.mx_setup.write_metadata"),
            mock.patch("mxtreme.scans.mx_setup.write_exp_description"),
            mock.patch("mxtreme.scans.mx_setup.write_stim_electrodes"),
            mock.patch(
                "mxtreme.scans.activity_scan._saved_file", side_effect=lambda params, t: params.h5_path
            ),
            mock.patch("mxtreme.experiments.associative.run.execute_schedule", return_value=([], False)),
        ):
            result = R.run(p, phases="session", on_progress=lambda m: None)
        assert inits == [1]  # the chip once
        assert [n.rsplit("_t_", 1)[1] for n in opened] == ["baseline", "encode_1", "encode_2", "retrieval"]
        assert closed == [1, 2, 3, 4]  # each file closed before the next opens
        # Nothing of the run is left on the server afterwards: the array and every sequence.
        assert "stimulation0" in deleted and any(t.startswith("probe_cs_") for t in deleted)
        assert [r.h5_path.name.rsplit("_t_", 1)[1] for r in result.phases] == [
            "baseline.raw.h5",
            "encode_1.raw.h5",
            "encode_2.raw.h5",
            "retrieval.raw.h5",
        ]
    finally:
        # Modules imported under the fake maxlab must not outlive this test.
        for name in set(sys.modules) - before:
            sys.modules.pop(name, None)


def test_routing_gives_up_recording_electrodes_to_route_every_stimulation_electrode(monkeypatch):
    """A fake router that refuses a stimulation electrode until its crowded neighbourhood clears."""
    import types
    from unittest import mock

    before = set(sys.modules)
    for name in (
        "maxlab",
        "maxlab.saving",
        "maxlab.system",
        "maxlab.chip",
        "maxlab.util",
        "maxlab.characterize",
    ):
        monkeypatch.setitem(sys.modules, name, mock.MagicMock())
    try:
        from mxtreme.scans.mx_setup import route_with_stimulation

        stim = [3858]
        crowd = [3858 + d for d in (-221, -220, -219, -1, 1, 219, 220, 221)]  # its eight neighbours
        far = [100, 200, 300]
        state = {"selected": set(), "stim": set(), "routes": 0}

        class Array:
            def clear_selected_electrodes(self):
                state["selected"], state["stim"] = set(), set()

            def select_electrodes(self, elecs, weight=1):
                state["selected"] |= set(elecs)

            def select_stimulation_electrodes(self, elecs):
                state["stim"] |= set(elecs)

            def route(self):
                state["routes"] += 1
                return "OK"

            def get_config(self):
                # The router keeps the stimulation electrode only once none of its neighbours ask.
                kept = set(state["selected"])
                if not (state["selected"] & set(crowd)):
                    kept |= state["stim"]
                return types.SimpleNamespace(mappings=[types.SimpleNamespace(electrode=e) for e in kept])

        routed = route_with_stimulation(Array(), crowd + far, stim)
        assert 3858 in routed and set(far) <= routed and not (routed & set(crowd))
        assert state["routes"] == 2  # once, then once more with the crowd given up

        class Never(Array):
            def get_config(self):
                return types.SimpleNamespace(
                    mappings=[types.SimpleNamespace(electrode=e) for e in state["selected"]]
                )

        with pytest.raises(RuntimeError, match=r"3858 at \(2065, 298\) um"):
            route_with_stimulation(Never(), far, stim)

        class Broken(Array):
            def route(self):
                return "ERROR"

        with pytest.raises(RuntimeError, match="routing failed"):
            route_with_stimulation(Broken(), far, stim)
    finally:
        for name in set(sys.modules) - before:
            sys.modules.pop(name, None)


def test_a_stimulation_unit_clash_is_solved_by_a_neighbouring_electrode(monkeypatch):
    """A fake chip on which two electrodes of the block map to the same stimulation unit."""
    import types
    from unittest import mock

    before = set(sys.modules)
    for name in (
        "maxlab",
        "maxlab.saving",
        "maxlab.system",
        "maxlab.chip",
        "maxlab.util",
        "maxlab.characterize",
    ):
        monkeypatch.setitem(sys.modules, name, mock.MagicMock())
    try:
        from mxtreme.scans import mx_setup

        def unit_of(e):
            return 7 if e in (3853, 3858) else e % 32  # 3853 and 3858 clash

        log = {"routes": 0, "downloads": 0}

        class Array:
            def __init__(self, token, persistent=False):
                self.connected = []

            def close(self):
                pass

            def reset(self):
                pass

            def clear_selected_electrodes(self):
                self.selected = set()

            def select_electrodes(self, elecs, weight=1):
                self.selected |= set(elecs)

            def select_stimulation_electrodes(self, elecs):
                self.selected |= set(elecs)

            def route(self):
                log["routes"] += 1
                self.connected = []
                return "OK"

            def get_config(self):
                return types.SimpleNamespace(
                    mappings=[types.SimpleNamespace(electrode=e, channel=e % 1024) for e in self.selected]
                )

            def connect_electrode_to_stimulation(self, e):
                self.connected.append(e)

            def disconnect_electrode_from_stimulation(self, e):
                self.connected.remove(e)

            def query_stimulation_at_electrode(self, e):
                return str(unit_of(e))

            def download(self, wells):
                log["downloads"] += 1

        fake_mx = types.SimpleNamespace(
            Array=Array,
            activate=lambda w: None,
            Timing=types.SimpleNamespace(waitAfterDownload=0),
        )
        monkeypatch.setattr(mx_setup, "mx", fake_mx)
        monkeypatch.setattr(mx_setup.time, "sleep", lambda s: None)

        block = [3853, 3858, 4953, 4958]
        _array, units, used = mx_setup.init_well_stim(0, [100, 200, 300], block)
        assert used[0] == 3853 and used[2:] == [4953, 4958]
        moved = used[1]
        assert max(abs(moved % 220 - 3858 % 220), abs(moved // 220 - 3858 // 220)) == 1  # one pitch away
        # ... and the neighbour that keeps the block's spacing: no closer than 5 pitches to the rest.
        rest = [3853, 4953, 4958]
        assert min(max(abs(moved % 220 - o % 220), abs(moved // 220 - o // 220)) for o in rest) >= 5
        assert len(set(units)) == 4
        # Routed once, the clash found no routed neighbour, routed once more with candidates, then
        # the neighbour was connected without a third routing; downloaded once.
        assert log["routes"] == 2 and log["downloads"] == 1
        assert sorted(_array.connected) == sorted(used)  # nothing left connected on a taken unit
    finally:
        for name in set(sys.modules) - before:
            sys.modules.pop(name, None)
