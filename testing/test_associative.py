"""The associative experiment: parameters, geometry and schedule, all without a rig."""

import json
import os
import sys

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
    }
    base.update(overrides)
    return AssociativeParams(**base)


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
    assert 200 <= protocol.total_minutes(protocol.build_schedule(default)[0]) <= 240


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
    assert protocol.electrode_at(col - 3, row - 3) in ring
    drive, ring = protocol.stim_site((1000, 1000), {"shape": "grid", "size": 3, "gap": 2})
    assert len(drive) == 9 and ring == []
    with pytest.raises(ValueError):
        protocol.stim_site((10, 10), {"shape": "focal", "inner": 2, "return_radius": 3})


def test_regions_take_focal_dacs_and_refuse_close_centres():
    regions = protocol.regions_for(_params())
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


def test_calibration_sweeps_amplitude_and_polarity_shuffled():
    params = _params(
        mode="calibration", calibration_amplitudes_mv=[50, 100], calibration_reps=2, calibration_iti=3
    )
    blocks, stimuli = protocol.build_schedule(params)
    cal = [b for b in blocks if b.label.startswith("calibrate_")]
    assert [b.label for b in cal] == ["calibrate_us", "calibrate_cs", "calibrate_ns"]
    tokens = {p.token for p in cal[0].presentations}
    assert tokens == {"stim_us_a50_ac", "stim_us_a50_ca", "stim_us_a100_ac", "stim_us_a100_ca"}
    assert len(cal[0].presentations) == 2 * 2 * 2
    assert all(stimuli[p.token].num_events == 1 for p in cal[0].presentations)


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
    assert f["units"] == 8 and f["units"] * 3 <= STIM_UNITS
    # The stimulation site fills the region whose response is measured, rather than being a point
    # inside it.
    assert f["driven_span_um"] == pytest.approx(87.5)
    assert f["ring_radius_um"] == pytest.approx(default.region_radius_um, abs=1.0)

    # A 3x3 focal site needs 13 units per region, 39 for three, and cannot route.
    with pytest.raises(ValueError, match="stimulation units"):
        _params(stim_site={"shape": "focal", "inner": 3, "inner_gap": 2, "return_radius": 5}).validate()

    # A 3x3 grid has no return ring, so it does fit.
    grid = _params(stim_site={"shape": "grid", "size": 3, "gap": 2})
    grid.validate()
    assert site_footprint(grid.stim_site)["units"] == 9
