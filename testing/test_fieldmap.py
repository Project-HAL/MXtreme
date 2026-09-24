"""The field-map measurement, offline: the electrode layout around a driven electrode, and the
analysis on a synthetic recording whose field falls as 1/r in every direction."""

import json
import math

import h5py
import numpy as np
import pytest

from mxtreme.experiments import fieldmap
from mxtreme.experiments.associative import protocol

FPS = 20000.0


def test_recording_electrodes_form_a_dense_disc_and_eight_rays():
    stim = protocol.nearest_electrode(1750.0, 1000.0)
    rec = fieldmap.recording_electrodes(stim)
    assert stim not in rec and len(rec) == len(set(rec))
    sx, sy = protocol.electrode_xy(stim)
    d = np.array([math.hypot(*(np.subtract(protocol.electrode_xy(e), (sx, sy)))) for e in rec])
    # Everything within six pitches is there, and the rays reach a millimetre.
    assert (d <= 6 * protocol.PITCH_UM + 1).sum() >= 100
    assert d.max() >= 59 * protocol.PITCH_UM
    # Eight directions are represented far out.
    far = [e for e, dist in zip(rec, d) if dist > 500]
    angles = {
        round(
            math.degrees(math.atan2(protocol.electrode_xy(e)[1] - sy, protocol.electrode_xy(e)[0] - sx)) / 45
        )
        % 8
        for e in far
    }
    assert len(angles) == 8
    # Near an edge the rays are simply shorter.
    assert len(fieldmap.recording_electrodes(protocol.nearest_electrode(20.0, 20.0))) < len(rec)


def _synthetic(tmp_path, exponent=1.0, anisotropy=0.0, lsb=1e-6):
    """A recording in which every pulse deflects each electrode by A / r^exponent uV (1 uV per
    count), plus a little noise; the driven electrode itself rails."""
    record = fieldmap.plan((1750.0, 1000.0), amplitudes=(10.0, 40.0), pulses=3, iti_sec=0.5)
    stim = record["stim_electrode"]
    electrodes = [stim] + record["rec_electrodes"]
    n_frames = int(6 * FPS)
    rng = np.random.default_rng(1)
    raw = 512 + rng.normal(0, 2, size=(len(electrodes), n_frames))
    sx, sy = protocol.electrode_xy(stim)
    events, frame = [], int(1 * FPS)
    for token, spec in record["tokens"].items():
        events.append((frame, {"start_stimulation": f"{token}_0"}))
        for k in range(record["pulses"]):
            at = frame + round(k * record["iti_sec"] * FPS) + 3
            for i, e in enumerate(electrodes):
                x, y = protocol.electrode_xy(e)
                r = math.hypot(x - sx, y - sy)
                if e == stim:
                    v = 5000.0
                else:
                    ang = math.atan2(y - sy, x - sx)
                    v = (
                        3000.0
                        * spec["amplitude_mv"]
                        / 10.0
                        * (protocol.PITCH_UM / r) ** exponent
                        * (1 + anisotropy * math.cos(2 * ang))
                    )
                raw[i, at : at + 3] += v
        frame += int(record["pulses"] * record["iti_sec"] * FPS) + int(FPS)
    raw = np.clip(raw, 0, 1023).astype(np.uint16)
    path = tmp_path / "field.raw.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("recordings/rec0000/well000")
        g.create_dataset(
            "spikes", data=np.zeros(0, dtype=[("frameno", "<i8"), ("channel", "<i4"), ("amplitude", "<f4")])
        )
        g.create_dataset(
            "settings/mapping",
            data=np.array(
                [(c, e, *protocol.electrode_xy(e)) for c, e in enumerate(electrodes)],
                dtype=[("channel", "<i4"), ("electrode", "<i4"), ("x", "<f8"), ("y", "<f8")],
            ),
        )
        g.create_dataset("settings/sampling", data=np.array([FPS]))
        g.create_dataset("settings/lsb", data=np.array([lsb]))
        g.create_dataset("start_time", data=np.array([0]))
        ev = np.array(
            [(fr, 1, 1, json.dumps(m).encode()) for fr, m in events],
            dtype=[
                ("frameno", "<i8"),
                ("eventtype", "<u4"),
                ("eventid", "<u4"),
                ("eventmessage", h5py.string_dtype()),
            ],
        )
        g.create_dataset("events", data=ev)
        grp = g.create_group("groups/raw_all_0")
        grp.create_dataset("channels", data=np.arange(len(electrodes)))
        grp.create_dataset("frame_nos", data=np.arange(n_frames, dtype=np.int64))
        grp.create_dataset("raw", data=raw)
        f.create_dataset(f"assay/{fieldmap.PROTOCOL_KEY}", data=np.array([json.dumps(record).encode()]))
    return str(path)


def test_report_recovers_the_exponent_and_a_round_field(tmp_path, capsys):
    path = _synthetic(tmp_path, exponent=1.0)
    out = fieldmap.report(path, str(tmp_path / "f.png"), str(tmp_path / "f.csv"))
    fits = out["fits"]["by_amplitude"]
    for amp, f in fits.items():
        assert f is not None
        assert abs(f["exponent"] - 1.0) < 0.1, (amp, f)
        assert abs(f["r10_um"] - 10 * protocol.PITCH_UM) / (10 * protocol.PITCH_UM) < 0.25
    # The near electrodes saturate at 40 mV (3000 * 4 uV = 12 mV at one pitch); those rows are hollow, not fitted.
    assert fits[40.0]["saturated_channels"] > 0
    assert all(b["spread"] < 0.2 for b in out["fits"]["angle_spread"])
    assert out["fits"]["linearity_cv"] < 0.1
    text = capsys.readouterr().out
    assert "exponent n" in text and "direction" in text and "linearity" in text
    assert (tmp_path / "f.png").exists() and (tmp_path / "f.csv").exists()


def test_report_sees_a_field_that_depends_on_direction(tmp_path):
    out = fieldmap.report(_synthetic(tmp_path, exponent=1.0, anisotropy=0.5))
    assert any(b["spread"] > 0.5 for b in out["fits"]["angle_spread"])


def test_the_model_leaves_out_an_amplitude_that_disagrees_with_the_others():
    """The medium is linear, so every amplitude should give the same field per mV. On the first
    real field map one amplitude came out about three times too large; averaging it in would drag
    the model with it."""
    fits = {
        "by_amplitude": {
            a: {
                "exponent": 0.8,
                "uv_at_one_pitch": uv,
                "measured_to_um": 1000.0,
                "r10_um": 0.0,
                "r100_um": 0.0,
                "points": 100,
            }
            for a, uv in ((5.0, 150.0), (10.0, 100.0), (20.0, 200.0), (40.0, 400.0), (80.0, 800.0))
        },
        "linearity_cv": 0.5,
    }
    m = fieldmap.model(fits)
    assert m["amplitudes_mv"] == [10.0, 20.0, 40.0, 80.0]
    assert list(m["disagreeing"]) == [5.0]
    assert abs(m["uv_per_mv_at_one_pitch"] - 10.0) < 0.01


def test_the_field_contour_moves_with_amplitude_and_with_more_electrodes():
    """A line of equal field, not a threshold: it is drawn where the potential crosses a stated
    level, so it moves when the amplitude does. A contour defined as a fraction of the site's own
    field would not move at all, the medium being linear."""
    model = {"exponent": 0.78, "uv_per_mv_at_one_pitch": 11.5, "pitch_um": 17.5}
    hex7, _ = protocol.stim_site((1750.0, 1000.0), {"shape": "hex", "gap": 5})
    grid4, _ = protocol.stim_site((1750.0, 1000.0), {"shape": "grid", "size": 2, "gap": 6})
    at60 = fieldmap.contour_um(model, hex7, 60.0, 100.0)
    at30 = fieldmap.contour_um(model, hex7, 30.0, 100.0)
    assert at60["median_um"] > at30["median_um"]
    # Seven electrodes firing together put more field out than four, at the same amplitude.
    assert at60["median_um"] > fieldmap.contour_um(model, grid4, 60.0, 100.0)["median_um"]
    # One electrode at one pitch is the model's own scale.
    single = fieldmap.field_at_um(model, [hex7[3]], 40.0, protocol.electrode_xy(hex7[3]))
    assert single == pytest.approx(11.5 * 40.0)
    # A site's field is the sum of its electrodes', so it exceeds any one of them.
    centre = protocol.electrode_xy(hex7[3])
    assert fieldmap.field_at_um(model, hex7, 40.0, centre) > single


def test_the_activating_function_falls_off_two_powers_faster_than_the_potential():
    """Which is how an artifact can cover the array while the stimulation stays local."""
    assert fieldmap.activating_falloff({"exponent": 0.78}) == pytest.approx(2.78)


def test_load_model_refuses_a_file_that_is_not_one(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"exponent": 0.8}))
    with pytest.raises(ValueError, match="not a field model"):
        fieldmap.load_model(path)


def test_amplitudes_are_interleaved_rather_than_run_as_blocks():
    """A block design confounds amplitude with time: on the first real field map the lowest
    amplitude fired first and came out 2.65x too large per mV, and nothing in those data can say
    whether that was the amplitude or the moment."""
    record = fieldmap.plan((1750.0, 1000.0), amplitudes=(5.0, 10.0, 20.0), pulses=4)
    order = record["order"]
    assert len(order) == 12
    # Every repeat holds each amplitude exactly once, and the repeats are not all the same order.
    repeats = [order[i : i + 3] for i in range(0, 12, 3)]
    for repeat in repeats:
        assert sorted(repeat) == sorted(record["tokens"])
    assert len({tuple(r) for r in repeats}) > 1
    # An amplitude's pulses are never consecutive, which is exactly what a block design makes them.
    for token in record["tokens"]:
        positions = [i for i, t in enumerate(order) if t == token]
        assert len(positions) == 4
        assert positions != list(range(positions[0], positions[0] + 4))
    # The same seed gives the same order; a different one does not.
    assert fieldmap.plan((1750.0, 1000.0), amplitudes=(5.0, 10.0, 20.0), pulses=4)["order"] == order
    assert fieldmap.plan((1750.0, 1000.0), amplitudes=(5.0, 10.0, 20.0), pulses=4, seed=7)["order"] != order


def test_measure_takes_pulse_frames_from_the_events_when_each_pulse_has_one(tmp_path):
    """Interleaved runs mark every pulse; the older block recordings marked only each train's
    start, and those still have to read back."""
    path = _synthetic(tmp_path, exponent=1.0)
    rows, record = fieldmap.measure(path)
    assert rows and {r["pulses"] for r in rows} == {record["pulses"]}


def test_warm_up_pulses_are_fired_and_kept_out_of_the_analysis():
    """The first stimulation of a recording is a transient. Shuffling only moves it into a random
    amplitude; discarding it removes it."""
    record = fieldmap.plan((1750.0, 1000.0), amplitudes=(5.0, 80.0), pulses=3)
    assert record["warmup_pulses"] == fieldmap.WARMUP_PULSES
    assert record["warmup_amplitude_mv"] == 80.0
    # It is not one of the measured tokens, so measure() never looks for it.
    assert record["warmup_token"] not in record["tokens"]
    assert record["warmup_token"] not in record["order"]
    assert fieldmap.plan((1750.0, 1000.0), amplitudes=(5.0,), pulses=1, warmup=0)["warmup_pulses"] == 0


def test_the_run_fires_warm_ups_first_then_every_amplitude_interleaved(monkeypatch, tmp_path):
    """Walks ``run`` against a fake maxlab: no hardware, but the real order of sequences fired.
    A run is only exercised on the rig, so a signature or ordering mistake would otherwise reach
    the chip before anything caught it."""
    import sys
    import types
    from unittest import mock

    before = set(sys.modules)
    for name in ("maxlab", "maxlab.saving", "maxlab.system", "maxlab.chip", "maxlab.util"):
        monkeypatch.setitem(sys.modules, name, mock.MagicMock())
    try:
        fired = []

        class Seq:
            def __init__(self, name, **kw):
                self.name = name

            def send(self):
                fired.append(self.name)

            def shutdown(self):
                pass

        class Arr:
            def __init__(self, *a, **k):
                pass

            def close(self):
                pass

            def get_config(self):
                return types.SimpleNamespace(get_channels=lambda: list(range(8)))

        class Saving:
            def open_directory(self, d):
                pass

            def start_file(self, n):
                pass

            def group_delete_all(self):
                pass

            def group_define(self, w, n, c):
                pass

            def write_assay_property(self, k, v):
                return "Ok"

            def start_recording(self, w):
                pass

            def stop_recording(self):
                pass

            def stop_file(self):
                pass

        mx = types.SimpleNamespace(
            initialize=lambda: None,
            send=lambda x: "Ok",
            Core=lambda: types.SimpleNamespace(enable_stimulation_power=lambda b: None),
            Timing=types.SimpleNamespace(waitInit=0, waitInMX2Offset=0, waitAfterRecording=0),
            clear_events=lambda: None,
            activate=lambda w: None,
            offset=lambda: None,
            Saving=Saving,
            Sequence=Seq,
            Array=Arr,
        )

        def build(name, well, regions, delays, num_events, hz, phase, amplitude_mv=None, **kw):
            # Interleaving needs one pulse per sequence; a train could not be shuffled.
            assert num_events == 1
            return Seq(name), 0.001

        with (
            mock.patch.object(
                fieldmap, "time", types.SimpleNamespace(sleep=lambda s: None, time=lambda: 0.0)
            ),
            mock.patch("mxtreme.stimulation.sequences.build_region_sequence", build),
            mock.patch("mxtreme.scans.mx_setup.init_well_stim", lambda w, r, s, **k: (Arr(), [3], list(s))),
            mock.patch("mxtreme.experiments.associative.run.release", lambda *a, **k: None),
        ):
            fieldmap.run(
                (1750.0, 1000.0),
                str(tmp_path),
                "fake",
                amplitudes=(5.0, 10.0, 20.0),
                pulses=4,
                seed=3,
                mx=mx,
                on_progress=lambda m: None,
            )

        assert fired[:2] == ["warmup_ac_0"] * 2, "the discarded warm-ups come first"
        measured = fired[2:]
        assert len(measured) == 3 * 4
        for a in (5, 10, 20):
            assert measured.count(f"field_a{a:g}_ac_0") == 4
        # Shuffling is per repeat, so the same amplitude may sit either side of a repeat
        # boundary; what it can never do is run three deep, which a block design always does.
        for i in range(len(measured) - 2):
            assert not measured[i] == measured[i + 1] == measured[i + 2]
    finally:
        for name in set(sys.modules) - before:
            sys.modules.pop(name, None)
