"""The field-map measurement, offline: the electrode layout around a driven electrode, and the
analysis on a synthetic recording whose field falls as 1/r in every direction."""

import json
import math

import h5py
import numpy as np

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
