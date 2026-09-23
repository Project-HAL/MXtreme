"""How far a pulse reaches through the medium: the field around one stimulation electrode.

A saline chip has no neurons, so what a recording electrode sees when another electrode is pulsed
is the pulse's field alone, attenuated by the medium and the electrode geometry. Mapping that
field -- the artifact's size against distance from the driven electrode, at several amplitudes
and in every direction -- says how large a stimulation site's *reach* is, which is what the
associative experiment's regions and their separation are sized against, and whether the field
is round (independent of direction) or not.

The measurement: one electrode is driven with single biphasic pulses at a ladder of amplitudes
while raw traces are kept on electrodes laid out around it -- a dense disc out to six pitches,
then eight rays out to a millimetre. For every pulse and recording electrode the report takes
the largest deflection from the pre-pulse level within the first 0.8 ms, discards channels the
amplifier saturated on, and gives the median over pulses. The figure is the field against
distance on log axes per amplitude, the same normalised by amplitude (they collapse if the medium
is linear), the field against direction at fixed distances, and a map. The numbers are the
power-law exponent, the distance at which the field is a tenth and a hundredth of its value one
pitch away, and how much the field varies with direction.

Run on the rig::

    python -m mxtreme.experiments.fieldmap run --centre 1750,1000 --out <dir> --name fieldmap_P003983

Read back (anywhere with MaxWell's HDF5 filter, i.e. the rig)::

    python -m mxtreme.experiments.fieldmap report <dir>/fieldmap_P003983.raw.h5 -o field.png --csv field.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from mxtreme.experiments.associative import protocol

PROTOCOL_KEY = "fieldmap_protocol"
#: mV per phase. Small ones so the electrodes nearest the driven one are not all saturated.
DEFAULT_AMPLITUDES = (5.0, 10.0, 20.0, 40.0, 80.0)
#: Rays out from the driven electrode, every 45 degrees.
RAYS = 8
#: Every electrode within this many pitches is recorded.
DENSE_RADIUS = 6
#: Along each ray: (step in electrodes, out to this many electrodes).
RAY_STEPS = ((1, 10), (2, 30), (4, 60))
#: The deflection is read this long after each pulse, ms; both 100 us phases and the buffer's
#: settling fall inside it.
PEAK_WINDOW_MS = 0.8
#: A deflection this close to the amplifier's full scale is saturated and not a measurement.
SATURATION = 0.9


# ---------------------------------------------------------------
#                            THE PLAN
# ---------------------------------------------------------------


def recording_electrodes(stim_electrode: int) -> list[int]:
    """The electrodes to record around a driven one: every electrode within :data:`DENSE_RADIUS`
    pitches, then :data:`RAYS` rays sampled per :data:`RAY_STEPS`. Off-array points are skipped,
    so a driven electrode near an edge simply has shorter rays that way."""
    col0, row0 = stim_electrode % protocol.COLS, stim_electrode // protocol.COLS
    chosen = set()
    for dy in range(-DENSE_RADIUS, DENSE_RADIUS + 1):
        for dx in range(-DENSE_RADIUS, DENSE_RADIUS + 1):
            if dx * dx + dy * dy <= DENSE_RADIUS * DENSE_RADIUS:
                e = protocol.electrode_at(col0 + dx, row0 + dy)
                if e is not None:
                    chosen.add(e)
    for k in range(RAYS):
        angle = 2 * math.pi * k / RAYS
        start = 1
        for step, upto in RAY_STEPS:
            for r in range(start, upto + 1, step):
                e = protocol.electrode_at(
                    col0 + round(r * math.cos(angle)), row0 + round(r * math.sin(angle))
                )
                if e is not None:
                    chosen.add(e)
            start = upto + step
    chosen.discard(stim_electrode)
    return sorted(chosen)


def plan(
    centre_um,
    amplitudes=DEFAULT_AMPLITUDES,
    pulses: int = 10,
    iti_sec: float = 2.0,
    polarity: str = "anodic-first",
    phase_us: float = 100.0,
) -> dict:
    """What a run will do, as the record written into the recording."""
    stim = protocol.nearest_electrode(*centre_um)
    return {
        "centre_um": [float(centre_um[0]), float(centre_um[1])],
        "stim_electrode": int(stim),
        "rec_electrodes": recording_electrodes(stim),
        "amplitudes_mv": [float(a) for a in amplitudes],
        "pulses": int(pulses),
        "iti_sec": float(iti_sec),
        "polarity": polarity,
        "phase_us": float(phase_us),
        "tokens": {
            f"field_a{a:g}_{protocol.POLARITY_TAG[polarity]}": {
                "amplitude_mv": float(a),
                "polarity": polarity,
            }
            for a in amplitudes
        },
    }


# ---------------------------------------------------------------
#                            THE RUN
# ---------------------------------------------------------------


def run(
    centre_um,
    out_dir: str,
    name: str,
    *,
    well: int = 0,
    amplitudes=DEFAULT_AMPLITUDES,
    pulses: int = 10,
    iti_sec: float = 2.0,
    polarity: str = "anodic-first",
    phase_us: float = 100.0,
    on_progress=print,
    mx=None,
) -> Path:
    """Drive one electrode and record the field around it. Returns the recording's path."""
    from mxtreme.experiments.associative.run import release
    from mxtreme.scans import mx_setup
    from mxtreme.scans.activity_scan import _require_maxlab
    from mxtreme.stimulation import sequences
    from mxtreme.stimulation.timeline import RegionUnits

    mx = mx or _require_maxlab()
    record = plan(centre_um, amplitudes, pulses, iti_sec, polarity, phase_us)
    stim = record["stim_electrode"]
    on_progress(
        f"=== field map === driving e{stim} at {protocol.electrode_xy(stim)} um; recording "
        f"{len(record['rec_electrodes'])} electrodes around it; {len(amplitudes)} amplitudes x {pulses} pulses "
        f"{iti_sec} s apart, {polarity}: about {len(amplitudes) * (pulses * iti_sec + 3) / 60 + 0.3:.1f} min"
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    on_progress("Initializing chip...")
    mx.initialize()
    if mx.send(mx.Core().enable_stimulation_power(True)) != "Ok":
        raise RuntimeError("the system did not enable stimulation power")
    time.sleep(mx.Timing.waitInit)
    mx.clear_events()
    array, units, used = mx_setup.init_well_stim(well, list(record["rec_electrodes"]), [stim])
    if used[0] != stim:
        on_progress(f"driving e{used[0]} instead of e{stim} (stimulation unit clash)")
        record["stim_electrode"] = stim = int(used[0])
    config = array.get_config()
    channels = sorted(config.get_channels())
    record["channels"] = channels
    on_progress(f"routed {len(channels)} channels; unit {units[0]} on e{stim}")

    registered = {}
    for token, spec in record["tokens"].items():
        seq, seconds = sequences.build_region_sequence(
            f"{token}_{well}",
            well,
            [RegionUnits(role="F", drive_dac=0, drive_units=list(units), amplitude_mv=spec["amplitude_mv"])],
            {"F": 0.0},
            pulses,
            1.0 / iti_sec,
            phase_us,
            amplitude_mv=spec["amplitude_mv"],
            polarity=polarity,
        )
        registered[token] = (seq, seconds)
        on_progress(f"  sequence {token}: {seconds:.1f} s")

    mx.activate([well])
    mx.offset()
    time.sleep(mx.Timing.waitInMX2Offset)

    s = mx.Saving()
    s.open_directory(str(out_dir))
    s.start_file(name)
    s.group_delete_all()
    s.group_define(well, f"raw_all_{well}", channels)
    reply = s.write_assay_property(PROTOCOL_KEY, json.dumps(record, separators=(",", ":")))
    plan_copy = Path(out_dir) / f"{name}_plan.json"
    plan_copy.write_text(json.dumps(record, indent=2))
    if str(reply).strip().lower() == "error":
        on_progress(f"WARNING: the plan is not inside the recording; keep {plan_copy}")
    started = time.time()
    s.start_recording([well])
    on_progress("recording; 5 s of quiet first")
    try:
        time.sleep(5.0)
        for token, (seq, seconds) in registered.items():
            on_progress(f"  {token}")
            seq.send()
            time.sleep(seconds + 3.0)
    finally:
        s.stop_recording()
        time.sleep(mx.Timing.waitAfterRecording)
        s.stop_file()
        s.group_delete_all()
        release(mx, well, list(registered))

    written = [p for p in Path(out_dir).glob("*.raw.h5") if p.stat().st_mtime >= started - 1]
    h5 = max(written, key=lambda p: p.stat().st_mtime) if written else Path(out_dir) / f"{name}.raw.h5"
    on_progress(
        f"done: {h5}\nread it back: python -m mxtreme.experiments.fieldmap report {h5} -o {Path(out_dir) / name}.png --csv {Path(out_dir) / name}.csv"
    )
    return h5


# ---------------------------------------------------------------
#                            THE ANALYSIS
# ---------------------------------------------------------------


def _load(h5_path: str, well: int):
    from mxtreme.experiments.associative.report import _load_well

    data = _load_well(h5_path, well)
    with h5py.File(h5_path, "r") as f:
        if f"assay/{PROTOCOL_KEY}" in f:
            raw = f[f"assay/{PROTOCOL_KEY}"][()]
            raw = raw[0] if getattr(raw, "shape", ()) else raw
            record = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
        else:
            side = Path(h5_path).with_name(Path(h5_path).name.split(".raw.h5")[0] + "_plan.json")
            if not side.exists():
                raise ValueError(f"{h5_path} carries no field-map plan and {side} is missing")
            record = json.loads(side.read_text())
    return data, record


def measure(h5_path: str, well: int = 0) -> tuple[list[dict], dict]:
    """Per recording electrode and amplitude: the median peak deflection over the pulses.

    :returns: Rows with ``token, amplitude_mv, polarity, channel, electrode, x_um, y_um,
        distance_um, angle_deg, peak_uv, pulses, saturated`` (the fraction of pulses at the
        amplifier's rail), and the plan.
    """
    data, record = _load(h5_path, well)
    fps, lsb = data["fps"], data["lsb"]
    full_scale_uv = 512 * lsb * 1e6
    stim = int(record["stim_electrode"])
    sx, sy = protocol.electrode_xy(stim)
    elec_of_chan = {int(m["channel"]): int(m["electrode"]) for m in data["mapping"]}
    starts = [
        (f, m["start_stimulation"].rsplit("_", 1)[0]) for f, m in data["events"] if "start_stimulation" in m
    ]
    if not data["segments"]:
        raise ValueError("the recording has no raw traces; the field map needs them")
    seg = data["segments"][0]
    frame_nos, channels = seg["frame_nos"], np.asarray(seg["channels"]).astype(int)
    pre, post = int(0.001 * fps), int(PEAK_WINDOW_MS / 1000 * fps)
    iti = float(record["iti_sec"])
    rows = []
    with h5py.File(seg["raw"][0], "r") as f:
        raw = f[seg["raw"][1]]
        for token, spec in record["tokens"].items():
            frames = [fr for fr, t in starts if t == token]
            pulses = [fr + round(k * iti * fps) for fr in frames for k in range(int(record["pulses"]))]
            peaks, saturated = [], []
            for frame in pulses:
                if frame - pre < frame_nos[0] or frame + post > frame_nos[-1]:
                    continue
                i0 = int(np.searchsorted(frame_nos, frame - pre))
                i1 = int(np.searchsorted(frame_nos, frame + post))
                block = raw[:, i0:i1].astype(float) * lsb * 1e6
                t = (frame_nos[i0:i1] - frame) / fps * 1000.0
                base = np.median(block[:, t < -0.2], axis=1)
                after = block[:, t >= 0] - base[:, None]
                peak = np.max(np.abs(after), axis=1)
                peaks.append(peak)
                saturated.append(peak >= SATURATION * full_scale_uv)
            if not peaks:
                continue
            peaks = np.array(peaks)
            saturated = np.array(saturated)
            for k, ch in enumerate(channels):
                e = elec_of_chan.get(int(ch))
                if e is None or e == stim:
                    continue
                x, y = protocol.electrode_xy(e)
                rows.append(
                    {
                        "token": token,
                        "amplitude_mv": float(spec["amplitude_mv"]),
                        "polarity": spec["polarity"],
                        "channel": int(ch),
                        "electrode": int(e),
                        "x_um": x,
                        "y_um": y,
                        "distance_um": math.hypot(x - sx, y - sy),
                        "angle_deg": math.degrees(math.atan2(y - sy, x - sx)) % 360.0,
                        "peak_uv": float(np.median(peaks[:, k])),
                        "pulses": len(peaks),
                        "saturated": float(saturated[:, k].mean()),
                    }
                )
    return rows, record


def fit_attenuation(rows: list[dict], min_distance_um: float = protocol.PITCH_UM) -> dict:
    """Per amplitude, a power law ``V = a * r^-n`` through the unsaturated points at least one
    pitch away, and the distances at which it falls to a tenth and a hundredth of its value one
    pitch away; then the direction dependence: per log-spaced distance band, the spread of the
    eight directions' medians over their overall median."""
    out = {"by_amplitude": {}, "angle_spread": []}
    for amp in sorted({r["amplitude_mv"] for r in rows}):
        pts = [
            r
            for r in rows
            if r["amplitude_mv"] == amp
            and r["saturated"] == 0
            and r["distance_um"] >= min_distance_um
            and r["peak_uv"] > 0
        ]
        if len(pts) < 5:
            out["by_amplitude"][amp] = None
            continue
        x = np.log10([r["distance_um"] for r in pts])
        y = np.log10([r["peak_uv"] for r in pts])
        slope, intercept = np.polyfit(x, y, 1)
        n = -slope
        v1 = 10 ** (intercept + slope * math.log10(protocol.PITCH_UM))
        out["by_amplitude"][amp] = {
            "exponent": float(n),
            "uv_at_one_pitch": float(v1),
            "r10_um": float(protocol.PITCH_UM * 10 ** (1 / n)) if n > 0 else float("inf"),
            "r100_um": float(protocol.PITCH_UM * 10 ** (2 / n)) if n > 0 else float("inf"),
            "points": len(pts),
            "saturated_channels": sum(1 for r in rows if r["amplitude_mv"] == amp and r["saturated"] > 0),
        }
    # Direction: use the largest amplitude with enough unsaturated points.
    usable = [a for a, f in out["by_amplitude"].items() if f]
    if usable:
        amp = max(usable)
        pts = [
            r
            for r in rows
            if r["amplitude_mv"] == amp and r["saturated"] == 0 and r["distance_um"] >= min_distance_um
        ]
        edges = protocol.PITCH_UM * 2.0 ** np.arange(0, 7)
        for lo, hi in itertools.pairwise(edges):
            band = [r for r in pts if lo <= r["distance_um"] < hi]
            sectors = {}
            for r in band:
                sectors.setdefault(int(r["angle_deg"] // 45), []).append(r["peak_uv"])
            if len(sectors) < 3:
                continue
            medians = np.array([np.median(v) for v in sectors.values()])
            out["angle_spread"].append(
                {
                    "from_um": float(lo),
                    "to_um": float(hi),
                    "sectors": len(sectors),
                    "spread": float((medians.max() - medians.min()) / np.median(medians))
                    if np.median(medians) > 0
                    else float("nan"),
                    "amplitude_mv": amp,
                }
            )
    # Linearity: peak / amplitude on channels unsaturated at every amplitude.
    by_e = {}
    for r in rows:
        by_e.setdefault(r["electrode"], []).append(r)
    ratios = []
    for rs in by_e.values():
        if len(rs) == len(out["by_amplitude"]) and all(r["saturated"] == 0 and r["peak_uv"] > 0 for r in rs):
            ratios.append(
                np.std([r["peak_uv"] / r["amplitude_mv"] for r in rs])
                / np.mean([r["peak_uv"] / r["amplitude_mv"] for r in rs])
            )
    out["linearity_cv"] = float(np.median(ratios)) if ratios else float("nan")
    out["linearity_channels"] = len(ratios)
    return out


def report(h5_path: str, out_png: str | None = None, out_csv: str | None = None, well: int = 0) -> dict:
    rows, record = measure(h5_path, well)
    fits = fit_attenuation(rows)
    stim = record["stim_electrode"]
    print(
        f"=== field map === e{stim} at {protocol.electrode_xy(stim)} um driven; {len({r['electrode'] for r in rows})} electrodes measured; {record['pulses']} pulses per amplitude"
    )
    print(
        f"  {'mV':>6}{'exponent n':>12}{'uV at 1 pitch':>15}{'r(10%) um':>11}{'r(1%) um':>10}{'points':>8}{'saturated':>11}"
    )
    for amp, f in fits["by_amplitude"].items():
        if f is None:
            print(f"  {amp:6g}   too few unsaturated points")
            continue
        print(
            f"  {amp:6g}{f['exponent']:12.2f}{f['uv_at_one_pitch']:15.0f}{f['r10_um']:11.0f}{f['r100_um']:10.0f}{f['points']:8d}{f['saturated_channels']:11d}"
        )
    print(
        "  V = a * r^-n fitted on the unsaturated electrodes at least one pitch (17.5 um) away; r(10%) and r(1%) are where the fit falls to a tenth and a hundredth of its value at one pitch."
    )
    if fits["angle_spread"]:
        print(
            f"  direction, at {fits['angle_spread'][0]['amplitude_mv']:g} mV: (max - min) / median over the 45-degree sectors, per distance band"
        )
        for b in fits["angle_spread"]:
            print(
                f"    {b['from_um']:5.0f}-{b['to_um']:5.0f} um  {b['sectors']} sectors  spread {b['spread']:.0%}"
            )
    if np.isfinite(fits["linearity_cv"]):
        print(
            f"  linearity: peak/amplitude varies by {fits['linearity_cv']:.0%} (median CV) over {fits['linearity_channels']} electrodes unsaturated at every amplitude; a few percent means the medium is linear and the ladder can be read as one curve"
        )
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv}")
    if out_png:
        _draw(rows, fits, record, out_png)
    return {"rows": rows, "fits": fits, "record": record}


def _draw(rows, fits, record, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    amps = sorted({r["amplitude_mv"] for r in rows})
    colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(amps)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax = axes[0, 0]
    for c, amp in zip(colours, amps):
        pts = [r for r in rows if r["amplitude_mv"] == amp]
        ok = [r for r in pts if r["saturated"] == 0]
        sat = [r for r in pts if r["saturated"] > 0]
        ax.plot(
            [r["distance_um"] for r in ok],
            [r["peak_uv"] for r in ok],
            "o",
            ms=3,
            color=c,
            label=f"{amp:g} mV",
        )
        ax.plot([r["distance_um"] for r in sat], [r["peak_uv"] for r in sat], "o", ms=3, mfc="none", color=c)
        f = fits["by_amplitude"].get(amp)
        if f:
            xs = np.logspace(
                math.log10(protocol.PITCH_UM), math.log10(max(r["distance_um"] for r in pts)), 50
            )
            ax.plot(
                xs, f["uv_at_one_pitch"] * (xs / protocol.PITCH_UM) ** -f["exponent"], "-", color=c, lw=0.8
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distance from the driven electrode (um)")
    ax.set_ylabel("peak deflection (uV)")
    ax.set_title(
        "the field against distance, per amplitude (hollow = amplifier saturated; line = power-law fit)",
        fontsize=9,
        loc="left",
    )
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for c, amp in zip(colours, amps):
        ok = [r for r in rows if r["amplitude_mv"] == amp and r["saturated"] == 0]
        ax.plot(
            [r["distance_um"] for r in ok],
            [r["peak_uv"] / amp for r in ok],
            "o",
            ms=3,
            color=c,
            label=f"{amp:g} mV",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distance from the driven electrode (um)")
    ax.set_ylabel("peak deflection / amplitude (uV per mV)")
    ax.set_title(
        "the same, normalised by amplitude: one curve if the medium is linear", fontsize=9, loc="left"
    )
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    if fits["angle_spread"]:
        amp = fits["angle_spread"][0]["amplitude_mv"]
        f = fits["by_amplitude"][amp]
        pts = [
            r
            for r in rows
            if r["amplitude_mv"] == amp and r["saturated"] == 0 and r["distance_um"] >= protocol.PITCH_UM
        ]
        bands = [(b["from_um"], b["to_um"]) for b in fits["angle_spread"]]
        cols = plt.cm.plasma(np.linspace(0.1, 0.9, len(bands)))
        for c, (lo, hi) in zip(cols, bands):
            band = [r for r in pts if lo <= r["distance_um"] < hi]
            expected = [
                f["uv_at_one_pitch"] * (r["distance_um"] / protocol.PITCH_UM) ** -f["exponent"] for r in band
            ]
            ax.plot(
                [r["angle_deg"] for r in band],
                [r["peak_uv"] / e for r, e in zip(band, expected)],
                "o",
                ms=3,
                color=c,
                label=f"{lo:.0f}-{hi:.0f} um",
            )
        ax.axhline(1.0, color="#9ca3af", lw=0.8)
        ax.set_xlabel("direction from the driven electrode (degrees)")
        ax.set_ylabel("peak / the distance fit")
        ax.set_title(
            f"the field against direction at {amp:g} mV, each electrode relative to the fit at its distance: "
            "a round field sits on the line at 1",
            fontsize=9,
            loc="left",
        )
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.axis("off")

    ax = axes[1, 1]
    amp = (
        max(a for a, f in fits["by_amplitude"].items() if f)
        if any(fits["by_amplitude"].values())
        else amps[-1]
    )
    pts = [r for r in rows if r["amplitude_mv"] == amp]
    sc = ax.scatter(
        [r["x_um"] for r in pts],
        [r["y_um"] for r in pts],
        c=np.log10(np.maximum([r["peak_uv"] for r in pts], 1.0)),
        s=12,
        cmap="magma",
        marker="s",
    )
    sx, sy = protocol.electrode_xy(record["stim_electrode"])
    ax.plot([sx], [sy], "c+", ms=12, mew=2)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(
        f"map at {amp:g} mV: log10 peak deflection (uV) on every recorded electrode; + is the driven one",
        fontsize=9,
        loc="left",
    )
    fig.colorbar(sc, ax=ax, shrink=0.7, label="log10 uV")
    fig.suptitle(
        f"field around e{record['stim_electrode']}, {record['polarity']}, {record['pulses']} pulses per amplitude",
        fontsize=10,
    )
    fig.savefig(out_png, dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


# ---------------------------------------------------------------
#                            THE CLI
# ---------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mxtreme.experiments.fieldmap", description=__doc__.split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="drive one electrode on the rig and record the field around it")
    r.add_argument("--centre", required=True, help="x,y in um of the electrode to drive")
    r.add_argument("--out", required=True, help="directory to record into (not the managed store)")
    r.add_argument("--name", required=True, help="the recording's file name, e.g. fieldmap_P003983")
    r.add_argument("--well", type=int, default=0)
    r.add_argument(
        "--amplitudes",
        default=",".join(f"{a:g}" for a in DEFAULT_AMPLITUDES),
        help="mV per phase, comma separated",
    )
    r.add_argument("--pulses", type=int, default=10)
    r.add_argument("--iti", type=float, default=2.0, help="seconds between pulses")
    r.add_argument("--polarity", default="anodic-first", choices=list(protocol.POLARITY_TAG))
    p = sub.add_parser("report", help="the field against distance, amplitude and direction")
    p.add_argument("h5")
    p.add_argument("-o", "--out", help="figure path")
    p.add_argument("--csv", help="one row per electrode and amplitude")
    p.add_argument("--well", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "run":
        x, y = (float(v) for v in args.centre.split(","))
        run(
            (x, y),
            args.out,
            args.name,
            well=args.well,
            amplitudes=[float(a) for a in args.amplitudes.split(",")],
            pulses=args.pulses,
            iti_sec=args.iti,
            polarity=args.polarity,
        )
    else:
        report(args.h5, args.out, args.csv, well=args.well)


if __name__ == "__main__":
    sys.exit(main())
