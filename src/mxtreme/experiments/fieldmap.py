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
#: Pulses fired and thrown away before the measured ones begin. The first stimulation of a
#: recording is a start-up transient: on the first field map it reached every routed channel on
#: the array and was fifteen times the size of the pulse that followed it. Shuffling alone only
#: moves that pulse into a random amplitude; discarding it removes it.
WARMUP_PULSES = 2
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
#: A window of the same length, ending this long before the pulse, gives each channel its own
#: noise floor: the largest deflection it shows when nothing is being driven.
NOISE_OFFSET_MS = 1.0
#: A peak has to beat the channel's noise floor by this much to be a measurement of the field
#: rather than of the amplifier. Below it the far field flattens, which reads as a smaller
#: exponent and as the medium being non-linear; both are artifacts.
NOISE_FACTOR = 3.0


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
    seed: int = 1,
    warmup: int = WARMUP_PULSES,
) -> dict:
    """What a run will do, as the record written into the recording.

    The amplitudes are **interleaved**: each repeat fires one pulse of every amplitude, in an
    order shuffled afresh, rather than running each amplitude as its own block. A block design
    confounds amplitude with time perfectly, and the first field map on this rig was run that
    way: its lowest amplitude fired first and came out 2.65x too large per mV, which no amount of
    DAC rounding can produce, so whether that was the amplitude or the moment cannot be told from
    those data. Calibration shuffles for the same reason (Ronchi et al. 2019).
    """
    import random

    stim = protocol.nearest_electrode(*centre_um)
    tokens = [f"field_a{a:g}_{protocol.POLARITY_TAG[polarity]}" for a in amplitudes]
    rng = random.Random(seed)
    order = []
    for _ in range(int(pulses)):
        repeat = list(tokens)
        rng.shuffle(repeat)
        order += repeat
    return {
        "order": order,
        "seed": int(seed),
        # Fired first and left out of the analysis; not in "tokens", so nothing reads it.
        "warmup_pulses": int(warmup),
        "warmup_token": f"warmup_{protocol.POLARITY_TAG[polarity]}",
        "warmup_amplitude_mv": float(max(amplitudes)) if len(amplitudes) else 0.0,
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
    seed: int = 1,
    warmup: int = WARMUP_PULSES,
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
    record = plan(centre_um, amplitudes, pulses, iti_sec, polarity, phase_us, seed, warmup)
    stim = record["stim_electrode"]
    on_progress(
        f"=== field map === driving e{stim} at {protocol.electrode_xy(stim)} um; recording "
        f"{len(record['rec_electrodes'])} electrodes around it; {len(amplitudes)} amplitudes x {pulses} pulses "
        f"{iti_sec} s apart, interleaved and shuffled, {polarity}, after "
        f"{record['warmup_pulses']} discarded warm-up pulse(s): about "
        f"{(len(amplitudes) * pulses + record['warmup_pulses']) * iti_sec / 60 + 0.3:.1f} min"
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

    # One single-pulse sequence per amplitude, fired repeatedly in the plan's shuffled order, so
    # each amplitude's pulses are spread over the whole run instead of clumped at one moment.
    registered = {}
    for token, spec in record["tokens"].items():
        seq, seconds = sequences.build_region_sequence(
            f"{token}_{well}",
            well,
            [RegionUnits(role="F", drive_dac=0, drive_units=list(units), amplitude_mv=spec["amplitude_mv"])],
            {"F": 0.0},
            1,
            1.0 / iti_sec,
            phase_us,
            amplitude_mv=spec["amplitude_mv"],
            polarity=polarity,
        )
        registered[token] = (seq, seconds)
        on_progress(f"  sequence {token}: one pulse, {seconds:.3f} s")
    warm_seq = None
    if record["warmup_pulses"]:
        warm_seq, _ = sequences.build_region_sequence(
            f"{record['warmup_token']}_{well}",
            well,
            [
                RegionUnits(
                    role="F",
                    drive_dac=0,
                    drive_units=list(units),
                    amplitude_mv=record["warmup_amplitude_mv"],
                )
            ],
            {"F": 0.0},
            1,
            1.0 / iti_sec,
            phase_us,
            amplitude_mv=record["warmup_amplitude_mv"],
            polarity=polarity,
        )

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
        if warm_seq is not None:
            on_progress(
                f"  {record['warmup_pulses']} warm-up pulse(s) at "
                f"{record['warmup_amplitude_mv']:g} mV, thrown away: the first stimulation of a "
                "recording is a transient"
            )
            for _ in range(record["warmup_pulses"]):
                warm_seq.send()
                time.sleep(iti_sec)
        for i, token in enumerate(record["order"]):
            seq, seconds = registered[token]
            seq.send()
            if (i + 1) % len(registered) == 0:
                on_progress(f"  repeat {(i + 1) // len(registered)} of {record['pulses']}")
            time.sleep(max(0.0, iti_sec - seconds))
    finally:
        s.stop_recording()
        time.sleep(mx.Timing.waitAfterRecording)
        s.stop_file()
        s.group_delete_all()
        release(
            mx,
            well,
            list(registered) + ([record["warmup_token"]] if warm_seq is not None else []),
        )

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
    """Per recording electrode and amplitude: the peak deflection of the pulse-averaged trace.

    The artifact is time-locked to the pulse and the noise is not, so the traces are combined
    over the pulses *before* the peak is taken. Taking a peak per pulse and then a median would
    keep each pulse's noise: a maximum of ``|signal + noise|`` is biased upward by the noise,
    which inflates the far field, flattens the curve and reads as a smaller exponent at small
    amplitudes. Combining ``n`` pulses cuts the noise by about ``sqrt(n)`` and the bias with it.

    The traces are combined by taking the **median at each sample**, not the mean. On the first
    real field map the very first pulse of the recording was an order of magnitude larger than
    the nine that followed it -- a start-up transient, on every channel at once -- and a mean
    carried it into the result, inflating that block by 2.65x. A median rejects it. The count of
    pulses that disagree with their block is reported.

    Each pulse is aligned on the artifact's own peak on a reference channel (the electrode
    nearest the driven one), so a sample of clock drift between the stimulation and sampling
    clocks cannot smear a 100 us phase across the average.

    :returns: Rows with ``token, amplitude_mv, polarity, channel, electrode, x_um, y_um,
        distance_um, angle_deg, peak_uv``, ``noise_uv`` (the same statistic over a window of the
        same length before the pulse, so the noise floor of the combined trace), ``pulses``,
        ``odd_pulses`` (how many disagreed with their own block) and ``saturated`` (the fraction
        of individual pulses at the amplifier's rail), and the plan.
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
    post = int(PEAK_WINDOW_MS / 1000 * fps)
    pre = int((NOISE_OFFSET_MS + PEAK_WINDOW_MS + 0.5) / 1000 * fps)
    slack = int(0.0005 * fps)  # room to realign each pulse on the artifact
    # The channel nearest the driven electrode carries the largest artifact: it is what every
    # pulse is aligned on, so one clock's drift cannot smear the average.
    ref = min(
        (k for k, ch in enumerate(channels) if elec_of_chan.get(int(ch)) is not None),
        key=lambda k: math.hypot(
            *np.subtract(protocol.electrode_xy(elec_of_chan[int(channels[k])]), (sx, sy))
        ),
    )
    iti = float(record["iti_sec"])
    rows = []
    with h5py.File(seg["raw"][0], "r") as f:
        raw = f[seg["raw"][1]]
        for token, spec in record["tokens"].items():
            frames = [fr for fr, t in starts if t == token]
            pulses = [fr + round(k * iti * fps) for fr in frames for k in range(int(record["pulses"]))]
            windows, saturated, used = [], None, 0
            for frame in pulses:
                if frame - pre - slack < frame_nos[0] or frame + post + slack > frame_nos[-1]:
                    continue
                i0 = int(np.searchsorted(frame_nos, frame - pre - slack))
                i1 = int(np.searchsorted(frame_nos, frame + post + slack))
                block = raw[:, i0:i1].astype(float) * lsb * 1e6
                t = (frame_nos[i0:i1] - frame) / fps * 1000.0
                base = np.median(block[:, t < -0.2], axis=1)
                block -= base[:, None]
                # Realign: the artifact's own peak on the reference channel is this pulse's zero.
                near = np.flatnonzero((t >= -0.2) & (t <= 0.5))
                shift = int(near[np.argmax(np.abs(block[ref, near]))]) - int(np.searchsorted(t, 0.0))
                lo = int(np.searchsorted(t, 0.0)) + shift
                window = block[:, lo - pre : lo + post]
                quiet = block[:, lo - pre : lo - pre + post]
                if window.shape[1] < post or quiet.shape[1] < post:
                    continue
                windows.append(window)
                this_peak = np.max(np.abs(window[:, pre:]), axis=1)
                saturated = (
                    (this_peak >= SATURATION * full_scale_uv).astype(float)
                    if saturated is None
                    else saturated + (this_peak >= SATURATION * full_scale_uv)
                )
                used += 1
            if not used:
                continue
            stack = np.stack(windows)
            # Median at each sample: the noise still falls with the pulse count, and a single
            # rogue pulse cannot carry the block.
            combined = np.median(stack, axis=0)
            peak = np.max(np.abs(combined[:, pre:]), axis=1)
            floor = np.max(np.abs(combined[:, :post]), axis=1)
            # How many pulses disagree with their own block, judged on the channels that carry a
            # real artifact, so a quiet far channel's noise cannot make a pulse look rogue.
            per_pulse = np.max(np.abs(stack[:, :, pre:]), axis=2)
            strong = peak >= 5 * max(float(np.median(floor)), 1e-9)
            odd = 0
            if strong.sum() >= 3 and used >= 4:
                scale = np.median(per_pulse[:, strong], axis=1)
                middle = float(np.median(scale))
                mad = float(np.median(np.abs(scale - middle)))
                # Both tests have to agree: five median-absolute-deviations catches an outlier in
                # a scattered block, and a quarter of the median stops a tight block, where the
                # deviation is near zero, from flagging its own ordinary spread.
                odd = int(np.sum(np.abs(scale - middle) > max(5 * mad, 0.25 * middle)))
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
                        "peak_uv": float(peak[k]),
                        "noise_uv": float(floor[k]),
                        "pulses": used,
                        "odd_pulses": odd,
                        "saturated": float(saturated[k] / used),
                    }
                )
    return rows, record


def usable(row, min_distance_um: float = protocol.PITCH_UM) -> bool:
    """Whether a row measures the field: far enough out that the electrode's own geometry is not
    the story, not at the amplifier's rail, and clear of the channel's own noise floor."""
    return (
        row["distance_um"] >= min_distance_um
        and row["saturated"] == 0
        and row["peak_uv"] > 0
        and row["peak_uv"] >= NOISE_FACTOR * max(row.get("noise_uv", 0.0), 0.0)
    )


def fit_attenuation(rows: list[dict], min_distance_um: float = protocol.PITCH_UM) -> dict:
    """Per amplitude, a power law ``V = a * r^-n`` through the usable points (see :func:`usable`),
    and the distances at which it falls to a tenth and a hundredth of its value one pitch away;
    then the direction dependence: per log-spaced distance band, the spread of the eight
    directions' medians over their overall median.

    Points at the channel's noise floor are excluded rather than fitted. Included, they flatten
    the far end of every curve, which reads as a smaller exponent at small amplitudes and as the
    medium being non-linear; both are artifacts of the floor, not of the medium.
    """
    out = {
        "by_amplitude": {},
        "angle_spread": [],
        "noise_uv": float(np.median([r.get("noise_uv", 0.0) for r in rows])) if rows else float("nan"),
    }
    for amp in sorted({r["amplitude_mv"] for r in rows}):
        mine = [r for r in rows if r["amplitude_mv"] == amp]
        pts = [r for r in mine if usable(r, min_distance_um)]
        reach = max((r["distance_um"] for r in pts), default=0.0)
        if len(pts) < 5:
            out["by_amplitude"][amp] = None
            continue
        x = np.log10([r["distance_um"] for r in pts])
        y = np.log10([r["peak_uv"] for r in pts])
        slope, intercept = np.polyfit(x, y, 1)
        n = -slope
        v1 = 10 ** (intercept + slope * math.log10(protocol.PITCH_UM))
        r10 = protocol.PITCH_UM * 10 ** (1 / n) if n > 0 else float("inf")
        r100 = protocol.PITCH_UM * 10 ** (2 / n) if n > 0 else float("inf")
        out["by_amplitude"][amp] = {
            "exponent": float(n),
            "uv_at_one_pitch": float(v1),
            "r10_um": float(r10),
            "r100_um": float(r100),
            "points": len(pts),
            "measured_to_um": float(reach),
            "r10_measured": bool(r10 <= reach),
            "r100_measured": bool(r100 <= reach),
            "saturated_channels": sum(1 for r in mine if r["saturated"] > 0),
            "at_noise_channels": sum(
                1
                for r in mine
                if r["saturated"] == 0
                and r["distance_um"] >= min_distance_um
                and not usable(r, min_distance_um)
            ),
        }
    # Direction: use the largest amplitude, which reaches furthest before the noise floor.
    usable_amps = [a for a, f in out["by_amplitude"].items() if f]
    if usable_amps:
        amp = max(usable_amps)
        pts = [r for r in rows if r["amplitude_mv"] == amp and usable(r, min_distance_um)]
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
    # Linearity: peak / amplitude on electrodes usable at every amplitude.
    by_e = {}
    for r in rows:
        by_e.setdefault(r["electrode"], []).append(r)
    ratios = []
    for rs in by_e.values():
        if len(rs) == len(out["by_amplitude"]) and all(usable(r, min_distance_um) for r in rs):
            per_mv = [r["peak_uv"] / r["amplitude_mv"] for r in rs]
            ratios.append(np.std(per_mv) / np.mean(per_mv))
    out["linearity_cv"] = float(np.median(ratios)) if ratios else float("nan")
    out["linearity_channels"] = len(ratios)
    return out


#: An amplitude whose field per mV differs from the others by more than this is left out of the
#: model and named, rather than averaged in.
AGREEMENT = 0.3


def model(fits: dict) -> dict | None:
    """One field model from the amplitudes that agree with each other: the medium is linear, so
    every amplitude should give the same field per mV, and the ones that do are the measurement.

    The scale is the median of ``uv_at_one_pitch / amplitude`` and the exponent the median of the
    exponents, over the amplitudes within :data:`AGREEMENT` of that median scale. Any amplitude
    outside it is listed in ``disagreeing`` and excluded: a pulse that did not come out at the
    amplitude it was asked for would otherwise drag the model toward itself.

    :returns: ``{"exponent", "uv_per_mv_at_one_pitch", "pitch_um", "measured_to_um",
        "amplitudes_mv", "disagreeing", "linearity_cv"}``, or ``None`` if nothing was usable.
    """
    good = {a: f for a, f in fits["by_amplitude"].items() if f}
    if not good:
        return None
    per_mv = {a: f["uv_at_one_pitch"] / a for a, f in good.items()}
    scale = float(np.median(list(per_mv.values())))
    agree = {a: f for a, f in good.items() if abs(per_mv[a] - scale) <= AGREEMENT * scale}
    if not agree:
        agree = good
    return {
        "exponent": float(np.median([f["exponent"] for f in agree.values()])),
        "uv_per_mv_at_one_pitch": float(np.median([per_mv[a] for a in agree])),
        "pitch_um": float(protocol.PITCH_UM),
        "measured_to_um": float(max(f["measured_to_um"] for f in agree.values())),
        "amplitudes_mv": sorted(agree),
        "disagreeing": {a: round(per_mv[a], 2) for a in good if a not in agree},
        "linearity_cv": fits["linearity_cv"],
    }


def field_uv(model_: dict, distance_um, amplitude_mv: float):
    """The model's field at a distance (or an array of them), in uV, for one amplitude."""
    d = np.maximum(np.asarray(distance_um, dtype=float), model_["pitch_um"])
    return model_["uv_per_mv_at_one_pitch"] * amplitude_mv * (d / model_["pitch_um"]) ** -model_["exponent"]


def load_model(path) -> dict:
    """A field model as :func:`report` ``--save-model`` wrote it."""
    model_ = json.loads(Path(path).read_text())
    missing = {"exponent", "uv_per_mv_at_one_pitch", "pitch_um"} - set(model_)
    if missing:
        raise ValueError(f"{path} is not a field model: it has no {sorted(missing)}")
    return model_


def site_field_uv(model_: dict, electrodes, amplitude_mv: float, points_xy) -> np.ndarray:
    """The field at each point from every driven electrode of a site together.

    The electrodes of one site are driven with the same pulse at the same instant, and the medium
    is linear (the field map measures that), so their fields add. The middle of a spread-out site
    is therefore raised by all of its electrodes at once, which is the point of spreading them.

    :param points_xy: (N, 2) positions in um.
    :returns: (N,) field in uV.
    """
    xy = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    total = np.zeros(len(xy))
    for e in electrodes:
        ex, ey = protocol.electrode_xy(e)
        total += field_uv(model_, np.hypot(xy[:, 0] - ex, xy[:, 1] - ey), amplitude_mv)
    return total


def field_at_um(model_: dict, electrodes, amplitude_mv: float, point_xy) -> float:
    """The site's field at one point, in uV."""
    return float(site_field_uv(model_, electrodes, amplitude_mv, [point_xy])[0])


#: The field level worth quoting as a round number when comparing sites, uV. A round number, deliberately:
#: it is a line of equal field, not a threshold. What field a neuron needs is not measured by a
#: saline field map, and calibration does not measure it either -- what calibration finds is the
#: lowest amplitude at which a whole site evokes a countable response over a 150 um disc.
CONTOUR_UV = 100.0


def activating_falloff(model_: dict) -> float:
    """The exponent with which the *activating function* falls off, given the potential's.

    A neuron is not driven by the extracellular potential itself but, to first order, by its
    second derivative along the axon: the activating function
    ``f = (d / 4 rho_i c_m) * d2Ve/dx2`` (Rattay 1986, *IEEE TBME* 33(10) 974-977; Rattay 1989,
    *IEEE TBME* 36(7) 676-682). Each derivative costs a power of distance, so a potential falling
    as ``r^-n`` gives an activating function falling as ``r^-(n+2)``. That is how a stimulation
    artifact can cover the whole array while the stimulation itself stays local: at ten times the
    distance the potential is down 6-fold but the activating function is down 600-fold.

    **This is the mid-axon case only.** At axon terminals and bends the activating function
    reduces to a term in the *first* derivative instead (Rattay 1999, *Neuroscience* 89(2)
    335-346), so ``r^-(n+1)``. A dissociated culture is mostly short, branching, terminal-rich
    neurites, so that regime may matter here at least as much. For comparison, the empirical
    current-distance law measured in vivo is ``I_threshold ~ r^2`` (Stoney, Thompson & Asanuma
    1968, *J Neurophysiol* 31(5) 659-669), an effective ``r^-2``, which sits between the two.

    Whichever holds, a quantity falling as ``r^-p`` puts the excited radius at ``A^(1/p)``: the
    radius grows far more slowly than the amplitude.

    This is a scaling argument, not a measurement. It assumes a straight fibre in a uniform
    medium and says nothing about what a real neuron on this array needs.
    """
    return float(model_["exponent"]) + 2.0


def contour_um(
    model_: dict,
    electrodes,
    amplitude_mv: float,
    level_uv: float = CONTOUR_UV,
    directions: int = 16,
    out_to_um: float = 3000.0,
) -> dict:
    """How far from a site's centre its field is still above ``level_uv``, along ``directions``
    rays.

    A line of equal field, nothing more. Because the medium is linear, this contour moves with
    the amplitude in a way that a contour defined as a *fraction* of the site's own field would
    not: at twice the amplitude the field is twice as large everywhere, so any given level is
    crossed further out.

    :returns: ``{"min_um", "median_um", "max_um", "level_uv", "beyond_range"}``.
    """
    xy = np.array([protocol.electrode_xy(e) for e in electrodes], dtype=float)
    centre = xy.mean(axis=0)
    radii = np.arange(0.0, out_to_um, 2.0)
    crossings, beyond = [], False
    for k in range(directions):
        angle = 2 * math.pi * k / directions
        points = centre + np.outer(radii, [math.cos(angle), math.sin(angle)])
        field = site_field_uv(model_, electrodes, amplitude_mv, points)
        past = np.flatnonzero(field < level_uv)
        if len(past):
            crossings.append(float(radii[past[0]]))
        else:
            crossings.append(float(radii[-1]))
            beyond = True
    return {
        "min_um": float(np.min(crossings)),
        "median_um": float(np.median(crossings)),
        "max_um": float(np.max(crossings)),
        "level_uv": float(level_uv),
        "beyond_range": beyond,
    }


def report(
    h5_path: str,
    out_png: str | None = None,
    out_csv: str | None = None,
    well: int = 0,
    save_model: str | None = None,
) -> dict:
    rows, record = measure(h5_path, well)
    fits = fit_attenuation(rows)
    field = model(fits)
    stim = record["stim_electrode"]
    print(
        f"=== field map === e{stim} at {protocol.electrode_xy(stim)} um driven; {len({r['electrode'] for r in rows})} electrodes measured; {record['pulses']} pulses per amplitude"
    )
    print(
        f"  {'mV':>6}{'exponent n':>12}{'uV at 1 pitch':>15}{'r(10%) um':>12}{'r(1%) um':>12}"
        f"{'points':>8}{'at noise':>10}{'saturated':>11}"
    )
    for amp, f in fits["by_amplitude"].items():
        if f is None:
            print(f"  {amp:6g}   too few usable points")
            continue
        r10 = f"{f['r10_um']:.0f}" + ("" if f["r10_measured"] else "*")
        r100 = f"{f['r100_um']:.0f}" + ("" if f["r100_measured"] else "*")
        print(
            f"  {amp:6g}{f['exponent']:12.2f}{f['uv_at_one_pitch']:15.0f}{r10:>12}{r100:>12}"
            f"{f['points']:8d}{f['at_noise_channels']:10d}{f['saturated_channels']:11d}"
        )
    odd = {
        a: max((r.get("odd_pulses", 0) for r in rows if r["amplitude_mv"] == a), default=0)
        for a in fits["by_amplitude"]
    }
    if any(odd.values()):
        print(
            "  pulses out of line with their own block, rejected by the trace median: "
            + ", ".join(f"{a:g} mV: {v}" for a, v in odd.items() if v)
            + ". The first pulse of a recording is the usual one: it follows the offset "
            "compensation and can be many times the size of the rest."
        )
    print(
        f"  V = a * r^-n, fitted on electrodes at least one pitch (17.5 um) away that are not at the "
        f"amplifier's rail and clear the channel's own noise floor (median {fits['noise_uv']:.1f} uV) by "
        f"{NOISE_FACTOR:g}x.\n  r(10%) and r(1%) are where the fit falls to a tenth and a hundredth of "
        f"its value at one pitch; * marks a value beyond the furthest usable electrode, so extrapolated."
    )
    if fits["angle_spread"]:
        print(
            f"  direction, at {fits['angle_spread'][0]['amplitude_mv']:g} mV: (max - min) / median over "
            f"the 45-degree sectors, per distance band"
        )
        for b in fits["angle_spread"]:
            print(
                f"    {b['from_um']:5.0f}-{b['to_um']:5.0f} um  {b['sectors']} sectors  spread {b['spread']:.0%}"
            )
    if np.isfinite(fits["linearity_cv"]):
        print(
            f"  linearity: peak/amplitude varies by {fits['linearity_cv']:.0%} (median CV) over "
            f"{fits['linearity_channels']} electrodes usable at every amplitude; a few percent means the "
            f"medium is linear and the ladder is one curve"
        )
    if field:
        print(
            f"  field model: {field['uv_per_mv_at_one_pitch']:.1f} uV per mV at one pitch, falling as "
            f"r^-{field['exponent']:.2f}, measured out to {field['measured_to_um']:.0f} um, from "
            + ", ".join(f"{a:g}" for a in field["amplitudes_mv"])
            + " mV"
        )
        if field["disagreeing"]:
            print(
                "  WARN these amplitudes disagree with the rest on the field per mV and are left out: "
                + ", ".join(
                    f"{a:g} mV gave {v:.1f} uV/mV against {field['uv_per_mv_at_one_pitch']:.1f}"
                    for a, v in field["disagreeing"].items()
                )
                + ".\n  The medium is linear, so an amplitude out of line means the pulse was not the size "
                "it was asked for; check the recording's dac_steps against its amp_mV."
            )
        if save_model:
            payload = dict(field, source=Path(h5_path).name, stim_electrode=record["stim_electrode"])
            Path(save_model).write_text(json.dumps(payload, indent=2) + "\n")
            print(f"wrote {save_model}; point a parameter file at it with --set field_model={save_model}")
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.30, wspace=0.22, top=0.91, bottom=0.07, left=0.07, right=0.97)

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
    ax.set_title("the field against distance, per amplitude", fontsize=10, loc="left")
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
    ax.set_title("the same, normalised by amplitude", fontsize=10, loc="left")
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
        ax.set_title(f"the field against direction at {amp:g} mV", fontsize=10, loc="left")
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
    ax.set_title(f"map at {amp:g} mV", fontsize=10, loc="left")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="log10 uV")
    fig.suptitle(
        f"field around e{record['stim_electrode']}, {record['polarity']}, "
        f"{record['pulses']} pulses per amplitude, averaged before the peak is taken",
        fontsize=11,
        y=0.965,
    )
    fig.savefig(out_png, dpi=110, facecolor="white")
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
    r.add_argument("--seed", type=int, default=1, help="which shuffle of the interleaved order")
    r.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_PULSES,
        help="pulses fired and discarded before the measured ones",
    )
    p = sub.add_parser("report", help="the field against distance, amplitude and direction")
    p.add_argument("h5")
    p.add_argument("-o", "--out", help="figure path")
    p.add_argument("--csv", help="one row per electrode and amplitude")
    p.add_argument("--save-model", help="write the fitted field model here, for preview to draw the reach")
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
            seed=args.seed,
            warmup=args.warmup,
        )
    else:
        report(args.h5, args.out, args.csv, well=args.well, save_model=args.save_model)


if __name__ == "__main__":
    sys.exit(main())
