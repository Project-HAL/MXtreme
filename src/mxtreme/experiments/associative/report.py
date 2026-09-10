"""Reading a recording back against what was meant to happen.

From the ``.h5`` and the ``_protocol.json`` beside it:

- which presentations fired, against the schedule, and the intervals between them;
- per presentation, spikes in every region in every response window, and the pulse-locked count
  (spikes in ``pulse_window_ms`` after each pulse, summed over the train) -- the readout: the
  experiment worked if "CS alone" rises in US after encoding and "NS alone" does not;
- the coupling between the regions before and after training, from the pre and post blocks;
- with ``out_csv``, every per-presentation number as a CSV, so the printed summaries can be
  checked against the raw counts independently;
- on a saline dry run, the first deflection on a driven and a return electrode of each region
  (sign, size and timing of the artifact) and spikes in the artifact window against the readout
  window. Raw traces need MaxWell's HDF5 compression filter, which MaxLab installs on the rig.
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np

from mxtreme.experiments.associative import display, protocol


def _load_well(h5_path, well):
    with h5py.File(h5_path, "r") as f:
        rec = f[f"/recordings/rec0000/well{well:03d}"]
        fps = float(np.asarray(rec["settings/sampling"][:]).ravel()[0])
        lsb = float(np.asarray(rec["settings/lsb"][:]).ravel()[0])
        mapping = rec["settings/mapping"][:]
        spikes = rec["spikes"][:]
        events = []
        for row in rec["events"][:]:
            raw = row["eventmessage"]
            try:
                message = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
            except (ValueError, AttributeError):
                message = {}
            events.append((int(row["frameno"]), message))
        group = next(iter(rec["groups"].values()))
        return {
            "fps": fps,
            "lsb": lsb,
            "mapping": mapping,
            "spikes": spikes,
            "events": events,
            "raw": (h5_path, group["raw"].name),
            "frame_nos": group["frame_nos"][:],
            "raw_channels": group["channels"][:],
        }


_raw_unreadable = [False]


def _trace(data, channel, frame, before, after):
    if _raw_unreadable[0]:
        return None, None
    frame_nos = data["frame_nos"]
    i0, i1 = int(np.searchsorted(frame_nos, frame - before)), int(np.searchsorted(frame_nos, frame + after))
    if i1 <= i0:
        return None, None
    row = (
        int(np.flatnonzero(data["raw_channels"] == channel)[0])
        if channel in data["raw_channels"]
        else channel
    )
    try:
        with h5py.File(data["raw"][0], "r") as f:
            v = f[data["raw"][1]][row, i0:i1].astype(float)
    except OSError:
        _raw_unreadable[0] = True
        print(
            "  (raw traces are not readable here: MaxWell's HDF5 compression filter is missing; run this on the rig for the artifact check)"
        )
        return None, None
    return (frame_nos[i0:i1] - frame) / data["fps"] * 1000.0, (v - np.median(v)) * data["lsb"] * 1e6


def readout(data, record, well) -> list[dict]:
    """One row per presentation fired: block, token, frame, and per region the count in each
    response window and the pulse-locked count."""
    fps = data["fps"]
    spikes = data["spikes"]
    frames = spikes["frameno"].astype(np.int64)
    elec_of_chan = {int(m["channel"]): int(m["electrode"]) for m in data["mapping"]}
    spike_elec = np.array([elec_of_chan.get(int(c), -1) for c in spikes["channel"]])
    members = {
        role: np.isin(spike_elec, np.asarray(r["rec_electrodes"])) for role, r in record["regions"].items()
    }
    windows = {
        k: (int(a / 1000 * fps), int(b / 1000 * fps)) for k, (a, b) in record["response_windows_ms"].items()
    }
    pw = record["params"]["pulse_window_ms"]
    pulse = (int(pw[0] / 1000 * fps), int(pw[1] / 1000 * fps))

    def count(member, a, b):
        return int(np.count_nonzero(member & (frames >= a) & (frames < b)))

    block_of = {}
    current = None
    for frame, m in data["events"]:
        for key in m:
            if key.endswith("_start") or key == "end_experiment":
                current = key.replace("_start", "")
        if "start_stimulation" in m:
            block_of[frame] = current
    rows = []
    for index, (frame, m) in enumerate((f, m) for f, m in data["events"] if "start_stimulation" in m):
        token = m["start_stimulation"].rsplit("_", 1)[0]
        stim = record["stimuli"].get(token)
        row = {"index": index, "block": block_of.get(frame), "token": token, "frame": frame, "sec": None}
        for role, member in members.items():
            for label, (a, b) in windows.items():
                row[f"{role}_{label}"] = count(member, frame + a, frame + b)
            total = 0
            for offset in stim["offsets_sec"] if stim else [0.0]:
                onset = frame + round(offset * fps)
                total += count(member, onset + pulse[0], onset + pulse[1])
            row[f"{role}_pulse"] = total
        rows.append(row)
    return rows


def _pulses_and_regions(data, record):
    """Every pulse delivered, as ``(token, frame)``, and the regions' recording electrodes."""
    fps = data["fps"]
    pulses = []
    for frame, message in data["events"]:
        if "start_stimulation" not in message:
            continue
        token = message["start_stimulation"].rsplit("_", 1)[0]
        stim = record["stimuli"].get(token)
        for offset in stim["offsets_sec"] if stim else [0.0]:
            pulses.append((token, frame + round(offset * fps)))
    regions = {role: r["rec_electrodes"] for role, r in record["regions"].items()}
    return pulses, regions


def _gate(data, record) -> list:
    """The calibration or connectivity analysis for a run in one of those modes.

    Prints the table the verdicts are drawn from, then returns the verdicts themselves; see
    :mod:`mxtreme.experiments.associative.checks`.
    """
    from mxtreme.experiments.associative import checks
    from mxtreme.scans import region_selection

    params = record["params"]
    fps = data["fps"]
    spikes = data["spikes"]
    elec_of_chan = {int(m["channel"]): int(m["electrode"]) for m in data["mapping"]}
    frames = spikes["frameno"].astype(np.int64)
    elecs = np.array([elec_of_chan.get(int(c), -1) for c in spikes["channel"]])
    pulses, regions = _pulses_and_regions(data, record)
    if not pulses:
        return [checks.Verdict("stop", "no stimulation events in this recording.")]

    silent = checks.silent_recording(frames, fps)
    if silent is not None:
        return [silent]

    window = tuple(params["pulse_window_ms"])
    bursts = region_selection.network_bursts(frames, fps)

    if params["mode"] == "connectivity":
        # Grouped by the region stimulated rather than by token.
        by_role = [(protocol.roles_in(t)[0], f) for t, f in pulses if protocol.roles_in(t)]
        responses = checks.pulse_responses(frames, elecs, regions, fps, by_role, window)
        rates = checks.burst_rate(by_role, bursts, fps, within_ms=500.0)
        names, evoked, ratio = checks.crosstalk(responses)
        print(
            "\nevoked spikes per pulse (row stimulated, column measured), above the matching "
            "window before each pulse:"
        )
        for name, row in zip(names, evoked):
            print(f"  {name:<4}" + "".join(f"{v:8.2f}" for v in row))
        print("as a fraction of the stimulated region's own response:")
        for name, row in zip(names, ratio):
            print(f"  {name:<4}" + "".join(("     nan" if np.isnan(v) else f"{v:8.0%}") for v in row))
        print("network bursts started: " + ", ".join(f"{k} {v:.0%}" for k, v in rates.items()))
        return checks.connectivity_verdicts(
            names,
            evoked,
            ratio,
            rates,
            crosstalk_warn=params.get("crosstalk_warn", 0.3),
            burst_warn=params.get("burst_warn", 0.2),
        )

    responses = checks.pulse_responses(frames, elecs, regions, fps, pulses, window)
    rates = checks.burst_rate(pulses, bursts, fps, within_ms=500.0)
    tokens = {}
    for token, stim in record["stimuli"].items():
        roles = protocol.roles_in(token)
        if roles:
            tokens[token] = (roles[0], stim.get("amplitude_mv") or 0.0, stim.get("polarity", ""))
    rows = checks.calibration_table(responses, rates, tokens)
    print("\ncalibration, per region and amplitude:")
    print(f"  {'role':<5}{'mV':>7}{'polarity':>16}{'spikes/pulse':>14}{'bursts':>9}{'pulses':>8}")
    for r in rows:
        print(
            f"  {r['role']:<5}{r['amplitude_mv']:7.0f}{r['polarity']:>16}{r['local']:14.2f}"
            f"{r['burst_rate']:9.0%}{r['pulses']:8d}"
        )
    chosen, verdicts = checks.calibration_verdicts(
        rows, list(record["regions"]), burst_warn=params.get("burst_warn", 0.2)
    )
    picked = {role: (row["amplitude_mv"] if row else None) for role, row in chosen.items()}
    if all(v is not None for v in picked.values()):
        print(f"\n  amplitudes_mv for the parameter file: {picked}")
    return verdicts


def report(
    h5_path: str,
    protocol_path: str,
    out_png: str | None = None,
    threshold_uv: float = 200.0,
    out_csv: str | None = None,
) -> list[dict]:
    with open(protocol_path) as f:
        record = json.load(f)
    params = record["params"]
    well = params["well"]
    data = _load_well(h5_path, well)
    fps = data["fps"]
    chan = {int(m["electrode"]): int(m["channel"]) for m in data["mapping"]}
    print(
        f"{os.path.basename(h5_path)} well {well}: {fps:.0f} Hz, {len(data['spikes'])} spikes, {len(data['events'])} events, {len(chan)} routed electrodes"
    )

    # --- what fired, against the schedule ---
    starts = [(frame, m["start_stimulation"]) for frame, m in data["events"] if "start_stimulation" in m]
    tokens = {}
    for frame, token in starts:
        tokens.setdefault(token, []).append(frame)
    expected = {}
    for block in record["blocks"]:
        for p in block["presentations"]:
            expected[f"{p['token']}_{well}"] = expected.get(f"{p['token']}_{well}", 0) + 1
    print("\npresentations, against the schedule:")
    for token, n in expected.items():
        frames = tokens.get(token, [])
        gaps = np.diff(frames) / fps if len(frames) > 1 else []
        print(
            f"  {token:<24} expected {n:4d}  fired {len(frames):4d}  {'ok' if len(frames) == n else 'MISMATCH'}"
            + (f"  interval median {np.median(gaps):.1f} s" if len(gaps) else "")
        )

    # --- artifacts, for the dry run ---
    inspect = {}
    for role, r in record["regions"].items():
        if r.get("drive_electrodes"):
            inspect[f"{role} drive"] = r["drive_electrodes"][0]
        if r.get("return_electrodes"):
            inspect[f"{role} return"] = r["return_electrodes"][0]
    before, after = int(0.001 * fps), int(0.004 * fps)
    print(
        f"\nfirst deflection beyond {threshold_uv:.0f} uV after the sequence event, on one electrode of each site:"
    )
    panels = []
    for token, frames in tokens.items():
        roles = protocol.roles_in(token.rsplit("_", 1)[0])
        for label, electrode in inspect.items():
            if label.split()[0] not in roles or electrode not in chan:
                continue
            found = []
            for frame in frames[:5]:
                t, v = _trace(data, chan[electrode], frame, before, after)
                if t is None:
                    break
                hit = np.flatnonzero((t >= 0) & (np.abs(v) >= threshold_uv))
                found.append((float(t[hit[0]]), float(v[hit[0]])) if len(hit) else None)
                if len(panels) < 12 and frame == frames[0]:
                    panels.append((f"{token} {label} (e{electrode})", t, v))
            seen = [x for x in found if x]
            if seen:
                print(
                    f"  {token:<24} {label:<12} e{electrode:<6} at {np.median([x[0] for x in seen]):6.2f} ms, "
                    f"{'+' if np.median([x[1] for x in seen]) > 0 else '-'}{abs(np.median([x[1] for x in seen])):.0f} uV  ({len(seen)}/{len(found)})"
                )
            elif found:
                print(
                    f"  {token:<24} {label:<12} e{electrode:<6} nothing above threshold in {len(found)} presentations"
                )

    # --- the gates: calibration and connectivity ---
    mode = params.get("mode", "conditioning")
    if mode in ("calibration", "connectivity"):
        verdicts = _gate(data, record)
        print(f"\n=== {mode} verdict ===")
        for verdict in verdicts:
            print(f"  {verdict}")
        if any(v.level == "stop" for v in verdicts):
            print("  A [STOP] means conditioning as things stand would waste the culture.")

    # --- the readout ---
    rows = readout(data, record, well)
    if rows:
        print(
            "\nreadout: spikes in US in the pulse window after each presentation (and in the stimulated region for calibration):"
        )
        for row in rows:
            stimulated = protocol.roles_in(row["token"])
            local = stimulated[0] if stimulated else "US"
            print(
                f"  {row['index']:4d} {row['block']!s:<12} {row['token']:<20} US {row['US_pulse']:5d}   local({local}) {row[f'{local}_pulse']:5d}"
            )

    # --- coupling before and after ---
    from mxtreme.scans import region_selection

    patches = {role: r["rec_electrodes"] for role, r in record["regions"].items()}
    spikes = data["spikes"]
    frames_all = spikes["frameno"].astype(np.int64)
    elec_of_chan = {int(m["channel"]): int(m["electrode"]) for m in data["mapping"]}
    elecs_all = np.array([elec_of_chan.get(int(c), -1) for c in spikes["channel"]])
    marks = {}
    for frame, m in data["events"]:
        for key in m:
            if key.endswith("_start") or key == "end_experiment":
                marks.setdefault(key.replace("_start", ""), frame)
    for label in ("pre", "post"):
        if label not in marks:
            continue
        a = marks[label]
        later = [f for k, f in marks.items() if f > a]
        b = min(later) if later else frames_all.max() + 1
        sel = (frames_all >= a) & (frames_all < b)
        if sel.sum() < 100:
            continue
        bursts = region_selection.network_bursts(frames_all[sel], fps)
        names, outside = region_selection.region_correlations(
            frames_all[sel], elecs_all[sel], patches, fps, exclude=bursts
        )
        _, _, lead, _used = region_selection.burst_onset_lags(
            frames_all[sel], elecs_all[sel], patches, fps, bursts
        )
        coupling = region_selection.coupling(outside, lead)
        print(f"\ncoupling during {label} ({(b - a) / fps / 60:.1f} min, {len(bursts)} bursts):")
        for name, row in zip(names, coupling):
            print(f"  {name:<4}" + "".join(f"{v:7.2f}" for v in row))

    if out_csv and rows:
        # Every number the printed tables are drawn from, so they can be checked independently:
        # one row per presentation, the frame it fired on, and each region's count in each window.
        import csv as _csv

        with open(out_csv, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out_csv} ({len(rows)} presentations x {len(rows[0])} columns)")

    if out_png:
        _draw(record, rows, panels, out_png, data, marks)
    return rows


def _draw(record, rows, panels, out_png, data, marks):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps = data["fps"]
    blocks = [
        protocol.Block(
            b["label"], b["duration_sec"], [protocol.Presentation(**p) for p in b["presentations"]]
        )
        for b in record["blocks"]
    ]
    n_panels = min(len(panels), 6)
    fig = plt.figure(figsize=(15, 10 + 1.5 * n_panels))
    grid = fig.add_gridspec(2 + n_panels, 1, height_ratios=[3, 1.4] + [0.8] * n_panels, hspace=0.45)
    ax_read = fig.add_subplot(grid[0])
    ax_time = fig.add_subplot(grid[1])

    calibrating = record["params"]["mode"] == "calibration"
    ax_read.set_title(
        "evoked response in the stimulated region, by amplitude"
        if calibrating
        else "readout: spikes in US after each presentation",
        loc="left",
        fontsize=10,
    )
    ax_read.set_xlabel("presentation")
    pw = record["params"]["pulse_window_ms"]
    ax_read.set_ylabel(f"spikes {pw[0]:.0f}-{pw[1]:.0f} ms after each pulse, summed")
    series = {}
    for row in rows:
        stimulated = protocol.roles_in(row["token"])
        read = stimulated[0] if calibrating and stimulated else "US"
        series.setdefault(row["token"], []).append((row["index"], row[f"{read}_pulse"]))
    for token, points in series.items():
        stimulated = protocol.roles_in(token)
        colour = (
            display.PAIR_COLOUR
            if len(stimulated) > 1
            else display.ROLE_COLOURS.get(stimulated[0] if stimulated else "US", "#888888")
        )
        label = (
            token
            if calibrating
            else (
                "+".join(stimulated)
                + (" (pairing)" if len(stimulated) > 1 else " alone")
                + (" probe" if token.startswith("probe") else " training")
            )
        )
        xs, ys = zip(*points)
        ax_read.plot(xs, ys, "o" if token.startswith("probe") else "s", color=colour, label=label, ms=6, lw=0)
    for label in ("baseline", "encode", "decay"):
        edge = next((r["index"] for r in rows if r["block"] == label), None)
        if edge is not None:
            ax_read.axvline(edge - 0.5, color="#cccccc", lw=0.8)
    if series:
        ax_read.legend(fontsize=7, loc="upper left")

    t0 = marks.get("pre", min(f for f, _ in data["events"]) if data["events"] else 0)
    starts = [(marks[b.label] - t0) / fps if b.label in marks else None for b in blocks]
    for i in range(len(starts)):
        if starts[i] is None:
            starts[i] = (starts[i - 1] + blocks[i - 1].duration_sec) if i else 0.0
    fired = [((r["frame"] - t0) / fps, r["token"]) for r in rows]
    display.draw_timeline(ax_time, blocks, starts, fired=fired, dt_cs_us_ms=record["params"]["dt_cs_us"])
    ax_time.set_title(f"{len(rows)} presentation(s) fired", loc="left", fontsize=10)

    for k, (title, t, v) in enumerate(panels[:n_panels]):
        ax = fig.add_subplot(grid[2 + k])
        ax.plot(t, v, lw=0.8, color="#333333")
        ax.axvline(0, color="#ef4444", lw=0.6)
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_ylabel("uV", fontsize=7)
        if k == n_panels - 1:
            ax.set_xlabel("ms after the sequence event")
    fig.savefig(out_png, dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_png}")
