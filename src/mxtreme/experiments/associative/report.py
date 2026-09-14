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


def load_protocol(h5_path: str) -> dict:
    """The protocol a run wrote into its recording (see :data:`.run.PROTOCOL_KEY`).

    :raises KeyError: If the recording carries none -- one made before protocols were embedded, or
        one where embedding failed, in which case pass the ``_protocol.json`` copy instead.
    """
    from mxtreme.experiments.associative.run import PROTOCOL_KEY

    with h5py.File(h5_path, "r") as f:
        key = f"/assay/{PROTOCOL_KEY}"
        if key not in f:
            raise KeyError(
                f"{os.path.basename(h5_path)} carries no {PROTOCOL_KEY}; pass the run's "
                f"_protocol.json with --protocol"
            )
        raw = f[key][:][0]
    return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))


def _load_well(h5_path, well):
    """One well's spikes, events, channel map and raw-trace segments, across every recording in
    the file.

    A run normally makes one continuous recording, but a file can hold several (``rec0000``,
    ``rec0001``, ...) when recording was stopped and restarted; frame numbers are the device's own
    running count, so the recordings' spikes and events line up when concatenated. Raw traces are
    kept per recording, since each has its own frame index, and there may be none at all -- a
    spikes-only recording (``raw_traces="none"``) defines no raw groups.
    """
    spikes, events, segments = [], [], []
    mapping = fps = lsb = start_ms = None
    with h5py.File(h5_path, "r") as f:
        names = sorted(k for k in f["/recordings"] if k.startswith("rec"))
        for name in names:
            key = f"/recordings/{name}/well{well:03d}"
            if key not in f:
                continue
            rec = f[key]
            if fps is None:
                fps = float(np.asarray(rec["settings/sampling"][:]).ravel()[0])
                lsb = float(np.asarray(rec["settings/lsb"][:]).ravel()[0])
                mapping = rec["settings/mapping"][:]
                if "start_time" in rec:
                    start_ms = float(np.asarray(rec["start_time"][()]).ravel()[0])
            spikes.append(rec["spikes"][:])
            for row in rec["events"][:] if "events" in rec else []:
                raw = row["eventmessage"]
                try:
                    message = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
                except (ValueError, AttributeError):
                    message = {}
                events.append((int(row["frameno"]), message))
            for group in rec["groups"].values() if "groups" in rec else []:
                segments.append(
                    {
                        "raw": (h5_path, group["raw"].name),
                        "frame_nos": group["frame_nos"][:],
                        "channels": group["channels"][:],
                    }
                )
    if fps is None:
        raise ValueError(f"{os.path.basename(h5_path)} has no recording of well {well}")
    spikes = np.concatenate(spikes) if spikes else np.zeros(0)
    if len(spikes):
        spikes = spikes[np.argsort(spikes["frameno"], kind="stable")]
    events.sort(key=lambda e: e[0])
    return {
        "fps": fps,
        "lsb": lsb,
        "mapping": mapping,
        "spikes": spikes,
        "events": events,
        "segments": segments,
        "recordings": len(names),
        "start_ms": start_ms,
        "name": os.path.basename(h5_path).removesuffix(".raw.h5").removesuffix(".h5"),
    }


_warned = set()


def _once(key, message):
    if key not in _warned:
        _warned.add(key)
        print(message)


def _trace(data, channel, frame, before, after):
    """Raw voltage in uV on one channel around one frame, or (None, None) when the recording kept
    no raw trace there: a spikes-only recording, a channel outside the raw group, or MaxWell's
    HDF5 compression filter missing on this machine."""
    if not data["segments"]:
        _once(
            "none",
            "  (no raw traces in this recording: it kept spikes only. The artifact check "
            "needs raw_traces 'regions' or 'all'.)",
        )
        return None, None
    for seg in data["segments"]:
        frame_nos = seg["frame_nos"]
        if not len(frame_nos) or frame - before < frame_nos[0] or frame + after > frame_nos[-1]:
            continue
        if channel not in seg["channels"]:
            continue
        i0 = int(np.searchsorted(frame_nos, frame - before))
        i1 = int(np.searchsorted(frame_nos, frame + after))
        row = int(np.flatnonzero(seg["channels"] == channel)[0])
        try:
            with h5py.File(seg["raw"][0], "r") as f:
                v = f[seg["raw"][1]][row, i0:i1].astype(float)
        except OSError:
            _once(
                "filter",
                "  (raw traces are not readable here: MaxWell's HDF5 compression filter "
                "is missing; run this on the rig for the artifact check)",
            )
            return None, None
        return (frame_nos[i0:i1] - frame) / data["fps"] * 1000.0, (v - np.median(v)) * data["lsb"] * 1e6
    return None, None


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
    # Presentations fire in schedule order, so the k-th one fired is the k-th one planned: that
    # names the block when a recording carries no block markers (an older run, or a file whose
    # events are not all in yet).
    planned = [b["label"] for b in record.get("blocks", []) for _ in b["presentations"]]
    rows = []
    for index, (frame, m) in enumerate((f, m) for f, m in data["events"] if "start_stimulation" in m):
        token = m["start_stimulation"].rsplit("_", 1)[0]
        stim = record["stimuli"].get(token)
        row = {
            "index": index,
            "block": block_of.get(frame) or (planned[index] if index < len(planned) else None),
            "checkpoint": "",  # filled in by checkpoints(), so the CSV can be grouped as plotted
            "token": token,
            "frame": frame,
            "sec": None,
        }
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


ALONE = ("probe", "checkpoint")


def checkpoints(rows, fps: float, stimuli: dict | None = None) -> list[dict]:
    """The probe-alone presentations grouped into the checkpoints they belong to, in time order,
    and each row tagged with its checkpoint.

    A checkpoint is a run of probes with no training between them: the baseline block, each
    mid-encode checkpoint, each retrieval block. Grouping by block is not enough, because the
    checkpoints inside the encode block are what turn three end-point measurements into a curve of
    the association forming.

    Counts are **per pulse**, so a short mid-encode checkpoint and a full probe block sit on the
    same axis: a checkpoint is deliberately fewer pulses, since an unpaired CS presentation during
    acquisition is a small extinction trial.

    :param stimuli: The protocol's stimuli, for each token's pulse count; without them a count is
        taken as one pulse and the numbers are per presentation.
    :returns: ``[{"label", "minutes", "n", "per_pulse": {role: mean US count per pulse}}]``.
    """
    groups: list[dict] = []
    current = None
    for row in rows:
        if not row["token"].startswith(ALONE):
            current = None  # a training presentation closes the checkpoint
            continue
        if current is None or current["block"] != row["block"]:
            same = sum(1 for g in groups if g["block"] == row["block"])
            current = {
                "block": row["block"],
                "label": f"{row['block']} {same + 1}" if row["block"] == "encode" else str(row["block"]),
                "rows": [],
            }
            groups.append(current)
        row["checkpoint"] = current["label"]
        current["rows"].append(row)
    if not groups:
        return []

    def pulses(token):
        stim = (stimuli or {}).get(token) or {}
        return max(1, int(stim.get("num_events", 1)) * int(stim.get("pulses_per_burst", 1)))

    first = groups[0]["rows"][0]
    timed = all(r.get("sec") is not None for g in groups for r in g["rows"])
    out = []
    for g in groups:
        head = g["rows"][0]
        minutes = (
            (head["sec"] - first["sec"]) / 60.0 if timed else (head["frame"] - first["frame"]) / fps / 60.0
        )
        per_pulse: dict[str, list[float]] = {}
        spikes: dict[str, int] = {}
        count: dict[str, int] = {}
        for row in g["rows"]:
            stimulated = protocol.roles_in(row["token"])
            if stimulated:
                role, n = stimulated[0], pulses(row["token"])
                per_pulse.setdefault(role, []).append(row["US_pulse"] / n)
                spikes[role] = spikes.get(role, 0) + int(row["US_pulse"])
                count[role] = count.get(role, 0) + n
        out.append(
            {
                "label": g["label"],
                "minutes": minutes,
                "n": len(g["rows"]),
                "per_pulse": {role: sum(v) / len(v) for role, v in per_pulse.items()},
                # Totals, so a verdict can put Poisson error bars on the per-pulse rate.
                "spikes": spikes,
                "pulses": count,
            }
        )
    return out


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
    _print_calibration(rows)
    chosen, verdicts = checks.calibration_verdicts(
        rows, list(record["regions"]), burst_warn=params.get("burst_warn", 0.2)
    )
    picked = {role: (row["amplitude_mv"] if row else None) for role, row in chosen.items()}
    if all(v is not None for v in picked.values()):
        print("\n  for the parameter file:")
        print(f"    amplitudes_mv = {picked}")
        print(f"    amplitudes_source = {data.get('name', 'this calibration')!r}")
    record["_calibration_rows"] = rows  # for compare()
    return verdicts


def _print_calibration(rows):
    print("\ncalibration, per region and amplitude (spikes per pulse above the window before it):")
    print(
        f"  {'role':<5}{'mV':>7}{'polarity':>16}{'local':>9}{'remote':>9}{'spread':>8}{'bursts':>9}{'pulses':>8}"
    )
    for r in rows:
        spread = (r["remote"] / r["local"]) if r["local"] > 0 and np.isfinite(r["remote"]) else float("nan")
        print(
            f"  {r['role']:<5}{r['amplitude_mv']:7.0f}{r['polarity']:>16}{r['local']:9.2f}"
            f"{r['remote']:9.2f}"
            + (f"{spread:8.0%}" if np.isfinite(spread) else f"{'-':>8}")
            + f"{r['burst_rate']:9.0%}{r['pulses']:8d}"
        )
    print("  local = the stimulated region; remote = the mean of the other two; spread = remote / local")


def site_label(record: dict) -> str:
    """A short name for the stimulation pattern a run used: ``focal 2x2 gap 4`` or ``grid 3x3 gap 2``."""
    site = record["params"].get("stim_site", {})
    if site.get("shape") == "focal":
        return f"focal {site.get('inner', 2)}x{site.get('inner', 2)} gap {site.get('inner_gap', 0)}"
    return f"grid {site.get('size', 3)}x{site.get('size', 3)} gap {site.get('gap', 2)}"


def compare(h5_paths: list[str], out_png: str | None = None) -> list:
    """Set calibration runs of the same regions side by side, one per stimulation pattern, and say
    which pattern to condition through.

    Each recording is read as :func:`report` reads it -- protocol from inside, responses per pulse
    -- and labelled by its ``stim_site``. :func:`checks.compare_calibrations` does the judging.

    :returns: The verdicts.
    """
    from mxtreme.experiments.associative import checks

    tables, labels, roles = {}, [], None
    for path in h5_paths:
        record = load_protocol(path)
        if record["params"].get("mode") != "calibration":
            raise ValueError(
                f"{os.path.basename(path)} is a {record['params'].get('mode')} run, not a calibration"
            )
        label = site_label(record)
        if label in tables:
            label = f"{label} ({os.path.basename(path)})"
        labels.append(label)
        data = _load_well(path, record["params"]["well"])
        print(f"\n=== {label}: {os.path.basename(path)} ===")
        verdicts = _gate(data, record)
        for v in verdicts:
            print(f"  {v}")
        tables[label] = record.get("_calibration_rows", [])
        roles = list(record["regions"])
    if not all(tables.values()):
        empty = [label for label, rows in tables.items() if not rows]
        print(f"\nno responses to compare in {', '.join(empty)} (silent recording?).")
        return [checks.Verdict("ok", "nothing to compare: at least one recording has no responses.")]

    # The same burst cap the runs' own verdicts used, so a row usable there is usable here.
    summary, verdicts = checks.compare_calibrations(
        tables, roles, burst_warn=record["params"].get("burst_warn", 0.2)
    )
    print("\n=== patterns compared ===")
    width = max(30, max(len(label) for label in labels) + 4)
    print(f"  {'region':<7}" + "".join(f"{label:>{width}}" for label in labels))
    for role in roles:
        line = f"  {role:<7}"
        for label in labels:
            e = next(s for s in summary if s["pattern"] == label and s["role"] == role)
            cell = "no response" if e["threshold_mv"] is None else f"threshold {e['threshold_mv']:.0f} mV"
            if e.get("local_at_shared") is not None:
                cell += f", {e['local_at_shared']:.2f}/pulse"
            if e.get("spread_at_shared") is not None:
                cell += f", spread {e['spread_at_shared']:.0%}"
            line += f"{cell:>{max(width, len(cell) + 2)}}"
        print(line)
    for v in verdicts:
        print(f"  {v}")
    if out_png:
        _draw_compare(tables, roles, out_png)
    return verdicts


def _draw_compare(tables, roles, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(roles), figsize=(5 * len(roles), 7), sharex=True, squeeze=False)
    for k, role in enumerate(roles):
        for label, rows in tables.items():
            mine = sorted(
                (r for r in rows if r["role"] == role and r["polarity"] != "cathodic-first"),
                key=lambda r: r["amplitude_mv"],
            )
            if not mine:
                continue
            amps = [r["amplitude_mv"] for r in mine]
            axes[0, k].plot(amps, [r["local"] for r in mine], "o-", label=label)
            axes[1, k].plot(amps, [r["remote"] for r in mine], "s--", label=label)
        axes[0, k].set_title(f"{role}: local response", fontsize=9, loc="left")
        axes[1, k].set_title(f"{role}: remote response (the other two regions)", fontsize=9, loc="left")
        axes[1, k].set_xlabel("mV per phase")
        axes[0, k].axhline(0.5, color="#cccccc", lw=0.8)
    axes[0, 0].set_ylabel("spikes per pulse")
    axes[1, 0].set_ylabel("spikes per pulse")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("calibration by stimulation pattern, anodic-first rows", fontsize=10)
    fig.savefig(out_png, dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def _read_one(h5_path: str, protocol_path: str | None, threshold_uv: float) -> dict:
    """Read one recording back: print what fired against its schedule, the artifact check, the
    gate verdict if it is a gate, the per-presentation readout and the pre/post coupling; return
    what the session-level sections need."""
    if protocol_path is None:
        record = load_protocol(h5_path)
    else:
        with open(protocol_path) as f:
            record = json.load(f)
    params = record["params"]
    well = params["well"]
    data = _load_well(h5_path, well)
    fps = data["fps"]
    chan = {int(m["electrode"]): int(m["channel"]) for m in data["mapping"]}
    phase = params.get("phase", "all")
    print(
        f"\n=== {os.path.basename(h5_path)} ===\n"
        f"well {well}, {params.get('mode', 'conditioning')}"
        + (f" phase {phase}" if phase != "all" else "")
        + f": {fps:.0f} Hz, {len(data['spikes'])} spikes, "
        f"{len(data['events'])} events, {len(chan)} routed electrodes, {data['recordings']} recording(s), "
        f"raw traces on {sum(len(sg['channels']) for sg in data['segments'][:1])} channel(s)"
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
    if not data["segments"]:
        # Said once, here, rather than only if a stimulation electrode happens to be routed.
        _once(
            "none",
            "\nno raw traces in this recording: it kept spikes only, which is all the readout "
            "needs. The artifact check below needs raw_traces 'regions' or 'all'.",
        )
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
    frames_all = spikes["frameno"].astype(np.int64) if len(spikes) else np.zeros(0, dtype=np.int64)
    elec_of_chan = {int(m["channel"]): int(m["electrode"]) for m in data["mapping"]}
    elecs_all = (
        np.array([elec_of_chan.get(int(c), -1) for c in spikes["channel"]])
        if len(spikes)
        else np.zeros(0, dtype=int)
    )
    marks = {}
    for frame, m in data["events"]:
        for key in m:
            if key.endswith("_start") or key == "end_experiment":
                marks.setdefault(key.replace("_start", ""), frame)
    for label in ("pre", "post"):
        if label not in marks or not len(frames_all):
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

    # Pulses followed by a network burst, by kind, for the session verdict.
    pulses_all, _regions = _pulses_and_regions(data, record)
    hits: dict[str, list[int]] = {"probe": [0, 0], "train": [0, 0]}
    if len(frames_all) and pulses_all:
        by_kind = [("probe" if t.startswith(ALONE) else "train", f) for t, f in pulses_all]
        bursts = region_selection.network_bursts(frames_all, fps)
        starts_b = np.array([a for a, _ in bursts], dtype=float)
        reach = 0.5 * fps
        for kind, frame in by_kind:
            hits[kind][1] += 1
            if len(starts_b) and np.any((starts_b >= frame) & (starts_b <= frame + reach)):
                hits[kind][0] += 1

    first_frame = min(
        [f for f, _ in data["events"]] + ([int(frames_all.min())] if len(frames_all) else []),
        default=0,
    )
    return {
        "path": h5_path,
        "record": record,
        "data": data,
        "rows": rows,
        "panels": panels,
        "marks": marks,
        "burst_hits": hits,
        "first_frame": first_frame,
        "duration_sec": sum(b["duration_sec"] for b in record["blocks"]),
    }


def report(
    h5_paths,
    protocol_path: str | None = None,
    out_png: str | None = None,
    threshold_uv: float = 200.0,
    out_csv: str | None = None,
) -> list[dict]:
    """Read one recording, or the recordings of one session, back against the protocol each was
    made with.

    A conditioning session split into phases (``baseline``, ``encode_1`` ..., ``retrieval``) is
    several recordings; given all of them, in any order, they are put in time order and read as
    one: the checkpoint curve, the conditioning verdict, the CSV and the figure span the session,
    while the schedule check, the artifact check and the coupling are printed per recording.

    :param h5_paths: One path, or a list.
    :param protocol_path: A ``_protocol.json`` for a single recording; by default the protocol is
        read from inside each recording, where :func:`.run.run` writes it.
    :param out_png: Where to draw the figure. Nothing is drawn without it.
    :param out_csv: Where to write the per-presentation counts. Nothing is written without it.
        Both are analysis, which can be regenerated from the recording, so neither is written
        unless asked for -- and never into the store unless you point them there.
    :returns: The per-presentation rows, over the whole session.
    """
    paths = [h5_paths] if isinstance(h5_paths, (str, os.PathLike)) else list(h5_paths)
    if protocol_path is not None and len(paths) > 1:
        raise ValueError("--protocol applies to a single recording; several carry their own")
    files = [_read_one(str(p), protocol_path, threshold_uv) for p in paths]

    # Time order: by the recordings' own clocks when they have them, else as given.
    starts = [f["data"]["start_ms"] for f in files]
    if all(s is not None for s in starts) and len(set(starts)) == len(starts):
        files.sort(key=lambda f: f["data"]["start_ms"])
    timed = all(s is not None for s in starts) and len(set(starts)) == len(starts)

    # One row list for the session, each row placed in session time: the recording's clock when
    # every file has one, else the recordings laid end to end (the gaps between them not shown).
    rows: list[dict] = []
    offset = 0.0
    for k, f in enumerate(files):
        fps = f["data"]["fps"]
        if timed and k:
            offset = (f["data"]["start_ms"] - files[0]["data"]["start_ms"]) / 1000.0
        for row in f["rows"]:
            row["file"] = f["data"]["name"]
            row["sec"] = offset + (row["frame"] - f["first_frame"]) / fps
            row["index"] = len(rows)
            rows.append(row)
        if not timed:
            offset += f["duration_sec"]
    if len(files) > 1:
        print(
            f"\n=== session: {len(files)} recordings, {len(rows)} presentations ===\n  "
            + ", ".join(f["data"]["name"] for f in files)
            + ("" if timed else "\n  (no start times: laid end to end, gaps between recordings not shown)")
        )

    # --- the association as it forms: probe-alone responses, checkpoint by checkpoint ---
    conditioning = [f for f in files if f["record"]["params"].get("mode", "conditioning") == "conditioning"]
    stimuli = {}
    for f in files:
        stimuli.update(f["record"]["stimuli"])
    fps = files[0]["data"]["fps"]
    curve = checkpoints(rows, fps, stimuli)
    if conditioning and curve:
        seen = [r for r in protocol.ROLES if any(r in c["per_pulse"] for c in curve)]
        print("\nassociation, by checkpoint: spikes in US per pulse, stimulating one region alone")
        print(
            "  "
            + f"{'checkpoint':<14}{'min':>6}"
            + "".join(f"{role + ' alone':>12}" for role in seen)
            + "  probes"
        )
        for c in curve:
            print(
                f"  {c['label']:<14}{c['minutes']:6.0f}"
                + "".join(f"{c['per_pulse'][r]:12.2f}" if r in c["per_pulse"] else f"{'-':>12}" for r in seen)
                + f"{c['n']:8d}"
            )
        print(
            "  learning is CS alone rising from baseline through the encode checkpoints to the "
            "retrievals while NS alone does not. A mid-encode checkpoint is 2 short probes: read "
            "the trend across them, never one point."
        )

        # The same checks a person would make from that table, made every time: independence at
        # baseline, bursting, excitability drift, and the result at two standard errors. Works on
        # a session that is still being recorded, as far as it has got.
        from mxtreme.experiments.associative import checks

        pooled = {"probe": [0, 0], "train": [0, 0]}
        for f in files:
            for kind, (hit, total) in f["burst_hits"].items():
                pooled[kind][0] += hit
                pooled[kind][1] += total
        burst_by_kind = {k: h / t for k, (h, t) in pooled.items() if t}
        params = conditioning[-1]["record"]["params"]
        print("\n=== conditioning verdict ===")
        for verdict in checks.conditioning_verdicts(
            curve,
            burst_by_kind,
            crosstalk_warn=params.get("crosstalk_warn", 0.3),
            burst_warn=params.get("burst_warn", 0.2),
        ):
            print(f"  {verdict}")

    if out_csv and rows:
        # Every number the printed tables are drawn from, so they can be checked independently:
        # one row per presentation, the recording and frame it fired on, and each region's count
        # in each window.
        import csv as _csv

        with open(out_csv, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out_csv} ({len(rows)} presentations x {len(rows[0])} columns)")

    if out_png:
        _draw(files, rows, out_png, timed)
    return rows


def _draw(files, rows, out_png, timed):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    record = files[0]["record"]
    panels = [p for f in files for p in f["panels"]]
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
                + (
                    " checkpoint"
                    if token.startswith("checkpoint")
                    else " probe"
                    if token.startswith("probe")
                    else " training"
                )
            )
        )
        xs, ys = zip(*points)
        marker = "^" if token.startswith("checkpoint") else "o" if token.startswith("probe") else "s"
        ax_read.plot(xs, ys, marker, color=colour, label=label, ms=6, lw=0)
    for label in ("baseline", "encode", "decay"):
        edge = next((r["index"] for r in rows if r["block"] == label), None)
        if edge is not None:
            ax_read.axvline(edge - 0.5, color="#cccccc", lw=0.8)
    if len(files) > 1:
        for f in files[1:]:
            edge = next((r["index"] for r in rows if r["file"] == f["data"]["name"]), None)
            if edge is not None:
                ax_read.axvline(edge - 0.5, color="#111827", lw=0.8, ls=":")
    if series:
        ax_read.legend(fontsize=7, loc="upper left")

    # The session's blocks end to end, each recording's blocks placed by its own markers where it
    # has them, the recordings themselves by their clocks when every one has one.
    blocks, starts, stimuli = [], [], {}
    offset = 0.0
    for k, f in enumerate(files):
        fps = f["data"]["fps"]
        if timed and k:
            offset = (f["data"]["start_ms"] - files[0]["data"]["start_ms"]) / 1000.0
        mine = [
            protocol.Block(
                b["label"], b["duration_sec"], [protocol.Presentation(**p) for p in b["presentations"]]
            )
            for b in f["record"]["blocks"]
        ]
        t0 = f["first_frame"]
        local = [(f["marks"][b.label] - t0) / fps if b.label in f["marks"] else None for b in mine]
        for i in range(len(local)):
            if local[i] is None:
                local[i] = (local[i - 1] + mine[i - 1].duration_sec) if i else 0.0
        blocks += mine
        starts += [offset + t for t in local]
        stimuli.update(f["record"]["stimuli"])
        if not timed:
            offset += f["duration_sec"]
    fired = [(r["sec"], r["token"]) for r in rows]
    display.draw_timeline(
        ax_time, blocks, starts, fired=fired, dt_cs_us_ms=record["params"]["dt_cs_us"], stimuli=stimuli
    )
    ax_time.set_title(
        f"{len(rows)} presentation(s) fired"
        + (
            f" over {len(files)} recordings" + ("" if timed else ", gaps not shown") if len(files) > 1 else ""
        ),
        loc="left",
        fontsize=10,
    )

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
