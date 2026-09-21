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
                        # Signed: MaxWell stores these uint64, and the samples before an event
                        # would otherwise wrap to 2^64 when the event's frame is subtracted.
                        "frame_nos": np.asarray(group["frame_nos"][:]).astype(np.int64),
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
    """Raw voltage in uV on one channel around one frame, zero being the level just before the
    pulse (when the window has any), so what the pulse leaves behind is read directly; or
    (None, None) when the recording kept no raw trace there."""
    t, v = _raw(data, channel, frame, before, after)
    if t is None:
        return None, None
    pre = v[t < -0.2]
    return t, v - (np.median(pre) if len(pre) >= 4 else np.median(v))


def _raw(data, channel, frame, before, after):
    """Raw voltage in uV, as recorded, on one channel around one frame, or (None, None) when the
    recording kept no raw trace there: a spikes-only recording, a channel outside the raw group,
    or MaxWell's HDF5 compression filter missing on this machine."""
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
        t = (frame_nos[i0:i1].astype(np.int64) - int(frame)) / data["fps"] * 1000.0
        return t, v * data["lsb"] * 1e6
    return None, None


def readout(data, record, well) -> list[dict]:
    """One row per presentation fired: block, token, frame, and per region the count in each
    response window and the pulse-locked count.

    A region's count is over its recording electrodes plus, when the region is not the one being
    pulsed, its stimulation electrodes (which record between pulses and sit on its liveliest
    spots)."""
    fps = data["fps"]
    spikes = data["spikes"]
    frames = spikes["frameno"].astype(np.int64)
    elec_of_chan = {int(m["channel"]): int(m["electrode"]) for m in data["mapping"]}
    spike_elec = np.array([elec_of_chan.get(int(c), -1) for c in spikes["channel"]])
    # A region's stimulation electrodes are routed to amplifiers like any other and record between
    # pulses -- and they sit on the region's liveliest spots, since select placed them there. They
    # count towards the region's readout whenever the region is not the one being pulsed; during
    # its own pulses they carry the artifact, so only the recording electrodes count then.
    members = {
        role: np.isin(spike_elec, np.asarray(r["rec_electrodes"])) for role, r in record["regions"].items()
    }
    members_quiet = {
        role: np.isin(spike_elec, np.asarray(list(r["rec_electrodes"]) + list(r.get("stim_electrodes", []))))
        for role, r in record["regions"].items()
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
        pulsed = set(protocol.roles_in(token))
        for role, member_rec in members.items():
            member = member_rec if role in pulsed else members_quiet[role]
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


def _crosstalk(frames, elecs, regions, fps, by_role, window, params) -> list:
    """The independence checks on single pulses to one region at a time: the evoked matrix (row
    stimulated, column measured), each region's own response, bursts, and whether CS and US are
    connected at all. Run on the baseline block's probes, where it gates the session."""
    from mxtreme.experiments.associative import checks
    from mxtreme.scans import region_selection

    bursts = region_selection.network_bursts(frames, fps)
    responses = checks.pulse_responses(frames, elecs, regions, fps, by_role, window)
    rates = checks.burst_rate(by_role, bursts, fps, within_ms=500.0)
    names, evoked, ratio = checks.crosstalk(responses)
    counts = {role: sum(1 for r, _ in by_role if r == role) for role in names}
    print(
        "evoked spikes per pulse (row stimulated alone, column measured), above the matching "
        "window before each pulse; pulses per row: " + ", ".join(f"{k} {v}" for k, v in counts.items())
    )
    for name, row in zip(names, evoked):
        print(f"  {name:<4}" + "".join(f"{v:8.2f}" for v in row))
    print("as a fraction of the stimulated region's own response:")
    for name, row in zip(names, ratio):
        print(f"  {name:<4}" + "".join(("     nan" if np.isnan(v) else f"{v:8.0%}") for v in row))
    print("network bursts started: " + ", ".join(f"{k} {v:.0%}" for k, v in rates.items()))
    return checks.independence_verdicts(
        names,
        evoked,
        ratio,
        rates,
        crosstalk_warn=params.get("crosstalk_warn", 0.3),
        burst_warn=params.get("burst_warn", 0.2),
    )


def _gate(data, record) -> list:
    """The calibration analysis: the table the verdicts are drawn from, then the verdicts.

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

    # Only detections large enough to be spikes count towards "not silent": the detector fires on
    # noise crossings of a few uV on every channel, and on the artifact everywhere a pulse reaches.
    # A recording's amplitudes are in ADC counts (clean.py and the scans scale them by lsb too).
    real = frames[np.abs(spikes["amplitude"]) * data["lsb"] * 1e6 >= checks.MIN_SPIKE_UV]
    silent = checks.silent_recording(checks.outside_artifacts(real, [f for _, f in pulses], fps), fps)
    if silent is not None:
        return [silent]

    window = tuple(params["pulse_window_ms"])
    bursts = region_selection.network_bursts(frames, fps)

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
        rows,
        list(record["regions"]),
        burst_warn=params.get("burst_warn", 0.2),
        crosstalk_warn=params.get("crosstalk_warn", 0.3),
    )
    picked = {role: (row["amplitude_mv"] if row else None) for role, row in chosen.items()}
    if all(v is not None for v in picked.values()):
        polarities = [row["polarity"] for row in chosen.values() if row and row.get("polarity")]
        majority = max(set(polarities), key=polarities.count) if polarities else params.get("pulse_polarity")
        if len(set(polarities)) > 1:
            verdicts.append(
                checks.Verdict(
                    "warn",
                    "the regions prefer different polarities ("
                    + ", ".join(f"{role} {row['polarity']}" for role, row in chosen.items() if row)
                    + f"); pulse_polarity is one setting for all three, so {majority} is taken for all.",
                )
            )
        record["_calibration_pick"] = {
            "amplitudes_mv": picked,
            "amplitudes_source": data.get("name", "this calibration"),
            "pulse_polarity": majority,
        }
        print("\n  for the parameter file (report --apply <params.json> writes them):")
        print(f"    amplitudes_mv = {picked}")
        print(f"    amplitudes_source = {data.get('name', 'this calibration')!r}")
        print(f"    pulse_polarity = {majority!r}")
    record["_calibration_rows"] = rows  # for compare()
    return verdicts


def apply_calibration(params_path: str, pick: dict) -> None:
    """Write a calibration's choice -- amplitudes, their source, the polarity -- into a parameter
    file in place, keeping its other keys and its ``_`` notes as they are."""
    from mxtreme.experiments.associative.params import AssociativeParams

    AssociativeParams.update_file(
        params_path, {key: pick[key] for key in ("amplitudes_mv", "amplitudes_source", "pulse_polarity")}
    )
    print(
        f"\nwrote amplitudes_mv {pick['amplitudes_mv']}, amplitudes_source {pick['amplitudes_source']!r} "
        f"and pulse_polarity {pick['pulse_polarity']!r} into {params_path}"
    )


def _print_calibration(rows):
    print(
        "\ncalibration, per region and amplitude (evoked spikes per pulse: the count 5-50 ms after the "
        "pulse minus the count in the same window before it):"
    )
    print(
        f"  {'role':<5}{'mV':>7}{'polarity':>16}{'local':>9}{'+/-':>6}{'remote':>9}{'spread':>8}"
        f"{'bursts':>9}{'pulses':>8}"
    )
    for r in rows:
        spread = r.get("spread", float("nan"))
        se = r.get("se", float("nan"))
        print(
            f"  {r['role']:<5}{r['amplitude_mv']:7.0f}{r['polarity']:>16}{r['local']:9.2f}"
            + (f"{se:6.2f}" if np.isfinite(se) else f"{'-':>6}")
            + f"{r['remote']:9.2f}"
            + (f"{spread:8.0%}" if np.isfinite(spread) else f"{'-':>8}")
            + f"{r['burst_rate']:9.0%}{r['pulses']:8d}"
        )
    print(
        "  local = the stimulated region, +/- its standard error over the pulses; remote = the mean of the "
        "other two; spread = remote / local.\n  usable = local >= 0.5 and >= 2 x its error, spread <= 30%, "
        "bursts <= 20%, at least 3 pulses"
    )


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


#: How far after a pulse the driven electrode's trace is read, at most.
RECOVERY_MS = 1500.0
#: Back to baseline means within this of the pre-pulse level for the rest of the window, on a
#: 5 ms running mean; and a shift larger than this from one pulse's baseline to the next's is
#: something the pulse left behind.
SETTLED_UV = 100.0
#: A recovery taking more than this fraction of the interval between pulses is warned about.
RECOVERY_FRACTION = 0.8


def _recovery(t, v, full_scale_uv) -> tuple[float, float | None]:
    """How long after the event the trace sits at the amplifier's rail, and when it is back
    within :data:`SETTLED_UV` of the pre-pulse level for good (``None`` if not within the trace)."""
    after = t >= 0
    railed = after & (np.abs(v) >= 0.9 * full_scale_uv)
    rail_ms = float(t[railed].max()) if railed.any() else 0.0
    # Settling is judged on a 5 ms running mean, so single noise crossings do not count.
    width = max(1, round(5.0 / max(float(t[1] - t[0]), 1e-9)))
    smooth = np.convolve(v, np.ones(width) / width, mode="same")
    off = after & (np.abs(smooth) >= SETTLED_UV)
    if not off.any():
        return rail_ms, 0.0
    last = int(np.flatnonzero(off).max())
    return rail_ms, (float(t[last + 1]) if last + 1 < len(t) else None)


def _panel_tokens(tokens, stimuli: dict) -> set:
    """Which presentations get an artifact panel in the figure. The panels are a hardware check
    (did the pulse reach the electrode?), not a response, so one per case is enough: in a
    calibration, per region the largest amplitude of each polarity; otherwise the first
    presentation of each combination of regions.

    :param tokens: ``{fired token: frames}`` in the order they fired.
    :param stimuli: The protocol's stimuli by token.
    """
    best = {}
    for token in tokens:
        base = token.rsplit("_", 1)[0]
        stim = stimuli.get(base, {})
        key = (tuple(protocol.roles_in(base)), stim.get("polarity"))
        amplitude = stim.get("amplitude_mv") or 0.0
        if key not in best or amplitude > best[key][0]:
            best[key] = (amplitude, token)
    return {token for _, token in best.values()}


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
    # The trace runs from 2 ms before the event to well after the pulse, bounded by the next one:
    # what the pulse leaves on the electrode, and how long it takes to go, is the question.
    all_pulses = sorted(f for f, _ in starts)
    interval_ms = float(np.median(np.diff(all_pulses)) / fps * 1000.0) if len(all_pulses) > 1 else RECOVERY_MS
    span_ms = min(RECOVERY_MS, 0.9 * interval_ms)
    before, after = int(0.002 * fps), int(span_ms / 1000 * fps)
    full_scale_uv = 512 * data["lsb"] * 1e6
    if not data["segments"]:
        # Said once, here, rather than only if a stimulation electrode happens to be routed.
        _once(
            "none",
            "\nno raw traces in this recording: it kept spikes only, which is all the readout "
            "needs. The artifact check below needs raw_traces 'regions' or 'all'.",
        )
    print(
        f"\nthe pulse on one driven electrode of each site: first deflection beyond {threshold_uv:.0f} uV "
        f"after the sequence event (its sign is the first phase's), how long the amplifier stays at its "
        f"rail (+/-{full_scale_uv / 1000:.1f} mV), when the electrode is back within {SETTLED_UV:.0f} uV of "
        f"where it was, and how far from that the next pulse starts (pulses {interval_ms / 1000:.1f} s apart):"
    )
    drawn = _panel_tokens(tokens, record["stimuli"])
    panels = []
    slowest, carry = {}, {}
    for token, frames in tokens.items():
        roles = protocol.roles_in(token.rsplit("_", 1)[0])
        for label, electrode in inspect.items():
            if label.split()[0] not in roles or electrode not in chan:
                continue
            found, recovery, carried = [], [], []
            for frame in frames[:5]:
                t, v = _trace(data, chan[electrode], frame, before, after)
                if t is None:
                    break
                hit = np.flatnonzero((t >= 0) & (t < 5.0) & (np.abs(v) >= threshold_uv))
                found.append((float(t[hit[0]]), float(v[hit[0]])) if len(hit) else None)
                recovery.append(_recovery(t, v, full_scale_uv))
                # The level just before the next pulse against the level just before this one:
                # the direct test of whether anything carries over.
                nxt = (
                    all_pulses[int(np.searchsorted(all_pulses, frame, side="right"))]
                    if frame < all_pulses[-1]
                    else None
                )
                if nxt is not None:
                    _, here = _raw(data, chan[electrode], frame, before, 0)
                    _, there = _raw(data, chan[electrode], nxt, before, 0)
                    if here is not None and there is not None and len(here) and len(there):
                        carried.append(float(np.median(there) - np.median(here)))
                if token in drawn and frame == frames[0]:
                    panels.append((f"artifact: {token}, {label} e{electrode}, first pulse", t, v))
            seen = [x for x in found if x]
            if seen:
                rails = [r[0] for r in recovery]
                settles = [r[1] for r in recovery]
                settled = max(settles, key=lambda x: (x is None, x or 0))
                slowest[electrode] = max(
                    slowest.get(electrode, 0.0), float("inf") if settled is None else settled
                )
                shift = float(np.median(carried)) if carried else float("nan")
                if np.isfinite(shift):
                    carry[electrode] = max(carry.get(electrode, 0.0), abs(shift))
                print(
                    f"  {token:<24} {label:<12} e{electrode:<6} at {np.median([x[0] for x in seen]):6.2f} ms, "
                    f"{'+' if np.median([x[1] for x in seen]) > 0 else '-'}{abs(np.median([x[1] for x in seen])):.0f} uV  "
                    f"({len(seen)}/{len(found)}); rail {np.median(rails):.0f} ms; back "
                    + (f"at {settled:.0f} ms" if settled is not None else f"not within {span_ms:.0f} ms")
                    + (f"; next pulse starts {shift:+.0f} uV from this one" if np.isfinite(shift) else "")
                )
            elif found:
                print(
                    f"  {token:<24} {label:<12} e{electrode:<6} nothing above threshold in {len(found)} presentations"
                )
    for electrode, shift in carry.items():
        if shift > SETTLED_UV:
            print(
                f"  WARN e{electrode} starts a pulse up to {shift:.0f} uV away from where it started the "
                f"previous one: what a pulse leaves on the electrode is carrying over across the "
                f"{interval_ms / 1000:.1f} s between pulses. Longer intervals, or a smaller amplitude."
            )
    for electrode, settled in slowest.items():
        if settled > RECOVERY_FRACTION * interval_ms and carry.get(electrode, 0.0) <= SETTLED_UV:
            print(
                f"  note: e{electrode} is still more than {SETTLED_UV:.0f} uV from its pre-pulse level "
                + (f"{settled / 1000:.1f} s" if np.isfinite(settled) else f"{span_ms / 1000:.1f} s")
                + " after a pulse, but is back by the next one."
            )

    # --- the calibration verdict ---
    mode = params.get("mode", "conditioning")
    if mode == "calibration":
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
    # --- the baseline gate: the independence checks, on the baseline block's probes ---
    # Each role is probed alone there (60 pulses per role), so the session's first recording is
    # its own go/no-go; `--phase baseline` on its own is the same measurement.
    if mode == "conditioning" and "baseline" in marks and len(frames_all):
        a = marks["baseline"]
        later = [f for f in marks.values() if f > a]
        b = min(later) if later else int(frames_all.max()) + 1
        pulses_base, regions_base = _pulses_and_regions(data, record)
        by_role = [
            (protocol.roles_in(t)[0], f)
            for t, f in pulses_base
            if a <= f < b and t.startswith(ALONE) and len(protocol.roles_in(t)) == 1
        ]
        if by_role:
            from mxtreme.experiments.associative import checks

            print("\n=== baseline gate: each region probed alone, from the baseline block ===")
            real = frames_all[np.abs(spikes["amplitude"]) * data["lsb"] * 1e6 >= checks.MIN_SPIKE_UV]
            silent = checks.silent_recording(
                checks.outside_artifacts(real, [f for _, f in by_role], fps), fps
            )
            verdicts = (
                [silent]
                if silent is not None
                else _crosstalk(
                    frames_all,
                    elecs_all,
                    regions_base,
                    fps,
                    by_role,
                    tuple(params["pulse_window_ms"]),
                    params,
                )
            )
            for verdict in verdicts:
                print(f"  {verdict}")
            if any(v.level == "stop" for v in verdicts):
                print(
                    "  A [STOP] here means: stop the run before encoding starts (Ctrl-C) and fix the cause."
                )

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
    apply_to: str | None = None,
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
    :param apply_to: A parameter file to write a calibration's choice into (amplitudes, their
        source, the polarity), so nothing has to be copied by hand. Refused unless every region
        got an amplitude.
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
    if apply_to is not None:
        picks = [
            f["record"].get("_calibration_pick")
            for f in files
            if f["record"]["params"].get("mode") == "calibration"
        ]
        if not picks:
            raise ValueError("--apply needs a calibration recording")
        if picks[-1] is None:
            raise ValueError("this calibration chose no amplitude for at least one region; nothing written")
        apply_calibration(apply_to, picks[-1])
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


def _draw_readout(ax_read, files, rows):
    record = files[0]["record"]
    ax_read.set_title(
        "readout: spikes in US in the pulse-locked window after each presentation, by what was presented",
        loc="left",
        fontsize=10,
    )
    ax_read.set_xlabel("presentation")
    pw = record["params"]["pulse_window_ms"]
    ax_read.set_ylabel(f"spikes {pw[0]:.0f}-{pw[1]:.0f} ms after each pulse, summed")
    series = {}
    for row in rows:
        stimulated = protocol.roles_in(row["token"])
        read = "US"
        series.setdefault(row["token"], []).append((row["index"], row[f"{read}_pulse"]))
    for token, points in series.items():
        stimulated = protocol.roles_in(token)
        colour = (
            display.PAIR_COLOUR
            if len(stimulated) > 1
            else display.ROLE_COLOURS.get(stimulated[0] if stimulated else "US", "#888888")
        )
        label = (
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


def _draw_calibration(fig, cell, record, roles):
    """Calibration's figure: per region, evoked spikes per pulse against amplitude, one line per
    polarity, the local response solid and the remote one dashed, with the usable threshold, the
    burst limit and the chosen amplitude marked. This is the table drawn, so the choice can be
    checked by eye."""
    from mxtreme.experiments.associative import checks

    rows = record.get("_calibration_rows") or []
    pick = record.get("_calibration_pick") or {}
    chosen = pick.get("amplitudes_mv", {})
    burst_warn = record["params"].get("burst_warn", 0.2)
    sub = cell.subgridspec(1, max(1, len(roles)), wspace=0.25)
    axes = [fig.add_subplot(sub[0, k]) for k in range(max(1, len(roles)))]
    if not rows:
        axes[0].text(
            0.0,
            0.5,
            "no evoked response to plot: the recording has no spikes outside the stimulation artifacts",
            transform=axes[0].transAxes,
            fontsize=9,
        )
        for ax in axes:
            ax.axis("off")
        return
    polarities = sorted({r["polarity"] for r in rows})
    colours = {"anodic-first": "#2563eb", "cathodic-first": "#dc2626"}
    for ax, role in zip(axes, roles):
        for polarity in polarities:
            mine = sorted(
                (r for r in rows if r["role"] == role and r["polarity"] == polarity),
                key=lambda r: r["amplitude_mv"],
            )
            if not mine:
                continue
            colour = colours.get(polarity, "#111827")
            amps = [r["amplitude_mv"] for r in mine]
            ax.errorbar(
                amps,
                [r["local"] for r in mine],
                yerr=[r.get("se", 0.0) if np.isfinite(r.get("se", 0.0)) else 0.0 for r in mine],
                fmt="o-",
                color=colour,
                capsize=3,
                label=f"{polarity}: local (+/- standard error)",
            )
            ax.plot(
                amps,
                [r["remote"] for r in mine],
                "s--",
                color=colour,
                alpha=0.5,
                ms=4,
                label=f"{polarity}: remote (mean of the other two)",
            )
            for r in mine:
                if r["burst_rate"] > burst_warn:
                    ax.plot(r["amplitude_mv"], r["local"], "x", color="#111827", ms=10, mew=2)
        ax.axhline(checks.MIN_LOCAL, color="#9ca3af", lw=0.8, ls=":")
        ax.text(
            0.01,
            checks.MIN_LOCAL,
            f" usable above {checks.MIN_LOCAL:g} spikes/pulse",
            transform=ax.get_yaxis_transform(),
            fontsize=7,
            color="#6b7280",
            va="bottom",
        )
        amplitude = chosen.get(role)
        if amplitude is not None:
            ax.axvline(amplitude, color="#16a34a", lw=1.2)
            amps_here = [r["amplitude_mv"] for r in rows if r["role"] == role] or [amplitude]
            on_right = amplitude > (min(amps_here) + max(amps_here)) / 2
            ax.text(
                amplitude,
                0.02,
                f" chosen {amplitude:g} mV ({pick.get('pulse_polarity', '')}) ",
                transform=ax.get_xaxis_transform(),
                fontsize=7,
                color="#16a34a",
                va="bottom",
                ha="right" if on_right else "left",
            )
        else:
            ax.set_facecolor("#fff7ed")
        ax.set_title(f"{role}: evoked spikes per pulse, by amplitude", fontsize=9, loc="left")
        ax.set_xlabel("mV per phase")
    axes[0].set_ylabel(
        "evoked spikes per pulse\n(count 5-50 ms after the pulse, minus the count in the 45 ms before it)"
    )
    axes[0].legend(fontsize=7, loc="upper left")
    fig.text(
        0.01,
        0.985,
        "calibration: how hard each region responds to its own pulses (solid) and how far the pulse "
        "reaches (dashed). Each point is the mean over that amplitude's pulses of: spikes the region fired "
        "5-50 ms after the pulse, minus spikes it fired in the same-length window just before, so 0 means "
        "the pulse added nothing and a negative value is chance. Read left to right: the curve should rise "
        "from nothing to a plateau while the dashed line stays low. The choice is the largest amplitude whose "
        "solid point is above the dotted threshold by at least two error bars, whose dashed point is under "
        "30% of it (the pulse stays in its region), and without an x (a pulse that starts a network burst "
        f"more than {burst_warn:.0%} of the time). Between polarities, the one that crosses the threshold "
        "first wins. A shaded panel found nothing usable.",
        fontsize=8,
        color="#374151",
        wrap=True,
    )


def _draw(files, rows, out_png, timed):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    record = files[0]["record"]
    panels = [p for f in files for p in f["panels"]]
    n_panels = min(len(panels), 8)
    calibrating = record["params"]["mode"] == "calibration"
    top = 0.6 if calibrating and not record.get("_calibration_rows") else 3
    fig = plt.figure(figsize=(15, 7 + top + 1.5 * n_panels))
    grid = fig.add_gridspec(2 + n_panels, 1, height_ratios=[top, 1.4] + [0.8] * n_panels, hspace=0.45)
    ax_time = fig.add_subplot(grid[1])
    if calibrating:
        _draw_calibration(fig, grid[0], record, list(record["regions"]))
    else:
        ax_read = fig.add_subplot(grid[0])
        _draw_readout(ax_read, files, rows)

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
        ax.axhspan(-SETTLED_UV, SETTLED_UV, color="#dcfce7", lw=0)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(t[0], t[-1])
        if k == 0:
            title += (
                "   [raw trace on one driven electrode, zero = its level before the pulse; the pulse at 0 ms "
                "(red), then what it leaves behind and how long until the trace is back in the green band]"
            )
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_ylabel("uV", fontsize=7)
        if k == n_panels - 1:
            ax.set_xlabel("ms after the sequence event (log scale beyond 1 ms)")
    fig.savefig(out_png, dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_png}")
