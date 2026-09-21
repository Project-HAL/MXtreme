"""Running the associative protocol on the rig, open loop.

One process, one well: route the electrodes, register one sequence per kind of presentation, open
the recording, then walk the schedule firing each presentation at its time and marking each
block's start with an event. The recording lands in the managed store like a scan does and is
registered as an ``experiment``.

**The recording is the only file the run leaves in the store.** Everything needed to read it back
travels inside it: the protocol (the parameters, which electrodes are US, CS and NS, every
stimulus and every block) is written to ``/assay/associative_protocol``, the routing is MaxLab's own
``settings/mapping``, and every presentation is an event with its frame. Given ``work``, a copy of
the protocol and a table of planned-against-actual fire times go there instead -- conveniences,
outside the store, that can be regenerated.

By default the recording keeps **spikes only** (``raw_traces="none"``): the readout is spike counts,
and a four-hour run is then a few hundred megabytes rather than tens of gigabytes.

The timing that matters -- pulses within a presentation, the offset between CS and US in a
pairing -- is inside each sequence and kept by the hardware. This loop only has to start
presentations that are seconds to minutes apart, which ``time.sleep`` on a monotonic clock does
to within tens of milliseconds. :func:`execute_schedule` is separated from the rig so that this
walk can be tested with a fake clock.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from mxtreme.experiments.associative import protocol
from mxtreme.experiments.associative.params import AssociativeParams

ProgressFn = Callable[[str], None]
StopFn = Callable[[], bool]

_POLL_SEC = 0.25


#: Where the protocol is written inside a recording; :mod:`.report` reads it from here.
PROTOCOL_KEY = "associative_protocol"


@dataclass
class RunResult:
    h5_path: Path
    fired: list[tuple[str, str, float, float]]  # (block, token, planned_sec, actual_sec)
    stopped_early: bool
    protocol_copy: Path | None = None  # in ``work``, when one was given
    fired_copy: Path | None = None
    phases: list[RunResult] = field(default_factory=list)  # every phase's result, in order
    stim_electrodes: dict[str, list[int]] = field(default_factory=dict)  # per role, as actually driven
    routing_cfg: str | None = None  # the routing used, saved or loaded, for the parameter file


#: What the MaxLab server's system type means, in the batch id's terms.
_SYSTEM_OF_DEVICE = {"MaxOne": "M1", "MaxTwo": "M2"}


def check_device(batch, mx=None) -> str:
    """Refuse to record when the connected system is not the one the batch id names.

    The ``M1`` / ``M2`` at the end of a batch id decides the ``chip_<M1|M2>_<chip>`` directory a
    recording is filed under, so a MaxTwo run under an ``_M1`` batch would land in the wrong place
    and be registered as the wrong system, with nothing to say so. Checked before the chip is
    touched.

    :param batch: The batch id, or a :class:`mxtreme.store.Batch`.
    :param mx: The imported ``maxlab`` module; imported here when not given.
    :raises RuntimeError: On a mismatch, or when no system answers.
    :returns: The device name, e.g. ``"MaxOne"``.
    """
    from mxtreme import store
    from mxtreme.scans.activity_scan import _connected_device, _require_maxlab

    mx = mx or _require_maxlab()
    device = _connected_device(mx)
    expected = store.Batch.parse(batch).system
    found = _SYSTEM_OF_DEVICE.get(device)
    if found != expected:
        raise RuntimeError(
            f"the connected system is {device} ({found}), but batch {store.Batch.parse(batch).id} "
            f"says {expected}. The recording would be filed under chip_{expected}_... as the wrong "
            f"system. Correct the batch id, or connect the right device."
        )
    return device


def execute_schedule(
    blocks,
    fire,
    mark,
    *,
    clock=time.monotonic,
    sleep=time.sleep,
    should_stop: StopFn | None = None,
    on_progress: ProgressFn = print,
):
    """Walk the blocks, calling ``mark(label)`` as each starts and ``fire(token)`` for each
    presentation on its second. Returns ``(fired, stopped_early)`` with ``fired`` as
    ``(block, token, planned_sec, actual_sec)`` from the schedule's origin.

    ``clock`` and ``sleep`` are injectable so the walk can be tested without waiting.
    """
    fired = []
    t0 = clock()

    def wait_until(t):
        while True:
            remaining = t0 + t - clock()
            if remaining <= 0:
                return True
            if should_stop is not None and should_stop():
                return False
            sleep(min(_POLL_SEC, remaining))

    block_start = 0.0
    try:
        return _walk(blocks, fire, mark, wait_until, clock, t0, fired, block_start, on_progress)
    except KeyboardInterrupt:
        on_progress("interrupted: closing this recording; no later phase will start")
        return fired, True


def _walk(blocks, fire, mark, wait_until, clock, t0, fired, block_start, on_progress):
    for block in blocks:
        if not wait_until(block_start):
            return fired, True
        mark(block.label)
        on_progress(
            f"[{clock() - t0:8.1f}s] {block.label} ({block.duration_sec / 60:.1f} min, {len(block.presentations)} presentation(s))"
        )
        for p in block.presentations:
            planned = block_start + p.t_sec
            if not wait_until(planned):
                return fired, True
            fire(p.token)
            actual = clock() - t0
            fired.append((block.label, p.token, planned, actual))
            on_progress(f"[{actual:8.1f}s]   {p.token}  ({actual - planned:+.3f}s)")
        block_start += block.duration_sec
    if not wait_until(block_start):
        return fired, True
    mark("end_experiment")
    return fired, False


def raw_channels(params: AssociativeParams, regions, config) -> list[int]:
    """The channels whose raw traces the recording keeps, per ``params.raw_traces``.

    :param regions: ``{role: RegionSpec}``, from :func:`.protocol.regions_for`.
    :param config: The routed array's ``maxlab`` :class:`Config` (``array.get_config()``), which
        maps electrodes to the channels they landed on.
    """
    if params.raw_traces == "none":
        return []
    if params.raw_traces == "all":
        return list(range(1024))
    electrodes = sorted({e for spec in regions.values() for e in spec.rec_electrodes + spec.stim_electrodes})
    return sorted(set(config.get_channels_for_electrodes(electrodes)))


def expand_phases(params: AssociativeParams, spec) -> list[str]:
    """The phases a ``run`` records, in order, from what was asked for.

    ``spec`` is a phase name, a list of them, or a comma-separated string, and may use two
    shorthands: ``"session"`` is the whole conditioning session -- ``baseline``, every
    ``encode_<k>``, ``retrieval`` -- and ``"encode"`` is every encode cycle. ``None`` means the
    parameters' own ``phase``.
    """
    if spec is None:
        spec = [params.phase]
    if isinstance(spec, str):
        spec = spec.split(",")
    encode = [f"encode_{k}" for k in range(1, params.encode_cycles + 1)] if params.encode_pattern else []
    out: list[str] = []
    for item in spec:
        item = item.strip()
        if item == "session":
            out += ["baseline", *encode, "retrieval"]
        elif item == "encode":
            out += encode
        elif item:
            out.append(item)
    if len(out) > 1 and "all" in out:
        raise ValueError("'all' is one recording of the whole session and cannot be combined with phases")
    return out or ["all"]


def run(
    params: AssociativeParams,
    config=None,
    *,
    phases=None,
    work: str | Path | None = None,
    on_progress: ProgressFn = print,
    should_stop: StopFn | None = None,
    clock=None,
    sleep=None,
) -> RunResult:
    """Run the protocol on the rig and record it: one recording, or one per phase.

    The rig is set up once -- device check, routing, the sequences -- and then each phase is its
    own recording, opened, written with its own protocol, run and closed before the next starts.
    ``phases=["session"]`` records a whole conditioning session as separate files without anyone
    at the keyboard between them; a single phase, or the default ``params.phase``, is one file.

    :param params: With ``regions``, ``rec_electrodes`` and ``amplitudes_mv`` set.
    :param config: The :class:`~mxtreme.config.Config` naming the store, unless ``params.save_path``
        says where to write instead.
    :param phases: See :func:`expand_phases`; ``None`` is ``params.phase``.
    :param work: A directory outside the store for a copy of each protocol and fire-time table.
        Optional; the recordings carry everything :mod:`.report` needs without it.
    :param should_stop: Polled while waiting; return ``True`` to end the run early. The recording
        being made is closed properly and no later phase starts.
    :param clock: What the schedule is walked against; ``time.monotonic`` unless given, or unless
        the ``maxlab`` module in use carries its own ``clock`` (a simulated rig keeps simulated
        time; MaxLab's API has no such attribute, so on hardware this is always the wall clock).
    :param sleep: Its partner, the same way; ``time.sleep`` on hardware.
    :returns: The last phase's result; every phase's is in its ``phases`` list.
    :raises ModuleNotFoundError: If ``maxlab`` is not installed (this is not the rig).
    :raises RuntimeError: If a protocol cannot be written into its recording and there is no
        ``work`` directory to keep it in. Raised before that recording starts, so nothing is lost.
    """
    from mxtreme.scans import mx_setup
    from mxtreme.scans.activity_scan import _require_maxlab
    from mxtreme.stimulation import sequences

    mx = _require_maxlab()
    clock = clock or getattr(mx, "clock", time.monotonic)
    sleep = sleep or getattr(mx, "sleep", time.sleep)
    names = expand_phases(params, phases)
    per_phase = [replace(params, phase=name) for name in names]
    for p in per_phase:
        p.validate(for_run=True)
    registry_path = config.registry_path if config is not None and params.save_path is None else None
    per_phase = [p.resolved(config) for p in per_phase]
    Path(per_phase[0].save_path).mkdir(parents=True, exist_ok=True)
    well = params.well

    # Every sequence any phase fires, built once; the regions are the same for all of them.
    schedules = {p.phase: protocol.build_schedule(p) for p in per_phase}
    stimuli: dict[str, protocol.Stimulus] = {}
    for _blocks, mine in schedules.values():
        stimuli.update(mine)
    regions = protocol.regions_for(params)
    on_progress(
        "=== Associative run ==="
        + (f" ({params.mode})" if params.mode != "conditioning" else "")
        + (
            f": {len(per_phase)} recordings, " + ", ".join(names)
            if len(per_phase) > 1
            else (f" phase {names[0]}" if names[0] != "all" else "")
        )
    )
    for p in per_phase:
        on_progress(f"--- {p.file_name}.raw.h5")
        on_progress(protocol.summary(schedules[p.phase][0], schedules[p.phase][1], params.stim_phase_us))
    for spec in regions.values():
        on_progress(
            f"  {spec.role}: centre {spec.center_um} um, {len(spec.drive_electrodes)} driven on dac {spec.drive_dac}"
            + (
                f", {len(spec.return_electrodes)} return on dac {spec.return_dac}"
                if spec.return_electrodes
                else ""
            )
            + f", {len(spec.rec_electrodes)} recording electrodes, {spec.amplitude_mv} mV"
        )

    # --- the chip, once ---
    on_progress(f"Device: {check_device(params.batch, mx)}, matching batch {params.batch_id}")
    on_progress("Initializing chip...")
    mx.initialize()
    if mx.send(mx.Core().enable_stimulation_power(True)) != "Ok":
        raise RuntimeError("the system did not enable stimulation power")
    sleep(mx.Timing.waitInit)
    mx.clear_events()

    groups = {}
    for role, spec in regions.items():
        groups[f"{role}_drive"] = spec.drive_electrodes
        if spec.return_electrodes:
            groups[f"{role}_return"] = spec.return_electrodes
    routing_cfg = params.routing_cfg
    if routing_cfg is None and work is not None:
        routing_cfg = str(Path(work) / f"{params.chip}_well{well}_DIV{params.div}_routing.cfg")
    array, units, used = sequences.init_well_regions(
        well, list(params.rec_electrodes), groups, config_path=routing_cfg, save_to=routing_cfg
    )
    routed = array.get_config()
    on_progress(f"routed {len(routed.get_channels())} channels")
    # A stimulation electrode may have been swapped for a neighbour to get its own unit; the
    # regions, and so the protocol written into every recording, say what was actually driven.
    for role, spec in regions.items():
        for kind in ("drive", "return"):
            before, after = list(getattr(spec, f"{kind}_electrodes")), used.get(f"{role}_{kind}", [])
            if after and after != before:
                setattr(spec, f"{kind}_electrodes", list(after))
                swapped = [(a, b) for a, b in zip(before, after) if a != b]
                on_progress(
                    f"  {role} {kind} electrodes adjusted for stimulation units: "
                    + ", ".join(f"{a} -> {b}" for a, b in swapped)
                )

    from mxtreme.stimulation.timeline import RegionUnits

    region_units = {
        role: RegionUnits(
            role,
            spec.drive_dac,
            units[f"{role}_drive"],
            spec.amplitude_mv,
            spec.return_dac,
            units.get(f"{role}_return", []),
        )
        for role, spec in regions.items()
    }
    registered = {}
    for stim in stimuli.values():
        seq, seconds = sequences.build_region_sequence(
            f"{stim.token}_{well}",
            well,
            [region_units[r] for r in stim.roles],
            stim.delays_sec,
            stim.num_events,
            stim.pulse_hz,
            params.stim_phase_us,
            amplitude_mv=stim.amplitude_mv,
            pulses_per_burst=stim.pulses_per_burst,
            burst_hz=stim.burst_hz,
            polarity=stim.polarity,
        )
        registered[stim.token] = seq
        on_progress(f"  sequence {stim.token}: {seconds:.3f} s")

    mx.activate([well])
    mx.offset()
    sleep(mx.Timing.waitInMX2Offset)

    def fire(token):
        registered[token].send()

    def mark(label):
        # `<block>_start <well>`, the key report looks for (and the scans' convention:
        # pre_recording_start, closed_loop_start); end_experiment is already its own key.
        tag = label if label == "end_experiment" else f"{label}_start"
        sequences.mark(well, f"{tag} {well}")

    # --- one recording per phase ---
    results: list[RunResult] = []
    try:
        for k, p in enumerate(per_phase):
            if len(per_phase) > 1:
                on_progress(f"\n=== {p.phase} ({k + 1} of {len(per_phase)}) ===")
            blocks, mine = schedules[p.phase]
            result = _record_phase(
                p,
                blocks,
                mine,
                regions,
                units,
                routed,
                mx,
                mx_setup,
                fire,
                mark,
                registry_path,
                work,
                on_progress,
                should_stop,
                clock,
                sleep,
            )
            results.append(result)
            if result.stopped_early:
                on_progress(f"stopped during {p.phase}; {len(per_phase) - k - 1} phase(s) not recorded")
                break
    finally:
        release(mx, well, list(registered))
    results[-1].phases = results
    results[-1].stim_electrodes = {role: list(spec.drive_electrodes) for role, spec in regions.items()}
    results[-1].routing_cfg = routing_cfg if routing_cfg and Path(routing_cfg).exists() else None
    if len(results) > 1:
        on_progress(
            f"session: {len(results)} recordings\n  "
            + "\n  ".join(str(r.h5_path) for r in results)
            + "\n  read them together: report "
            + " ".join(str(r.h5_path) for r in results)
        )
    return results[-1]


def release(mx, well: int, tokens) -> None:
    """Delete this run's objects from the MaxLab server: the array ``init_well`` made and every
    sequence that was built.

    Both are created ``persistent=True`` so they outlive the Python objects, which a run needs
    while it fires them -- but nothing deletes them afterwards, and an ``Array`` on the server
    costs seconds to create and real memory to keep. Left behind, one accumulates per session.
    Deleting is the ``close`` of a non-persistent object with the same token.
    """
    for token in tokens:
        try:
            # A Sequence deletes its server-side twin in shutdown() (and __del__) when not persistent.
            mx.Sequence(f"{token}_{well}", persistent=False, initial_delay=0).shutdown()
        except Exception as e:  # noqa: BLE001 -- clean-up; the recording is already safe
            print(f"could not delete sequence {token}_{well}: {e}")
    try:
        mx.Array(f"stimulation{well}", persistent=False).close()
    except Exception as e:  # noqa: BLE001
        print(f"could not delete array stimulation{well}: {e}")


def _record_phase(
    params,
    blocks,
    stimuli,
    regions,
    units,
    routed,
    mx,
    mx_setup,
    fire,
    mark,
    registry_path,
    work,
    on_progress,
    should_stop,
    clock=time.monotonic,
    sleep=time.sleep,
) -> RunResult:
    """Open one recording, write its metadata and protocol into it, run its schedule, close it."""
    from mxtreme import io

    well = params.well
    s = mx.Saving()
    s.open_directory(params.save_path)
    s.start_file(params.file_name)
    s.group_delete_all()
    mx_setup.write_metadata(s, params.as_metadata())
    mx_setup.write_exp_description(s, params.description)
    mx_setup.write_stim_electrodes(s, [e for spec in regions.values() for e in spec.stim_electrodes])
    channels = raw_channels(params, regions, routed)
    if channels:
        s.group_define(well, f"raw_{params.raw_traces}_{well}", channels)
    on_progress(f"keeping spikes on every channel and raw traces on {len(channels)} ({params.raw_traces!r})")

    record = protocol.schedule_record(params, blocks, stimuli, regions, {"units": units})
    text = json.dumps(record, separators=(",", ":"))
    reply = s.write_assay_property(PROTOCOL_KEY, text)
    embedded = str(reply).strip().lower() != "error"

    protocol_copy = fired_copy = None
    if work is not None:
        Path(work).mkdir(parents=True, exist_ok=True)
        protocol_copy = Path(work) / f"{params.file_name}_protocol.json"
        protocol_copy.write_text(json.dumps(record, indent=2))
    if not embedded:
        if protocol_copy is None:
            s.stop_file()
            raise RuntimeError(
                "the protocol could not be written into the recording, and there is no work "
                "directory to keep it in; without it the recording cannot be read back. Nothing "
                "was recorded. Re-run with --work <dir>."
            )
        on_progress(f"WARNING: the protocol is not inside the recording; keep {protocol_copy}")

    started_at = time.time()
    s.start_recording([well])
    on_progress(f"recording to {params.h5_path}")
    try:
        fired, stopped = execute_schedule(
            blocks, fire, mark, clock=clock, sleep=sleep, should_stop=should_stop, on_progress=on_progress
        )
    finally:
        s.stop_recording()
        sleep(mx.Timing.waitAfterRecording)
        s.stop_file()
        s.group_delete_all()

    from mxtreme.scans.activity_scan import _saved_file

    h5_path = _saved_file(params, started_at)
    stem = h5_path.name.split(".raw.h5")[0]
    if stem != params.file_name:
        # MaxLab does not overwrite: asked for a name that exists it appends _1, _2 and so on.
        on_progress(f"note: {params.file_name} was taken, so this run is {stem}")

    if work is not None:
        # Named after the recording actually written, so a copy can never describe another run.
        if protocol_copy is not None and stem != params.file_name:
            protocol_copy = protocol_copy.rename(protocol_copy.with_name(f"{stem}_protocol.json"))
        fired_copy = Path(work) / f"{stem}_fired.csv"
        with open(fired_copy, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["block", "token", "planned_sec", "actual_sec"])
            w.writerows(fired)

    if registry_path is not None:
        io.register_scan(params, registry_path, kind="experiment")
    on_progress(f"{'stopped early' if stopped else 'done'}: {len(fired)} presentation(s); {h5_path}")
    out = Path(work) if work is not None else h5_path.parent
    on_progress(
        f"read it back: python -m mxtreme.experiments.associative report {h5_path} "
        f"-o {out / stem}.png --csv {out / stem}.csv"
        + (" --apply <params.json>" if params.mode == "calibration" else "")
    )
    return RunResult(Path(h5_path), fired, stopped, protocol_copy, fired_copy)
