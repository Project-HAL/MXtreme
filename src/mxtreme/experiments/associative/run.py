"""Running the associative protocol on the rig, open loop.

One process, one well: route the electrodes, register one sequence per kind of presentation, open
the recording, then walk the schedule firing each presentation at its time and marking each
block's start with an event. The recording lands in the managed store like a scan does and is
registered as an ``experiment``. Beside it go ``<stem>_protocol.json`` (everything that was
decided, which :mod:`.report` reads) and ``<stem>_fired.csv`` (when each presentation was actually
sent).

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
from dataclasses import dataclass
from pathlib import Path

from mxtreme.experiments.associative import protocol
from mxtreme.experiments.associative.params import AssociativeParams

ProgressFn = Callable[[str], None]
StopFn = Callable[[], bool]

_POLL_SEC = 0.25


@dataclass
class RunResult:
    h5_path: Path
    protocol_path: Path
    fired_path: Path
    fired: list[tuple[str, str, float, float]]  # (block, token, planned_sec, actual_sec)
    stopped_early: bool


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


def run(
    params: AssociativeParams,
    config=None,
    *,
    on_progress: ProgressFn = print,
    should_stop: StopFn | None = None,
) -> RunResult:
    """Run the protocol on the rig and record it.

    :param params: With ``regions``, ``rec_electrodes`` and ``amplitudes_mv`` set.
    :param config: The :class:`~mxtreme.config.Config` naming the store, unless ``params.save_path``
        says where to write instead.
    :param should_stop: Polled while waiting; return ``True`` to end the run early. The recording
        is closed properly either way.
    :raises ModuleNotFoundError: If ``maxlab`` is not installed (this is not the rig).
    """
    from mxtreme import io
    from mxtreme.scans import mx_setup
    from mxtreme.scans.activity_scan import _require_maxlab
    from mxtreme.stimulation import sequences

    mx = _require_maxlab()
    params.validate(for_run=True)
    registry_path = config.registry_path if config is not None and params.save_path is None else None
    params = params.resolved(config)
    Path(params.save_path).mkdir(parents=True, exist_ok=True)
    well = params.well

    blocks, stimuli = protocol.build_schedule(params)
    regions = protocol.regions_for(params)
    on_progress("=== Associative run ===")
    on_progress(protocol.summary(blocks, stimuli, params.stim_phase_us))
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

    # --- the chip ---
    on_progress("Initializing chip...")
    mx.initialize()
    if mx.send(mx.Core().enable_stimulation_power(True)) != "Ok":
        raise RuntimeError("the system did not enable stimulation power")
    time.sleep(mx.Timing.waitInit)
    mx.clear_events()

    groups = {}
    for role, spec in regions.items():
        groups[f"{role}_drive"] = spec.drive_electrodes
        if spec.return_electrodes:
            groups[f"{role}_return"] = spec.return_electrodes
    array, units = sequences.init_well_regions(well, list(params.rec_electrodes), groups)
    cfg_path = Path(params.save_path) / f"{params.file_name}.cfg"
    array.save_config(str(cfg_path))
    on_progress(f"routed; config saved to {cfg_path}")

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
    time.sleep(mx.Timing.waitInMX2Offset)

    # --- the recording ---
    s = mx.Saving()
    s.open_directory(params.save_path)
    s.start_file(params.file_name)
    s.group_delete_all()
    mx_setup.write_metadata(s, params.as_metadata())
    mx_setup.write_exp_description(s, params.description)
    mx_setup.write_stim_electrodes(s, [e for spec in regions.values() for e in spec.stim_electrodes])
    s.group_define(well, f"all_channels_{well}", list(range(1024)))

    protocol_path = Path(params.save_path) / f"{params.file_name}_protocol.json"
    with open(protocol_path, "w") as f:
        json.dump(
            protocol.schedule_record(
                params, blocks, stimuli, regions, {"config": str(cfg_path), "units": units}
            ),
            f,
            indent=2,
        )

    started_at = time.time()
    s.start_recording([well])
    on_progress(f"recording to {params.h5_path}")

    def fire(token):
        registered[token].send()

    def mark(label):
        sequences.mark(well, f"{label} {well}")

    try:
        fired, stopped = execute_schedule(
            blocks, fire, mark, should_stop=should_stop, on_progress=on_progress
        )
    finally:
        s.stop_recording()
        time.sleep(mx.Timing.waitAfterRecording)
        s.stop_file()
        s.group_delete_all()

    fired_path = Path(params.save_path) / f"{params.file_name}_fired.csv"
    with open(fired_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "token", "planned_sec", "actual_sec"])
        w.writerows(fired)

    from mxtreme.scans.activity_scan import _saved_file

    h5_path = _saved_file(params, started_at)
    # MaxLab does not overwrite: asked for a name that exists it appends _1, _2 and so on. The
    # protocol and fired files have to follow, or a second run into the same directory would
    # leave the first recording described by the second run's protocol -- and `report` would
    # read the wrong schedule without any sign that it had.
    stem = h5_path.name.split(".raw.h5")[0]
    if stem != params.file_name:
        for path, tail in ((protocol_path, "_protocol.json"), (fired_path, "_fired.csv")):
            path.rename(path.with_name(f"{stem}{tail}"))
        protocol_path = protocol_path.with_name(f"{stem}_protocol.json")
        fired_path = fired_path.with_name(f"{stem}_fired.csv")
        on_progress(f"note: {params.file_name} was taken, so this run is {stem}")
    if registry_path is not None:
        io.register_scan(params, registry_path, kind="experiment")
    on_progress(f"{'stopped early' if stopped else 'done'}: {len(fired)} presentation(s); {h5_path}")
    return RunResult(Path(h5_path), protocol_path, fired_path, fired, stopped)
