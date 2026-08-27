"""Activity Scan screen: choose a scan protocol, review its parameters, start it on the rig.

Each protocol is one entry in :data:`SCANS`. A protocol is just a name, a description and a factory
for a :class:`~mxtreme.scans.activity_scan.ActivityScanParams` -- adding another one is a single
entry here, not a new screen.

Scans run in the background (:mod:`mxtreme_cli.jobs`). Starting one hands the user straight back to
the main menu with the scan's progress in the banner, so a long sweep no longer costs them the
terminal. The consequence is that this screen no longer sees a scan through: :func:`report_job` is
called later, by whichever menu notices the job has ended.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from mxtreme.scans import activity_scan as scan
from mxtreme.scans.activity_scan import ActivityScanParams
from mxtreme_cli import jobs, prompts, ui
from mxtreme_cli.prompts import Param
from mxtreme_cli.screens import electrode_selection as selection_screen


@dataclass(frozen=True)
class ScanProtocol:
    """One selectable scan protocol.

    :param key: Stable identifier, used internally.
    :param name: Name shown in the menu.
    :param summary: One-line description shown beside the name.
    :param detail: Longer explanation shown once the protocol is chosen.
    :param defaults: Builds the protocol's starting parameters.
    """

    key: str
    name: str
    summary: str
    detail: str
    defaults: Callable[[], ActivityScanParams]


#: The scan protocols on offer. Add new protocols here.
SCANS: list[ScanProtocol] = [
    ScanProtocol(
        key="kam",
        name="Kam Scan",
        summary="random-subset sweep of the array",
        detail=(
            "Covers the array with a series of short recordings, each routing a different random\n"
            "subset of the available electrodes. Subsets are drawn without replacement, so the\n"
            "recordings of a well do not overlap until the array has been covered once. All\n"
            "selected wells are recorded simultaneously and land in a single .h5 file."
        ),
        defaults=ActivityScanParams,
    ),
]


PARAM_FIELDS = [
    Param("chip", "Chip ID", prompts.parse_text, help="Chip serial, e.g. M07460"),
    Param("plate_date", "Plate date", prompts.parse_int(0), help="YYMMDD"),
    Param("div", "DIV", prompts.parse_int(0), help="Days in vitro"),
    Param("exp_id", "Experiment ID", prompts.parse_text, help="Task name stored in the file"),
    Param(
        "wells",
        "Wells",
        prompts.parse_int_list(0, scan.MAX_WELLS - 1),
        format=lambda v: ", ".join(str(w) for w in v),
        help=f"Comma-separated, 0..{scan.MAX_WELLS - 1}; recorded simultaneously",
    ),
    Param(
        "conditions",
        "Conditions",
        prompts.parse_str_list,
        format=lambda v: ", ".join(v) if v else "(none)",
        help="Optional per-well labels, one per well",
        allow_blank=True,
    ),
    Param("description", "Description", prompts.parse_text, help="Free text stored in the file"),
    Param("n_scans", "Number of scans", prompts.parse_int(1, 200), help="Recordings in the sweep"),
    Param("rec_length_sec", "Recording length", prompts.parse_int(1, 3600), help="Seconds per recording"),
    Param(
        "electrode_spacing",
        "Electrode spacing",
        prompts.parse_int(0, 20),
        help="0 = every electrode eligible, 1 = every other, ...",
    ),
    Param(
        "electrodes_per_scan",
        "Electrodes per scan",
        prompts.parse_int(1, scan.MAX_ROUTED_ELECTRODES),
        help=f"Per well per recording; max {scan.MAX_ROUTED_ELECTRODES}",
    ),
    Param("save_path", "Save directory", prompts.parse_dir, help="Created if it does not exist"),
]


def screen() -> None:
    """Run the Activity Scan screen: pick a protocol, then start it in the background."""
    while True:
        ui.banner("Activity Scan", "Sweep the array to find where the culture is firing")

        if jobs.active() and not _confirm_second_scan():
            return

        try:
            choice = prompts.menu(
                [(protocol, protocol.name, protocol.summary) for protocol in SCANS],
                prompt="Select a scan",
                back_label="Back to main menu",
            )
        except prompts.Cancelled:
            return

        if choice is None:
            return

        if _run_protocol(choice):
            return  # a scan is now running; the main menu is where it is watched


def _confirm_second_scan() -> bool:
    """Warn that a scan is already running and ask whether to start another anyway.

    :returns: ``True`` to carry on to the protocol list, ``False`` to go back.
    """
    running = jobs.active()
    ui.warn(f"{len(running)} activity scan(s) already running:")
    for job in running:
        ui.bullet(job.status_line())
    ui.hint(
        "\nOne rig drives one chip. A second scan started now shares that chip with the first: "
        "both will route electrodes and record through the same MaxLab server, and neither will "
        "record what its parameters say. Unless the first scan is on its way out, going back and "
        "waiting is what you want."
    )

    try:
        choice = prompts.menu(
            [
                ("new", "Start a new activity scan", "runs alongside the one above"),
                ("stop", "Stop the running scan first", "ends it after the current recording"),
            ],
            prompt="What now",
            back_label="Back to main menu",
        )
    except prompts.Cancelled:
        return False

    if choice is None:
        return False
    if choice == "stop":
        for job in running:
            job.stop()
        ui.info("\nStopping after the current recording; the file will be finalized.")
        prompts.pause()
        return False
    return True


def _run_protocol(protocol: ScanProtocol) -> bool:
    """Review the protocol's parameters and start the scan.

    :returns: ``True`` if a scan was started, ``False`` if the user backed out.
    """
    ui.banner(protocol.name, protocol.summary)
    ui.block(protocol.detail)

    params = protocol.defaults()

    try:
        if not prompts.edit_params(
            params,
            PARAM_FIELDS,
            title=f"{protocol.name} parameters",
            validate=lambda p: p.validate(),
        ):
            ui.warn("Cancelled.")
            return False

        ui.hint("\nA seed makes the electrode subsets reproducible across runs.")
        seed = prompts.ask(
            "Random seed (blank for none)", parse=prompts.parse_optional_int, allow_blank=True
        )

        ui.section("Ready to run")
        ui.block(scan.describe(params))
        ui.info("")
        if params.array_coverage < 1:
            ui.warn(
                f"This scan reaches {params.array_coverage:.0%} of the available electrodes. "
                "Raise the number of scans, or the spacing, for full coverage."
            )
        _warn_about_clashes(params)
        ui.warn("The chip will be initialized and recorded from. Make sure the plate is ready.")

        if not prompts.confirm("\nStart the scan?", default=True):
            ui.warn("Cancelled.")
            return False

    except prompts.Cancelled:
        ui.warn("Cancelled.")
        return False

    job = jobs.start(params, seed)
    ui.success(f"\nStarted: {job.name}")
    ui.info(f"Estimated {params.estimated_minutes:.1f} min for {params.n_scans} recordings.")
    ui.hint(
        "It runs in the background. Progress is on the main menu, and the results appear there "
        "as soon as it finishes."
    )
    prompts.pause("Press Enter to return to the main menu")
    return True


def _warn_about_clashes(params: ActivityScanParams) -> None:
    """Say what will happen to an existing file of the same name, before anything is written."""
    clashing = jobs.clashing(params)
    if clashing:
        ui.warn(
            f"A running scan is already writing {params.file_name}. MaxLab never overwrites an "
            "open file: this one will be saved as a numbered variant instead."
        )
        return

    if params.h5_path.exists():
        ui.warn(
            f"{params.h5_path} already exists. MaxLab does not overwrite it -- the new scan is "
            "saved alongside it with _1, _2 ... appended. Reuse the exact name only if that is "
            "what you want; the old file stays either way."
        )


def report_job(job: jobs.ScanJob) -> None:
    """Show what a background scan produced, once it has ended.

    Called from the menu that notices the job finished, not from this screen, since by then the user
    is somewhere else entirely.
    """
    job.reported = True

    if job.state is jobs.State.FAILED:
        _report_failure(job)
        return

    result = job.result
    if result is None:  # defensive: a finished job always has one
        ui.error(f"Scan {job.name} ended without a result.")
        return

    _report(result, job.seed)

    try:
        h5_path = _resolve_h5(result)
        if h5_path is not None and prompts.confirm(
            "\nSelect recording electrodes from this scan now?", default=True
        ):
            selection_screen.screen(h5_path)
            return
    except prompts.Cancelled:
        pass

    prompts.pause()


def _report_failure(job: jobs.ScanJob) -> None:
    """Explain a background scan that raised, in the terms the foreground would have used."""
    exc = job.error
    ui.section(f"Scan failed: {job.name}")

    if isinstance(exc, ModuleNotFoundError):
        ui.error("This machine cannot run a scan: MaxWell's 'maxlab' API is not installed.")
        ui.hint(str(exc))
    elif isinstance(exc, RuntimeError) and str(exc).startswith("No MaxWell device"):
        # The device check fails this way; it is a plain message, not a defect worth a type name.
        ui.error(str(exc))
        ui.hint("Nothing was recorded. Fix that and start the scan again.")
    else:
        ui.error(f"{type(exc).__name__}: {exc}")

    tail = job.log()[-8:]
    if tail:
        ui.info("\nLast output before it stopped:")
        ui.block("\n".join(tail))
    prompts.pause()


def _report(result: scan.ActivityScanResult, seed: int | None) -> None:
    """Summarize a finished scan and record which electrodes each recording covered."""
    ui.section("Scan complete" if result.complete else "Scan ended early")

    params = result.params
    ui.key_values(
        [
            ("File", str(result.h5_path)),
            ("Recordings", f"{result.completed_scans} of {params.n_scans}"),
            ("Duration", f"{result.duration_sec / 60:.1f} min"),
            ("Wells", ", ".join(str(w) for w in params.wells)),
            ("Seed", "none" if seed is None else str(seed)),
        ]
    )

    # Which electrodes each recording covered is not stored in the .h5 in a form that survives the
    # scan, so keep it beside the file: it is the only record of what was *attempted* per round.
    manifest = Path(params.save_path) / f"{params.file_name}_scan_electrodes.npz"
    try:
        np.savez(
            manifest,
            **{f"well{well}": np.array(scans) for well, scans in result.scan_electrodes.items()},
        )
        ui.info("")
        ui.success(f"Routed-electrode record saved to {manifest}")
    except (OSError, ValueError) as exc:
        ui.warn(f"Could not write the routed-electrode record: {exc}")


def _resolve_h5(result: scan.ActivityScanResult) -> Path | None:
    """Find the file the scan actually produced.

    The expected name follows MaxLab's ``<name>.raw.h5`` convention, but fall back to the newest
    ``.h5`` in the save directory rather than giving up if the software named it differently.
    """
    if result.h5_path.is_file():
        return result.h5_path

    candidates = sorted(
        Path(result.params.save_path).glob("**/*.h5"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        ui.warn(f"No .h5 found under {result.params.save_path}.")
        return None

    ui.warn(f"Expected {result.h5_path}, found {candidates[0]} instead.")
    return candidates[0]
