"""Activity Scan screen: choose a scan protocol, review its parameters, run it on the rig.

Each protocol is one entry in :data:`SCANS`. A protocol is just a name, a description and a factory
for a :class:`~mxtreme.scans.activity_scan.ActivityScanParams` -- adding another one is a single
entry here, not a new screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from mxtreme.scans import activity_scan as scan
from mxtreme.scans.activity_scan import ActivityScanParams
from mxtreme_cli import electrode_selection as selection_screen
from mxtreme_cli import prompts, ui
from mxtreme_cli.prompts import Param


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
    """Run the Activity Scan screen: pick a protocol, then run it."""
    while True:
        ui.banner("Activity Scan", "Sweep the array to find where the culture is firing")
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

        _run_protocol(choice)


def _run_protocol(protocol: ScanProtocol) -> None:
    """Review the protocol's parameters, run the scan, then offer to select electrodes from it."""
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
            return

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
        ui.warn("The chip will be initialized and recorded from. Make sure the plate is ready.")

        if not prompts.confirm("\nStart the scan?", default=True):
            ui.warn("Cancelled.")
            return

    except prompts.Cancelled:
        ui.warn("Cancelled.")
        return

    result = _run_scan(params, seed)
    if result is None:
        prompts.pause()
        return

    _report(result, seed)

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


def _run_scan(params: ActivityScanParams, seed: int | None) -> scan.ActivityScanResult | None:
    """Run the scan, turning the rig's failure modes into readable messages.

    :returns: The result, or ``None`` if the scan could not run.
    """
    ui.section("Scanning")
    try:
        return scan.run_activity_scan(params, seed=seed, on_progress=ui.info)
    except ModuleNotFoundError as exc:
        ui.error("This machine cannot run a scan: MaxWell's 'maxlab' API is not installed.")
        ui.hint(str(exc))
        return None
    except KeyboardInterrupt:
        # run_activity_scan() finalizes the .h5 in its own `finally`, so whatever was recorded before
        # the interrupt is already on disk and readable.
        ui.error("\nInterrupted. The file was finalized; recordings completed so far are saved.")
        return None
    except (RuntimeError, ValueError, OSError) as exc:
        ui.error(f"Scan failed: {type(exc).__name__}: {exc}")
        return None


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
