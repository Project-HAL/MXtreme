"""Top-level menu: the three things the CLI can do, plus whatever is running in the background.

The screens themselves live in :mod:`mxtreme_cli.screens`, one module per entry. They are
imported lazily, so a machine that is missing a screen's dependencies (matplotlib for electrode
selection, ``maxlab`` on the rig side) can still start the CLI and use the screens that do work.

An activity scan runs on its own thread (:mod:`mxtreme_cli.jobs`) rather than holding the terminal
for the length of the scan. That makes this module responsible for two things a single-threaded menu
never had to think about: showing the user what is running, and refusing to quit out from under it.
"""

from __future__ import annotations

import time

from mxtreme import __version__
from mxtreme_cli import jobs, prompts, ui

TITLE = "MXtreme"
SUBTITLE = "HD-MEA scans, electrode selection and analysis"

#: How often the main menu repaints while a scan is running, in seconds. Fast enough that the bar
#: looks live, slow enough that an idle terminal is not being rewritten constantly.
REFRESH_SEC = 1.0


def _activity_scan() -> None:
    from mxtreme_cli.screens import activity_scan

    activity_scan.screen()


def _electrode_selection() -> None:
    from mxtreme_cli.screens import electrode_selection

    electrode_selection.screen()


def _data_analysis() -> None:
    from mxtreme_cli.screens import data_analysis

    data_analysis.screen()


MAIN_MENU = [
    (_activity_scan, "Activity scan", "sweep the array on the rig"),
    (_electrode_selection, "Electrode selection", "requires an activity scan .h5"),
    (_data_analysis, "Data analysis", "arbitrary .h5 recordings"),
]


def main() -> int:
    """Run the menu loop until the user quits.

    :returns: A process exit code.
    """
    ui.PRODUCT = f"{TITLE} {__version__}"
    ui.status_provider = jobs.status_lines  # every banner now carries the running scans

    ui.banner("Main menu", SUBTITLE)

    while True:
        _report_finished()  # anything that ended while a screen was open

        try:
            choice = prompts.menu(
                MAIN_MENU,
                prompt="Select an option",
                back_label="Quit",
            )
        except prompts.Cancelled:
            choice = None

        # The menu blocks for as long as the user leaves it up, which is exactly when a background
        # scan is likely to end. Report before acting on the choice, so its results are not held
        # back behind whichever screen they open next.
        _report_finished()

        if choice is None:
            if _ready_to_quit():
                ui.info("\nBye.")
                return 0
            ui.banner("Main menu", SUBTITLE)
            continue

        try:
            choice()
        except prompts.Cancelled:
            ui.warn("Cancelled.")
        except ImportError as exc:
            # A screen whose dependencies are not installed on this machine should not take the
            # whole CLI down with it.
            ui.error(f"That screen is unavailable here: {exc}")
        except Exception as exc:  # noqa: BLE001 -- a screen crash must not kill the session
            ui.error(f"Unexpected error: {type(exc).__name__}: {exc}")
            ui.hint("The menu is still usable; re-run with MXTREME_CLI_TRACEBACK=1 for a traceback.")
            import os

            if os.environ.get("MXTREME_CLI_TRACEBACK"):
                import traceback

                traceback.print_exc()

        ui.banner("Main menu", SUBTITLE)


def _report_finished() -> None:
    """Show the outcome of any scan that ended while the user was somewhere else.

    A background scan finishing is easy to miss, and its result is the whole point of running it, so
    it is reported at the first opportunity rather than left for the user to go looking for.
    """
    pending = [job for job in jobs.finished() if not job.reported]
    if not pending:
        return

    from mxtreme_cli.screens import activity_scan

    for job in pending:
        activity_scan.report_job(job)


def _ready_to_quit() -> bool:
    """Decide whether the CLI can exit, given whatever is still running.

    :returns: ``True`` if nothing is running, or the user waited for it.
    """
    running = jobs.active()
    if not running:
        return True

    ui.banner("Quit", "A scan is still running")
    ui.warn(f"{len(running)} activity scan(s) have not finished:")
    for job in running:
        ui.bullet(job.status_line())
    ui.hint(
        "\nThe recording in progress has to end before the .h5 can be closed, so quitting now "
        "would leave the file unreadable. Both options below finish it properly first."
    )

    try:
        choice = prompts.menu(
            [
                ("wait", "Wait for the scan to finish", "leaves every recording intact"),
                ("stop", "Stop after the current recording", "keeps what has been recorded so far"),
            ],
            prompt="What now",
            back_label="Back to the menu (leave it running)",
        )
    except prompts.Cancelled:
        return False

    if choice is None:
        return False
    if choice == "stop":
        for job in running:
            job.stop()
        ui.info("\nStopping after the current recording...")

    return _wait_for(running)


def _wait_for(running: list[jobs.ScanJob]) -> bool:
    """Block until every job ends, repainting its status line as it goes.

    :returns: ``True`` once all of them ended, ``False`` if the user interrupted the wait -- the
        scans keep running in that case, so it is a return to the menu, not a cancellation.
    """
    ui.section("Waiting")
    drawn = False
    try:
        while True:
            if drawn:
                ui.cursor_up(len(running))
            for job in running:
                ui.info(ui.CLEAR_LINE + "  " + job.status_line())
            drawn = True

            if all(job.state.done for job in running):
                _report_finished()
                return True
            time.sleep(REFRESH_SEC)
    except KeyboardInterrupt:
        print()
        ui.warn("Still running. Back to the menu.")
        return False
