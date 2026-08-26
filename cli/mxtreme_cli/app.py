"""Top-level menu: the three things the CLI can do.

Screens are imported lazily, one per menu entry, so a machine that is missing a screen's
dependencies (matplotlib for electrode selection, ``maxlab`` on the rig side) can still start the CLI
and use the screens that do work.
"""

from __future__ import annotations

from mxtreme import __version__
from mxtreme_cli import prompts, ui

TITLE = "MXtreme"
SUBTITLE = "HD-MEA scans, electrode selection and analysis"


def _activity_scan() -> None:
    from mxtreme_cli import activity_scan

    activity_scan.screen()


def _electrode_selection() -> None:
    from mxtreme_cli import electrode_selection

    electrode_selection.screen()


def _data_analysis() -> None:
    from mxtreme_cli import data_analysis

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
    ui.banner(f"{TITLE} {__version__}", SUBTITLE)

    while True:
        try:
            choice = prompts.menu(MAIN_MENU, prompt="Select an option", back_label="Quit")
        except prompts.Cancelled:
            choice = None

        if choice is None:
            ui.info("\nBye.")
            return 0

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

        ui.banner(TITLE, SUBTITLE)
