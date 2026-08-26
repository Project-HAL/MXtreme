"""Data Analysis screen -- placeholder.

This will front the offline half of the package (:mod:`mxtreme.extract`, :mod:`mxtreme.clean`,
:mod:`mxtreme.bursting`, :mod:`mxtreme.analysis`) for arbitrary ``.h5`` recordings, rather than the
activity-scan-specific path the other two screens cover. Nothing here calls into those modules yet.
"""

from __future__ import annotations

from mxtreme_cli import prompts, ui

#: What this screen is expected to grow into, shown so the stub is self-documenting.
PLANNED = [
    "Extract wells from an .h5 and clean them into the managed store (mxtreme.extract / clean)",
    "Run burst detection over cleaned recordings (mxtreme.bursting)",
    "Activity, spatial and stimulation analyses (mxtreme.analysis)",
    "Build a report for a culture or an experiment (mxtreme.analysis.report)",
]


def screen() -> None:
    """Show the placeholder for the not-yet-implemented analysis screen."""
    ui.banner("Data Analysis", "Analyse arbitrary .h5 recordings")
    ui.warn("Not implemented yet.")
    ui.info("\nPlanned:")
    for item in PLANNED:
        ui.bullet(item)
    ui.hint(
        "\nUntil this lands, the analysis modules are driven from Python directly -- "
        "see the notebooks under examples/."
    )
    prompts.pause("Press Enter to return to the main menu")
