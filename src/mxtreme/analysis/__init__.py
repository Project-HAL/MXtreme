"""Topic-oriented analyses plus a PDF report assembler.

Each topic module exposes a culture-level function (operates on one ``CulturePaths``, writes a tidy
per-DIV CSV under ``config.analysis_dir``) and, where meaningful, a population-level function that
reads those CSVs back:

    from mxtreme.analysis import activity, spatial, stimulation, performance

The high-level entry point assembles a multi-section PDF report for a single culture or a group:

    from mxtreme.analysis import generate_report

Importing this package pulls in matplotlib (every submodule plots), so only import it when you intend
to run analyses/plots -- not on the fast ``import mxtreme`` path.
"""

from mxtreme.analysis import activity, performance, spatial, stimulation
from mxtreme.analysis.report import DEFAULT_SECTIONS, generate_report

__all__ = [
    "activity",
    "performance",
    "spatial",
    "stimulation",
    "generate_report",
    "DEFAULT_SECTIONS",
]
