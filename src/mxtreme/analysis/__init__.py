"""Topic-oriented analyses plus a PDF report assembler.

Each topic module works at the same three levels:

- recording level: functions take a ``Recording`` and return its data;
- group level: ``summarize_*`` / ``plot_*_summary`` take a ``RecordingID`` / ``CultureID`` /
  ``CultureSelector`` (or several selectors) plus the ``Config``, and cache per-culture CSVs under
  ``config.analysis_dir``;
- culture level: plots of one culture's recordings side by side.

    from mxtreme.analysis import activity, network, spatial, stimulation, performance

The high-level entry point assembles a multi-section PDF report for a single culture or a group:

    from mxtreme.analysis import generate_report

Importing this package pulls in matplotlib (every submodule plots), so only import it when you intend
to run analyses/plots -- not on the fast ``import mxtreme`` path.
"""

from mxtreme.analysis import activity, network, performance, spatial, stimulation
from mxtreme.analysis.report import DEFAULT_SECTIONS, generate_report

__all__ = [
    "activity",
    "network",
    "performance",
    "spatial",
    "stimulation",
    "generate_report",
    "DEFAULT_SECTIONS",
]
