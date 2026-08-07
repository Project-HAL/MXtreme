"""Topic-oriented analyses, one module per topic.

Each module exposes a culture-level function (operates on one `CulturePaths`, writes a tidy
per-DIV CSV) and, where a population aggregate is meaningful, a population-level function that
reads those CSVs back. See `claude_configs/analysis_protocol.md`.

Submodules are not imported here -- each pulls in matplotlib. Import what you need:

    from mxtreme.analysis import activity, spatial, stimulation, learning
"""
