"""Thin drivers that run the preprocessing pipeline: raw .h5 -> npz -> bursts -> burst features.

These wrap `mxtreme.etl` and `mxtreme.recording` and are what establishes the on-disk layout that
`mxtreme.paths` and `mxtreme.analysis` expect.

Planned: collapse to a single `mxtreme/pipeline.py` with CLI entry points, alongside the config
work (`claude_configs/structure_feedback.md` §1c).
"""
