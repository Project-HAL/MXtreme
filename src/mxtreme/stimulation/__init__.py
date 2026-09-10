"""Rig-side stimulation: pulse timelines, region sequences, and stimulation-unit routing.

Deliberately imports nothing. :mod:`mxtreme.stimulation.timeline` is pure Python -- a timeline is
built and inspected anywhere -- while :mod:`mxtreme.stimulation.sequences` turns one into MaxLab
commands and needs the proprietary ``maxlab`` API, imported on first use rather than at module
import, the way :mod:`mxtreme.scans.activity_scan` does. Import what you need::

    from mxtreme.stimulation import timeline     # offline -- no maxlab
    from mxtreme.stimulation import sequences    # rig only on use -- requires maxlab
"""
