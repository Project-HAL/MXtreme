"""Rig-side acquisition: activity and network scans, electrode selection, and MaxLab setup.

Deliberately imports nothing. :mod:`mxtreme.scans.mx_setup` needs the proprietary ``maxlab`` API that
ships with MaxWell's MaxLab Live software, so re-exporting it here would make ``import mxtreme.scans``
fail on any machine that is not the rig. Import what you need::

    from mxtreme.scans import activity_scan        # planning offline; running needs maxlab
    from mxtreme.scans import network_scan          # planning offline; running needs maxlab
    from mxtreme.scans import electrode_selection   # offline -- no maxlab
    from mxtreme.scans import mx_config             # offline -- no maxlab
    from mxtreme.scans import mx_setup              # rig only -- requires maxlab

This mirrors the policy in ``mxtreme`` itself, where the heavy submodules stay behind explicit
imports.
"""
