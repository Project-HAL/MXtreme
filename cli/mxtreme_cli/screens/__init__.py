"""The CLI's screens: one module per entry in the main menu.

A screen owns a whole interaction -- it collects its own parameters, calls into :mod:`mxtreme`, and
reports what happened. Screens are imported lazily by :mod:`mxtreme_cli.app`, so a machine missing
one screen's dependencies (``matplotlib`` for electrode selection, ``maxlab`` on the rig side) can
still start the CLI and use the rest. Importing them here eagerly would undo that, so this module
deliberately imports nothing.
"""

#: Screen modules, in menu order. Names only -- see the note above about lazy imports.
__all__ = ["activity_scan", "electrode_selection", "data_analysis"]
