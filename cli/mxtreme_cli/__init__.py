"""Menu-driven front end for MXtreme.

A thin layer over the package: every screen collects parameters and then calls into
:mod:`mxtreme`, so nothing scientific lives here. :mod:`~mxtreme_cli.app` holds the main menu,
:mod:`~mxtreme_cli.screens` the screens it dispatches to, and :mod:`~mxtreme_cli.prompts` /
:mod:`~mxtreme_cli.ui` the terminal plumbing they share. See ``cli/README.md`` for how to run it.
"""

__all__ = ["app", "prompts", "screens", "ui"]
