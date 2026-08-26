"""Menu-driven front end for MXtreme.

A thin layer over the package: every screen collects parameters and then calls into
:mod:`mxtreme`, so nothing scientific lives here. See ``cli/README.md`` for how to run it.
"""

__all__ = ["app", "prompts", "ui"]
