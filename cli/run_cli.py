#!/usr/bin/env python3
"""Launcher for the MXtreme CLI: ``python cli/run_cli.py``.

Works from any directory, and whether or not the package is installed: running this file already puts
``cli/`` on ``sys.path`` (so ``mxtreme_cli`` imports), and ``src/`` is appended as a fallback when
``mxtreme`` itself is not installed in the active environment.

Deliberately *not* named ``mxtreme.py``: a script's own directory leads ``sys.path``, so that name
would shadow the package it is meant to drive.
"""

from __future__ import annotations

import sys
from pathlib import Path

CLI_DIR = Path(__file__).resolve().parent
REPO_ROOT = CLI_DIR.parent

if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))

try:  # prefer the installed package, so the CLI matches whatever the environment has
    import mxtreme  # noqa: F401
except ModuleNotFoundError:
    sys.path.append(str(REPO_ROOT / "src"))

from mxtreme_cli.app import main  # noqa: E402  -- import after sys.path is set up

if __name__ == "__main__":
    sys.exit(main())
