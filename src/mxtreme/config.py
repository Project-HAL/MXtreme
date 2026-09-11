"""Project configuration for the preprocessing scope.

`Config` decouples the package from any particular machine's directory layout. It holds a single
``data_root`` -- the root of the *package-managed data store*, i.e. the directory the package writes
its outputs to (raw ``.h5`` recordings, cleaned ``.npz`` files, burst CSVs, analysis outputs, the
registry) and later reads those same outputs back from. A scan MXtreme ran itself lands in the
store's ``recordings/`` tree; a raw ``.h5`` acquired *outside* MXtreme joins it through
:func:`mxtreme.store.ingest_recording`.

The root is supplied by a small TOML file so it never has to be hard-coded in library code:

    [data]
    root = "/abs/path/to/managed/store"
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Resolved locations of the package-managed data store.

    :param data_root: Root directory of the managed store. Outputs are written beneath it and read
        back from the same place. It is never silently defaulted to a lab-specific path -- callers
        must provide it (typically via :meth:`from_toml`).
    """

    data_root: Path

    @property
    def recordings_dir(self) -> Path:
        """Root of the recordings tree: raw ``.h5`` files (scans and ingested experiments), one
        file per well, laid out by plating batch -- see :mod:`mxtreme.store`."""
        return self.data_root / "recordings"

    @property
    def preprocessed_dir(self) -> Path:
        """Directory holding cleaned/transformed ``.npz`` files (one per well per recording)."""
        return self.data_root / "preprocessed"

    @property
    def burst_data_dir(self) -> Path:
        """Directory holding per-recording burst CSVs and per-experiment burst logs."""
        return self.data_root / "burst_data"

    @property
    def analysis_dir(self) -> Path:
        """Directory holding analysis outputs (per-culture summary CSVs, plots, PDF reports)."""
        return self.data_root / "analysis"

    @property
    def registry_path(self) -> Path:
        """Path to the CSV index of what has been processed."""
        return self.data_root / "registry.csv"

    @property
    def transactions_path(self) -> Path:
        """Path to the append-only log of decisions made about the data (a culture marked dead,
        a chip's device kind) -- see :mod:`mxtreme.transactions`."""
        return self.data_root / "transactions.jsonl"

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Build a :class:`Config` from a TOML file.

        The file must contain a ``[data]`` table with a ``root`` key pointing at the managed store::

            [data]
            root = "/abs/path/to/managed/store"

        :param path: Path to the TOML configuration file.
        :type path: str or Path
        :raises FileNotFoundError: If ``path`` does not exist.
        :raises KeyError: If the ``[data].root`` key is missing.
        :returns: A populated configuration object.
        :rtype: Config
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file does not exist: {path}")

        with path.open("rb") as f:
            cfg = tomllib.load(f)

        try:
            root = cfg["data"]["root"]
        except KeyError as exc:
            raise KeyError("Config TOML must define [data].root (the managed-store directory).") from exc

        return cls(data_root=Path(root).expanduser())
