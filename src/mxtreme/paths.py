"""Resolve a selection (recording / culture / group) into concrete on-disk paths.

Path resolution is driven entirely by a :class:`~mxtreme.config.Config`: the managed-store layout
(``preprocessed/``, ``burst_data/``, ``experimental_conditions/``, ``registry.csv``) comes from the
config, so nothing here is coupled to a particular machine or lab share.

Entry point: :func:`resolve_paths`.

    RecordingID      -> RecordingPaths
    CultureID        -> CulturePaths  (all available DIVs)
    CultureSelector  -> dict[exp_id, ExperimentPaths]
"""

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path

import pandas as pd

from mxtreme.config import Config
from mxtreme.identity import CultureID, CultureSelector, RecordingID


@dataclass(frozen=True)
class RecordingPaths:
    recording_id: RecordingID
    npz: Path          # cleaned spike/lfp data
    burst_stats: Path  # per-recording burst CSV

@dataclass(frozen=True)
class CulturePaths:
    culture_id: CultureID
    experimental_conditions: Path           # culture-level experimental-condition CSV
    recordings: dict[int, RecordingPaths]  # keyed by DIV

    def recording(self, div: int) -> RecordingPaths:
        return self.recordings[div]

@dataclass(frozen=True)
class ExperimentPaths:
    exp_id: str
    burst_summary: Path        # per-experiment burst log
    cultures: dict[CultureID, CulturePaths]

    def culture(self, culture_id: CultureID) -> CulturePaths:
        return self.cultures[culture_id]


def _one(matches: list[str], what: str) -> Path:
    """Return the single matching path, with a clear error if zero (or many) matched."""
    if not matches:
        raise FileNotFoundError(f"No {what} found (checked pattern had no matches).")
    return Path(matches[0])


def _recording_paths(rid: RecordingID, config: Config) -> RecordingPaths:
    npz_glob = str(
        config.preprocessed_dir / rid.exp_id / rid.chip / f"well{rid.well}" / f"DIV{rid.div}*exp_data.npz"
    )
    burst_glob = str(
        config.burst_data_dir / rid.exp_id / rid.chip / f"well{rid.well}" / f"DIV{rid.div}*burst_data.csv"
    )
    return RecordingPaths(
        recording_id=rid,
        npz=_one(glob(npz_glob), f"preprocessed npz for {rid}"),
        burst_stats=_one(glob(burst_glob), f"burst CSV for {rid}"),
    )

def _culture_paths(cid: CultureID, divs: list[int] | int, config: Config) -> CulturePaths:
    if isinstance(divs, int):
        divs = [divs]
    return CulturePaths(
        culture_id=cid,
        experimental_conditions=(
            config.experimental_conditions_dir / cid.exp_id / f"{cid.chip}_well{cid.well}_exp_conditions.csv"
        ),
        recordings={
            div: _recording_paths(RecordingID(cid.exp_id, cid.chip, cid.well, div), config)
            for div in divs
        },
    )

def _experiment_paths(exp_id: str, cultures: list[CulturePaths], config: Config) -> ExperimentPaths:
    return ExperimentPaths(
        exp_id=exp_id,
        burst_summary=config.burst_data_dir / f"{exp_id}_burst_log.csv",
        cultures={c.culture_id: c for c in cultures},
    )


def resolve_paths(
    target: CultureSelector | CultureID | RecordingID,
    config: Config,
) -> RecordingPaths | CulturePaths | dict[str, ExperimentPaths]:
    """Resolve a selection into concrete paths using ``config``'s managed-store layout.

    :param target: What to resolve -- a single :class:`~mxtreme.identity.RecordingID`, a
        :class:`~mxtreme.identity.CultureID` (expands to all DIVs on record), or a
        :class:`~mxtreme.identity.CultureSelector` (a group, possibly across experiments).
    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :returns: ``RecordingPaths`` / ``CulturePaths`` / ``dict[exp_id, ExperimentPaths]`` per the
        target type.
    """
    df = pd.read_csv(config.registry_path)
    # Registry rows may be written by more than one producer (`io.register`,
    # `utils.build_registry_from_disk`) with differing dtypes/duplicates; normalise + dedupe so a
    # culture's DIVs aren't double-counted.
    key = ["exp_id", "chip", "well", "div"]
    df = df.drop_duplicates(subset=key)

    def get_divs(cid: CultureID, divs: list[int] | int | None = None) -> list[int]:
        mask = (
            (df.exp_id.astype(str) == str(cid.exp_id))
            & (df.chip.astype(str) == str(cid.chip))
            & (df.well.astype(str) == str(cid.well))
        )
        available = sorted(int(d) for d in df[mask]["div"].tolist())

        if divs:
            if isinstance(divs, int):
                return [divs] if divs in available else []
            return [d for d in divs if d in available]
        return available

    def get_culture_ids(sel: CultureSelector) -> list[CultureID]:
        if sel.cultures:
            return sel.cultures
        if sel.exp_ids:
            rows = df[df.exp_id.astype(str).isin([str(e) for e in sel.exp_ids])]
        else:
            rows = df
        rows = rows[["exp_id", "chip", "well"]].drop_duplicates()
        return [CultureID(str(r.exp_id), str(r.chip), str(r.well)) for r in rows.itertuples()]

    if isinstance(target, RecordingID):
        return _recording_paths(target, config)

    if isinstance(target, CultureID):
        divs = get_divs(target)
        return _culture_paths(target, divs, config)

    if isinstance(target, CultureSelector):
        culture_ids = get_culture_ids(target)
        by_exp: dict[str, list[CulturePaths]] = {}
        for cid in culture_ids:
            divs = get_divs(cid, divs=target.divs)
            cpaths = _culture_paths(cid, divs, config)
            by_exp.setdefault(cid.exp_id, []).append(cpaths)
        return {
            exp_id: _experiment_paths(exp_id, cultures, config)
            for exp_id, cultures in by_exp.items()
        }

    raise TypeError(f"Unsupported target type: {type(target).__name__}")
