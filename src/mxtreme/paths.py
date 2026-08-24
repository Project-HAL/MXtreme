"""Resolve a selection (recording / culture / group) into concrete on-disk paths.

Path resolution is driven entirely by a :class:`~mxtreme.config.Config`: the managed-store layout
(``preprocessed/``, ``burst_data/``, ``registry.csv``) comes from the config, so nothing here is
coupled to a particular machine or lab share.

Two entry points, for the two shapes callers need:

- :func:`resolve_paths` mirrors the selection's own structure, which is what reporting and
  per-experiment grouping want::

      RecordingID      -> RecordingPaths
      CultureID        -> CulturePaths  (all available DIVs)
      CultureSelector  -> dict[exp_id, ExperimentPaths]

- :func:`resolve_recordings` flattens any of those into a :class:`RecordingSet` -- an ordered list of
  recordings with ``.npz`` / ``.burst_stats`` accessors, for the common case of "just give me the
  files to loop over".
"""

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterator

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


@dataclass(frozen=True)
class RecordingSet:
    """A flat, ordered collection of :class:`RecordingPaths` -- what analyses actually loop over.

    Built by :func:`resolve_recordings`. Iterates, indexes, and reports its length like a list, and
    exposes each path column directly so the common case is a single expression::

        recs = resolve_recordings(selector, config)
        for rec in recs:                  # RecordingPaths, ordered by exp_id / chip / well / DIV
            print(rec.recording_id.div, rec.npz)

        npz_paths = recs.npz              # list[Path]

    :param recordings: The resolved recordings, in resolution order.
    """

    recordings: tuple[RecordingPaths, ...]

    def __iter__(self) -> Iterator[RecordingPaths]:
        return iter(self.recordings)

    def __len__(self) -> int:
        return len(self.recordings)

    def __getitem__(self, index):
        return self.recordings[index]

    @property
    def ids(self) -> list[RecordingID]:
        """The :class:`~mxtreme.identity.RecordingID` of each recording."""
        return [r.recording_id for r in self.recordings]

    @property
    def npz(self) -> list[Path]:
        """Every preprocessed ``.npz`` path, in order."""
        return [r.npz for r in self.recordings]

    @property
    def burst_stats(self) -> list[Path]:
        """Every per-recording burst CSV path, in order."""
        return [r.burst_stats for r in self.recordings]

    def to_frame(self) -> pd.DataFrame:
        """The set as a tidy ``exp_id, chip, well, div, npz, burst_stats`` DataFrame.

        Convenient when the selection came from the registry in the first place -- filter or join on
        the frame, then feed the paths straight to the loading code.
        """
        return pd.DataFrame(
            [
                {
                    "exp_id": r.recording_id.exp_id,
                    "chip": r.recording_id.chip,
                    "well": r.recording_id.well,
                    "div": r.recording_id.div,
                    "npz": r.npz,
                    "burst_stats": r.burst_stats,
                }
                for r in self.recordings
            ],
            columns=["exp_id", "chip", "well", "div", "npz", "burst_stats"],
        )


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
    # An older registry on disk may carry duplicate rows for a recording (written before `io.register`
    # deduped on write); dedupe so a culture's DIVs aren't double-counted.
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


def _iter_recordings(resolved) -> Iterator[RecordingPaths]:
    """Yield every recording in a :func:`resolve_paths` result, ordered exp_id / chip / well / DIV."""
    if isinstance(resolved, RecordingPaths):
        yield resolved
        return

    if isinstance(resolved, CulturePaths):
        for div in sorted(resolved.recordings):
            yield resolved.recordings[div]
        return

    for exp_id in sorted(resolved):
        cultures = resolved[exp_id].cultures
        # CultureID is frozen but not ordered, so sort on its fields rather than the object.
        for cid in sorted(cultures, key=lambda c: (str(c.chip), str(c.well))):
            for div in sorted(cultures[cid].recordings):
                yield cultures[cid].recordings[div]


def resolve_recordings(
    target: CultureSelector | CultureID | RecordingID,
    config: Config,
) -> RecordingSet:
    """Resolve a selection into a flat :class:`RecordingSet`, whatever its shape.

    The same resolution as :func:`resolve_paths` -- registry-driven, with a clear error for a
    registered recording whose file is missing -- but flattened, so reaching the files takes one
    expression instead of walking experiment / culture / DIV::

        selector = CultureSelector(cultures=[CultureID("burstTrainer", "M07140", "0")])
        npz_paths = resolve_recordings(selector, config).npz

    Use :func:`resolve_paths` instead when the grouping itself matters (per-experiment reports,
    per-culture summaries).

    :param target: A :class:`~mxtreme.identity.RecordingID`, :class:`~mxtreme.identity.CultureID`, or
        :class:`~mxtreme.identity.CultureSelector`.
    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :returns: Every matching recording, ordered by exp_id, chip, well, then DIV.
    :rtype: RecordingSet
    """
    return RecordingSet(tuple(_iter_recordings(resolve_paths(target, config))))
