"""Resolve a selection (recording / culture / group) into concrete on-disk paths.

Path resolution is driven entirely by a :class:`~mxtreme.config.Config`: the managed-store layout
(``preprocessed/``, ``burst_data/``, ``registry.csv``) comes from the config, so nothing here is
coupled to a particular machine or lab share.

Two entry points, for the two shapes callers need:

- :func:`resolve_paths` mirrors the selection's own structure, which is what reporting and
  per-batch grouping want::

      RecordingID      -> RecordingPaths
      CultureID        -> CulturePaths  (all available DIVs)
      CultureSelector  -> dict[batch_id, BatchPaths]

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
from mxtreme.io import read_registry


@dataclass(frozen=True)
class RecordingPaths:
    recording_id: RecordingID
    npz: Path                        # cleaned spike/lfp data
    burst_stats: Path | None = None  # per-recording burst CSV; None until burst detection has run

    def require_burst_stats(self) -> Path:
        """Return :attr:`burst_stats`, or raise if burst detection has not been run on this recording."""
        if self.burst_stats is None:
            raise FileNotFoundError(
                f"No burst CSV for {self.recording_id}; run burst detection on it first "
                "(see mxtreme.bursting) before analyses that need bursts."
            )
        return self.burst_stats

@dataclass(frozen=True)
class CulturePaths:
    culture_id: CultureID
    recordings: dict[int, RecordingPaths]  # keyed by DIV

    def recording(self, div: int) -> RecordingPaths:
        return self.recordings[div]

@dataclass(frozen=True)
class BatchPaths:
    batch_id: str
    burst_summary: Path        # per-batch burst log
    cultures: dict[CultureID, CulturePaths]

    def culture(self, culture_id: CultureID) -> CulturePaths:
        return self.cultures[culture_id]


#: Former name of :class:`BatchPaths`, from when the store grouped recordings by experiment.
ExperimentPaths = BatchPaths


@dataclass(frozen=True)
class RecordingSet:
    """A flat, ordered collection of :class:`RecordingPaths` -- what analyses actually loop over.

    Built by :func:`resolve_recordings`. Iterates, indexes, and reports its length like a list, and
    exposes each path column directly so the common case is a single expression::

        recs = resolve_recordings(selector, config)
        for rec in recs:                  # RecordingPaths, ordered by batch_id / chip / well / DIV
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
    def burst_stats(self) -> list[Path | None]:
        """Every per-recording burst CSV path, in order (``None`` where bursts have not been detected)."""
        return [r.burst_stats for r in self.recordings]

    def to_frame(self) -> pd.DataFrame:
        """The set as a tidy ``batch_id, chip, well, div, experiment, npz, burst_stats`` DataFrame.

        Convenient when the selection came from the registry in the first place -- filter or join on
        the frame, then feed the paths straight to the loading code.
        """
        return pd.DataFrame(
            [
                {
                    "batch_id": r.recording_id.batch_id,
                    "chip": r.recording_id.chip,
                    "well": r.recording_id.well,
                    "div": r.recording_id.div,
                    "experiment": r.recording_id.experiment,
                    "npz": r.npz,
                    "burst_stats": r.burst_stats,
                }
                for r in self.recordings
            ],
            columns=["batch_id", "chip", "well", "div", "experiment", "npz", "burst_stats"],
        )


def _one(matches: list[str], what: str) -> Path:
    """Return the single matching path, with a clear error if zero (or many) matched."""
    if not matches:
        raise FileNotFoundError(f"No {what} found (checked pattern had no matches).")
    return Path(matches[0])


def _recording_paths(rid: RecordingID, config: Config) -> RecordingPaths:
    # File names end ``_well<w>[_<experiment>]_<suffix>`` (see mxtreme.io.recording_file_name), so a
    # scan's recording and an experiment's on the same DIV never match each other's pattern.
    tail = f"well{rid.well}_{rid.experiment}_" if rid.experiment else f"well{rid.well}_"
    well_tail = Path(rid.batch_id) / rid.chip / f"well{rid.well}"
    npz_glob = str(config.preprocessed_dir / well_tail / f"DIV{rid.div}_*{tail}exp_data.npz")
    burst_glob = str(config.burst_data_dir / well_tail / f"DIV{rid.div}_*{tail}burst_data.csv")
    # The npz is what a registry row promises, so its absence is an error. Burst data is downstream
    # and optional: a store of freshly preprocessed recordings has none yet.
    bursts = glob(burst_glob)
    return RecordingPaths(
        recording_id=rid,
        npz=_one(glob(npz_glob), f"preprocessed npz for {rid}"),
        burst_stats=Path(bursts[0]) if bursts else None,
    )

def _culture_paths(
    cid: CultureID, divs: list[int] | int, config: Config, experiment: str = ""
) -> CulturePaths:
    if isinstance(divs, int):
        divs = [divs]
    return CulturePaths(
        culture_id=cid,
        recordings={
            div: _recording_paths(RecordingID(cid.batch_id, cid.chip, cid.well, div, experiment), config)
            for div in divs
        },
    )

def _batch_paths(batch_id: str, cultures: list[CulturePaths], config: Config) -> BatchPaths:
    return BatchPaths(
        batch_id=batch_id,
        burst_summary=config.burst_data_dir / f"{batch_id}_burst_log.csv",
        cultures={c.culture_id: c for c in cultures},
    )


def resolve_paths(
    target: CultureSelector | CultureID | RecordingID,
    config: Config,
    *,
    experiment: str = "",
) -> RecordingPaths | CulturePaths | dict[str, BatchPaths]:
    """Resolve a selection into concrete paths using ``config``'s managed-store layout.

    :param target: What to resolve -- a single :class:`~mxtreme.identity.RecordingID`, a
        :class:`~mxtreme.identity.CultureID` (expands to all DIVs on record), or a
        :class:`~mxtreme.identity.CultureSelector` (a group, possibly across batches).
    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :param experiment: For a :class:`~mxtreme.identity.CultureID`, which of its recordings to
        resolve: ``""`` (the default) for its scans, or an ingested experiment's name. A
        ``RecordingID`` and a ``CultureSelector`` carry their own.
    :returns: ``RecordingPaths`` / ``CulturePaths`` / ``dict[batch_id, BatchPaths]`` per the
        target type.
    """
    if not Path(config.registry_path).is_file():
        raise FileNotFoundError(f"No registry at {config.registry_path}; nothing to resolve.")
    # Read through the registry's own reader, so a registry written before batch_id was the identity
    # (an exp_id column) is migrated on the way in.
    df = read_registry(config.registry_path)
    # Resolution here is about preprocessed recordings, and the registry also indexes activity scans
    # -- raw .h5 files with no .npz behind them. Including those rows would invent DIVs (and whole
    # cultures) that resolve to preprocessed paths which do not exist. A registry written before
    # scans were indexed has no `kind` column and holds recordings only.
    if "kind" in df.columns:
        df = df[df["kind"].astype(str) == "preprocessed"]
    # An older registry on disk may carry duplicate rows for a recording (written before `io.register`
    # deduped on write); dedupe so a culture's DIVs aren't double-counted.
    key = ["batch_id", "chip", "well", "div", "experiment"]
    df = df.drop_duplicates(subset=key)

    def get_divs(cid: CultureID, divs: list[int] | int | None = None, experiment: str = "") -> list[int]:
        mask = (
            (df.batch_id.astype(str) == str(cid.batch_id))
            & (df.chip.astype(str) == str(cid.chip))
            & (df.well.astype(str) == str(cid.well))
            & (df.experiment.astype(str) == str(experiment))
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
        rows = df[df.experiment.astype(str) == str(sel.experiment)]
        if sel.batch_ids:
            rows = rows[rows.batch_id.astype(str).isin([str(b) for b in sel.batch_ids])]
        rows = rows[["batch_id", "chip", "well"]].drop_duplicates()
        return [CultureID(str(r.batch_id), str(r.chip), str(r.well)) for r in rows.itertuples()]

    if isinstance(target, RecordingID):
        return _recording_paths(target, config)

    if isinstance(target, CultureID):
        divs = get_divs(target, experiment=experiment)
        return _culture_paths(target, divs, config, experiment)

    if isinstance(target, CultureSelector):
        culture_ids = get_culture_ids(target)
        by_batch: dict[str, list[CulturePaths]] = {}
        for cid in culture_ids:
            divs = get_divs(cid, divs=target.divs, experiment=target.experiment)
            cpaths = _culture_paths(cid, divs, config, target.experiment)
            by_batch.setdefault(cid.batch_id, []).append(cpaths)
        return {
            batch_id: _batch_paths(batch_id, cultures, config)
            for batch_id, cultures in by_batch.items()
        }

    raise TypeError(f"Unsupported target type: {type(target).__name__}")


def _iter_recordings(resolved) -> Iterator[RecordingPaths]:
    """Yield every recording in a :func:`resolve_paths` result, ordered batch_id / chip / well / DIV."""
    if isinstance(resolved, RecordingPaths):
        yield resolved
        return

    if isinstance(resolved, CulturePaths):
        for div in sorted(resolved.recordings):
            yield resolved.recordings[div]
        return

    for batch_id in sorted(resolved):
        cultures = resolved[batch_id].cultures
        # CultureID is frozen but not ordered, so sort on its fields rather than the object.
        for cid in sorted(cultures, key=lambda c: (str(c.chip), str(c.well))):
            for div in sorted(cultures[cid].recordings):
                yield cultures[cid].recordings[div]


def resolve_recordings(
    target: CultureSelector | CultureID | RecordingID,
    config: Config,
    *,
    experiment: str = "",
) -> RecordingSet:
    """Resolve a selection into a flat :class:`RecordingSet`, whatever its shape.

    The same resolution as :func:`resolve_paths` -- registry-driven, with a clear error for a
    registered recording whose ``.npz`` is missing, and ``burst_stats=None`` where bursts have not
    been detected yet -- but flattened, so reaching the files takes one
    expression instead of walking batch / culture / DIV::

        selector = CultureSelector(cultures=[CultureID("fall2026_batch1_DRG", "M07140", "0")])
        npz_paths = resolve_recordings(selector, config).npz

    Use :func:`resolve_paths` instead when the grouping itself matters (per-batch reports,
    per-culture summaries).

    :param target: A :class:`~mxtreme.identity.RecordingID`, :class:`~mxtreme.identity.CultureID`, or
        :class:`~mxtreme.identity.CultureSelector`.
    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :param experiment: As for :func:`resolve_paths`.
    :returns: Every matching recording, ordered by batch_id, chip, well, then DIV.
    :rtype: RecordingSet
    """
    return RecordingSet(tuple(_iter_recordings(resolve_paths(target, config, experiment=experiment))))
