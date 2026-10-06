"""Resolve a selection (recording / culture / group) into concrete on-disk paths.

Path resolution is driven entirely by a :class:`~mxtreme.config.Config`: the managed-store layout
(``preprocessed/``, ``burst_data/``, ``registry.csv``) comes from the config, so nothing here is
coupled to a particular machine or lab share.

Two entry points, for the two shapes callers need:

- :func:`resolve_paths` mirrors the selection's own structure, which is what reporting and
  per-batch grouping want::

      RecordingID      -> RecordingPaths
      CultureID        -> CulturePaths  (every recording, by DIV then experiment label)
      CultureSelector  -> dict[batch_id, BatchPaths]

- :func:`resolve_paths_flat` flattens any of those into a :class:`RecordingSet` -- an ordered list of
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
class RecordingSet:
    """A flat, ordered collection of :class:`RecordingPaths` -- what analyses actually loop over.

    Built by :func:`resolve_paths_flat`. Iterates, indexes, and reports its length like a list, and
    exposes each path column directly so the common case is a single expression::

        recs = resolve_paths_flat(selector, config)
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


@dataclass(frozen=True)
class CulturePaths:
    """Every resolved recording of one culture, ordered by DIV then ``experiment`` label.

    A culture can have several recordings on one DIV (e.g. ``NS_ATP1hr`` and ``NS_ATP6hr`` on DIV
    27), each told apart by its label, so recordings are a flat :class:`RecordingSet` rather than a
    DIV-keyed dict::

        for rec in cpath.recordings:      # RecordingPaths
            rec.recording_id.div, rec.recording_id.experiment, rec.npz

        cpath.divs                        # [7, 14, 27]
        cpath.at_div(27)                  # every recording on DIV 27
        cpath.recording(14)               # the one recording on DIV 14
        cpath.recording(27, "NS_ATP1hr")  # one of several, by label
    """

    culture_id: CultureID
    recordings: RecordingSet

    @property
    def divs(self) -> list[int]:
        """The distinct DIVs with a recording, sorted."""
        return sorted({r.recording_id.div for r in self.recordings})

    @property
    def experiments(self) -> list[str]:
        """The distinct ``experiment`` labels among the recordings, sorted (``""`` for unlabelled)."""
        return sorted({r.recording_id.experiment or "" for r in self.recordings})

    def at_div(self, div: int) -> RecordingSet:
        """Every recording on ``div`` (possibly several, one per label)."""
        return RecordingSet(tuple(r for r in self.recordings if r.recording_id.div == int(div)))

    def recording(self, div: int, experiment: str | None = None) -> RecordingPaths:
        """The recording on ``div`` -- labelled ``experiment``, or the only one there when ``None``.

        :raises KeyError: If no recording matches.
        :raises ValueError: If ``experiment`` is ``None`` and the DIV has several recordings, naming
            their labels so one can be picked.
        """
        matches = [
            r for r in self.at_div(div)
            if experiment is None or (r.recording_id.experiment or "") == str(experiment)
        ]
        if not matches:
            label = "" if experiment is None else f" labelled {_label(str(experiment))}"
            raise KeyError(f"No recording of {self.culture_id} on DIV {div}{label}.")
        if len(matches) > 1:
            labels = ", ".join(_label(r.recording_id.experiment or "") for r in matches)
            raise ValueError(
                f"{self.culture_id} has {len(matches)} recordings on DIV {div} ({labels}). "
                "Pick one with experiment=..., or use at_div() to get them all."
            )
        return matches[0]


@dataclass(frozen=True)
class BatchPaths:
    batch_id: str
    burst_summary: Path        # per-batch burst log
    cultures: dict[CultureID, CulturePaths]

    def culture(self, culture_id: CultureID) -> CulturePaths:
        return self.cultures[culture_id]


#: Former name of :class:`BatchPaths`, from when the store grouped recordings by experiment.
ExperimentPaths = BatchPaths


def _one(matches: list[str], what: str) -> Path:
    """Return the single matching path, with a clear error if zero (or many) matched."""
    if not matches:
        raise FileNotFoundError(f"No {what} found (checked pattern had no matches).")
    return Path(matches[0])


def _recording_paths(rid: RecordingID, config: Config) -> RecordingPaths:
    """Find one recording's files. ``rid.experiment`` must be a concrete label here, not ``None``."""
    # File names end ``_well<w>[_<experiment>]_<suffix>`` (see mxtreme.io.recording_file_name), so
    # recordings with different labels on the same DIV never match each other's pattern.
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

def _label(experiment: str) -> str:
    return repr(experiment) if experiment else "'' (unlabelled)"

def _one_per_div(cid: CultureID, rids: list[RecordingID]) -> dict[int, RecordingID]:
    """Key recordings by DIV, refusing a DIV that has several (differently labelled) ones.

    Only a bare :class:`RecordingID` (``experiment=None``) needs this: it must name exactly one
    recording. Cultures and selectors keep every recording.

    :raises ValueError: Naming the DIV and its labels, so the caller can pick one with ``experiment=``.
    """
    by_div: dict[int, list[RecordingID]] = {}
    for rid in rids:
        by_div.setdefault(rid.div, []).append(rid)
    clashes = {div: group for div, group in by_div.items() if len(group) > 1}
    if clashes:
        detail = "; ".join(
            f"DIV {div}: {', '.join(_label(r.experiment) for r in group)}" for div, group in sorted(clashes.items())
        )
        raise ValueError(
            f"{cid} has more than one recording on a DIV ({detail}), so a RecordingID without a label "
            "is ambiguous. Pick one with experiment=..., or resolve the CultureID to get them all."
        )
    return {div: group[0] for div, group in by_div.items()}

def _culture_paths(cid: CultureID, rids: list[RecordingID], config: Config) -> CulturePaths:
    # `rids` is already ordered by (div, experiment) -- see _recording_ids.
    return CulturePaths(
        culture_id=cid,
        recordings=RecordingSet(tuple(_recording_paths(rid, config) for rid in rids)),
    )

def _batch_paths(batch_id: str, cultures: list[CulturePaths], config: Config) -> BatchPaths:
    return BatchPaths(
        batch_id=batch_id,
        burst_summary=config.burst_data_dir / f"{batch_id}_burst_log.csv",
        cultures={c.culture_id: c for c in cultures},
    )


def _preprocessed_rows(config: Config) -> pd.DataFrame:
    """The registry's ``preprocessed`` rows -- the recordings resolution can hand back."""
    if not Path(config.registry_path).is_file():
        raise FileNotFoundError(f"No registry at {config.registry_path}; nothing to resolve.")
    # Read through the registry's own reader, so a registry written before batch_id was the identity
    # (an exp_id column) is migrated on the way in.
    df = read_registry(config.registry_path)
    # Resolution here is about preprocessed recordings, and the registry also indexes scans -- raw
    # .h5 files with no .npz behind them. Including those rows would invent DIVs (and whole cultures)
    # that resolve to preprocessed paths which do not exist. A registry written before scans were
    # indexed has no `kind` column and holds recordings only.
    if "kind" in df.columns:
        df = df[df["kind"].astype(str) == "preprocessed"]
    # An older registry on disk may carry duplicate rows for a recording (written before `io.register`
    # deduped on write); dedupe so a culture's DIVs aren't double-counted.
    return df.drop_duplicates(subset=["batch_id", "chip", "well", "div", "experiment"])


def _with_label(df: pd.DataFrame, experiment: str | None) -> pd.DataFrame:
    """Rows carrying ``experiment`` as their label; every row when it is ``None``."""
    return df if experiment is None else df[df.experiment.astype(str) == str(experiment)]


def _recording_ids(
    df: pd.DataFrame, cid: CultureID, divs: list[int] | int | None, experiment: str | None
) -> list[RecordingID]:
    """One :class:`RecordingID` per registered recording of ``cid``, ordered by DIV then label."""
    rows = _with_label(df, experiment)
    rows = rows[
        (rows.batch_id.astype(str) == cid.batch_id)
        & (rows.chip.astype(str) == cid.chip)
        & (rows.well.astype(str) == cid.well)
    ]
    rids = sorted(
        (RecordingID(cid.batch_id, cid.chip, cid.well, int(r.div), str(r.experiment)) for r in rows.itertuples()),
        key=lambda r: (r.div, r.experiment),
    )
    if divs:
        wanted = {divs} if isinstance(divs, int) else set(divs)
        rids = [r for r in rids if r.div in wanted]
    return rids


def _culture_ids(df: pd.DataFrame, sel: CultureSelector) -> list[CultureID]:
    if sel.cultures:
        return list(sel.cultures)
    rows = _with_label(df, sel.experiment)
    if sel.batch_ids:
        rows = rows[rows.batch_id.astype(str).isin([str(b) for b in sel.batch_ids])]
    rows = rows[["batch_id", "chip", "well"]].drop_duplicates()
    return [CultureID(str(r.batch_id), str(r.chip), str(r.well)) for r in rows.itertuples()]


def _concrete(rid: RecordingID, df: pd.DataFrame) -> RecordingID:
    """``rid`` with its label filled in from the registry when it was left ``None``."""
    if rid.experiment is not None:
        return rid
    matches = _recording_ids(df, rid.culture, rid.div, None)
    if not matches:
        raise FileNotFoundError(f"No preprocessed recording of {rid.culture} on DIV {rid.div} is registered.")
    _one_per_div(rid.culture, matches)  # raises, naming the labels, if the DIV has several
    return matches[0]


def resolve_paths(
    target: CultureSelector | CultureID | RecordingID,
    config: Config,
    *,
    experiment: str | None = None,
) -> RecordingPaths | CulturePaths | dict[str, BatchPaths]:
    """Resolve a selection into concrete paths using ``config``'s managed-store layout.

    Recordings are matched whatever their ``experiment`` label unless one is asked for, and a
    culture keeps every one of them -- including several differently labelled recordings on one
    DIV, which :class:`CulturePaths` lists side by side (see :meth:`CulturePaths.at_div`).

    :param target: What to resolve -- a single :class:`~mxtreme.identity.RecordingID`, a
        :class:`~mxtreme.identity.CultureID` (expands to all recordings on record), or a
        :class:`~mxtreme.identity.CultureSelector` (a group, possibly across batches).
    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :param experiment: For a :class:`~mxtreme.identity.CultureID`, which recordings to resolve:
        ``None`` (the default) for any label, ``""`` for unlabelled ones only, or one label. A
        ``RecordingID`` and a ``CultureSelector`` carry their own.
    :returns: ``RecordingPaths`` / ``CulturePaths`` / ``dict[batch_id, BatchPaths]`` per the
        target type.
    :raises ValueError: If a ``RecordingID`` without a label names a DIV with several recordings.
    """
    if not isinstance(target, (RecordingID, CultureID, CultureSelector)):
        raise TypeError(f"Unsupported target type: {type(target).__name__}")
    df = _preprocessed_rows(config)

    if isinstance(target, RecordingID):
        return _recording_paths(_concrete(target, df), config)

    if isinstance(target, CultureID):
        return _culture_paths(target, _recording_ids(df, target, None, experiment), config)

    by_batch: dict[str, list[CulturePaths]] = {}
    for cid in _culture_ids(df, target):
        rids = _recording_ids(df, cid, target.divs, target.experiment)
        by_batch.setdefault(cid.batch_id, []).append(_culture_paths(cid, rids, config))
    return {
        batch_id: _batch_paths(batch_id, cultures, config)
        for batch_id, cultures in by_batch.items()
    }


def resolve_paths_flat(
    target: CultureSelector | CultureID | RecordingID,
    config: Config,
    *,
    experiment: str | None = None,
) -> RecordingSet:
    """Resolve a selection into a flat :class:`RecordingSet`, whatever its shape.

    The same registry-driven resolution as :func:`resolve_paths` -- with a clear error for a
    registered recording whose ``.npz`` is missing, and ``burst_stats=None`` where bursts have not
    been detected yet -- but flattened, so reaching the files takes one expression instead of
    walking batch / culture / DIV::

        selector = CultureSelector(cultures=[CultureID("fall2026_batch1_DRG", "M07140", "0")])
        npz_paths = resolve_paths_flat(selector, config).npz

    With the default ``experiment=None`` every label is listed, including several recordings of one
    culture on the same DIV; each recording's ``recording_id.experiment`` (or the ``experiment``
    column of :meth:`RecordingSet.to_frame`) says which it is.

    Use :func:`resolve_paths` instead when the grouping itself matters (per-batch reports,
    per-culture summaries).

    :param target: A :class:`~mxtreme.identity.RecordingID`, :class:`~mxtreme.identity.CultureID`, or
        :class:`~mxtreme.identity.CultureSelector`.
    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :param experiment: As for :func:`resolve_paths`.
    :returns: Every matching recording, ordered by batch_id, chip, well, DIV, then label.
    :rtype: RecordingSet
    """
    if not isinstance(target, (RecordingID, CultureID, CultureSelector)):
        raise TypeError(f"Unsupported target type: {type(target).__name__}")
    df = _preprocessed_rows(config)

    if isinstance(target, RecordingID):
        return RecordingSet((_recording_paths(_concrete(target, df), config),))

    if isinstance(target, CultureID):
        rids = _recording_ids(df, target, None, experiment)
    else:
        cids = sorted(_culture_ids(df, target), key=lambda c: (c.batch_id, c.chip, c.well))
        rids = [rid for cid in cids for rid in _recording_ids(df, cid, target.divs, target.experiment)]
    return RecordingSet(tuple(_recording_paths(rid, config) for rid in rids))
