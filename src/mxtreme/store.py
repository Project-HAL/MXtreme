"""The ``recordings/`` tree of the managed store: batches, layout, naming, splitting, and ingest.

Every raw ``.h5`` the store manages -- activity scans, network scans, and exogenous experiment
recordings -- lives in one tree, keyed by the *plating batch* the culture came from rather than by an
experiment name::

    <data_root>/recordings/
        plating_<plate_date>_<batch_id>/
            chip_<M1|M2>_<chip>/
                well_<well>/
                    DIV_<div>/
                        plating_<plate_date>_<batch_id>_chip_<chip>_well_<well>_DIV_<div>_activity_scan.raw.h5
                        plating_<plate_date>_<batch_id>_chip_<chip>_well_<well>_DIV_<div>_network_scan.raw.h5
                        plating_<plate_date>_<batch_id>_chip_<chip>_well_<well>_DIV_<div>_<exp_id>.raw.h5

A *batch* is one plating event, named at the bench when it happens -- see :class:`Batch`. The
``<exp_id>`` tail is a free string naming one experiment, supplied when an exogenous recording is
ingested (see :func:`ingest_recording`); scans carry no exp id, their kind is the tail instead.

Each file in the tree holds exactly one well, so one culture's whole history sits in one directory.
A multi-well recording (a MaxTwo) is recorded as a single ``.h5`` and then broken apart by
:func:`split_by_well`.

The module is import-light on purpose: everything here except splitting/ingest is pure path and
string logic, and ``h5py``/``numpy``/``mxtreme.io`` are imported inside the functions that need
them -- so ``import mxtreme`` (which re-exports :class:`Batch`) stays fast and rig-safe.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

#: Semesters a batch id may name, in calendar order.
SEMESTERS = ("spring", "summer", "fall", "winter")

#: MaxWell systems a batch may be plated on: MaxOne or MaxTwo.
SYSTEMS = ("M1", "M2")

#: What a file in the recordings tree can be. ``"experiment"`` is an ingested exogenous recording;
#: the other two are scans MXtreme ran itself.
RECORDING_KINDS = ("activity_scan", "network_scan", "experiment")

#: The batch-id convention, e.g. ``fall2026_batch1_DRG_M1``. The cell type is the one free-form
#: field; it may itself contain underscores, which parses unambiguously because everything around it
#: is fixed-format.
_BATCH_ID = re.compile(
    r"^(?P<semester>spring|summer|fall|winter)(?P<year>\d{4})"
    r"_batch(?P<number>\d+)"
    r"_(?P<cell_type>.+)"
    r"_(?P<system>M[12])$"
)

#: Matches the scan-kind tail of a store file name (``..._activity_scan.raw.h5``,
#: ``..._network_scan_1.raw.h5`` -- the numeric suffix is MaxLab's collision rename).
_SCAN_TAIL = re.compile(r"_(activity|network)_scan(_\d+)?\.raw\.h5$")

#: Extracts the experiment-id tail of a store file name: everything after the ``DIV_<div>_`` marker.
_EXP_TAIL = re.compile(r"_DIV_\d+_(?P<exp_id>.+?)\.raw\.h5$")

#: Plating dates are ``YYMMDD``, matching :func:`mxtreme.scans.mx_setup.write_metadata`.
_PLATE_DATE = re.compile(r"^\d{6}$")

#: The directory names the layout functions below produce, for reading the tree back.
_PLATING_DIR = re.compile(r"^plating_(?P<plate_date>\d{6})_(?P<batch_id>.+)$")
_CHIP_DIR = re.compile(r"^chip_M[12]_(?P<chip>.+)$")


@dataclass(frozen=True)
class Batch:
    """One plating batch: the identity every recording in the store hangs off.

    Created manually at the start of a batch -- this constructor *is* the generator, validating each
    field so a malformed id never reaches a directory name::

        batch = Batch(semester="fall", year=2026, number=1, cell_type="DRG", system="M1")
        batch.id  # "fall2026_batch1_DRG_M1"

    An id someone already has as a string round-trips through :meth:`parse`.

    :param semester: One of :data:`SEMESTERS`.
    :param year: Four-digit calendar year.
    :param number: Batch number within the semester, starting at 1.
    :param cell_type: Free-form cell type label, e.g. ``"DRG"``. Underscores are allowed; path
        separators and whitespace are not.
    :param system: ``"M1"`` (MaxOne) or ``"M2"`` (MaxTwo) -- see :data:`SYSTEMS`.
    """

    semester: str
    year: int
    number: int
    cell_type: str
    system: str

    def __post_init__(self) -> None:
        if self.semester not in SEMESTERS:
            raise ValueError(f"semester must be one of {SEMESTERS}, got {self.semester!r}.")
        if not 1000 <= int(self.year) <= 9999:
            raise ValueError(f"year must be a four-digit year, got {self.year!r}.")
        if int(self.number) < 1:
            raise ValueError(f"batch number must be at least 1, got {self.number!r}.")
        if not self.cell_type or not re.fullmatch(r"[A-Za-z0-9_-]+", self.cell_type):
            raise ValueError(
                f"cell_type must be a non-empty label of letters, digits, '-' or '_', "
                f"got {self.cell_type!r}."
            )
        if self.system not in SYSTEMS:
            raise ValueError(f"system must be one of {SYSTEMS}, got {self.system!r}.")

    @property
    def id(self) -> str:
        """The batch id string, e.g. ``fall2026_batch1_DRG_M1``."""
        return f"{self.semester}{self.year}_batch{self.number}_{self.cell_type}_{self.system}"

    def __str__(self) -> str:
        return self.id

    @classmethod
    def parse(cls, batch_id: str | Batch) -> Batch:
        """Parse a batch id string back into a :class:`Batch`, validating it on the way.

        A :class:`Batch` passes through unchanged, so call sites can accept either form.

        :param batch_id: e.g. ``"fall2026_batch1_DRG_M1"``.
        :raises ValueError: If the string does not follow the batch-id convention.
        """
        if isinstance(batch_id, Batch):
            return batch_id
        match = _BATCH_ID.match(str(batch_id))
        if match is None:
            raise ValueError(
                f"Not a valid batch id: {batch_id!r}. Expected "
                "<semester><year>_batch<n>_<cell_type>_<M1|M2>, e.g. 'fall2026_batch1_DRG_M1'."
            )
        return cls(
            semester=match["semester"],
            year=int(match["year"]),
            number=int(match["number"]),
            cell_type=match["cell_type"],
            system=match["system"],
        )


def _validate_plate_date(plate_date) -> int:
    """Check a plating date is ``YYMMDD`` (the format the metadata blob is validated against)."""
    if not _PLATE_DATE.fullmatch(str(plate_date)):
        raise ValueError(f"plate_date must be YYMMDD, got {plate_date!r}.")
    return int(plate_date)


# --- layout ---------------------------------------------------------------------------------------


def plating_dirname(batch: Batch | str, plate_date) -> str:
    """The top-level directory of one plating: ``plating_<plate_date>_<batch_id>``."""
    return f"plating_{_validate_plate_date(plate_date)}_{Batch.parse(batch).id}"


def chip_dirname(batch: Batch | str, chip: str) -> str:
    """One chip's directory within a plating: ``chip_<M1|M2>_<chip>``."""
    return f"chip_{Batch.parse(batch).system}_{chip}"


def chip_dir(config, batch: Batch | str, plate_date, chip: str) -> Path:
    """Directory holding one chip's wells: ``<recordings_dir>/plating_.../chip_...``."""
    return config.recordings_dir / plating_dirname(batch, plate_date) / chip_dirname(batch, chip)


def recording_dir(config, batch: Batch | str, plate_date, chip: str, well: int, div: int) -> Path:
    """Directory one culture's recordings for one DIV land in: ``.../well_<well>/DIV_<div>``."""
    return chip_dir(config, batch, plate_date, chip) / f"well_{int(well)}" / f"DIV_{int(div)}"


def recording_stem(batch: Batch | str, plate_date, chip: str, well, div: int) -> str:
    """Base name shared by every file of one well on one DIV, without the kind/exp-id tail.

    ``plating_<plate_date>_<batch_id>_chip_<chip>_well_<well>_DIV_<div>``. ``well`` is normally an
    int; a multi-well recording awaiting :func:`split_by_well` uses a joined token (``"0-3"``)
    instead, which is why the parameter is not coerced.
    """
    return (
        f"plating_{_validate_plate_date(plate_date)}_{Batch.parse(batch).id}"
        f"_chip_{chip}_well_{well}_DIV_{int(div)}"
    )


@dataclass(frozen=True)
class Plating:
    """One plating found in the recordings tree: a batch, its plate date, and its directory.

    What :func:`list_platings` returns -- the read-back counterpart of :func:`plating_dirname`,
    for a front end offering "which plating is this session about" from what the store already
    holds.

    :param batch: The plating batch, parsed from the directory name.
    :param plate_date: Plating date (``YYMMDD``), from the directory name.
    :param path: The plating's directory.
    """

    batch: Batch
    plate_date: int
    path: Path

    def chips(self) -> list[str]:
        """Chip serials this plating has recordings for, from its ``chip_*`` directories."""
        if not self.path.is_dir():
            return []
        found = (_CHIP_DIR.match(entry.name) for entry in self.path.iterdir() if entry.is_dir())
        return sorted(match["chip"] for match in found if match)


def list_platings(config) -> list[Plating]:
    """Every plating in the recordings tree, most recently plated first.

    A directory that does not follow the plating naming convention is skipped rather than fatal:
    the tree is also a place people look at in a file browser, and a stray folder should not stop
    the real platings from being listed.

    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :returns: The platings, sorted by plate date, newest first.
    """
    root = config.recordings_dir
    if not root.is_dir():
        return []

    platings = []
    for entry in root.iterdir():
        match = _PLATING_DIR.match(entry.name)
        if not entry.is_dir() or match is None:
            continue
        try:
            batch = Batch.parse(match["batch_id"])
        except ValueError:
            continue
        platings.append(Plating(batch=batch, plate_date=int(match["plate_date"]), path=entry))
    return sorted(platings, key=lambda p: p.plate_date, reverse=True)


def recording_kind(h5_path: str | Path) -> str:
    """What a file in the recordings tree is, from its name -- see :data:`RECORDING_KINDS`.

    The tail is the only thing on disk that tells the kinds apart: the embedded metadata blob
    carries the culture's identity, not what was run on it. A name with neither scan tail is an
    ingested experiment recording, whose tail is its exp id.
    """
    match = _SCAN_TAIL.search(Path(h5_path).name)
    return f"{match.group(1)}_scan" if match else "experiment"


@dataclass(frozen=True)
class RecordingLocation:
    """One store file's identity, as read back from its place in the recordings tree.

    :param batch: The plating batch, parsed from the plating directory.
    :param plate_date: Plating date (``YYMMDD``), from the plating directory.
    :param chip: Chip serial, from the chip directory.
    :param well: Well number, from the well directory.
    :param div: Days *in vitro*, from the DIV directory.
    :param kind: See :data:`RECORDING_KINDS`.
    :param exp_id: The experiment id tail for ``kind="experiment"``; ``""`` for scans.
    """

    batch: Batch
    plate_date: int
    chip: str
    well: int
    div: int
    kind: str
    exp_id: str


def parse_recording_path(h5_path: Path, recordings_dir: Path) -> RecordingLocation | None:
    """Read a file's identity back out of its path in the recordings tree.

    The inverse of :func:`recording_dir` + :func:`recording_stem`: the tree encodes the full
    identity in fixed-format directory names, so -- unlike the old flat scans tree -- a file here
    *can* be identified without opening it. Used by :func:`mxtreme.io.rebuild_registry`.

    :param h5_path: The file, anywhere under ``recordings_dir``.
    :param recordings_dir: The tree root (``config.recordings_dir``).
    :returns: The parsed identity, or ``None`` if the path does not follow the layout.
    """
    try:
        plating, chip_part, well_part, div_part = h5_path.relative_to(recordings_dir).parts[:-1]
    except ValueError:
        return None

    plating_match = _PLATING_DIR.match(plating)
    chip_match = _CHIP_DIR.match(chip_part)
    well_match = re.match(r"^well_(\d+)$", well_part)
    div_match = re.match(r"^DIV_(\d+)$", div_part)
    if not (plating_match and chip_match and well_match and div_match):
        return None
    try:
        batch = Batch.parse(plating_match["batch_id"])
    except ValueError:
        return None

    kind = recording_kind(h5_path)
    exp_id = ""
    if kind == "experiment":
        exp_match = _EXP_TAIL.search(h5_path.name)
        # A foreign file name still has a full identity from the directories; falling back to its
        # stem keeps it indexable rather than skipped.
        exp_id = exp_match["exp_id"] if exp_match else h5_path.name.removesuffix(".raw.h5").removesuffix(".h5")

    return RecordingLocation(
        batch=batch,
        plate_date=int(plating_match["plate_date"]),
        chip=chip_match["chip"],
        well=int(well_match.group(1)),
        div=int(div_match.group(1)),
        kind=kind,
        exp_id=exp_id,
    )


# --- splitting a multi-well recording -------------------------------------------------------------


def _well_number(group_name: str) -> int:
    """``"well000"`` -> 0, matching how :mod:`mxtreme.extract` reads the same groups."""
    return int(group_name.removeprefix("well"))


def wells_in_file(h5_path: str | Path) -> list[int]:
    """The well numbers a MaxWell ``.h5`` holds data for, ascending.

    :raises ValueError: If the file has no ``/wells`` group (not a MaxWell wells-format file).
    """
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        if "wells" not in f:
            raise ValueError(
                f"{h5_path} has no /wells group, so its wells cannot be determined. "
                "Is it a MaxWell recording?"
            )
        return sorted(_well_number(name) for name in f["wells"])


def _rewrite_metadata_wells(out, well: int) -> None:
    """Narrow a split file's ``/assay/metadata`` blob to the one well it now holds.

    The blob is stored as ``str(dict)`` (see :func:`mxtreme.scans.mx_setup.write_metadata`) and read
    back with quotes swapped (see :func:`mxtreme.io._embedded_metadata`); the same round-trip is
    used here. A file with no parseable blob is left as it is -- the path already carries the
    identity, so an unreadable blob costs nothing.
    """
    import json

    import numpy as np

    try:
        raw = out["/assay/metadata"][:][0]
        metadata = json.loads(raw.decode("utf-8").strip().replace("'", '"'))
    except (KeyError, ValueError, IndexError):
        return

    well_ids = list(metadata.get("Well IDs") or [])
    conditions = list(metadata.get("Conditions") or [])
    condition = None
    if well in well_ids:
        index = well_ids.index(well)
        if index < len(conditions):
            condition = conditions[index]

    metadata["Well IDs"] = [well]
    metadata["Conditions"] = [] if condition is None else [condition]

    del out["/assay/metadata"]
    out["assay"].create_dataset("metadata", data=np.array([str(metadata).encode("utf-8")]))


def split_by_well(
    h5_path: str | Path,
    dest_for_well: Callable[[int], Path],
    *,
    on_progress: Callable[[str], None] = print,
    delete_original: bool = False,
) -> dict[int, Path]:
    """Break a multi-well MaxWell ``.h5`` into one file per well.

    A MaxTwo records every well into a single file, but in the store each culture gets its own --
    so the shared parts (``/assay``, settings, version, root attributes) are copied into each output
    and only that well's data comes along. Both of MaxWell's views of the data are preserved:
    ``/wells/wellNNN/...`` is copied, and ``/recordings/recMMMM/wellNNN`` is re-created as a hard
    link to it, so no data is duplicated inside the output file. The metadata blob's well list is
    narrowed to the one well each file now holds.

    :param h5_path: The multi-well file. A single-well file splits fine too (one output).
    :param dest_for_well: Called with each well number; returns the full destination file path.
        Parent directories are created as needed.
    :param on_progress: Called with each progress line.
    :param delete_original: Remove ``h5_path`` after every well has been written successfully.
        On any failure the original is always kept.
    :returns: ``{well: destination}`` for every well the file held.
    :raises ValueError: If the file has no ``/wells`` group.
    :raises FileExistsError: If a destination already exists.
    """
    import h5py

    h5_path = Path(h5_path)

    written: dict[int, Path] = {}
    with h5py.File(str(h5_path), "r") as f:
        if "wells" not in f:
            raise ValueError(f"{h5_path} has no /wells group; nothing to split by.")
        well_groups = sorted(f["wells"], key=_well_number)

        for group_name in well_groups:
            well = _well_number(group_name)
            dest = Path(dest_for_well(well))
            if dest.exists():
                raise FileExistsError(f"Refusing to overwrite {dest}.")
            dest.parent.mkdir(parents=True, exist_ok=True)
            on_progress(f"  Well {well}: writing {dest.name}")

            try:
                with h5py.File(str(dest), "w") as out:
                    for key, value in f.attrs.items():
                        out.attrs[key] = value
                    for key in f:
                        if key not in ("wells", "recordings"):
                            f.copy(f[key], out, name=key)

                    out.require_group("wells")
                    f.copy(f["wells"][group_name], out["wells"], name=group_name)

                    # /recordings/recMMMM/wellNNN mirrors /wells/wellNNN/recMMMM (mxtreme.extract
                    # reads the former). Hard-link to the copy just made where the two line up; fall
                    # back to a real copy for any recording group the wells view doesn't carry.
                    if "recordings" in f:
                        for rec in f["recordings"]:
                            if group_name not in f["recordings"][rec]:
                                continue
                            rec_group = out.require_group(f"recordings/{rec}")
                            copied = f"wells/{group_name}/{rec}"
                            if copied in out:
                                rec_group[group_name] = out[copied]
                            else:
                                f.copy(f["recordings"][rec][group_name], rec_group, name=group_name)

                    _rewrite_metadata_wells(out, well)
            except BaseException:
                # A half-written output would block the retry (FileExistsError) and could be
                # mistaken for real data; the original still holds everything, so drop it.
                dest.unlink(missing_ok=True)
                raise

            written[well] = dest

    if delete_original:
        h5_path.unlink()
        on_progress(f"  Removed the combined file {h5_path.name}")

    return written


def _ensure_metadata_blob(h5_path: Path, metadata: dict) -> None:
    """Write ``/assay/metadata`` into a file that has none, making it self-describing.

    The blob is what :func:`mxtreme.extract.extract` reads for a recording's identity and what a
    registry rebuild reads for condition labels -- a scan gets it from
    :func:`mxtreme.scans.mx_setup.write_metadata`, so an ingested file should carry the same. A blob
    already present is left untouched: it was written by whoever recorded the file, and second-hand
    identity should not overwrite first-hand.
    """
    import h5py
    import numpy as np

    with h5py.File(str(h5_path), "r+") as f:
        if "/assay/metadata" in f:
            return
        f.require_group("assay").create_dataset(
            "metadata", data=np.array([str(metadata).encode("utf-8")])
        )


# --- removing a recording -------------------------------------------------------------------------


@dataclass(frozen=True)
class Removal:
    """What :func:`remove_recording` removed -- or, with ``dry_run``, would remove.

    :param recording: The raw ``.h5`` the removal was about.
    :param location: Its identity, read from its place in the tree.
    :param files: Every file removed, the recording first, then what was derived from it.
    :param trashed_to: The directory the files were moved into, or ``None`` when purged (or dry run).
    :param registry_rows: Registry rows dropped.
    :param burst_log_rows: Rows dropped from the batch's burst log.
    :param pruned_dirs: Directories left empty by the removal and removed too.
    """

    recording: Path
    location: RecordingLocation
    files: tuple[Path, ...]
    trashed_to: Path | None
    registry_rows: int
    burst_log_rows: int
    pruned_dirs: tuple[Path, ...]


def derived_files(config, h5_path: str | Path) -> list[Path]:
    """The files elsewhere in the store that were computed from one raw recording.

    An activity scan's derivatives are its electrode selection (the figures and the electrode
    list in ``electrode_selection/`` beside it). A network scan's, or an ingested experiment's,
    are the preprocessed ``.npz`` and the burst CSV for that well and DIV -- under the batch id for
    a scan (the pipeline files a scan by its batch id, see :func:`mxtreme.scans.activity_scan`),
    under the experiment name for an experiment. Analysis summaries and reports are per culture
    across DIVs, and are left alone: they are rebuilt from what remains.

    :returns: The files that exist, in a stable order. Empty if the path is not in the tree.
    """
    h5_path = Path(h5_path)
    location = parse_recording_path(h5_path, config.recordings_dir)
    if location is None:
        return []
    out: list[Path] = []
    if location.kind == "activity_scan":
        selection = h5_path.parent / "electrode_selection"
        if selection.is_dir():
            out.extend(sorted(p for p in selection.iterdir() if p.is_file() and f"well{location.well}" in p.name))
        return out
    exp = location.exp_id or location.batch.id
    well_tail = Path(location.chip) / f"well{location.well}"
    for root, pattern in (
        (config.preprocessed_dir / exp / well_tail, f"DIV{location.div}_*exp_data.npz"),
        (config.burst_data_dir / exp / well_tail, f"DIV{location.div}_*burst_data.csv"),
    ):
        if root.is_dir():
            out.extend(sorted(root.glob(pattern)))
    return out


def _drop_burst_log_rows(config, exp: str, chip: str, well: int, div: int, *, dry_run: bool) -> int:
    """Drop one recording's rows from the batch's burst log; return how many."""
    log_path = config.burst_data_dir / f"{exp}_burst_log.csv"
    if not log_path.is_file():
        return 0
    import pandas as pd

    df = pd.read_csv(log_path)
    if df.empty or not {"chip", "well", "DIV"} <= set(df.columns):
        return 0
    mask = (df["chip"].astype(str) == str(chip)) & (df["well"].astype(str) == str(well)) & (df["DIV"].astype(str) == str(div))
    n = int(mask.sum())
    if n and not dry_run:
        df[~mask].to_csv(log_path, index=False)
    return n


def remove_recording(
    h5_path: str | Path,
    config,
    *,
    derived: bool = True,
    purge: bool = False,
    dry_run: bool = False,
    reason: str = "",
    actor: str | None = None,
    on_progress: Callable[[str], None] = print,
) -> Removal:
    """Take one raw recording out of the managed store, cleanly.

    Deleting the file alone leaves a store that lies: registry rows for a recording that is gone,
    a preprocessed ``.npz`` and burst table computed from it, a burst-log row counting its bursts.
    This removes all of that together -- the recording, what was derived from it
    (:func:`derived_files`, unless ``derived=False``), their registry rows and burst-log rows --
    and journals the removal (``recording.removed`` in :mod:`mxtreme.transactions`).

    Nothing is destroyed by default: the files are *moved* into
    :attr:`~mxtreme.config.Config.trash_dir`, under a directory named for the moment and the
    recording, keeping their paths relative to the store, so a removal can be undone by moving
    them back and re-registering (:func:`mxtreme.io.rebuild_registry`). ``purge=True`` deletes
    them instead. Directories the removal leaves empty (the DIV directory, an emptied
    ``electrode_selection/``) are removed too, so the tree does not show a day that has nothing.

    The typical use is an aborted or empty scan -- a network scan with no frames, a sweep stopped
    on its first recording -- that would otherwise sit in the tree forever, registered, and fail
    every analysis pointed at it.

    :param h5_path: The recording, inside ``config.recordings_dir``.
    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :param derived: Also remove what was computed from the recording. ``False`` leaves those files
        and their registry rows in place -- they then describe a recording that is gone.
    :param purge: Delete outright instead of moving to the trash.
    :param dry_run: Work out what would be removed and report it, touching nothing.
    :param reason: Why, for the journal.
    :param actor: Who, for the journal; the OS user when ``None``.
    :param on_progress: Called with each progress line.
    :returns: What was (or would be) removed.
    :raises ValueError: If the path is not a file inside the recordings tree, or does not follow
        its layout.
    """
    import shutil
    from datetime import datetime

    from mxtreme import io, transactions

    h5_path = Path(h5_path).resolve()
    recordings_dir = Path(config.recordings_dir).resolve()
    if recordings_dir not in h5_path.parents:
        raise ValueError(f"{h5_path} is not inside the recordings tree ({recordings_dir}); nothing removed.")
    if not h5_path.is_file():
        raise ValueError(f"{h5_path} is not a file; nothing removed.")
    location = parse_recording_path(h5_path, recordings_dir)
    if location is None:
        raise ValueError(f"{h5_path} does not follow the recordings-tree layout, so its identity is unknown; nothing removed.")

    files = [h5_path] + (derived_files(config, h5_path) if derived else [])
    exp = location.exp_id or location.batch.id
    verb = "Would remove" if dry_run else "Removing"
    for f in files:
        on_progress(f"{verb} {f}")

    trash_root = None
    if not dry_run:
        if purge:
            for f in files:
                f.unlink()
        else:
            stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
            trash_root = Path(config.trash_dir) / f"{stamp}_{h5_path.name.removesuffix('.raw.h5').removesuffix('.h5')}"
            data_root = Path(config.data_root).resolve()
            for f in files:
                dest = trash_root / f.relative_to(data_root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest))
            on_progress(f"Moved to {trash_root}")

    # The registry: the recording's own rows, and its preprocessed rows when those files went.
    registry_rows = 0
    if not dry_run:
        registry_rows += io.unregister(
            config.registry_path, chip=location.chip, well=location.well, div=location.div,
            kind=location.kind, exp_id=location.exp_id, batch_id=location.batch.id,
        )
        if derived and location.kind != "activity_scan":
            registry_rows += io.unregister(
                config.registry_path, chip=location.chip, well=location.well, div=location.div,
                kind="preprocessed", exp_id=exp,
            )
    burst_log_rows = 0
    if derived and location.kind != "activity_scan":
        burst_log_rows = _drop_burst_log_rows(config, exp, location.chip, location.well, location.div, dry_run=dry_run)

    pruned: list[Path] = []
    if not dry_run:
        for d in (h5_path.parent / "electrode_selection", h5_path.parent):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
                pruned.append(d)

    removal = Removal(
        recording=h5_path,
        location=location,
        files=tuple(files),
        trashed_to=trash_root,
        registry_rows=registry_rows,
        burst_log_rows=burst_log_rows,
        pruned_dirs=tuple(pruned),
    )
    if not dry_run:
        transactions.record(
            config,
            "recording.removed",
            batch_id=location.batch,
            plate_date=location.plate_date,
            exp_id=location.exp_id,
            chip=location.chip,
            well=location.well,
            div=location.div,
            actor=actor,
            note=reason,
            data={
                "path": str(h5_path),
                "kind": location.kind,
                "trashed_to": None if trash_root is None else str(trash_root),
                "purged": bool(purge),
                "derived": [str(f) for f in files[1:]],
                "registry_rows": registry_rows,
                "burst_log_rows": burst_log_rows,
            },
        )
        on_progress(
            f"Removed {h5_path.name}: {len(files)} file(s), {registry_rows} registry row(s), "
            f"{burst_log_rows} burst-log row(s)."
        )
    return removal


# --- ingesting an exogenous recording -------------------------------------------------------------


def _validate_exp_id(exp_id: str) -> str:
    """Check an experiment id can serve as a file-name tail, and is not a reserved scan tail."""
    if not isinstance(exp_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", exp_id):
        raise ValueError(
            f"exp_id must be a non-empty string of letters, digits, '.', '-' or '_' (and start "
            f"with a letter or digit), got {exp_id!r}."
        )
    if re.fullmatch(r"(activity|network)_scan(_\d+)?", exp_id):
        raise ValueError(
            f"exp_id {exp_id!r} is reserved for scans; an ingested recording needs its own name."
        )
    return exp_id


def ingest_recording(
    h5_path: str | Path,
    config,
    *,
    batch: Batch | str,
    plate_date,
    chip: str,
    div: int,
    exp_id: str,
    conditions: dict[int, object] | None = None,
    move: bool = False,
    on_progress: Callable[[str], None] = print,
) -> dict[int, Path]:
    """Bring an exogenous ``.h5`` (recorded outside MXtreme) into the managed store.

    The file is placed in the recordings tree under the identity given here, named
    ``<stem>_<exp_id>.raw.h5``, and one ``experiment`` row per well is upserted into the registry --
    exactly as if MXtreme had recorded it. A file holding several wells is split into one file per
    well on the way in (see :func:`split_by_well`), so each culture's directory holds its own data.
    An ingested copy with no ``/assay/metadata`` blob gets one written from the identity given here,
    so :func:`mxtreme.extract.extract` and a registry rebuild can read it back without being handed
    the metadata again; a blob the file already carries is left as recorded. ::

        from mxtreme.config import Config
        from mxtreme.store import ingest_recording

        config = Config.from_toml("mxtreme.toml")
        ingest_recording(
            "/data/exports/stim_session.raw.h5", config,
            batch="fall2026_batch1_DRG_M1", plate_date=260810,
            chip="M07460", div=21, exp_id="burstTrainer_trial3",
        )

    :param h5_path: The recording to ingest. Must be a MaxWell wells-format file (``/wells/...``).
    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :param batch: The plating batch the culture belongs to, as a :class:`Batch` or its id string.
    :param plate_date: Plating date, ``YYMMDD``.
    :param chip: Chip serial, e.g. ``"M07460"``.
    :param div: Days *in vitro* at the time of the recording.
    :param exp_id: Free-form name for this experiment -- becomes the file-name tail and the
        registry row's ``exp_id``. Letters, digits, ``.``, ``-``, ``_``.
    :param conditions: Optional per-well condition labels, keyed by well number.
    :param move: Move the file instead of copying it. A multi-well source is removed after a
        successful split either way when this is set; on any failure the original is kept.
    :param on_progress: Called with each progress line.
    :returns: ``{well: destination path}`` for every well ingested.
    :raises FileNotFoundError: If ``h5_path`` does not exist.
    :raises FileExistsError: If a destination file already exists -- nothing is overwritten.
    :raises ValueError: On a malformed batch id, plate date, or exp id, or a file with no wells.
    """
    import shutil

    from mxtreme import io

    h5_path = Path(h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"No such file: {h5_path}")

    batch = Batch.parse(batch)
    _validate_exp_id(exp_id)
    conditions = conditions or {}
    wells = wells_in_file(h5_path)

    def destination(well: int) -> Path:
        stem = recording_stem(batch, plate_date, chip, well, div)
        return recording_dir(config, batch, plate_date, chip, well, div) / f"{stem}_{exp_id}.raw.h5"

    if len(wells) == 1:
        # One well: a straight file copy (or move) is faster and bit-exact; nothing to split.
        well = wells[0]
        dest = destination(well)
        if dest.exists():
            raise FileExistsError(f"Refusing to overwrite {dest}.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        on_progress(f"{'Moving' if move else 'Copying'} {h5_path.name} -> {dest}")
        if move:
            shutil.move(str(h5_path), str(dest))
        else:
            shutil.copy2(str(h5_path), str(dest))
        written = {well: dest}
    else:
        on_progress(f"Splitting {h5_path.name} into {len(wells)} per-well files...")
        written = split_by_well(
            h5_path, destination, on_progress=on_progress, delete_original=move
        )

    for well, dest in written.items():
        condition = conditions.get(well)
        _ensure_metadata_blob(dest, {
            "Exp ID": exp_id,
            "Batch ID": batch.id,
            "Chip ID": chip,
            "Plate date": _validate_plate_date(plate_date),
            "DIV": int(div),
            "Well IDs": [well],
            "Conditions": [] if condition is None else [condition],
        })

    io.register(
        {
            well: {
                "exp_id": exp_id,
                "chip": chip,
                "DIV": int(div),
                "experimental_condition": conditions.get(well),
                "batch_id": batch.id,
                "plate_date": _validate_plate_date(plate_date),
            }
            for well in written
        },
        config.registry_path,
        kind="experiment",
    )
    on_progress(f"Registered wells {sorted(written)} in {config.registry_path}")

    from mxtreme import transactions

    for well, dest in written.items():
        transactions.record(
            config,
            "recording.ingested",
            batch_id=batch,
            plate_date=plate_date,
            exp_id=exp_id,
            chip=chip,
            well=well,
            div=div,
            data={"source": str(h5_path), "path": str(dest), "moved": bool(move)},
        )

    return written
