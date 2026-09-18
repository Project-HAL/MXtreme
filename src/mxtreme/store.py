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

A culture can be network-scanned more than once on one DIV. The first such file is
``..._network_scan.raw.h5``; when a second is ingested the two are numbered in arrival order,
``..._network_scan_0.raw.h5`` and ``..._network_scan_1.raw.h5``, and later ones take the next
index (:func:`network_scan_slot`). Every one of them is a network scan -- the number is not an
experiment name -- and the readers here accept either form.

Each file in the tree holds exactly one well, so one culture's whole history sits in one directory.
A multi-well recording (a MaxTwo) is recorded as a single ``.h5`` and then broken apart by
:func:`split_by_well`.

The module is import-light on purpose: everything here except splitting/ingest is pure path and
string logic, and ``h5py``/``numpy``/``mxtreme.io`` are imported inside the functions that need
them -- so ``import mxtreme`` (which re-exports :class:`Batch`) stays fast and rig-safe.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

#: Semesters a batch id may name, in calendar order.
SEMESTERS = ("spring", "summer", "fall", "winter")

#: MaxWell systems a batch may be plated on: MaxOne or MaxTwo.
SYSTEMS = ("M1", "M2")

#: What a file in the recordings tree can be. ``"experiment"`` is an ingested exogenous recording;
#: the other two are scans -- run by MXtreme itself, or recorded by MaxLab Live's own assays and
#: ingested as scans (see :func:`ingest_recording`).
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
#: ``..._network_scan_1.raw.h5`` -- the numeric suffix numbers several scans of one DIV, see
#: :func:`network_scan_slot`; MaxLab's own collision rename produces the same shape).
_SCAN_TAIL = re.compile(r"_(activity|network)_scan(_\d+)?\.raw\.h5$")

#: The index a network scan file carries: ``None`` for the unnumbered ``_network_scan``, else the
#: number after it.
_NETWORK_INDEX = re.compile(r"_network_scan(?:_(\d+))?\.raw\.h5$")

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


def network_scan_index(h5_path: str | Path) -> int | None:
    """Where a network scan file sits among its DIV's: ``0`` for the unnumbered
    ``..._network_scan.raw.h5``, ``n`` for ``..._network_scan_<n>.raw.h5``, ``None`` for a file
    that is not a network scan. Sorting a DIV's network scans by this gives arrival order."""
    match = _NETWORK_INDEX.search(Path(h5_path).name)
    if match is None:
        return None
    return 0 if match.group(1) is None else int(match.group(1))


def network_scan_slot(div_dir: str | Path, stem: str) -> tuple[str, list[tuple[Path, Path]]]:
    """The tail the next network scan of one well and DIV takes, and the renames that make room.

    A DIV with no network scan gets the plain ``network_scan`` tail. A DIV that already has one
    numbers them from zero in arrival order: the existing unnumbered file is renamed to
    ``network_scan_0`` and the new one becomes ``network_scan_1``; a DIV whose scans are already
    numbered hands out the next free index. Nothing is renamed here -- the caller applies the
    renames it is handed (:func:`ingest_recording` does, and journals them) -- so a front end can
    ask where a file *would* land.

    :param div_dir: The culture's ``DIV_<n>`` directory (need not exist yet).
    :param stem: The well's file stem for that DIV, from :func:`recording_stem`.
    :returns: ``(tail, renames)``: the tail to put after ``<stem>_`` and the ``(old, new)`` pairs
        to move first, in order.
    """
    div_dir = Path(div_dir)
    numbered: dict[int, Path] = {}
    plain = None
    for path in div_dir.glob(f"{stem}_network_scan*.raw.h5") if div_dir.is_dir() else []:
        match = _NETWORK_INDEX.search(path.name)
        if match is None:
            continue
        if match.group(1) is None:
            plain = path
        else:
            numbered[int(match.group(1))] = path
    if plain is None and not numbered:
        return "network_scan", []
    renames: list[tuple[Path, Path]] = []
    if plain is not None:
        index = 0
        while index in numbered:  # a stray ``_0`` beside the plain file: give the plain one the next gap
            index += 1
        target = div_dir / f"{stem}_network_scan_{index}.raw.h5"
        renames.append((plain, target))
        numbered[index] = target
    return f"network_scan_{max(numbered) + 1}", renames


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
    wells: Iterable[int] | None = None,
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
    :param wells: Only split out these wells; ``None`` means every well the file holds. A well
        asked for that the file does not hold is an error, not silently skipped.
    :returns: ``{well: destination}`` for every well written.
    :raises ValueError: If the file has no ``/wells`` group, or ``wells`` names one it lacks.
    :raises FileExistsError: If a destination already exists.
    """
    import h5py

    h5_path = Path(h5_path)

    written: dict[int, Path] = {}
    with h5py.File(str(h5_path), "r") as f:
        if "wells" not in f:
            raise ValueError(f"{h5_path} has no /wells group; nothing to split by.")
        well_groups = sorted(f["wells"], key=_well_number)
        if wells is not None:
            wanted = sorted({int(w) for w in wells})
            present = {_well_number(name) for name in well_groups}
            missing = [w for w in wanted if w not in present]
            if missing:
                raise ValueError(
                    f"{h5_path.name} holds no well {missing} (it has {sorted(present)})."
                )
            well_groups = [name for name in well_groups if _well_number(name) in wanted]

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


# --- describing and ingesting an exogenous recording ----------------------------------------------


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


def _text(dataset) -> str | None:
    """The string a one-element MaxLab text dataset holds (``|S`` or variable-length), or ``None``."""
    try:
        value = dataset[()]
    except Exception:  # noqa: BLE001 -- an unreadable dataset is simply not a value
        return None
    try:
        if hasattr(value, "shape") and value.shape != ():
            value = value[0]
    except (IndexError, TypeError):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    return str(value).strip() or None


def _scope_plating_date(raw: str | None) -> int | None:
    """MaxLab Live's per-well ``Plating Date`` (``28.3.2024``, ``2024-03-28``...) as ``YYMMDD``."""
    if not raw:
        return None
    from datetime import date

    for pattern in (r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", r"^(\d{4})-(\d{1,2})-(\d{1,2})$", r"^(\d{1,2})/(\d{1,2})/(\d{4})$"):
        match = re.match(pattern, raw.strip())
        if not match:
            continue
        parts = [int(x) for x in match.groups()]
        year, month, day = (parts[0], parts[1], parts[2]) if parts[0] > 31 else (parts[2], parts[1], parts[0])
        try:
            when = date(year, month, day)
        except ValueError:
            return None
        return int(f"{when:%y%m%d}")
    if re.fullmatch(r"\d{6}", raw.strip()):
        return int(raw.strip())
    return None


@dataclass(frozen=True)
class WellDescription:
    """What one well of a MaxWell ``.h5`` holds, for :class:`RecordingDescription`.

    :param well: Well number.
    :param n_recordings: How many ``recNNNN`` groups the well has.
    :param n_spikes: Spikes across all of them.
    :param n_channels: Channels routed in the first recording (its ``settings/mapping`` size).
    :param n_electrodes: Distinct electrodes across every recording's mapping.
    :param sampling_hz: The first recording's sampling rate, or ``None``.
    :param seconds: Total recorded seconds across the well's recordings, from MaxLab's per-recording
        start/stop stamps, or ``None`` when the file carries none.
    :param has_raw: Whether any recording saved raw traces (``groups/*/raw``), as a network scan does.
    :param plating_date: The well's ``Plating Date`` from MaxLab Live's wellplate metadata, as
        ``YYMMDD``, when someone filled it in in Scope and it parses; else ``None``.
    :param wellplate: Every other string MaxLab Live stored about the well (``Plating Type``,
        ``Cells``, ``group_name``...), verbatim, for a front end to show.
    """

    well: int
    n_recordings: int
    n_spikes: int
    n_channels: int
    n_electrodes: int
    sampling_hz: float | None
    seconds: float | None
    has_raw: bool
    plating_date: int | None
    wellplate: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RecordingDescription:
    """What a MaxWell ``.h5`` says about itself, before anyone tells the store what it is.

    Read by :func:`describe_recording` from the file alone. ``problems`` is the verdict: an empty
    list means the file is a MaxWell recording the store can hold (``/wells`` with recordings
    that carry spikes and settings); anything listed there is a reason it cannot be ingested as it
    is. ``warnings`` are oddities worth showing that do not block an ingest.

    :param path: The file.
    :param size: Its size in bytes.
    :param chip: The chip serial MaxLab wrote to ``/wellplate/id`` (Scope's own chip readout), or
        ``None``.
    :param wellplate_version: MaxLab's plate description, e.g. ``"MaxTwo 6 multi-well MEA"``.
    :param system: ``"M1"`` or ``"M2"`` as far as the plate description says, else ``None``.
    :param script_id: The MaxLab Live assay that recorded the file (``ActivityScan_v1.0``...), if
        it was one of Scope's assays; ``None`` for a plain recording or an MXtreme scan.
    :param mxw_version: MaxLab Live's version string.
    :param recorded_at: When the first recording started, ISO 8601 in local time, or ``None``.
    :param recorded_date: The same as a calendar date (``YYMMDD``), or ``None``.
    :param record_time: ``/assay/inputs/record_time`` in seconds, when the file has it.
    :param kind_guess: What the file looks like -- one of :data:`RECORDING_KINDS`: several
        recordings per well on different electrode sets is an activity scan, one recording with
        raw traces is a network scan, anything else an experiment. A guess to confirm, not a fact.
    :param metadata: The ``/assay/metadata`` blob MXtreme writes into its own scans, decoded, or
        ``None`` for a file recorded outside MXtreme.
    :param wells: One :class:`WellDescription` per well the file holds data for, ascending.
    :param problems: Why the file cannot be ingested; empty when it can.
    :param warnings: Oddities that do not block an ingest.
    """

    path: Path
    size: int
    chip: str | None
    wellplate_version: str | None
    system: str | None
    script_id: str | None
    mxw_version: str | None
    recorded_at: str | None
    recorded_date: int | None
    record_time: int | None
    kind_guess: str
    metadata: dict | None
    wells: list[WellDescription] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the file can be ingested as it is."""
        return not self.problems

    @property
    def plating_date(self) -> int | None:
        """The one plating date the wells agree on, or ``None`` when they disagree or have none."""
        dates = {w.plating_date for w in self.wells if w.plating_date is not None}
        return dates.pop() if len(dates) == 1 else None


def describe_recording(h5_path: str | Path) -> RecordingDescription:
    """Read what a MaxWell ``.h5`` says about itself, for an ingest that asks before it files.

    Nothing here needs the store: this is the file's own account -- which chip Scope read, which
    wells hold data, how many recordings each has, when it was recorded, whether it looks like an
    activity scan -- so a front end can fill in what it can and ask only for what the file cannot
    say (the batch, and with it the DIV). It never writes.

    A file that is not HDF5, has no ``/wells`` group, or whose wells hold no recording with spikes
    and settings is described with ``problems`` listing why; it does not raise.

    :param h5_path: The recording.
    :returns: The description; check :attr:`RecordingDescription.ok` before ingesting.
    """
    import json
    from datetime import datetime

    h5_path = Path(h5_path)
    problems: list[str] = []
    warnings: list[str] = []
    wells: list[WellDescription] = []
    chip = wellplate_version = script_id = mxw_version = None
    record_time = None
    metadata = None
    starts: list[int] = []

    def _describe(path: Path, size: int, **kw) -> RecordingDescription:
        return RecordingDescription(path=path, size=size, **kw)

    try:
        size = h5_path.stat().st_size
    except OSError as exc:
        return _describe(
            h5_path, 0, chip=None, wellplate_version=None, system=None, script_id=None,
            mxw_version=None, recorded_at=None, recorded_date=None, record_time=None,
            kind_guess="experiment", metadata=None, problems=[f"Cannot read the file: {exc}"],
        )

    import h5py

    try:
        f = h5py.File(str(h5_path), "r")
    except OSError as exc:
        return _describe(
            h5_path, size, chip=None, wellplate_version=None, system=None, script_id=None,
            mxw_version=None, recorded_at=None, recorded_date=None, record_time=None,
            kind_guess="experiment", metadata=None,
            problems=[f"Not an HDF5 file (h5py could not open it: {str(exc).splitlines()[0]})."],
        )

    with f:
        if "wellplate" in f:
            chip = _text(f["wellplate/id"]) if "id" in f["wellplate"] else None
            wellplate_version = _text(f["wellplate/version"]) if "version" in f["wellplate"] else None
        if "assay" in f:
            if "script_id" in f["assay"]:
                script_id = _text(f["assay/script_id"])
            if "inputs" in f["assay"] and "record_time" in f["assay/inputs"]:
                try:
                    record_time = int(float(_text(f["assay/inputs/record_time"]) or ""))
                except ValueError:
                    warnings.append("/assay/inputs/record_time is present but not a number.")
            if "metadata" in f["assay"]:
                try:
                    metadata = json.loads(
                        (_text(f["assay/metadata"]) or "").replace("'", '"')
                    )
                except ValueError:
                    warnings.append("/assay/metadata is present but not readable.")
        mxw_version = _text(f["mxw_version"]) if "mxw_version" in f else None
        if "version" not in f and mxw_version is None:
            warnings.append("No /version or /mxw_version: the file was not written by MaxLab Live.")

        if "wells" not in f:
            problems.append("No /wells group: not a MaxWell recording (or a very old one).")
        else:
            well_names = []
            for name in f["wells"]:
                try:
                    well_names.append((_well_number(name), name))
                except ValueError:
                    warnings.append(f"/wells/{name} is not a wellNNN group; ignored.")
            if not well_names:
                problems.append("/wells holds no wellNNN groups.")
            for well, name in sorted(well_names):
                group = f["wells"][name]
                recordings = [r for r in group if isinstance(group[r], h5py.Group)]
                if not recordings:
                    warnings.append(f"Well {well} has no recordings; it will be skipped.")
                    continue
                n_spikes = n_channels = 0
                electrodes: set[int] = set()
                sampling = None
                seconds = 0.0
                have_seconds = False
                has_raw = False
                bad = []
                for rec in sorted(recordings):
                    r = group[rec]
                    if "spikes" not in r or "settings" not in r or "mapping" not in r["settings"]:
                        bad.append(rec)
                        continue
                    try:
                        n_spikes += int(r["spikes"].shape[0])
                        mapping = r["settings/mapping"]
                        if not n_channels:
                            n_channels = int(mapping.shape[0])
                        if "electrode" in (mapping.dtype.names or ()):
                            electrodes.update(int(e) for e in mapping["electrode"][:])
                        if sampling is None and "sampling" in r["settings"]:
                            sampling = float(r["settings/sampling"][0])
                    except Exception as exc:  # noqa: BLE001 -- one broken recording, reported below
                        bad.append(f"{rec} ({type(exc).__name__})")
                        continue
                    if "start_time" in r and "stop_time" in r:
                        try:
                            start, stop = int(r["start_time"][0]), int(r["stop_time"][0])
                        except (IndexError, ValueError, TypeError):
                            start = stop = 0
                        if stop > start > 0:
                            seconds += (stop - start) / 1000
                            have_seconds = True
                            starts.append(start)
                    if "groups" in r and any("raw" in r["groups"][g] for g in r["groups"]):
                        has_raw = True
                if bad:
                    problems.append(
                        f"Well {well}: recording(s) {', '.join(bad)} lack spikes or settings/mapping."
                    )
                good = len(recordings) - len(bad)
                if good and n_spikes == 0:
                    warnings.append(f"Well {well} recorded no spikes at all.")
                plate = None
                info: dict[str, str] = {}
                wp = f"wellplate/{name}"
                if wp in f:
                    for key in f[wp]:
                        value = _text(f[wp][key])
                        if value is not None:
                            info[key] = value
                    plate = _scope_plating_date(info.get("Plating Date"))
                wells.append(
                    WellDescription(
                        well=well, n_recordings=good, n_spikes=n_spikes, n_channels=n_channels,
                        n_electrodes=len(electrodes), sampling_hz=sampling,
                        seconds=seconds if have_seconds else None, has_raw=has_raw,
                        plating_date=plate, wellplate=info,
                    )
                )
            if not wells and not problems:
                problems.append("No well holds a recording.")

    system = None
    if wellplate_version:
        lowered = wellplate_version.lower()
        system = "M2" if "maxtwo" in lowered else "M1" if "maxone" in lowered else None

    recorded_at = recorded_date = None
    if starts:
        when = datetime.fromtimestamp(min(starts) / 1000).astimezone()
        recorded_at = when.isoformat(timespec="seconds")
        recorded_date = int(f"{when:%y%m%d}")

    if script_id and "activityscan" in script_id.replace(" ", "").lower():
        kind_guess = "activity_scan"
    elif wells and all(w.n_recordings >= 2 and w.n_electrodes > w.n_channels for w in wells):
        kind_guess = "activity_scan"
    elif wells and all(w.n_recordings == 1 and w.has_raw for w in wells):
        kind_guess = "network_scan"
    else:
        kind_guess = "experiment"

    return RecordingDescription(
        path=h5_path, size=size, chip=chip, wellplate_version=wellplate_version, system=system,
        script_id=script_id, mxw_version=mxw_version, recorded_at=recorded_at,
        recorded_date=recorded_date, record_time=record_time, kind_guess=kind_guess,
        metadata=metadata, wells=wells, problems=problems, warnings=warnings,
    )


def ingest_recording(
    h5_path: str | Path,
    config,
    *,
    batch: Batch | str,
    plate_date,
    chip: str,
    div: int,
    kind: str = "experiment",
    exp_id: str | None = None,
    wells: Iterable[int] | None = None,
    conditions: dict[int, object] | None = None,
    move: bool = False,
    actor: str | None = None,
    note: str = "",
    on_progress: Callable[[str], None] = print,
) -> dict[int, Path]:
    """Bring an exogenous ``.h5`` (recorded outside MXtreme) into the managed store.

    The file is placed in the recordings tree under the identity given here and one row per well
    is upserted into the registry -- exactly as if MXtreme had recorded it. What it is filed *as*
    is ``kind``: an ``experiment`` (the default) is named ``<stem>_<exp_id>.raw.h5`` after the
    free-form experiment name; an ``activity_scan`` or ``network_scan`` recorded by MaxLab Live's
    own assays (or any other tool) takes the scan tail instead, ``<stem>_activity_scan.raw.h5``,
    and registers as that kind, so it is indistinguishable in the store from a scan MXtreme ran
    -- electrode selection, the analysis chain and a registry rebuild all read it the same way.
    An activity scan lacking ``/assay/inputs/record_time`` gets it derived from its own
    timestamps (:func:`mxtreme.scans.activity_scan.ensure_record_time`), since the selection
    pipeline needs it. A network scan may join others of the same well and DIV: they are
    numbered in arrival order (:func:`network_scan_slot`), the DIV's existing unnumbered scan
    becoming ``_network_scan_0`` as the new one lands as ``_network_scan_1``; that rename is
    noted in the new file's journal record. The registry keeps one ``network_scan`` row per well
    and DIV whichever way, as it does for a scan MaxLab re-ran.

    A file holding several wells is split into one file per well on the way in (see
    :func:`split_by_well`), so each culture's directory holds its own data; ``wells`` narrows that
    to the wells worth keeping (the plated ones of a MaxTwo plate). An ingested copy with no
    ``/assay/metadata`` blob gets one written from the identity given here, so
    :func:`mxtreme.extract.extract` and a registry rebuild can read it back without being handed
    the metadata again; a blob the file already carries is left as recorded. ::

        from mxtreme.config import Config
        from mxtreme.store import ingest_recording

        config = Config.from_toml("mxtreme.toml")
        ingest_recording(
            "/data/exports/stim_session.raw.h5", config,
            batch="fall2026_batch1_DRG_M1", plate_date=260810,
            chip="M07460", div=21, exp_id="burstTrainer_trial3",
        )
        ingest_recording(                      # a Scope activity scan
            "/data/scope/M07460_260831.h5", config,
            batch="fall2026_batch1_DRG_M1", plate_date=260810,
            chip="M07460", div=21, kind="activity_scan",
        )

    :param h5_path: The recording to ingest. Must be a MaxWell wells-format file (``/wells/...``);
        :func:`describe_recording` says beforehand whether it is, and what it looks like.
    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :param batch: The plating batch the culture belongs to, as a :class:`Batch` or its id string.
    :param plate_date: Plating date, ``YYMMDD``.
    :param chip: Chip serial, e.g. ``"M07460"``.
    :param div: Days *in vitro* at the time of the recording.
    :param kind: What to file it as -- one of :data:`RECORDING_KINDS`.
    :param exp_id: For an ``experiment``, its free-form name -- becomes the file-name tail and the
        registry row's ``exp_id``. Letters, digits, ``.``, ``-``, ``_``. Required for an
        experiment; not accepted for a scan (a scan's identity is its batch).
    :param wells: Which of the file's wells to ingest; ``None`` means all of them.
    :param conditions: Optional per-well condition labels, keyed by well number.
    :param move: Move the file instead of copying it. A multi-well source is removed after a
        successful split either way when this is set; on any failure the original is kept.
    :param actor: Who, for the journal; the OS user when ``None``.
    :param note: A remark for the journal (where the file came from, say).
    :param on_progress: Called with each progress line.
    :returns: ``{well: destination path}`` for every well ingested.
    :raises FileNotFoundError: If ``h5_path`` does not exist.
    :raises FileExistsError: If a destination file already exists -- nothing is overwritten. A
        network scan never collides (it takes the next index instead).
    :raises ValueError: On a malformed batch id, plate date, kind or exp id, a file with no wells,
        or ``wells`` naming one the file lacks.
    """
    import shutil

    from mxtreme import io

    h5_path = Path(h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"No such file: {h5_path}")

    batch = Batch.parse(batch)
    if kind not in RECORDING_KINDS:
        raise ValueError(f"kind must be one of {RECORDING_KINDS}, got {kind!r}.")
    if kind == "experiment":
        if exp_id is None:
            raise ValueError("An experiment needs an exp_id; a scan takes kind='activity_scan' "
                             "or 'network_scan' instead.")
        _validate_exp_id(exp_id)
        tail = exp_id
    else:
        if exp_id is not None:
            raise ValueError(f"A {kind} carries no exp_id (its identity is the batch); got {exp_id!r}.")
        tail = kind
    conditions = conditions or {}
    present = wells_in_file(h5_path)
    if wells is None:
        chosen = present
    else:
        chosen = sorted({int(w) for w in wells})
        missing = [w for w in chosen if w not in present]
        if missing:
            raise ValueError(f"{h5_path.name} holds no well {missing} (it has {present}).")
        if not chosen:
            raise ValueError("wells is empty: nothing to ingest.")

    slots: dict[int, tuple[str, list[tuple[Path, Path]]]] = {}

    def destination(well: int) -> Path:
        stem = recording_stem(batch, plate_date, chip, well, div)
        div_dir = recording_dir(config, batch, plate_date, chip, well, div)
        well_tail = tail
        if kind == "network_scan":
            if well not in slots:
                slots[well] = network_scan_slot(div_dir, stem)
            well_tail = slots[well][0]
        return div_dir / f"{stem}_{well_tail}.raw.h5"

    # A network scan joining others of its DIV: number what is there before anything lands, so the
    # new file's name and the renames it caused are settled together (and journaled together).
    renamed: dict[int, dict[str, str]] = {}
    for well in chosen:
        destination(well)
        for old, new in slots.get(well, ("", []))[1]:
            if new.exists():
                raise FileExistsError(f"Refusing to overwrite {new}.")
            on_progress(f"Numbering the DIV's earlier network scan: {old.name} -> {new.name}")
            old.rename(new)
            renamed.setdefault(well, {})[str(old)] = str(new)

    if len(present) == 1 and chosen == present:
        # One well: a straight file copy (or move) is faster and bit-exact; nothing to split.
        well = present[0]
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
        on_progress(f"Splitting {h5_path.name} into {len(chosen)} per-well file(s)...")
        written = split_by_well(
            h5_path, destination, on_progress=on_progress, delete_original=move, wells=chosen
        )

    for well, dest in written.items():
        condition = conditions.get(well)
        _ensure_metadata_blob(dest, {
            # A scan has no experiment name -- its identity is the batch, as ActivityScanParams
            # writes it. An experiment carries its own.
            "Exp ID": exp_id if kind == "experiment" else batch.id,
            "Batch ID": batch.id,
            "Chip ID": chip,
            "Plate date": _validate_plate_date(plate_date),
            "DIV": int(div),
            "Well IDs": [well],
            "Conditions": [] if condition is None else [condition],
        })
        if kind == "activity_scan":
            from mxtreme.scans.activity_scan import ensure_record_time, has_record_time

            if not has_record_time(dest):
                try:
                    seconds = ensure_record_time(dest)
                except ValueError as exc:
                    on_progress(f"  Well {well}: {exc}")
                else:
                    on_progress(f"  Well {well}: wrote record_time = {seconds} s from the file's timestamps")

    io.register(
        {
            well: {
                # The blank exp_id is what separates scan rows from `experiment` rows, as
                # io.register_scan writes them.
                "exp_id": exp_id if kind == "experiment" else "",
                "chip": chip,
                "DIV": int(div),
                "experimental_condition": conditions.get(well),
                "batch_id": batch.id,
                "plate_date": _validate_plate_date(plate_date),
            }
            for well in written
        },
        config.registry_path,
        kind=kind,
    )
    on_progress(f"Registered wells {sorted(written)} as {kind} in {config.registry_path}")

    from mxtreme import transactions

    for well, dest in written.items():
        data = {"source": str(h5_path), "path": str(dest), "moved": bool(move), "kind": kind}
        if well in renamed:
            data["renamed"] = renamed[well]
        transactions.record(
            config,
            "recording.ingested",
            batch_id=batch,
            plate_date=plate_date,
            exp_id=exp_id if kind == "experiment" else batch.id,
            chip=chip,
            well=well,
            div=div,
            actor=actor,
            note=note,
            data=data,
        )

    return written


# --- renaming a batch -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchRename:
    """What :func:`rename_batch` changed, or -- with ``dry_run`` -- would change.

    :param old: The batch as it was.
    :param new: The batch as it is now.
    :param paths: Every directory and file renamed, as ``(before, after)`` pairs.
    :param h5_files: Raw recordings whose ``/assay/metadata`` blob was rewritten (paths as before).
    :param npz_files: Preprocessed files whose embedded identity was rewritten (paths as before).
    :param csv_files: Burst logs and analysis summaries whose rows named the batch (paths as before).
    :param registry_rows: How many registry rows named the batch.
    :param plate_dates: Every plating date the batch has in the recordings tree (an id can be plated
        more than once); the ``batch.renamed`` journal record is written once per date.
    """

    old: Batch
    new: Batch
    paths: list[tuple[Path, Path]]
    h5_files: list[Path]
    npz_files: list[Path]
    csv_files: list[Path]
    registry_rows: int
    plate_dates: tuple[int, ...]


def _id_token(batch_id: str) -> re.Pattern[str]:
    """Match ``batch_id`` where it stands as a whole token in a name or a CSV cell.

    Every place the store writes a batch id butts it against ``_``, ``,``, a quote or an end, never
    against a letter or digit -- so this is what separates ``fall2026_batch1_E18_M1`` from
    ``fall2026_batch12_E18_M1``. It cannot separate an id from one whose cell type *contains* it
    (``fall2026_batch1_E18_M1_E18_M1``); :func:`rename_batch` checks the store for that case first.
    """
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(batch_id)}(?![A-Za-z0-9])")


def _batch_ids_on_disk(config) -> set[str]:
    """Every batch id a top-level directory in the store is named by."""
    found = set()
    if config.recordings_dir.is_dir():
        for entry in config.recordings_dir.iterdir():
            match = _PLATING_DIR.match(entry.name)
            if match:
                found.add(match["batch_id"])
    for root in (config.preprocessed_dir, config.burst_data_dir):
        if root.is_dir():
            found.update(entry.name for entry in root.iterdir() if entry.is_dir())
    return found


def rename_batch(
    config,
    old: Batch | str,
    new: Batch | str,
    *,
    dry_run: bool = False,
    reason: str = "",
    actor: str | None = None,
    on_progress: Callable[[str], None] = lambda _: None,
) -> BatchRename:
    """Give a plating batch a new id everywhere the store names it.

    A batch id is written into far more than its plating directory: every file name under
    ``recordings/``, ``preprocessed/``, ``burst_data/`` and ``analysis/`` carries it; so do the
    registry rows, the ``/assay/metadata`` blob inside every raw ``.h5``, the ``exp_id`` and
    ``path_to_h5`` fields inside every preprocessed ``.npz`` (which is where a registry rebuild reads
    identity from), and the ``exp_id`` / ``culture_id`` columns of the burst log and the analysis
    summary CSVs. This rewrites all of them, so that afterwards :func:`list_platings`,
    :func:`mxtreme.io.rebuild_registry` and :func:`mxtreme.paths.resolve_paths` all agree on the
    new name. Report PDFs are renamed but not regenerated: their text still says the old name until
    they are produced again.

    The plate date is not part of the batch id and is left alone -- the batch keeps its place in the
    calendar, only its label changes. The system (``M1``/``M2``) is refused: it names the hardware
    the recordings came off, and the chip directories and well counts assume it.

    Contents (blobs, ``.npz`` fields, CSV rows, the registry) are rewritten before any path moves,
    and paths are renamed deepest-first with the top-level directories last, so the failures most
    likely on a live store -- a file held open, a permission -- surface before anything has moved.
    The operation is not transactional beyond that ordering: a failure part-way leaves a store that
    is partly renamed, and ``dry_run`` is the way to see the whole plan first. Nothing here checks
    whether a scan is still writing into the batch; the caller knows that, and must not rename a
    batch it is recording into.

    The rename is journaled last (``batch.renamed`` in :mod:`mxtreme.transactions`, against the new
    id, one record per plating date the batch has), and the log's readers follow it: everything
    journaled under the old id -- scans, analyses, marks -- reads back under the new one, so a
    culture's history and its alive/dead state survive the rename without the log being rewritten.

    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :param old: The batch as it is named now.
    :param new: The batch id it should have. Validated by :meth:`Batch.parse`, so a malformed id
        is refused before anything is touched.
    :param dry_run: Collect and report what would change without changing it.
    :param reason: Why, in the caller's words; goes into the journal record's ``note``.
    :param actor: Who is renaming, for the journal. ``None`` records the OS user.
    :param on_progress: Called with a line of progress text as each group of files is done.
    :raises ValueError: If the new id is malformed, equals the old one, changes the system, or is a
        substring of another batch's id in the store (which the rename could not tell apart).
    :raises FileNotFoundError: If nothing in the store is named by ``old``.
    :raises FileExistsError: If something in the store is already named by ``new``.
    :returns: A :class:`BatchRename` listing everything changed (or, with ``dry_run``, to change).
    """
    import os

    import h5py
    import numpy as np

    old, new = Batch.parse(old), Batch.parse(new)
    if old.id == new.id:
        raise ValueError(f"Batch is already named {old.id!r}.")
    if old.system != new.system:
        raise ValueError(
            f"Cannot move batch {old.id!r} from {old.system} to {new.system}: the system names the "
            "hardware the recordings came off, and the store's layout assumes it."
        )

    old_token, new_token = _id_token(old.id), _id_token(new.id)
    on_disk = _batch_ids_on_disk(config)
    nested = sorted(b for b in on_disk if b != old.id and old_token.search(b))
    if nested:
        raise ValueError(
            f"Batch id {old.id!r} occurs inside other batch ids in this store ({nested}); "
            "a rename could not tell their files apart."
        )

    plate_dates = tuple(sorted({
        int(m["plate_date"])
        for entry in (config.recordings_dir.iterdir() if config.recordings_dir.is_dir() else ())
        if (m := _PLATING_DIR.match(entry.name)) and m["batch_id"] == old.id
    }))

    roots = [config.recordings_dir, config.preprocessed_dir, config.burst_data_dir, config.analysis_dir]
    registry_text = config.registry_path.read_text() if config.registry_path.is_file() else ""

    def _named(root: Path, token: re.Pattern[str]) -> list[Path]:
        return sorted(p for p in root.rglob("*") if token.search(p.name)) if root.is_dir() else []

    paths = [p for root in roots for p in _named(root, old_token)]
    registry_rows = sum(bool(old_token.search(line)) for line in registry_text.splitlines())
    if not paths and not registry_rows:
        raise FileNotFoundError(f"Nothing in {config.data_root} is named by batch {old.id!r}.")

    taken = [p for root in roots for p in _named(root, new_token)]
    if taken or new_token.search(registry_text):
        where = taken[0] if taken else config.registry_path
        raise FileExistsError(f"Batch {new.id!r} is already present in the store: {where}")

    def _keep_mtime(path: Path, write: Callable[[], None]) -> None:
        stat = path.stat()
        write()
        os.utime(path, (stat.st_atime, stat.st_mtime))

    # --- contents first: the identity each file carries inside it -----------------------------
    old_bytes = re.compile(old_token.pattern.encode())
    h5_files = []
    for h5_path in (p for p in paths if p.suffix == ".h5" and p.is_file()):
        with h5py.File(str(h5_path), "r") as f:
            raw = bytes(f["/assay/metadata"][:][0]) if "/assay/metadata" in f else b""
        if not old_bytes.search(raw):
            continue
        h5_files.append(h5_path)
        if dry_run:
            continue

        def _rewrite_blob(h5_path=h5_path, raw=raw):
            with h5py.File(str(h5_path), "r+") as f:
                del f["/assay/metadata"]
                f["assay"].create_dataset(
                    "metadata", data=np.array([old_bytes.sub(new.id.encode(), raw)])
                )

        _keep_mtime(h5_path, _rewrite_blob)
    on_progress(f"Metadata blobs: {len(h5_files)} raw recording(s)")

    npz_files = []
    for npz_path in (p for p in paths if p.suffix == ".npz" and p.is_file()):
        with np.load(npz_path, allow_pickle=True) as npz:
            contents = {key: npz[key] for key in npz.files}
        changed = False
        for key, value in contents.items():
            if value.ndim == 0 and isinstance(value.item(), str) and old_token.search(value.item()):
                # A fresh array, not the old dtype: a fixed-width ``<U24`` would truncate a longer id.
                renamed_value = old_token.sub(new.id, value.item())
                contents[key] = np.array(renamed_value, dtype=object if value.dtype == object else None)
                changed = True
        if not changed:
            continue  # an electrode-selection cache, not a preprocessed recording
        npz_files.append(npz_path)
        if dry_run:
            continue

        def _rewrite_npz(npz_path=npz_path, contents=contents):
            tmp = npz_path.with_name(npz_path.name + ".tmp")
            with open(tmp, "wb") as fh:  # a file object, so savez does not re-append ".npz"
                np.savez_compressed(fh, **contents)
            os.replace(tmp, npz_path)

        _keep_mtime(npz_path, _rewrite_npz)
    on_progress(f"Embedded identity: {len(npz_files)} preprocessed file(s)")

    csv_files = []
    for csv_path in (
        p for root in (config.burst_data_dir, config.analysis_dir) if root.is_dir()
        for p in sorted(root.rglob("*.csv"))
    ):
        text = csv_path.read_text()
        if not old_token.search(text):
            continue
        csv_files.append(csv_path)
        if not dry_run:
            _keep_mtime(csv_path, lambda p=csv_path, t=text: p.write_text(old_token.sub(new.id, t)))
    on_progress(f"CSV rows: {len(csv_files)} burst log(s) and summary table(s)")

    if registry_rows and not dry_run:
        tmp = config.registry_path.with_name(config.registry_path.name + ".tmp")
        tmp.write_text(old_token.sub(new.id, registry_text))
        os.replace(tmp, config.registry_path)
    on_progress(f"Registry: {registry_rows} row(s)")

    # --- then the paths, deepest first so every parent is still where its children expect --------
    renamed = []
    for path in sorted(paths, key=lambda p: len(p.parts), reverse=True):
        dest = path.with_name(old_token.sub(new.id, path.name))
        renamed.append((path, dest))
        if not dry_run:
            os.rename(path, dest)
    on_progress(f"Paths: {len(renamed)} directory(ies) and file(s) renamed")

    # --- and the journal, once the store really is renamed: one record per plating of the batch ----
    if not dry_run:
        from mxtreme import transactions

        for plate_date in plate_dates or (None,):
            transactions.record(
                config,
                "batch.renamed",
                batch_id=new,
                plate_date=plate_date,
                actor=actor,
                note=reason,
                data={
                    "old": old.id,
                    "new": new.id,
                    "paths": len(renamed),
                    "h5_files": len(h5_files),
                    "npz_files": len(npz_files),
                    "csv_files": len(csv_files),
                    "registry_rows": registry_rows,
                },
            )
        on_progress(f"Journal: {len(plate_dates) or 1} batch.renamed record(s)")

    return BatchRename(
        old=old, new=new, paths=renamed, h5_files=h5_files, npz_files=npz_files,
        csv_files=csv_files, registry_rows=registry_rows, plate_dates=plate_dates,
    )
