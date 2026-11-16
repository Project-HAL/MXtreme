"""The ``recordings/`` tree of the managed store: batches, layout, naming, and ingest.

Every raw ``.h5`` the store manages -- activity scans, network scans, and exogenous experiment
recordings -- lives in one tree, keyed by the *plating batch* the culture came from rather than by an
experiment name::

    <data_root>/recordings/
        plating_<plate_date>_<batch_id>/
            chip_<M1|M2>_<chip>/
                DIV_<div>/
                    plating_<plate_date>_<batch_id>_chip_<chip>_well_<wells>_DIV_<div>_activity_scan.raw.h5
                    plating_<plate_date>_<batch_id>_chip_<chip>_well_<wells>_DIV_<div>_network_scan.raw.h5
                    plating_<plate_date>_<batch_id>_chip_<chip>_well_<wells>_DIV_<div>_<experiment>.raw.h5

A *batch* is one plating event, named at the bench when it happens -- see :class:`Batch` -- and
its id is the identity everything downstream is keyed by. The ``<experiment>`` tail is a free string
naming one experiment, supplied when an exogenous recording is ingested (see
:func:`ingest_recording`); scans carry no experiment name, their kind is the tail instead.

A culture can be network-scanned more than once on one DIV. The first such file is
``..._network_scan.raw.h5``; when a second is ingested the two are numbered in arrival order,
``..._network_scan_0.raw.h5`` and ``..._network_scan_1.raw.h5``, and later ones take the next
index (:func:`network_scan_slot`). Every one of them is a network scan -- the number is not an
experiment name -- and the readers here accept either form.

A file is kept exactly as the rig wrote it: a MaxOne recording holds its one well, a MaxTwo
recording holds every well that was recorded together. The ``well_<wells>`` token in the name says
which (``well_0``, ``well_0-1-2-3-4-5``; see :func:`well_token`), so one chip's DIV directory holds
that day's files for all of its cultures, and a reader can tell which cultures a file covers without
opening it. Files from before 2026-10-28 were split into one per well, under a ``well_<N>/``
directory between the chip and the DIV; :func:`relayout_by_div` brings such a store into this layout,
stitching the pieces of one recording back together (:func:`stitch_wells`).

The module is import-light on purpose: everything here except stitching/ingest is pure path and
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

#: MaxWell systems a chip's recordings come off: MaxOne (``M1``) or MaxTwo (``M2``). Named at the
#: chip level of the tree (``chip_<system>_<chip>``), never in the batch id: one plating can put
#: cultures on both.
SYSTEMS = ("M1", "M2")

#: What a file in the recordings tree can be. ``"experiment"`` is an ingested exogenous recording;
#: the other two are scans -- run by MXtreme itself, or recorded by MaxLab Live's own assays and
#: ingested as scans (see :func:`ingest_recording`).
RECORDING_KINDS = ("activity_scan", "network_scan", "experiment")

#: The batch-id convention, e.g. ``fall2026_batch1_DRG``. The cell type is the one free-form
#: field, and the last; it may itself contain underscores, which parses unambiguously because
#: everything before it is fixed-format. (Ids written before 2026-09-21 ended in ``_M1``/``_M2``;
#: such an id still parses, with the suffix read as part of the cell type -- see :func:`rename_batch`
#: for taking it off.)
_BATCH_ID = re.compile(
    r"^(?P<semester>spring|summer|fall|winter)(?P<year>\d{4})"
    r"_batch(?P<number>\d+)"
    r"_(?P<cell_type>.+)$"
)

#: Matches the scan-kind tail of a store file name (``..._activity_scan.raw.h5``,
#: ``..._network_scan_1.raw.h5`` -- the numeric suffix numbers several scans of one DIV, see
#: :func:`network_scan_slot`; MaxLab's own collision rename produces the same shape).
_SCAN_TAIL = re.compile(r"_(activity|network)_scan(_\d+)?\.raw\.h5$")

#: The index a network scan file carries: ``None`` for the unnumbered ``_network_scan``, else the
#: number after it.
_NETWORK_INDEX = re.compile(r"_network_scan(?:_(\d+))?\.raw\.h5$")

#: Extracts the experiment-name tail of a store file name: everything after the ``DIV_<div>_`` marker.
_EXP_TAIL = re.compile(r"_DIV_\d+_(?P<experiment>.+?)\.raw\.h5$")

#: Plating dates are ``YYMMDD``, matching :func:`mxtreme.scans.mx_setup.write_metadata`.
#: The wells a store file holds, from its name (see :func:`well_token`).
_WELLS_IN_NAME = re.compile(r"_well_(?P<wells>\d+(?:-\d+)*)_DIV_\d+_")

_PLATE_DATE = re.compile(r"^\d{6}$")

#: The directory names the layout functions below produce, for reading the tree back.
_PLATING_DIR = re.compile(r"^plating_(?P<plate_date>\d{6})_(?P<batch_id>.+)$")
_CHIP_DIR = re.compile(r"^chip_(?P<system>M[12])_(?P<chip>.+)$")


@dataclass(frozen=True)
class Batch:
    """One plating batch: the identity every recording in the store hangs off.

    Created manually at the start of a batch -- this constructor *is* the generator, validating each
    field so a malformed id never reaches a directory name::

        batch = Batch(semester="fall", year=2026, number=1, cell_type="DRG")
        batch.id  # "fall2026_batch1_DRG"

    An id someone already has as a string round-trips through :meth:`parse`.

    The system a culture is recorded on (MaxOne or MaxTwo) is deliberately not part of the id: a
    plating can put cultures on both, and the tree records it one level down, in the chip
    directory (see :func:`chip_dirname`).

    :param semester: One of :data:`SEMESTERS`.
    :param year: Four-digit calendar year.
    :param number: Batch number within the semester, starting at 1.
    :param cell_type: Free-form cell type label, e.g. ``"DRG"``. Underscores are allowed; path
        separators and whitespace are not.
    """

    semester: str
    year: int
    number: int
    cell_type: str

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

    @property
    def id(self) -> str:
        """The batch id string, e.g. ``fall2026_batch1_DRG``."""
        return f"{self.semester}{self.year}_batch{self.number}_{self.cell_type}"

    def __str__(self) -> str:
        return self.id

    @classmethod
    def parse(cls, batch_id: str | Batch) -> Batch:
        """Parse a batch id string back into a :class:`Batch`, validating it on the way.

        A :class:`Batch` passes through unchanged, so call sites can accept either form.

        :param batch_id: e.g. ``"fall2026_batch1_DRG"``.
        :raises ValueError: If the string does not follow the batch-id convention.
        """
        if isinstance(batch_id, Batch):
            return batch_id
        match = _BATCH_ID.match(str(batch_id))
        if match is None:
            raise ValueError(
                f"Not a valid batch id: {batch_id!r}. Expected "
                "<semester><year>_batch<n>_<cell_type>, e.g. 'fall2026_batch1_DRG'."
            )
        return cls(
            semester=match["semester"],
            year=int(match["year"]),
            number=int(match["number"]),
            cell_type=match["cell_type"],
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


def chip_dirname(system: str, chip: str) -> str:
    """One chip's directory within a plating: ``chip_<M1|M2>_<chip>``.

    The system is the chip's, not the batch's: it is what the rig reported when the chip was
    recorded from (:func:`mxtreme.scans.activity_scan.run_activity_scan`), or what an ingested
    file says about its plate (:func:`describe_recording`).

    :raises ValueError: If ``system`` is not one of :data:`SYSTEMS`.
    """
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}, got {system!r}.")
    return f"chip_{system}_{chip}"


def chip_dir(config, batch: Batch | str, plate_date, chip: str, *, system: str) -> Path:
    """Directory holding one chip's wells: ``<recordings_dir>/plating_.../chip_...``.

    For *writing*: the caller says which system the chip is on. To find the directory a chip
    already has, whichever system it is on, use :func:`find_chip_dir`.
    """
    return config.recordings_dir / plating_dirname(batch, plate_date) / chip_dirname(system, chip)


def find_chip_dir(config, batch: Batch | str, plate_date, chip: str) -> Path | None:
    """The directory a chip already has in a plating, or ``None`` if it has none yet.

    A chip lives on one system, so at most one ``chip_M?_<chip>`` exists; this is how a reader
    that knows the chip but not the system (a registry row, a journal record) gets to its files.
    """
    plating_dir = config.recordings_dir / plating_dirname(batch, plate_date)
    return _find_chip_dir_in(plating_dir, chip)


def _find_chip_dir_in(plating_dir: Path, chip: str) -> Path | None:
    for system in SYSTEMS:
        candidate = plating_dir / chip_dirname(system, chip)
        if candidate.is_dir():
            return candidate
    return None


def recording_dir(config, batch: Batch | str, plate_date, chip: str, div: int, *, system: str) -> Path:
    """Directory one chip's recordings for one DIV land in: ``.../chip_<system>_<chip>/DIV_<div>``."""
    return chip_dir(config, batch, plate_date, chip, system=system) / f"DIV_{int(div)}"


def well_token(wells: int | Iterable[int]) -> str:
    """The wells a file holds, as its name carries them: ``"0"`` for one, ``"0-1-3"`` for several.

    Ascending and joined with ``-``; never a range, so ``"0-3"`` is wells 0 and 3. The token is
    how a reader learns which cultures a file covers without opening it (:func:`parse_recording_path`).
    """
    if isinstance(wells, int):
        return str(wells)
    return "-".join(str(w) for w in sorted({int(w) for w in wells}))


def parse_well_token(token: str) -> tuple[int, ...]:
    """The inverse of :func:`well_token`: ``"0-1-3"`` -> ``(0, 1, 3)``."""
    return tuple(sorted({int(part) for part in token.split("-")}))


def recording_stem(batch: Batch | str, plate_date, chip: str, wells: int | Iterable[int], div: int) -> str:
    """Base name shared by every file of these wells on one DIV, without the kind/experiment tail.

    ``plating_<plate_date>_<batch_id>_chip_<chip>_well_<token>_DIV_<div>``, the token from
    :func:`well_token`: the one well of a MaxOne recording, or every well a MaxTwo recording holds.
    """
    return (
        f"plating_{_validate_plate_date(plate_date)}_{Batch.parse(batch).id}"
        f"_chip_{chip}_well_{well_token(wells)}_DIV_{int(div)}"
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
        return sorted(self.systems())

    def systems(self) -> dict[str, str]:
        """``{chip: system}`` for every chip directory: which of :data:`SYSTEMS` each chip is on."""
        if not self.path.is_dir():
            return {}
        found = (_CHIP_DIR.match(entry.name) for entry in self.path.iterdir() if entry.is_dir())
        return {match["chip"]: match["system"] for match in found if match}

    def system_of(self, chip: str) -> str | None:
        """The system one chip is on, or ``None`` if the plating has no directory for it."""
        return self.systems().get(chip)

    def chip_dir(self, chip: str) -> Path | None:
        """One chip's directory in this plating, whichever system it is on; ``None`` if absent."""
        return _find_chip_dir_in(self.path, chip)


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
    ingested experiment recording, whose tail is its experiment name.
    """
    match = _SCAN_TAIL.search(Path(h5_path).name)
    return f"{match.group(1)}_scan" if match else "experiment"


@dataclass(frozen=True)
class RecordingLocation:
    """One store file's identity, as read back from its place in the recordings tree.

    :param batch: The plating batch, parsed from the plating directory.
    :param plate_date: Plating date (``YYMMDD``), from the plating directory.
    :param chip: Chip serial, from the chip directory.
    :param system: ``M1`` or ``M2``, from the chip directory.
    :param wells: The wells the file holds, ascending, from its name's ``well_`` token.
    :param div: Days *in vitro*, from the DIV directory.
    :param kind: See :data:`RECORDING_KINDS`.
    :param experiment: The experiment-name tail for ``kind="experiment"``; ``""`` for scans.
    """

    batch: Batch
    plate_date: int
    chip: str
    system: str
    wells: tuple[int, ...]
    div: int
    kind: str
    experiment: str


def parse_recording_path(h5_path: Path, recordings_dir: Path) -> RecordingLocation | None:
    """Read a file's identity back out of its path in the recordings tree.

    The inverse of :func:`recording_dir` + :func:`recording_stem`: the directories are fixed-format
    and the file name carries the wells, so a file here is identified without being opened. Used by
    :func:`mxtreme.io.rebuild_registry`.

    :param h5_path: The file, anywhere under ``recordings_dir``.
    :param recordings_dir: The tree root (``config.recordings_dir``).
    :returns: The parsed identity, or ``None`` if the path does not follow the layout.
    """
    try:
        plating, chip_part, div_part = h5_path.relative_to(recordings_dir).parts[:-1]
    except ValueError:
        return None

    plating_match = _PLATING_DIR.match(plating)
    chip_match = _CHIP_DIR.match(chip_part)
    div_match = re.match(r"^DIV_(\d+)$", div_part)
    wells_match = _WELLS_IN_NAME.search(h5_path.name)
    if not (plating_match and chip_match and div_match and wells_match):
        return None
    try:
        batch = Batch.parse(plating_match["batch_id"])
    except ValueError:
        return None

    kind = recording_kind(h5_path)
    experiment = ""
    if kind == "experiment":
        exp_match = _EXP_TAIL.search(h5_path.name)
        experiment = exp_match["experiment"] if exp_match else ""
        if not experiment:
            return None

    return RecordingLocation(
        batch=batch,
        plate_date=int(plating_match["plate_date"]),
        chip=chip_match["chip"],
        system=chip_match["system"],
        wells=parse_well_token(wells_match["wells"]),
        div=int(div_match.group(1)),
        kind=kind,
        experiment=experiment,
    )


# --- the wells inside a file, and putting a split recording back together -------------------------


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


def _read_metadata_blob(f) -> dict | None:
    """The ``/assay/metadata`` blob of an open file as a dict, or ``None`` if it has none that parses.

    Stored as ``str(dict)`` (see :func:`mxtreme.scans.mx_setup.write_metadata`) and read back with
    quotes swapped, the same round-trip as :func:`mxtreme.io._embedded_metadata`.
    """
    import json

    try:
        raw = f["/assay/metadata"][:][0]
        return json.loads(raw.decode("utf-8").strip().replace("'", '"'))
    except (KeyError, ValueError, IndexError):
        return None


def _write_metadata_blob(f, metadata: dict) -> None:
    """Replace an open file's ``/assay/metadata`` blob, in the format the readers expect."""
    import numpy as np

    if "/assay/metadata" in f:
        del f["/assay/metadata"]
    f.require_group("assay").create_dataset("metadata", data=np.array([str(metadata).encode("utf-8")]))


def stitch_wells(
    parts: Iterable[str | Path],
    dest: str | Path,
    *,
    on_progress: Callable[[str], None] = print,
) -> Path:
    """Put the per-well pieces of one multi-well recording back into a single ``.h5``.

    The inverse of the split the store used to do on the way in: each piece holds one well's
    ``/wells/wellNNN/...`` and a ``/recordings/recMMMM/wellNNN`` view of it, plus the shared parts
    (``/assay``, settings, version) copied from the original. The shared parts are taken from the
    first piece, every piece's wells are copied in, the recordings view is re-created as hard links
    to the copies (no data is duplicated inside the output), and the metadata blob's ``Well IDs`` /
    ``Conditions`` are merged back into one list in well order.

    The pieces must be of one recording -- the same chip, DIV and kind, recorded together. Nothing
    here checks the timestamps; :func:`relayout_by_div` groups the pieces before calling this.

    :param parts: The per-well files, in any order.
    :param dest: Where to write the combined file. Its parent is created; it must not exist.
    :param on_progress: Called with each progress line.
    :returns: ``dest``.
    :raises ValueError: If ``parts`` is empty, a piece has no ``/wells`` group, or two pieces hold
        the same well.
    :raises FileExistsError: If ``dest`` exists.
    """
    import h5py

    parts = [Path(p) for p in parts]
    dest = Path(dest)
    if not parts:
        raise ValueError("stitch_wells needs at least one file.")
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite {dest}.")
    dest.parent.mkdir(parents=True, exist_ok=True)

    well_ids: dict[int, object] = {}  # well -> condition (None when the piece had none)
    metadata: dict | None = None
    try:
        with h5py.File(str(dest), "w") as out:
            out.require_group("wells")
            for index, part in enumerate(parts):
                with h5py.File(str(part), "r") as f:
                    if "wells" not in f:
                        raise ValueError(f"{part} has no /wells group; not a MaxWell recording.")
                    if index == 0:
                        for key, value in f.attrs.items():
                            out.attrs[key] = value
                        for key in f:
                            if key not in ("wells", "recordings"):
                                f.copy(f[key], out, name=key)
                        metadata = _read_metadata_blob(f)
                    blob = _read_metadata_blob(f) or {}
                    ids = list(blob.get("Well IDs") or [])
                    conditions = list(blob.get("Conditions") or [])
                    for group_name in sorted(f["wells"], key=_well_number):
                        well = _well_number(group_name)
                        if group_name in out["wells"]:
                            raise ValueError(f"Well {well} appears in more than one of the pieces ({part.name}).")
                        on_progress(f"  Well {well}: from {part.name}")
                        f.copy(f["wells"][group_name], out["wells"], name=group_name)
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
                        condition = conditions[ids.index(well)] if well in ids and ids.index(well) < len(conditions) else None
                        well_ids[well] = condition
            if metadata is not None:
                metadata["Well IDs"] = sorted(well_ids)
                labelled = [well_ids[w] for w in sorted(well_ids)]
                metadata["Conditions"] = labelled if any(c is not None for c in labelled) else []
                _write_metadata_blob(out, metadata)
    except BaseException:
        dest.unlink(missing_ok=True)  # a half-written output must not pass for the recording
        raise
    return dest


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
    lists in ``electrode_selection/`` beside it, named after the scan file). A network scan's, or
    an ingested experiment's, are the preprocessed ``.npz`` and the burst CSV of each of its wells
    on that DIV, under the batch id
    (an experiment's carry its name as a file-name tail, see :func:`mxtreme.io.recording_file_name`;
    one preprocessed before that sits under a directory named for the experiment instead). Analysis
    summaries and reports are per culture across DIVs, and are left alone: they are rebuilt from
    what remains.

    :returns: The files that exist, in a stable order. Empty if the path is not in the tree.
    """
    h5_path = Path(h5_path)
    location = parse_recording_path(h5_path, config.recordings_dir)
    if location is None:
        return []
    out: list[Path] = []
    if location.kind == "activity_scan":
        selection = h5_path.parent / "electrode_selection"
        prefix = selection_prefix(h5_path) + "_"
        if selection.is_dir():
            out.extend(sorted(p for p in selection.iterdir() if p.is_file() and p.name.startswith(prefix)))
        return out
    for well in location.wells:
        well_tail = Path(location.chip) / f"well{well}"
        tail = f"well{well}_{location.experiment}_" if location.experiment else f"well{well}_"
        places = [(location.batch.id, tail)]
        if location.experiment:  # preprocessed before batch_id was the identity: filed under its name
            places.append((location.experiment, f"well{well}_"))
        for top, tail in places:
            for root, pattern in (
                (config.preprocessed_dir / top / well_tail, f"DIV{location.div}_*{tail}exp_data.npz"),
                (config.burst_data_dir / top / well_tail, f"DIV{location.div}_*{tail}burst_data.csv"),
            ):
                if root.is_dir():
                    out.extend(sorted(root.glob(pattern)))
    return out


def selection_prefix(h5_path: str | Path) -> str:
    """The name an activity scan's electrode-selection outputs start with: its file name without
    MaxWell's ``.raw.h5``. They sit in ``electrode_selection/`` beside the scan as
    ``<prefix>_AS_well<N>.png``, ``<prefix>_network_well<N>.png`` and ``<prefix>_network_well<N>.npz``,
    one of each per well selected."""
    name = Path(h5_path).name
    for suffix in (".raw.h5", ".h5"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _drop_burst_log_rows(
    config, batch_id: str, chip: str, well: int, div: int, experiment: str, *, dry_run: bool
) -> int:
    """Drop one recording's rows from the batch's burst log; return how many."""
    from mxtreme.io import read_burst_log

    # An experiment preprocessed before batch_id was the identity logged under its own name.
    places = [(batch_id, experiment)] + ([(experiment, "")] if experiment else [])
    n = 0
    for top, exp in places:
        log_path = config.burst_data_dir / f"{top}_burst_log.csv"
        if not log_path.is_file():
            continue
        df = read_burst_log(log_path)
        if df.empty or not {"chip", "well", "DIV"} <= set(df.columns):
            continue
        mask = (
            (df["chip"].astype(str) == str(chip))
            & (df["well"].astype(str) == str(well))
            & (df["DIV"].astype(str) == str(div))
            & (df["experiment"].astype(str) == exp)
        )
        hits = int(mask.sum())
        if hits and not dry_run:
            df[~mask].to_csv(log_path, index=False)
        n += hits
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

    # The registry: the recording's own rows (one per well), and its preprocessed rows when those
    # files went.
    registry_rows = 0
    burst_log_rows = 0
    for well in location.wells:
        if not dry_run:
            registry_rows += io.unregister(
                config.registry_path, chip=location.chip, well=well, div=location.div,
                kind=location.kind, batch_id=location.batch.id, experiment=location.experiment,
            )
            if derived and location.kind != "activity_scan":
                registry_rows += io.unregister(
                    config.registry_path, chip=location.chip, well=well, div=location.div,
                    kind="preprocessed", batch_id=location.batch.id, experiment=location.experiment,
                )
                if location.experiment:  # preprocessed before batch_id was the identity
                    registry_rows += io.unregister(
                        config.registry_path, chip=location.chip, well=well, div=location.div,
                        kind="preprocessed", batch_id=location.experiment, experiment="",
                    )
        if derived and location.kind != "activity_scan":
            burst_log_rows += _drop_burst_log_rows(
                config, location.batch.id, location.chip, well, location.div, location.experiment,
                dry_run=dry_run,
            )

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
        for well in location.wells:  # one record per culture the file held, like its registration
            transactions.record(
                config,
                "recording.removed",
                batch_id=location.batch,
                plate_date=location.plate_date,
                experiment=location.experiment,
                chip=location.chip,
                well=well,
                div=location.div,
                actor=actor,
                note=reason,
                data={
                    "path": str(h5_path),
                    "kind": location.kind,
                    "wells": list(location.wells),
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


def _validate_experiment(experiment: str) -> str:
    """Check an experiment name can serve as a file-name tail, and is not a reserved scan tail."""
    if not isinstance(experiment, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment):
        raise ValueError(
            f"experiment must be a non-empty string of letters, digits, '.', '-' or '_' (and start "
            f"with a letter or digit), got {experiment!r}."
        )
    if re.fullmatch(r"(activity|network)_scan(_\d+)?", experiment):
        raise ValueError(
            f"experiment {experiment!r} is reserved for scans; an ingested recording needs its own name."
        )
    return experiment


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
    experiment: str | None = None,
    wells: Iterable[int] | None = None,
    conditions: dict[int, object] | None = None,
    system: str | None = None,
    move: bool = False,
    actor: str | None = None,
    note: str = "",
    on_progress: Callable[[str], None] = print,
) -> dict[int, Path]:
    """Bring an exogenous ``.h5`` (recorded outside MXtreme) into the managed store.

    The file is placed in the recordings tree under the identity given here and one row per well
    is upserted into the registry -- exactly as if MXtreme had recorded it. What it is filed *as*
    is ``kind``: an ``experiment`` (the default) is named ``<stem>_<experiment>.raw.h5`` after the
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

    The file is kept whole, however many wells it holds; ``wells`` says which of them are cultures
    worth registering (the plated ones of a MaxTwo plate) -- those name the file and get registry
    rows, the rest ride along as the rig recorded them. An ingested copy with no ``/assay/metadata``
    blob gets one written from the identity given here, so :func:`mxtreme.extract.extract` and a
    registry rebuild can read it back without being handed the metadata again; a blob the file
    already carries is left as recorded. ::

        from mxtreme.config import Config
        from mxtreme.store import ingest_recording

        config = Config.from_toml("mxtreme.toml")
        ingest_recording(
            "/data/exports/stim_session.raw.h5", config,
            batch="fall2026_batch1_DRG", plate_date=260810,
            chip="M07460", div=21, experiment="burstTrainer_trial3",
        )
        ingest_recording(                      # a Scope activity scan
            "/data/scope/M07460_260831.h5", config,
            batch="fall2026_batch1_DRG", plate_date=260810,
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
    :param experiment: For an ``experiment``, its free-form name -- becomes the file-name tail and
        the registry row's ``experiment``. Letters, digits, ``.``, ``-``, ``_``. Required for an
        experiment; not accepted for a scan. Either way the recording's identity is its batch.
    :param wells: Which of the file's wells to register (and name the file by); ``None`` means
        all of them. The file itself is copied whole either way.
    :param conditions: Optional per-well condition labels, keyed by well number.
    :param system: Which system the chip is on, ``"M1"`` or ``"M2"`` -- it names the chip
        directory. ``None`` takes what the file says about its plate
        (:attr:`RecordingDescription.system`); a file that does not say needs it given.
    :param move: Move the file instead of copying it.
    :param actor: Who, for the journal; the OS user when ``None``.
    :param note: A remark for the journal (where the file came from, say).
    :param on_progress: Called with each progress line.
    :returns: ``{well: destination path}`` for every well registered -- one file, so every well
        maps to the same path.
    :raises FileNotFoundError: If ``h5_path`` does not exist.
    :raises FileExistsError: If the destination file already exists -- nothing is overwritten. A
        network scan never collides (it takes the next index instead).
    :raises ValueError: On a malformed batch id, plate date, kind or experiment, a file with no wells,
        ``wells`` naming one the file lacks, or no way to tell which system the chip is on.
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
        if experiment is None:
            raise ValueError("An experiment needs a name (experiment=...); a scan takes "
                             "kind='activity_scan' or 'network_scan' instead.")
        _validate_experiment(experiment)
        tail = experiment
    else:
        if experiment is not None:
            raise ValueError(f"A {kind} carries no experiment name; got {experiment!r}.")
        experiment = ""
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

    # The chip directory names the system. The file usually says (Scope and the MaxLab server
    # both write /wellplate/version); a caller that knows better, or has a file that does not
    # say, passes it.
    if system is None:
        system = describe_recording(h5_path).system
        if system is None:
            raise ValueError(
                f"{h5_path.name} does not say whether it was recorded on a MaxOne or a MaxTwo; "
                "pass system='M1' or 'M2' to name the chip directory."
            )
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}, got {system!r}.")

    stem = recording_stem(batch, plate_date, chip, chosen, div)
    div_dir = recording_dir(config, batch, plate_date, chip, div, system=system)
    renames: list[tuple[Path, Path]] = []
    if kind == "network_scan":
        tail, renames = network_scan_slot(div_dir, stem)
    dest = div_dir / f"{stem}_{tail}.raw.h5"
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite {dest}.")

    # A network scan joining others of its DIV: number what is there before anything lands, so the
    # new file's name and the renames it caused are settled together (and journaled together).
    renamed: dict[str, str] = {}
    for old, new in renames:
        if new.exists():
            raise FileExistsError(f"Refusing to overwrite {new}.")
        on_progress(f"Numbering the DIV's earlier network scan: {old.name} -> {new.name}")
        old.rename(new)
        renamed[str(old)] = str(new)

    dest.parent.mkdir(parents=True, exist_ok=True)
    on_progress(f"{'Moving' if move else 'Copying'} {h5_path.name} -> {dest}")
    if move:
        shutil.move(str(h5_path), str(dest))
    else:
        shutil.copy2(str(h5_path), str(dest))
    written = {well: dest for well in chosen}

    labels = [conditions.get(well) for well in chosen]
    _ensure_metadata_blob(dest, {
        "Batch ID": batch.id,
        **({"Experiment": experiment} if experiment else {}),
        "Chip ID": chip,
        "Plate date": _validate_plate_date(plate_date),
        "DIV": int(div),
        "Well IDs": list(chosen),
        "Conditions": labels if any(label is not None for label in labels) else [],
    })
    if kind == "activity_scan":
        from mxtreme.scans.activity_scan import ensure_record_time, has_record_time

        if not has_record_time(dest):
            try:
                seconds = ensure_record_time(dest)
            except ValueError as exc:
                on_progress(f"  {exc}")
            else:
                on_progress(f"  Wrote record_time = {seconds} s from the file's timestamps")

    io.register(
        {
            well: {
                "experiment": experiment,
                "chip": chip,
                "DIV": int(div),
                "exp_condition": conditions.get(well),
                "batch_id": batch.id,
                "plate_date": _validate_plate_date(plate_date),
            }
            for well in chosen
        },
        config.registry_path,
        kind=kind,
    )
    on_progress(f"Registered wells {chosen} as {kind} in {config.registry_path}")

    from mxtreme import transactions

    for well in chosen:
        data = {"source": str(h5_path), "path": str(dest), "moved": bool(move), "kind": kind, "system": system}
        if renamed:
            data["renamed"] = renamed
        transactions.record(
            config,
            "recording.ingested",
            batch_id=batch,
            plate_date=plate_date,
            experiment=experiment,
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
    against a letter or digit -- so this is what separates ``fall2026_batch1_E18`` from
    ``fall2026_batch12_E18``. It cannot separate an id from one whose cell type *contains* it
    (``fall2026_batch1_E18_E18``); :func:`rename_batch` checks the store for that case first.
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
    h5_blobs: bool = True,
    on_progress: Callable[[str], None] = lambda _: None,
) -> BatchRename:
    """Give a plating batch a new id everywhere the store names it.

    A batch id is written into far more than its plating directory: every file name under
    ``recordings/``, ``preprocessed/``, ``burst_data/`` and ``analysis/`` carries it; so do the
    registry rows, the ``/assay/metadata`` blob inside every raw ``.h5``, the ``batch_id`` and
    ``path_to_h5`` fields inside every preprocessed ``.npz`` (which is where a registry rebuild reads
    identity from), and the ``batch_id`` / ``culture_id`` columns of the burst log and the analysis
    summary CSVs. This rewrites all of them, so that afterwards :func:`list_platings`,
    :func:`mxtreme.io.rebuild_registry` and :func:`mxtreme.paths.resolve_paths` all agree on the
    new name. Report PDFs are renamed but not regenerated: their text still says the old name until
    they are produced again.

    The plate date is not part of the batch id and is left alone -- the batch keeps its place in the
    calendar, only its label changes. Nor is the system: the chip directories name it, and they are
    not touched. (Taking a pre-2026-09-21 ``_M1``/``_M2`` suffix off an id is an ordinary rename:
    ``rename_batch(config, "fall2026_batch1_E18_M1", "fall2026_batch1_E18")``.)

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
    :param h5_blobs: Rewrite the ``/assay/metadata`` blob inside each raw ``.h5``. Set it to
        ``False`` for a copy of the store that cannot be opened read-write in place -- an archive
        on a network share, say -- so its paths, registry, CSVs and ``.npz`` files follow the
        rename now, and the rewritten originals replace its raw files when they are next synced.
        Until then the blobs there still carry the old id; nothing in the tree reads identity from
        them (:func:`parse_recording_path` reads the path).
    :param on_progress: Called with a line of progress text as each group of files is done.
    :raises ValueError: If the new id is malformed, equals the old one, or is a substring of
        another batch's id in the store (which the rename could not tell apart).
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

    # Names carrying the *old* id are masked before looking for the new one: taking a suffix off
    # an id (``fall2026_batch1_E18_M1`` -> ``fall2026_batch1_E18``) leaves the new id standing as
    # a whole token inside every old name, which is the rename itself, not a collision.
    def _has_new(text: str) -> bool:
        return bool(new_token.search(old_token.sub("", text)))

    taken = [p for root in roots for p in _named(root, new_token) if _has_new(p.name)]
    if taken or _has_new(registry_text):
        where = taken[0] if taken else config.registry_path
        raise FileExistsError(f"Batch {new.id!r} is already present in the store: {where}")

    def _keep_mtime(path: Path, write: Callable[[], None]) -> None:
        stat = path.stat()
        write()
        os.utime(path, (stat.st_atime, stat.st_mtime))

    # --- contents first: the identity each file carries inside it -----------------------------
    old_bytes = re.compile(old_token.pattern.encode())
    h5_files = []
    for h5_path in (p for p in paths if h5_blobs and p.suffix == ".h5" and p.is_file()):
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
    on_progress(f"Metadata blobs: {len(h5_files)} raw recording(s)" + ("" if h5_blobs else " (left as they were)"))

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


# --- moving a store from per-well directories to per-DIV ones ------------------------------------


@dataclass(frozen=True)
class Relayout:
    """What :func:`relayout_by_div` changed, or -- with ``dry_run`` -- would change.

    :param moved: Single-piece files, ``(old, new)``: their directory changed, their name did not.
    :param stitched: Recordings put back together, ``(pieces, new)``: the per-well files of one
        recording and the combined file written from them.
    :param redundant: Per-well pieces of a recording whose whole file was also still there (a
        split that failed part-way left both): the whole file is kept, these go to the trash.
    :param selection_files: Electrode-selection outputs moved beside their scan, ``(old, new)``.
    :param npz_files: Preprocessed files whose embedded ``path_to_h5`` was pointed at the new file.
    :param skipped: Files left where they were, each with the reason.
    :param trashed_to: Where the stitched recordings' pieces and the redundant files went, or
        ``None`` (dry run, or nothing to trash).
    :param pruned_dirs: Directories the move left empty and removed.
    """

    moved: tuple[tuple[Path, Path], ...]
    stitched: tuple[tuple[tuple[Path, ...], Path], ...]
    redundant: tuple[Path, ...]
    selection_files: tuple[tuple[Path, Path], ...]
    npz_files: tuple[Path, ...]
    skipped: tuple[tuple[Path, str], ...]
    trashed_to: Path | None
    pruned_dirs: tuple[Path, ...]


_OLD_WELL_DIR = re.compile(r"^well_(\d+)$")
_DIV_DIR = re.compile(r"^DIV_(\d+)$")
_OLD_TAIL = re.compile(r"_DIV_\d+_(?P<tail>.+?)\.raw\.h5$")

#: Pieces of one recording were written by one ``start_recording``; their first recordings start
#: within a few milliseconds of each other. Separate recordings of one chip are minutes apart.
_SAME_RECORDING_MS = 2000


def _recording_starts(h5_path: Path) -> list[int]:
    """Start stamps (ms) of a file's recordings, from its first well, in recording order."""
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        if "wells" not in f or not len(f["wells"]):
            raise ValueError(f"{h5_path.name} has no wells.")
        first = f["wells"][min(f["wells"], key=_well_number)]
        starts = []
        for rec in sorted(first):
            group = first[rec]
            starts.append(int(group["start_time"][0]) if isinstance(group, h5py.Group) and "start_time" in group else 0)
        return starts


def relayout_by_div(
    config,
    *,
    dry_run: bool = False,
    reason: str = "",
    actor: str | None = None,
    on_progress: Callable[[str], None] = print,
) -> Relayout:
    """Move a store recorded under ``chip_*/well_<N>/DIV_<d>/`` into the per-DIV layout.

    Until 2026-10-28 the tree had a well directory between the chip and the DIV, and a MaxTwo
    recording was split into one file per well on its way in. The layout is now
    ``chip_*/DIV_<d>/`` with each recording kept whole (see the module docstring). This brings an
    existing store across:

    - the pieces of one split recording -- the files of one chip, DIV and kind whose recordings
      started together -- are stitched back into a single file (:func:`stitch_wells`), named for
      every well it holds, and the pieces go to :attr:`~mxtreme.config.Config.trash_dir`;
    - a file that is the whole recording already (a MaxOne scan, a network scan that only ever
      covered well 0, a multi-well file a failed split left in the chip directory) is moved into
      the DIV directory unchanged, name and all; per-well pieces of a recording whose whole file
      is still there are redundant and go to the trash with nothing stitched;
    - electrode-selection outputs follow their scan into ``DIV_<d>/electrode_selection/``, renamed
      to the scan's new prefix where it changed; outputs left in a chip directory by a selection
      run on the unsplit file go there too;
    - a preprocessed ``.npz`` whose ``path_to_h5`` named a piece is pointed at the new file;
    - the move is journaled (``recording.moved``, one record per well of each file).

    Nothing here checks whether a scan is still recording into the store; a file the rig holds
    open cannot be read and is skipped, but run this between recordings, not during one.

    Two recordings of the same wells and kind on one DIV (a scan re-run after its split failed)
    both survive: the earlier keeps the plain name, the later takes ``_1``, ``_2`` after its tail,
    the way MaxLab names a collision. The registry is not touched: its rows carry no paths. A file
    that cannot be read, or whose name does not carry a DIV and tail, is left where it is and
    reported in ``skipped``; the rest proceed. Run it with ``dry_run=True`` first and read the plan.

    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :param dry_run: Work out and report the plan without changing anything.
    :param reason: Why, for the journal.
    :param actor: Who, for the journal; the OS user when ``None``.
    :param on_progress: Called with each progress line.
    :returns: What was (or would be) changed.
    """
    import os
    import shutil
    from datetime import datetime

    import numpy as np

    root = Path(config.recordings_dir)
    verb = "Would" if dry_run else "Will"

    # -- find every file in the old layout, with what is needed to group the pieces ----------------
    @dataclass
    class _Piece:
        path: Path
        chip_dir: Path
        batch: Batch
        plate_date: int
        chip: str
        div: int
        tail: str
        wells: list[int]
        starts: list[int]

    pieces: list[_Piece] = []
    skipped: list[tuple[Path, str]] = []
    for plating_dir in sorted(root.iterdir()) if root.is_dir() else []:
        plating_match = _PLATING_DIR.match(plating_dir.name)
        if not plating_dir.is_dir() or plating_match is None:
            continue
        try:
            batch = Batch.parse(plating_match["batch_id"])
        except ValueError:
            continue
        for chip_path in sorted(plating_dir.iterdir()):
            chip_match = _CHIP_DIR.match(chip_path.name)
            if not chip_path.is_dir() or chip_match is None:
                continue
            # The old per-well directories, plus any file sitting in the chip directory itself: a
            # multi-well scan recorded there and never split (the split failed and kept it).
            candidates: list[tuple[Path, int]] = []
            for h5_path in sorted(chip_path.glob("*.h5")):
                div_match = re.search(r"_DIV_(\d+)_", h5_path.name)
                if div_match is not None:
                    candidates.append((h5_path, int(div_match.group(1))))
            for well_dir in sorted(chip_path.iterdir()):
                if not well_dir.is_dir() or not _OLD_WELL_DIR.match(well_dir.name):
                    continue
                for div_dir in sorted(well_dir.iterdir()):
                    div_match = _DIV_DIR.match(div_dir.name)
                    if div_dir.is_dir() and div_match is not None:
                        candidates.extend((h5_path, int(div_match.group(1))) for h5_path in sorted(div_dir.glob("*.h5")))
            for h5_path, div in candidates:
                tail_match = _OLD_TAIL.search(h5_path.name)
                if tail_match is None:
                    skipped.append((h5_path, "name does not end in _DIV_<n>_<tail>.raw.h5"))
                    continue
                try:
                    wells, starts = wells_in_file(h5_path), _recording_starts(h5_path)
                except OSError as exc:  # locked by a recording in progress, unreadable, not HDF5
                    skipped.append((h5_path, f"cannot be read: {exc}"))
                    continue
                except (ValueError, KeyError):
                    # An aborted scan with no wells inside: nothing to stitch it with, but its
                    # name still says where it belongs, so it moves on its own.
                    wells_match = _WELLS_IN_NAME.search(h5_path.name)
                    wells, starts = list(parse_well_token(wells_match["wells"])) if wells_match else [], []
                pieces.append(_Piece(
                    path=h5_path, chip_dir=chip_path, batch=batch,
                    plate_date=int(plating_match["plate_date"]), chip=chip_match["chip"],
                    div=div, tail=tail_match["tail"], wells=wells, starts=starts,
                ))

    # -- group the pieces of one recording: same chip, DIV and tail, recordings started together ---
    groups: list[list[_Piece]] = []
    by_key: dict[tuple, list[_Piece]] = {}
    for piece in pieces:
        by_key.setdefault((piece.chip_dir, piece.div, piece.tail, len(piece.starts)), []).append(piece)
    for members in by_key.values():
        members.sort(key=lambda m: (m.starts[0] if m.starts else 0, m.path))
        current: list[_Piece] = []
        for piece in members:
            if current and abs((piece.starts[0] if piece.starts else 0) - (current[0].starts[0] if current[0].starts else 0)) <= _SAME_RECORDING_MS:
                current.append(piece)
            else:
                if current:
                    groups.append(current)
                current = [piece]
        if current:
            groups.append(current)

    # -- the plan ------------------------------------------------------------------------------------
    moved: list[tuple[Path, Path]] = []
    stitched: list[tuple[tuple[Path, ...], Path]] = []
    redundant: list[Path] = []
    selection_moves: list[tuple[Path, Path]] = []
    npz_rewrites: list[tuple[Path, Path, dict]] = []  # (npz, new h5, contents)
    journal: list[tuple[_Piece, list[Path], Path, bool]] = []
    taken: set[Path] = set()
    selection_sources: set[Path] = set()
    selection_targets: set[Path] = set()
    stray_claims: list[tuple[Path, str, Path]] = []  # (chip_dir, base prefix, dest) -- settled last

    def _plan_selection(src: Path, target: Path) -> None:
        """Move a selection output beside its scan -- or, if that name is taken (the same figure
        from the whole file and from a piece), trash the extra rather than overwrite anything."""
        if src in selection_sources:
            return
        selection_sources.add(src)
        if target.exists() or target in selection_targets:
            redundant.append(src)
        else:
            selection_targets.add(target)
            selection_moves.append((src, target))

    def _free_name(div_dir: Path, stem: str, tail: str) -> Path:
        """``<stem>_<tail>.raw.h5``, or -- when two recordings of the same wells and kind met on
        one DIV (a scan re-run after a failed split, say) -- the next ``_1``, ``_2`` after it, the
        way MaxLab itself names a collision. The earlier recording keeps the plain name."""
        base = re.sub(r"_\d+$", "", tail)
        candidates = [div_dir / f"{stem}_{tail}.raw.h5"]
        candidates += (div_dir / f"{stem}_{base}_{n}.raw.h5" for n in range(1, 1000))
        return next(c for c in candidates if not c.exists() and c not in taken)

    def _starts(group: list[_Piece]) -> int:
        return min((piece.starts[0] if piece.starts else 0) for piece in group)

    for group in sorted(groups, key=lambda g: (g[0].chip_dir, g[0].div, g[0].tail, _starts(g))):
        first = group[0]
        wells = sorted({w for piece in group for w in piece.wells})
        whole = [piece for piece in group if sorted(piece.wells) == wells]
        if not whole and len(wells) != sum(len(piece.wells) for piece in group):
            for piece in group:
                skipped.append((piece.path, "two pieces of this recording hold the same well"))
            continue
        div_dir = first.chip_dir / f"DIV_{first.div}"
        stem = recording_stem(first.batch, first.plate_date, first.chip, wells, first.div)
        dest = _free_name(div_dir, stem, first.tail)
        taken.add(dest)
        new_prefix = selection_prefix(dest)
        if whole:
            # The recording as the rig wrote it is still here: keep it, and the pieces a failed
            # split made of it are copies with nothing to add.
            keep = whole[0]
            moved.append((keep.path, dest))
            pieces_of = [piece for piece in group if piece is not keep]
            redundant.extend(piece.path for piece in pieces_of)
            for piece in pieces_of:
                old_selection = piece.path.parent / "electrode_selection"
                if old_selection.is_dir():
                    redundant.extend(
                        f for f in sorted(old_selection.iterdir())
                        if f.is_file() and f.name.startswith(selection_prefix(piece.path) + "_")
                    )
            group = [keep]
        elif len(group) == 1:
            moved.append((first.path, dest))
        else:
            stitched.append((tuple(piece.path for piece in group), dest))
        for piece in group:
            old_prefix = selection_prefix(piece.path)
            old_selection = piece.path.parent / "electrode_selection"
            if old_selection.is_dir() and old_selection != div_dir / "electrode_selection":
                for f in sorted(old_selection.iterdir()):
                    if f.is_file() and f.name.startswith(old_prefix + "_"):
                        _plan_selection(f, div_dir / "electrode_selection" / (new_prefix + f.name[len(old_prefix):]))
        if not whole:
            # Outputs a selection left in the chip directory, from a run on the unsplit file before
            # it was split, are named for the whole recording; a stitched one claims them -- after
            # every whole file still present has claimed its own.
            stray_claims.append((first.chip_dir, selection_prefix(f"{stem}_{first.tail}.raw.h5"), dest))
        old_paths = {str(piece.path) for piece in group} | {str(p) for p in redundant}
        for well in wells:
            well_dir = Path(config.preprocessed_dir) / first.batch.id / first.chip / f"well{well}"
            for npz_path in sorted(well_dir.glob(f"DIV{first.div}_*exp_data.npz")) if well_dir.is_dir() else []:
                with np.load(npz_path, allow_pickle=True) as npz:
                    if "path_to_h5" not in npz.files or str(np.asarray(npz["path_to_h5"]).item()) not in old_paths:
                        continue
                    contents = {key: npz[key] for key in npz.files}
                contents["path_to_h5"] = np.array(str(dest), dtype=object if contents["path_to_h5"].dtype == object else None)
                npz_rewrites.append((npz_path, dest, contents))
        journal.append((first, [piece.path for piece in group], dest, len(group) > 1))

    for chip_path, base_prefix, dest in stray_claims:
        stray = chip_path / "electrode_selection"
        new_prefix = selection_prefix(dest)
        if stray.is_dir():
            for f in sorted(stray.iterdir()):
                if f.is_file() and f.name.startswith(base_prefix + "_"):
                    _plan_selection(f, dest.parent / "electrode_selection" / (new_prefix + f.name[len(base_prefix):]))

    for src, dest in moved:
        on_progress(f"{verb} move {src.relative_to(root)} -> {dest.relative_to(root)}")
    for parts, dest in stitched:
        on_progress(f"{verb} stitch {len(parts)} files -> {dest.relative_to(root)}")
        for part in parts:
            on_progress(f"    {part.relative_to(root)}")
    for src, dest in selection_moves:
        on_progress(f"{verb} move selection output {src.relative_to(root)} -> {dest.relative_to(root)}")
    for src in redundant:
        on_progress(f"{verb} trash {src.relative_to(root)} (a copy of part of a recording that is kept whole)")
    for npz_path, dest, _ in npz_rewrites:
        on_progress(f"{verb} point {npz_path.name} at {dest.name}")
    for path, why in skipped:
        on_progress(f"Skipping {path}: {why}")

    trash_root = None
    pruned: list[Path] = []
    npz_done: list[Path] = []
    if not dry_run:
        for parts, dest in stitched:
            on_progress(f"Stitching {dest.name}...")
            stitch_wells(parts, dest, on_progress=on_progress)
        for src, dest in moved:
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dest)
        for src, dest in selection_moves:
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dest)
        for npz_path, _, contents in npz_rewrites:
            stat = npz_path.stat()
            tmp = npz_path.with_name(npz_path.name + ".tmp")
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **contents)
            os.replace(tmp, npz_path)
            os.utime(npz_path, (stat.st_atime, stat.st_mtime))
            npz_done.append(npz_path)
        to_trash = [part for parts, _ in stitched for part in parts] + redundant
        if to_trash:
            stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
            trash_root = Path(config.trash_dir) / f"{stamp}_relayout"
            data_root = Path(config.data_root).resolve()
            for part in to_trash:
                target = trash_root / part.resolve().relative_to(data_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(part), str(target))
            on_progress(f"Pieces of the stitched recordings and redundant copies moved to {trash_root}")
        # Prune what the move emptied, deepest first: electrode_selection/, DIV_*/, then well_*/.
        prunable = re.compile(r"^(electrode_selection|DIV_\d+|well_\d+)$")
        candidates = sorted(
            {d for piece in pieces for d in (piece.path.parent / "electrode_selection", piece.path.parent, piece.path.parent.parent)}
            | {piece.chip_dir / "electrode_selection" for piece in pieces},
            key=lambda d: len(d.parts), reverse=True,
        )
        for d in candidates:
            if d.is_dir() and prunable.match(d.name) and not any(d.iterdir()):
                d.rmdir()
                pruned.append(d)

        from mxtreme import transactions

        for first, sources, dest, was_stitched in journal:
            location = parse_recording_path(dest, root)
            for well in (location.wells if location else []):
                transactions.record(
                    config,
                    "recording.moved",
                    batch_id=first.batch,
                    plate_date=first.plate_date,
                    experiment=location.experiment,
                    chip=first.chip,
                    well=well,
                    div=first.div,
                    actor=actor,
                    note=reason,
                    data={"from": [str(s) for s in sources], "path": str(dest), "stitched": was_stitched},
                )
        on_progress(
            f"Relayout done: {len(moved)} file(s) moved, {len(stitched)} recording(s) stitched, "
            f"{len(redundant)} redundant file(s) trashed, {len(selection_moves)} selection output(s) "
            f"moved, {len(npz_done)} .npz re-pointed, {len(skipped)} skipped."
        )

    return Relayout(
        moved=tuple(moved),
        stitched=tuple(stitched),
        redundant=tuple(redundant),
        selection_files=tuple(selection_moves),
        npz_files=tuple(npz_done if not dry_run else [n for n, _, _ in npz_rewrites]),
        skipped=tuple(skipped),
        trashed_to=trash_root,
        pruned_dirs=tuple(pruned),
    )
