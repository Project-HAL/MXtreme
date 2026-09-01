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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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

    plating_match = re.match(r"^plating_(\d{6})_(.+)$", plating)
    chip_match = re.match(r"^chip_M[12]_(.+)$", chip_part)
    well_match = re.match(r"^well_(\d+)$", well_part)
    div_match = re.match(r"^DIV_(\d+)$", div_part)
    if not (plating_match and chip_match and well_match and div_match):
        return None
    try:
        batch = Batch.parse(plating_match.group(2))
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
        plate_date=int(plating_match.group(1)),
        chip=chip_match.group(1),
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

    return written
