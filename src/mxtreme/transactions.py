"""The store's transaction log: an append-only journal of everything that changed the managed store.

Two kinds of thing land here. What MXtreme *did* to the store -- a scan registered, an outside
recording ingested, a well preprocessed, bursts detected, a report written, the registry rebuilt --
journaled by the function that did it, right after it did it. And what people *decided* about
their data that no file records -- a culture marked dead, a batch ended, a chip declared a MaxOne+ --
recorded by a front end on their behalf. Both are the same record shape, in the same file, in the
order they happened, so a culture's history reads top to bottom: scanned, analysed, reported, died.

The file is ``<data_root>/transactions.jsonl`` (:attr:`~mxtreme.config.Config.transactions_path`):
one JSON object per line, appended under an advisory lock and never rewritten. Nothing is edited or
deleted; a correction is another line. A partial line from a crash mid-write costs that one record,
and a line that does not parse is skipped on read rather than hiding everything before it.

Each record carries ``time`` (ISO 8601, local time with offset), ``op`` (see :data:`OPS`), as much
identity as the operation has -- ``batch_id``, ``plate_date``, ``exp_id``, ``chip``, ``well``,
``div`` -- who did it (``actor``, defaulting to the OS user for MXtreme's own journal entries), a
free-text ``note`` and any op-specific ``data`` (paths written, counts, the device declared).

Identity is recorded as the writer knows it. A scan knows its batch; a preprocessed ``.npz`` written
from an older flow may know only its ``exp_id`` (which, for scans, *is* the batch id). Readers that
want one culture's entries use :func:`for_culture`, which matches either way.

The current state of the decision-type records is a fold, computed on read: :func:`batch_states`,
:func:`culture_states`, :func:`chip_devices`, combined by :func:`is_dead`. The file is small enough
(a handful of lines per recording) that there is no index and no cache to invalidate.
"""

from __future__ import annotations

import getpass
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from mxtreme.store import Batch, _validate_plate_date

#: File name of the log inside the store; see :attr:`mxtreme.config.Config.transactions_path`.
TRANSACTIONS_FILE = "transactions.jsonl"

#: Every operation the log knows, mapped to the identity it requires: ``"store"`` needs nothing,
#: ``"batch"`` a batch id, ``"chip"`` a chip too, ``"culture"`` a well too, ``"recording"`` a DIV
#: too. A record with an op outside this table is refused on write and skipped on read, so a typo
#: cannot masquerade as an event.
OPS: dict[str, str] = {
    # -- decisions people record, through a front end --------------------------------------------
    "culture.mark_dead": "culture",  # this well on this chip is no longer a living culture
    "culture.mark_alive": "culture",  # correction of the above
    "batch.mark_dead": "batch",  # the whole plating is over: every culture in it is dead
    "batch.mark_alive": "batch",  # correction of the above; cultures keep their own marks
    "chip.set_device": "chip",  # data: {"device": one of DEVICES}
    "note": "batch",  # a remark against a batch, chip or culture; changes no state
    # -- what MXtreme did to the store, journaled by the function that did it ---------------------
    "activity_scan.registered": "recording",  # data: {"path"}; one per well
    "network_scan.registered": "recording",  # data: {"path"}; one per well
    "recording.ingested": "recording",  # data: {"source", "path", "moved"}; one per well
    "preprocessed.saved": "recording",  # data: {"path", "source"}
    "preprocessed.repaired": "recording",  # data: {"path"}; spike order fixed in place
    "bursts.saved": "recording",  # data: {"path", "n_bursts"}
    "report.written": "culture",  # data: {"path", "divs", "sections"}; one per culture in it
    "registry.rebuilt": "store",  # data: {"registry", "n_recordings", "n_raw", "n_unparseable"}
    "batch.renamed": "batch",  # data: {"old", "new"}; recorded against the new id
}

#: Ops whose identity is validated the way the recordings tree validates it (a real batch id, a
#: ``YYMMDD`` plate date). These come from people, typed or clicked, and a placeholder identity
#: would make a decision about nothing. MXtreme's own journal entries carry whatever the writer
#: knew, which for an older flow can be an ``exp_id`` alone.
STRICT_OPS = frozenset(
    {
        "culture.mark_dead",
        "culture.mark_alive",
        "batch.mark_dead",
        "batch.mark_alive",
        "chip.set_device",
        "note",
    }
)

#: What a chip can be declared to be through ``chip.set_device``. The batch id's ``system`` field
#: (``M1``/``M2``) only tells MaxOne from MaxTwo apart; a MaxOne+ is a MaxOne as far as the id
#: convention is concerned, so it has to be said explicitly.
DEVICES = ("MaxOne", "MaxOne+", "MaxTwo")

#: The ops that a person records rather than MXtreme -- for a reader that wants to show the two
#: kinds apart.
DECISION_OPS = STRICT_OPS


@dataclass(frozen=True)
class Transaction:
    """One line of the log.

    :param time: When it was recorded, ISO 8601.
    :param op: One of :data:`OPS`.
    :param batch_id: The batch the record is about; ``""`` when the writer knew only an exp id.
    :param plate_date: Its plating date (``YYMMDD``), or ``None`` when unknown.
    :param exp_id: The experiment the record is about (the batch id, for scans); ``""`` if none.
    :param chip: The chip, for chip-, culture- and recording-level ops; ``None`` otherwise.
    :param well: The well, for culture- and recording-level ops; ``None`` otherwise.
    :param div: Days *in vitro*, for recording-level ops; ``None`` otherwise.
    :param actor: Who did it -- a user name, free-form.
    :param note: Why, in the writer's words. Optional.
    :param data: Op-specific fields: paths written, counts, the device declared.
    """

    time: str
    op: str
    batch_id: str = ""
    plate_date: int | None = None
    exp_id: str = ""
    chip: str | None = None
    well: int | None = None
    div: int | None = None
    actor: str = ""
    note: str = ""
    data: dict = field(default_factory=dict)

    @property
    def scope(self) -> str:
        return OPS.get(self.op, "store")

    @property
    def batch_key(self) -> tuple[str, int | None]:
        return (self.batch_id, self.plate_date)

    @property
    def chip_key(self) -> tuple[str, int | None, str]:
        return (self.batch_id, self.plate_date, str(self.chip))

    @property
    def culture_key(self) -> tuple[str, int | None, str, int]:
        return (self.batch_id, self.plate_date, str(self.chip), int(self.well))


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _default_actor() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 -- no user database, no login name; the record still goes in
        return ""


def transactions_path_for(registry_path: str | Path) -> Path:
    """The log that goes with a registry: the two sit side by side at the store's root.

    For the writers that are handed a registry path or a store subdirectory rather than a
    :class:`~mxtreme.config.Config` -- the same convention :func:`mxtreme.io.save_preprocessed`
    uses to find the registry from its ``datastore``.
    """
    return Path(registry_path).parent / TRANSACTIONS_FILE


def _resolve_path(config) -> Path:
    """Accept a :class:`~mxtreme.config.Config`, a path to the log file, or a path to the store root."""
    path = getattr(config, "transactions_path", None)
    if path is not None:
        return Path(path)
    path = Path(config)
    return path if path.suffix == ".jsonl" else path / TRANSACTIONS_FILE


def _clean_chip(chip) -> str | None:
    if chip is None:
        return None
    chip = str(chip)
    if "/" in chip or chip in (".", ".."):
        raise ValueError(f"Not a chip serial: {chip!r}.")
    return chip


def _int_or_none(value, what: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be an integer, got {value!r}.") from exc


def record(
    config,
    op: str,
    *,
    batch_id: str | Batch | None = None,
    plate_date=None,
    exp_id: str | None = None,
    chip: str | None = None,
    well: int | None = None,
    div: int | None = None,
    actor: str | None = None,
    note: str = "",
    data: dict | None = None,
    time: str | None = None,
) -> Transaction:
    """Append one transaction to the store's log.

    Every writer in MXtreme that changes the store calls this after the change is safely on disk,
    so the journal never claims something the store does not have. An op that needs a chip, a well
    or a DIV refuses to go in without one -- a record that names nothing is not a record of
    anything -- and the decision ops (:data:`STRICT_OPS`) validate their batch identity in full.

    :param config: The :class:`~mxtreme.config.Config` of the store, or a path to the log file (see
        :func:`transactions_path_for`), or the store's root directory.
    :param op: One of :data:`OPS`.
    :param batch_id: The batch, as a :class:`~mxtreme.store.Batch` or its id string. Optional for
        MXtreme's journal ops, where an older flow may know only ``exp_id``.
    :param plate_date: The plating date, ``YYMMDD``. Required with ``batch_id`` for strict ops.
    :param exp_id: The experiment name (a scan's is its batch id).
    :param chip: Chip serial; required for chip-, culture- and recording-level ops.
    :param well: Well number; required for culture- and recording-level ops.
    :param div: Days *in vitro*; required for recording-level ops.
    :param actor: Who did it. ``None`` records the OS user; ``""`` records nobody.
    :param note: Free text.
    :param data: Op-specific fields. ``chip.set_device`` requires ``{"device": <DEVICES>}``.
    :param time: Override the timestamp (for tests, or for back-filling a decision made earlier).
    :returns: The record as written.
    :raises ValueError: If the op, identity or data is malformed.
    """
    scope = OPS.get(op)
    if scope is None:
        raise ValueError(f"Unknown transaction op {op!r}; expected one of {tuple(OPS)}.")
    data = dict(data or {})

    if op in STRICT_OPS:
        if batch_id is None or plate_date is None:
            raise ValueError(f"{op} needs a batch id and a plate date.")
        batch_id = Batch.parse(batch_id).id
        plate_date = _validate_plate_date(plate_date)
    else:
        batch_id = "" if batch_id is None else str(getattr(batch_id, "id", batch_id))
        plate_date = _int_or_none(plate_date, "plate_date")

    chip = _clean_chip(chip)
    well = _int_or_none(well, "well")
    div = _int_or_none(div, "div")
    if scope in ("chip", "culture", "recording") and not chip:
        raise ValueError(f"{op} needs a chip.")
    if scope in ("culture", "recording") and well is None:
        raise ValueError(f"{op} needs a well.")
    if scope == "recording" and div is None:
        raise ValueError(f"{op} needs a DIV.")
    if scope in ("batch", "chip", "culture", "recording") and not batch_id and not exp_id:
        raise ValueError(f"{op} needs a batch id or an exp id.")
    if op == "chip.set_device" and data.get("device") not in DEVICES:
        raise ValueError(f"chip.set_device needs data['device'] in {DEVICES}, got {data.get('device')!r}.")

    tx = Transaction(
        time=time or _now(),
        op=op,
        batch_id=batch_id,
        plate_date=plate_date,
        exp_id="" if exp_id is None else str(exp_id),
        chip=chip,
        well=well,
        div=div,
        actor=_default_actor() if actor is None else str(actor),
        note=str(note or ""),
        data=data,
    )
    _append(
        _resolve_path(config), json.dumps(asdict(tx), ensure_ascii=False, separators=(",", ":"), default=str)
    )
    return tx


def _append(path: Path, line: str) -> None:
    """Append one line under an advisory lock, so two writers (a scan thread and a web server,
    say) cannot interleave halves of a record. Best effort where the platform has no ``fcntl``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # Windows
        fcntl = None
    with path.open("a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line + "\n")
            f.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def iter_transactions(config) -> Iterator[Transaction]:
    """Read the log back, oldest first, skipping lines that are not valid records.

    A torn last line (the process died mid-append) or a hand-edited line that does not parse is
    skipped rather than fatal: the records before it are still good, and a front end should show
    them rather than nothing.
    """
    path = _resolve_path(config)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict) or obj.get("op") not in OPS:
                    continue
                yield Transaction(
                    time=str(obj["time"]),
                    op=str(obj["op"]),
                    batch_id=str(obj.get("batch_id") or ""),
                    plate_date=_int_or_none(obj.get("plate_date"), "plate_date"),
                    exp_id=str(obj.get("exp_id") or ""),
                    chip=None if obj.get("chip") is None else str(obj["chip"]),
                    well=_int_or_none(obj.get("well"), "well"),
                    div=_int_or_none(obj.get("div"), "div"),
                    actor=str(obj.get("actor") or ""),
                    note=str(obj.get("note") or ""),
                    data=dict(obj.get("data") or {}),
                )
            except (ValueError, KeyError, TypeError, AttributeError):
                continue


def read(config) -> list[Transaction]:
    """The whole log, oldest first."""
    return list(iter_transactions(config))


# --- selecting ------------------------------------------------------------------------------------


def _same_batch(tx: Transaction, batch_id: str, plate_date: int | None) -> bool:
    """Whether a record is about ``batch_id``.

    A record that knows its batch says so; one from an older flow knows only its ``exp_id``, which
    for anything MXtreme scanned is the batch id. A plate date, when both sides have one, has to
    agree -- a chip is re-plated across batches and an id could in principle be reused.
    """
    if tx.batch_id:
        if tx.batch_id != batch_id:
            return False
    elif tx.exp_id != batch_id:
        return False
    return plate_date is None or tx.plate_date is None or int(tx.plate_date) == int(plate_date)


def for_batch(
    transactions: Iterable[Transaction], batch_id: str | Batch, plate_date=None
) -> list[Transaction]:
    """Every record about one batch: its own, and every chip's, culture's and recording's in it."""
    batch_id = str(getattr(batch_id, "id", batch_id))
    plate_date = _int_or_none(plate_date, "plate_date")
    return [tx for tx in transactions if _same_batch(tx, batch_id, plate_date)]


def for_culture(
    transactions: Iterable[Transaction], batch_id: str | Batch, plate_date, chip: str, well: int
) -> list[Transaction]:
    """Every record about one culture, plus the batch-level records that apply to it.

    A culture's history is what happened to that well on that chip -- scans, analyses, reports,
    marks -- and the batch-wide events that changed its state too (the batch ended, or was
    renamed). Chip-level records for its chip are included for the same reason. Records about
    *other* wells on the same chip are not.
    """
    batch_id = str(getattr(batch_id, "id", batch_id))
    plate_date = _int_or_none(plate_date, "plate_date")
    chip, well = str(chip), int(well)
    out = []
    for tx in transactions:
        if not _same_batch(tx, batch_id, plate_date):
            continue
        if tx.chip is None:
            out.append(tx)  # batch-level
        elif str(tx.chip) == chip and (tx.well is None or int(tx.well) == well):
            out.append(tx)
    return out


# --- back-filling an existing store ---------------------------------------------------------------

#: Registry ``kind`` -> the journal op that would have been written when the row was.
_KIND_OPS = {
    "activity_scan": "activity_scan.registered",
    "network_scan": "network_scan.registered",
    "experiment": "recording.ingested",
    "preprocessed": "preprocessed.saved",
}

BACKFILL_NOTE = "back-filled from the registry"


def backfill(config, *, on_progress=print) -> int:
    """Give a store that predates the log a history, from what its registry and burst logs know.

    One record per registry row -- the op the row's ``kind`` would have journaled, timestamped
    with the row's own ``timestamp`` -- and one ``bursts.saved`` per burst-log row, each noted as
    back-filled so nobody mistakes it for a live record. Rows the log already has a matching
    record for (same op and identity) are skipped, so this is safe to run more than once, and
    safe to run on a store that has been journaling for a while: it only adds what is missing.

    Reports are not back-filled: nothing on disk says which cultures a PDF was about.

    :returns: How many records were added.
    """
    import pandas as pd

    from mxtreme.store import chip_dir

    def _when(value) -> str:
        """A registry or burst-log timestamp as ISO 8601; a stamp that does not parse is 'now'."""
        if value is None or value == "":
            return _now()
        try:
            stamp = (
                pd.Timestamp(float(value), unit="s")
                if str(value).replace(".", "", 1).isdigit()
                else pd.Timestamp(value)
            )
            return stamp.isoformat(timespec="seconds")
        except (ValueError, TypeError, OverflowError):
            return _now()

    existing = {(t.op, t.batch_id or t.exp_id, str(t.chip), t.well, t.div) for t in iter_transactions(config)}
    added = 0

    registry = Path(config.registry_path)
    if registry.is_file():
        df = pd.read_csv(registry).fillna("")
        for row in df.to_dict("records"):
            op = _KIND_OPS.get(str(row.get("kind") or "preprocessed"))
            if op is None:
                continue
            batch_id, exp_id = str(row.get("batch_id") or ""), str(row.get("exp_id") or "")
            chip, well, div = (
                str(row.get("chip")),
                _int_or_none(row.get("well"), "well"),
                _int_or_none(row.get("div"), "div"),
            )
            key = (op, batch_id or exp_id, chip, well, div)
            if key in existing or well is None or div is None:
                continue
            path = ""
            plate_date = _int_or_none(row.get("plate_date") or None, "plate_date")
            if op != "preprocessed.saved" and batch_id and plate_date:
                try:
                    tail = {
                        "activity_scan.registered": "_activity_scan",
                        "network_scan.registered": "_network_scan",
                    }.get(op, f"_{exp_id}")
                    div_dir = chip_dir(config, batch_id, plate_date, chip) / f"well_{well}" / f"DIV_{div}"
                    matches = (
                        sorted(div_dir.glob(f"*{tail}*.h5"), key=lambda p: p.stat().st_mtime)
                        if div_dir.is_dir()
                        else []
                    )
                    path = str(matches[-1]) if matches else ""
                except (ValueError, OSError):
                    path = ""
            elif op == "preprocessed.saved":
                well_dir = Path(config.preprocessed_dir) / (exp_id or batch_id) / chip / f"well{well}"
                matches = sorted(well_dir.glob(f"DIV{div}_*exp_data.npz")) if well_dir.is_dir() else []
                path = str(matches[-1]) if matches else ""
            record(
                config,
                op,
                batch_id=batch_id or None,
                plate_date=plate_date,
                exp_id=exp_id,
                chip=chip,
                well=well,
                div=div,
                actor="",
                note=BACKFILL_NOTE,
                data={"path": path},
                time=_when(row.get("timestamp")),
            )
            existing.add(key)
            added += 1

    burst_root = Path(config.burst_data_dir)
    for log_path in sorted(burst_root.glob("*_burst_log.csv")) if burst_root.is_dir() else []:
        df = pd.read_csv(log_path).fillna("")
        for row in df.to_dict("records"):
            exp_id, chip = str(row.get("exp_id") or ""), str(row.get("chip"))
            well, div = _int_or_none(row.get("well"), "well"), _int_or_none(row.get("DIV"), "div")
            key = ("bursts.saved", exp_id, chip, well, div)
            if key in existing or well is None or div is None or not exp_id:
                continue
            well_dir = burst_root / exp_id / chip / f"well{well}"
            matches = sorted(well_dir.glob(f"DIV{div}_*burst_data.csv")) if well_dir.is_dir() else []
            n_bursts = _int_or_none(row.get("n_bursts") or None, "n_bursts")
            record(
                config,
                "bursts.saved",
                exp_id=exp_id,
                plate_date=_int_or_none(row.get("plate_date") or None, "plate_date"),
                chip=chip,
                well=well,
                div=div,
                actor="",
                note=BACKFILL_NOTE,
                data={"path": str(matches[-1]) if matches else "", "n_bursts": n_bursts},
                time=_when(row.get("features_computed_at") or row.get("detection_completed_at")),
            )
            existing.add(key)
            added += 1

    on_progress(f"Transaction log: back-filled {added} record(s) into {_resolve_path(config)}")
    return added


# --- folding the decisions ------------------------------------------------------------------------


@dataclass(frozen=True)
class Mark:
    """The latest word on whether something is alive: what was said, when, and by whom."""

    dead: bool
    time: str
    actor: str = ""
    note: str = ""


def batch_states(config, transactions: list[Transaction] | None = None) -> dict[tuple[str, int], Mark]:
    """``{(batch_id, plate_date): Mark}`` for every batch the log has a ``batch.mark_*`` for.

    A batch absent from the result has never been marked and is alive.
    """
    marks: dict[tuple[str, int], Mark] = {}
    for tx in transactions if transactions is not None else iter_transactions(config):
        if tx.op == "batch.mark_dead":
            marks[tx.batch_key] = Mark(True, tx.time, tx.actor, tx.note)
        elif tx.op == "batch.mark_alive":
            marks[tx.batch_key] = Mark(False, tx.time, tx.actor, tx.note)
    return marks


def culture_states(
    config, transactions: list[Transaction] | None = None
) -> dict[tuple[str, int, str, int], Mark]:
    """``{(batch_id, plate_date, chip, well): Mark}`` for every culture the log has marked.

    Only the culture's *own* marks are folded here. A batch marked dead makes every culture in it
    dead regardless of these -- combine with :func:`batch_states` (see :func:`is_dead`) -- and
    marking the batch alive again restores each culture to its own last mark, so a culture that
    died before the batch was ended stays dead after a mistaken end is undone.
    """
    marks: dict[tuple[str, int, str, int], Mark] = {}
    for tx in transactions if transactions is not None else iter_transactions(config):
        if tx.op == "culture.mark_dead":
            marks[tx.culture_key] = Mark(True, tx.time, tx.actor, tx.note)
        elif tx.op == "culture.mark_alive":
            marks[tx.culture_key] = Mark(False, tx.time, tx.actor, tx.note)
    return marks


def is_dead(
    batch_marks: dict[tuple[str, int], Mark],
    culture_marks: dict[tuple[str, int, str, int], Mark],
    batch_id: str,
    plate_date: int,
    chip: str,
    well: int,
) -> Mark | None:
    """The mark that makes a culture dead, or ``None`` if it is alive.

    The batch's mark wins when it says dead; otherwise the culture's own mark decides.
    """
    batch = batch_marks.get((batch_id, int(plate_date)))
    if batch is not None and batch.dead:
        return batch
    own = culture_marks.get((batch_id, int(plate_date), str(chip), int(well)))
    if own is not None and own.dead:
        return own
    return None


def chip_devices(config, transactions: list[Transaction] | None = None) -> dict[tuple[str, int, str], str]:
    """``{(batch_id, plate_date, chip): device}`` for every chip the log has a ``chip.set_device`` for.

    A chip absent from the result is whatever its batch's system says: ``M1`` is a MaxOne, ``M2`` a
    MaxTwo (:func:`default_device`).
    """
    devices: dict[tuple[str, int, str], str] = {}
    for tx in transactions if transactions is not None else iter_transactions(config):
        if tx.op == "chip.set_device":
            devices[tx.chip_key] = str(tx.data["device"])
    return devices


def default_device(batch: Batch | str) -> str:
    """The device a batch's id implies for its chips: ``M1`` -> MaxOne, ``M2`` -> MaxTwo."""
    return "MaxTwo" if Batch.parse(batch).system == "M2" else "MaxOne"
