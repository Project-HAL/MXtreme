"""The store's transaction log: an append-only record of what people *decided* about their data.

Everything else in the managed store is derived from a recording: the registry is rebuilt from the
``.h5`` files, the analyses from the preprocessed ``.npz``. But some facts about a culture are not in
any file -- that it died on DIV 31, that a chip is a MaxOne+ rather than a MaxOne, that a batch was
ended. Those are told to MXtreme by a person, and they are kept here so that they survive a registry
rebuild and so that the *history* of them is never lost: nothing is ever edited or deleted, a
correction is another line.

The log is ``<data_root>/transactions.jsonl`` (:attr:`~mxtreme.config.Config.transactions_path`):
one JSON object per line, appended and never rewritten, so a partial line from a crash mid-write
costs one record and not the file. Each record carries ``time`` (ISO 8601, local time with offset),
``op`` (see :data:`OPS`), the identity it is about (``batch_id``, ``plate_date``, and for a chip or
culture ``chip`` and ``well``), who recorded it (``actor``), a free-text ``note`` and any op-specific
``data``.

The current state is the fold of the log -- :func:`culture_states`, :func:`batch_states`,
:func:`chip_devices` -- computed on read. The file is small (a few lines per culture over its life),
so there is no index and no cache to invalidate.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from mxtreme.store import Batch, _validate_plate_date

#: File name of the log inside the store; see :attr:`mxtreme.config.Config.transactions_path`.
TRANSACTIONS_FILE = "transactions.jsonl"

#: Every operation the log knows. A record with an op outside this list is refused on write and
#: skipped on read, so a typo cannot masquerade as a decision.
OPS = (
    "culture.mark_dead",  # this well on this chip is no longer a living culture
    "culture.mark_alive",  # correction of the above
    "batch.mark_dead",  # the whole plating is over: every culture in it is dead
    "batch.mark_alive",  # correction of the above; cultures keep their own individual marks
    "chip.set_device",  # data: {"device": "MaxOne" | "MaxOne+" | "MaxTwo"}
    "note",  # a free-text remark against a batch, chip or culture; changes no state
)

#: What a chip can be declared to be through ``chip.set_device``. The batch id's ``system`` field
#: (``M1``/``M2``) only tells MaxOne from MaxTwo apart; a MaxOne+ is a MaxOne as far as the id
#: convention is concerned, so it has to be said explicitly.
DEVICES = ("MaxOne", "MaxOne+", "MaxTwo")


@dataclass(frozen=True)
class Transaction:
    """One line of the log.

    :param time: When it was recorded, ISO 8601.
    :param op: One of :data:`OPS`.
    :param batch_id: The batch the record is about.
    :param plate_date: Its plating date (``YYMMDD``), because a batch id can in principle be reused.
    :param chip: The chip, for chip- and culture-level ops; ``None`` for batch-level ones.
    :param well: The well, for culture-level ops; ``None`` otherwise.
    :param actor: Who recorded it -- a user name, free-form.
    :param note: Why, in the recorder's words. Optional.
    :param data: Op-specific fields, e.g. the device for ``chip.set_device``.
    """

    time: str
    op: str
    batch_id: str
    plate_date: int
    chip: str | None = None
    well: int | None = None
    actor: str = ""
    note: str = ""
    data: dict = field(default_factory=dict)

    @property
    def batch_key(self) -> tuple[str, int]:
        return (self.batch_id, self.plate_date)

    @property
    def chip_key(self) -> tuple[str, int, str]:
        return (self.batch_id, self.plate_date, str(self.chip))

    @property
    def culture_key(self) -> tuple[str, int, str, int]:
        return (self.batch_id, self.plate_date, str(self.chip), int(self.well))


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def record(
    config,
    op: str,
    *,
    batch_id: str | Batch,
    plate_date,
    chip: str | None = None,
    well: int | None = None,
    actor: str = "",
    note: str = "",
    data: dict | None = None,
    time: str | None = None,
) -> Transaction:
    """Append one transaction to the store's log.

    The identity is validated the way the recordings tree validates it (a real batch id, a
    ``YYMMDD`` plate date), and an op that needs a chip or a well refuses to go in without one --
    a record that names nothing is not a decision about anything.

    :param config: The :class:`~mxtreme.config.Config` of the store.
    :param op: One of :data:`OPS`.
    :param batch_id: The batch, as a :class:`~mxtreme.store.Batch` or its id string.
    :param plate_date: The plating date, ``YYMMDD``.
    :param chip: Chip serial; required for ``chip.*`` and ``culture.*`` ops.
    :param well: Well number; required for ``culture.*`` ops.
    :param actor: Who is recording this.
    :param note: Free text.
    :param data: Op-specific fields. ``chip.set_device`` requires ``{"device": <DEVICES>}``.
    :param time: Override the timestamp (for tests or for back-filling a decision made earlier).
    :returns: The record as written.
    :raises ValueError: If the op, identity or data is malformed.
    """
    if op not in OPS:
        raise ValueError(f"Unknown transaction op {op!r}; expected one of {OPS}.")
    batch = Batch.parse(batch_id)
    plate_date = _validate_plate_date(plate_date)
    data = dict(data or {})

    level = op.split(".", 1)[0]
    if level in ("chip", "culture") and not chip:
        raise ValueError(f"{op} needs a chip.")
    if level == "culture" and well is None:
        raise ValueError(f"{op} needs a well.")
    if op == "chip.set_device" and data.get("device") not in DEVICES:
        raise ValueError(f"chip.set_device needs data['device'] in {DEVICES}, got {data.get('device')!r}.")
    if chip is not None and ("/" in str(chip) or str(chip) in (".", "..")):
        raise ValueError(f"Not a chip serial: {chip!r}.")

    tx = Transaction(
        time=time or _now(),
        op=op,
        batch_id=batch.id,
        plate_date=plate_date,
        chip=None if chip is None else str(chip),
        well=None if well is None else int(well),
        actor=str(actor or ""),
        note=str(note or ""),
        data=data,
    )

    path = Path(config.transactions_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(tx), ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
    return tx


def iter_transactions(config) -> Iterator[Transaction]:
    """Read the log back, oldest first, skipping lines that are not valid records.

    A torn last line (the process died mid-append) or a hand-edited line that does not parse is
    skipped rather than fatal: the decisions before it are still good, and a front end should
    show them rather than nothing.
    """
    path = Path(config.transactions_path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                if obj.get("op") not in OPS:
                    continue
                yield Transaction(
                    time=str(obj["time"]),
                    op=str(obj["op"]),
                    batch_id=str(obj["batch_id"]),
                    plate_date=int(obj["plate_date"]),
                    chip=None if obj.get("chip") is None else str(obj["chip"]),
                    well=None if obj.get("well") is None else int(obj["well"]),
                    actor=str(obj.get("actor") or ""),
                    note=str(obj.get("note") or ""),
                    data=dict(obj.get("data") or {}),
                )
            except (ValueError, KeyError, TypeError, AttributeError):
                continue


def read(config) -> list[Transaction]:
    """The whole log, oldest first."""
    return list(iter_transactions(config))


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
