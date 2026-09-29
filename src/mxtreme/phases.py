"""
A *phase* is a named interval of a recording. 

Phases are anchored to **maxlab event tags** already present in a recording's event data or can be specified in minutes 
from the first frame. The user defines a dictionary keyed by the phase name which 'start' and 'stop'
values to specify the beginning and end of the phse. :func:`phases_from_spec` turns that choice into a :class:`Phases`. 

:meth:`Phases.label_for` returns the phase given a frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Optional, Sequence, Union


@dataclass(frozen=True)
class Phase:
    """A single named closed interval ``[start_frame, end_frame]`` of a recording.

    Both bounds are inclusive, so ``end_frame`` is the last frame that belongs to the phase.

    :param name: Phase label.
    :param start_frame: First frame of the phase (inclusive).
    :param end_frame: Last frame of the phase (inclusive).
    """

    name: str
    start_frame: int
    end_frame: int

    @property
    def n_frames(self) -> int:
        """Number of frames in the phase (``end_frame - start_frame + 1``)."""
        return self.end_frame - self.start_frame + 1

    def contains(self, frame: float) -> bool:
        """Return whether ``frame`` falls in ``[start_frame, end_frame]``."""
        return self.start_frame <= frame <= self.end_frame


@dataclass(frozen=True)
class Phases:
    """An ordered collection of :class:`Phase` intervals covering (part of) a recording.

    Construct via :meth:`full` (the default single ``"full"`` phase) or :func:`phases_from_spec`.
    """

    phases: tuple[Phase, ...]

    @classmethod
    def full(cls, start_frame: int, end_frame: int, name: str = "full") -> "Phases":
        """Return a :class:`Phases` with one interval spanning ``[start_frame, end_frame]``.

        This is the default when the user injects no phases.
        """
        return cls((Phase(name, int(start_frame), int(end_frame)),))

    def label_for(self, frame: float) -> Optional[str]:
        """Return the name of the phase containing ``frame``, or ``None`` if none does.

        ``None`` only occurs when phases do not cover ``frame``. Where phases overlap, the earliest-starting one
        wins. The default ``full`` phase spans the whole recording, so it always matches.
        """
        for phase in self.phases:
            if phase.contains(frame):
                return phase.name
        return None

    def get_phase(self, label: str) -> Phase:
        """Return the phase named ``label``.

        Replicated phases are named ``name_1``, ``name_2``, ... (see :func:`phases_from_spec`), so pass
        the suffixed name to get one occurrence.

        :param label: The phase name, as listed in :attr:`names`.
        :raises KeyError: If no phase is named ``label``.
        :rtype: Phase
        """
        for phase in self.phases:
            if phase.name == label:
                return phase
        raise KeyError(f"No phase named {label!r}; available phases: {list(self.names)}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.phases)

    def __iter__(self) -> Iterator[Phase]:
        return iter(self.phases)

    def __len__(self) -> int:
        return len(self.phases)




# A phase spec (the value embedded in npz metadata as ``phase_spec``): an ordered ``name -> definition``
# mapping. A definition is a boundary (start only) or ``{"start": boundary, "end": boundary}``; a
# boundary is minutes (number), an event key (str), or a key/value matcher (Mapping). See
# :func:`phases_from_spec`.
PhaseSpec = Mapping[str, object]

_DEF_KEYS = frozenset({"start", "end"})


def _is_number(value) -> bool:
    """Return whether ``value`` is a real number (JSON int/float) and not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _minutes_to_frame(minutes, *, start_frame: int, samp_rate: float) -> int:
    """Return the frame ``minutes`` after the recording's first frame."""
    return int(start_frame + round(float(minutes) * 60.0 * samp_rate))


def _matcher(boundary) -> dict:
    """Normalize a tag boundary to a ``{key: value | None}`` matcher.

    A ``str`` is shorthand for ``{tag: None}`` (key present, any value). A Mapping is used as is.
    """
    if isinstance(boundary, str):
        return {boundary: None}
    return dict(boundary)


def _values_match(actual, expected) -> bool:
    """Compare an event value to a matcher value; maxlab stores values as strings (``'5'``)."""
    return actual == expected or str(actual) == str(expected)


def _message_matches(message, matcher: Mapping) -> bool:
    """Return whether ``message`` carries every pair of ``matcher`` (``None`` = key present)."""
    if not isinstance(message, dict):
        return False
    for key, expected in matcher.items():
        if key not in message:
            return False
        if expected is not None and not _values_match(message[key], expected):
            return False
    return True


def _event_frames(event_df, matcher: Mapping) -> list[int]:
    """Return every ``eventtime`` (frame) whose ``eventmessage`` matches ``matcher``, sorted."""
    mask = event_df["eventmessage"].apply(lambda d: _message_matches(d, matcher))
    return sorted(int(f) for f in event_df.loc[mask, "eventtime"])


def _check_boundary(name: str, boundary) -> None:
    """Raise ``ValueError`` unless ``boundary`` is minutes, an event key, or a non-empty matcher."""
    if _is_number(boundary) or (isinstance(boundary, str) and boundary):
        return
    if isinstance(boundary, Mapping) and boundary and all(isinstance(k, str) for k in boundary):
        return
    raise ValueError(
        f"Phase {name!r}: invalid boundary {boundary!r}. A boundary is a number of minutes, an event "
        "key (str), or a {key: value} matcher (value None = key present with any value)."
    )


def _normalize_spec(spec: PhaseSpec) -> list[tuple[str, object, object]]:
    """Validate ``spec`` and return its phases as ordered ``(name, start, end | None)`` triples."""
    if not isinstance(spec, Mapping):
        raise ValueError(f"A phase spec must be a mapping of name -> definition, got {spec!r}.")
    defs = []
    for name, definition in spec.items():
        if isinstance(definition, Mapping):
            if "start" not in definition or not set(definition) <= _DEF_KEYS:
                raise ValueError(
                    f"Phase {name!r}: a definition mapping must be {{'start': boundary}} with an "
                    f"optional 'end', got {definition!r}. To match an event by several key/value "
                    "pairs, put the matcher under 'start' (e.g. {'start': {'key': 'value'}})."
                )
            start, end = definition["start"], definition.get("end")
        else:
            start, end = definition, None
        _check_boundary(str(name), start)
        if end is not None:
            _check_boundary(str(name), end)
        defs.append((str(name), start, end))
    return defs


def phases_from_spec(
    spec: PhaseSpec,
    event_df,
    *,
    start_frame: int,
    end_frame: int,
    samp_rate: float,
) -> Phases:
    """Build :class:`Phases` from a phase ``spec``.

    ``spec`` is an ordered ``name -> definition`` mapping, e.g.::

        {
            "pre":   {"start": "pre_recording_start", "end": "pre_recording_end"},
            "train": {"start": {"closed_loop_start": None, "side": "left"},
                      "end": {"closed_loop_end": None, "side": "left"}},
            "post":  "post_recording_start",
            "base":  {"start": 0, "end": 20},
        }

    A **definition** is either a boundary (shorthand for a start with no explicit end) or a mapping
    with a required ``"start"`` and an optional ``"end"`` boundary. A **boundary** is:

    - a **number**: minutes from the recording's first frame;
    - a **str**: an event key; any event whose message contains that key matches;
    - a **mapping**: a key/value matcher; an event matches when its message contains every pair
      (a value of ``None`` means the key is present with any value; values compare equal as strings,
      so ``5`` matches maxlab's ``'5'``).

    A tag start matches every event it applies to, so a phase that repeats yields one interval per
    occurrence, named ``name_1``, ``name_2``, ... in time order (a single occurrence keeps ``name``).
    Phase bounds are inclusive (see :class:`Phase`). Each occurrence ends at its ``end`` boundary:

    - minutes: the last frame *before* that time, so ``{"start": 0, "end": 20}`` is exactly 20
      minutes long and does not overlap a phase starting at minute 20;
    - an event: the frame of the first matching event after its start (that frame is included).

    Without an ``end`` (or when the end event never follows the start), it runs to the frame before
    the next start of any phase, else to ``end_frame`` (included). Explicit ends may leave gaps, where
    :meth:`Phases.label_for` returns ``None``; where intervals overlap it returns the earliest-starting
    one.

    Starts that don't resolve (a tag absent from ``event_df``) are skipped, so the same spec can be
    applied across a batch of recordings that don't all carry every tag. If no start resolves, a single
    ``"full"`` phase spanning ``[start_frame, end_frame]`` is returned.

    :param spec: The phase spec.
    :param event_df: ``Recording.event_df`` (columns ``eventtime`` / ``eventmessage``).
    :param start_frame: The recording's first frame (anchor for minute-based boundaries).
    :param end_frame: The recording's last frame (inclusive); where a phase with no resolved end
        stops.
    :param samp_rate: Sampling rate (Hz), used to convert minutes to frames.
    :returns: The resolved phases.
    :rtype: Phases
    :raises ValueError: If ``spec`` is not in the format above.
    """
    to_frame = lambda minutes: _minutes_to_frame(minutes, start_frame=start_frame, samp_rate=samp_rate)

    # Every occurrence of every phase start: (name, start_frame, end_boundary).
    occurrences: list[tuple[str, int, object]] = []
    for name, start, end in _normalize_spec(spec):
        starts = [to_frame(start)] if _is_number(start) else _event_frames(event_df, _matcher(start))
        if len(starts) == 1:
            occurrences.append((name, starts[0], end))
        else:
            occurrences.extend((f"{name}_{i}", frame, end) for i, frame in enumerate(starts, 1))

    if not occurrences:
        return Phases.full(start_frame, end_frame)

    occurrences.sort(key=lambda o: o[1])
    all_starts = [frame for _, frame, _ in occurrences]

    phases: list[Phase] = []
    for name, start, end in occurrences:
        stop = None
        if _is_number(end):
            stop = to_frame(end) - 1
        elif end is not None:
            stop = next((f for f in _event_frames(event_df, _matcher(end)) if f > start), None)
        if stop is None:
            next_start = next((f for f in all_starts if f > start), None)
            stop = end_frame if next_start is None else next_start - 1
        if stop >= start:
            phases.append(Phase(name, int(start), int(stop)))

    return Phases(tuple(phases)) if phases else Phases.full(start_frame, end_frame)
