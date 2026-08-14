"""Experiment phases as an injected, experiment-agnostic value object.

A *phase* is a named interval of a recording (e.g. ``"pre"`` / ``"train"`` / ``"post"`` in a
closed-loop training experiment). The package never assumes any particular phase names or that phases
exist at all: :class:`Recording <mxtreme.recording.Recording>` takes a :class:`Phases` object as an
optional input and, when none is given, uses a single ``"full"`` phase spanning the whole recording.

Phases are anchored to **maxlab event tags** already present in a recording's event data. The user
chooses which tags mark phase boundaries and what to call each phase; :func:`phases_from_event_tags`
turns that choice into a :class:`Phases`. A burst's ``phase`` is then simply
:meth:`Phases.label_for` of its peak frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Optional, Sequence, Union


@dataclass(frozen=True)
class Phase:
    """A single named half-open interval ``[start_frame, end_frame)`` of a recording.

    :param name: Phase label (whatever the user chose; not interpreted by the package).
    :param start_frame: First frame of the phase (inclusive).
    :param end_frame: One past the last frame of the phase (exclusive).
    """

    name: str
    start_frame: int
    end_frame: int

    def contains(self, frame: float) -> bool:
        """Return whether ``frame`` falls in ``[start_frame, end_frame)``."""
        return self.start_frame <= frame < self.end_frame


@dataclass(frozen=True)
class Phases:
    """An ordered collection of :class:`Phase` intervals covering (part of) a recording.

    Construct via :meth:`full` (the default single ``"full"`` phase) or :func:`phases_from_event_tags`.
    """

    phases: tuple[Phase, ...]

    @classmethod
    def full(cls, start_frame: int, end_frame: int, name: str = "full") -> "Phases":
        """Return a :class:`Phases` with one interval spanning ``[start_frame, end_frame)``.

        This is the default when the user injects no phases: every burst then gets ``phase == name``.
        """
        return cls((Phase(name, int(start_frame), int(end_frame)),))

    def label_for(self, frame: float) -> Optional[str]:
        """Return the name of the phase containing ``frame``, or ``None`` if none does.

        ``None`` only occurs when phases do not cover ``frame`` (e.g. a burst before the first event
        tag). The default ``full`` phase spans the whole recording, so it always matches.
        """
        for phase in self.phases:
            if phase.contains(frame):
                return phase.name
        return None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.phases)

    def __iter__(self) -> Iterator[Phase]:
        return iter(self.phases)

    def __len__(self) -> int:
        return len(self.phases)


# Ordered spec accepted by phases_from_event_tags: either a name->tag mapping (order preserved) or a
# sequence of (phase_name, event_tag) pairs.
TagSpec = Union[Mapping[str, str], Sequence[tuple[str, str]]]


def _first_frame_for_tag(event_df, tag: str) -> Optional[int]:
    """Return the earliest ``eventtime`` (frame) whose ``eventmessage`` dict contains ``tag``.

    The event data stores one dict per row (e.g. ``{'pre_recording_start': '5'}``); a tag is present
    when it is a key of that dict. Returns ``None`` if the tag never appears.
    """
    mask = event_df["eventmessage"].apply(lambda d: isinstance(d, dict) and tag in d)
    hits = event_df.loc[mask, "eventtime"]
    return int(hits.iloc[0]) if len(hits) else None


def phases_from_event_tags(
    event_df,
    tags: TagSpec,
    *,
    end_frame: int,
    end_tag: Optional[str] = None,
) -> Phases:
    """Build :class:`Phases` from maxlab event tags in ``event_df``.

    Each ``(phase_name, event_tag)`` in ``tags`` starts a phase at that tag's first event frame; phases
    run contiguously in the given order, and the final phase runs to ``end_tag``'s frame if supplied
    and present, otherwise to ``end_frame``. Tags not found in the data are skipped, so the same spec
    can be applied across a batch of recordings that don't all carry every tag. If no requested tag is
    present, a single ``"full"`` phase spanning ``[0, end_frame)`` is returned.

    :param event_df: DataFrame with ``eventtime`` (frame) and ``eventmessage`` (dict) columns
        (``Recording.event_df``).
    :param tags: Ordered ``name -> tag`` mapping or sequence of ``(name, tag)`` pairs.
    :param end_frame: Frame at which the last phase ends (typically the last spike/recording frame).
    :param end_tag: Optional event tag marking the end of the last phase (e.g. ``"end_experiment"``).
    :returns: The resolved phases.
    :rtype: Phases
    """
    pairs = list(tags.items()) if isinstance(tags, Mapping) else list(tags)

    found: list[tuple[str, int]] = []
    for name, tag in pairs:
        frame = _first_frame_for_tag(event_df, tag)
        if frame is not None:
            found.append((name, frame))

    if not found:
        return Phases.full(0, end_frame)

    # Order phases by their start frame so contiguous intervals are well-formed even if the spec order
    # and the event order disagree.
    found.sort(key=lambda nf: nf[1])

    last_end = end_frame
    if end_tag is not None:
        tag_frame = _first_frame_for_tag(event_df, end_tag)
        if tag_frame is not None:
            last_end = tag_frame

    phases: list[Phase] = []
    for i, (name, start) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else last_end
        phases.append(Phase(name, int(start), int(end)))

    return Phases(tuple(phases))


def _is_number(value) -> bool:
    """Return whether ``value`` is a real number (JSON int/float) and not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _boundary_frame(boundary, event_df, *, start_frame: int, samp_rate: float) -> Optional[int]:
    """Resolve a polymorphic phase boundary to a frame.

    A **number** is minutes from the recording's first frame:
    ``start_frame + round(boundary * 60 * samp_rate)``. A **string** is a maxlab event tag, resolved to
    its first event frame (``None`` when the tag is absent from ``event_df``).
    """
    if _is_number(boundary):
        return int(start_frame + round(float(boundary) * 60.0 * samp_rate))
    return _first_frame_for_tag(event_df, str(boundary))


def phases_from_spec(
    spec: Mapping,
    event_df,
    *,
    start_frame: int,
    end_frame: int,
    samp_rate: float,
) -> Phases:
    """Build :class:`Phases` from a polymorphic phase ``spec`` (the value embedded in npz metadata).

    ``spec`` is a mapping with:

    - ``"starts"``: an ordered ``name -> boundary`` mapping. Each ``boundary`` is either a maxlab event
      **tag** (a ``str``, resolved from ``event_df``) or a number of **minutes** from the recording's
      first frame (see :func:`_boundary_frame`). Boundaries may mix within one spec.
    - ``"end"`` (optional): the terminal boundary of the last phase, resolved the same way. When
      omitted, absent (tag not found), the last phase runs to ``end_frame``.

    Boundaries that don't resolve (a tag absent from ``event_df``) are skipped, so the same spec can be
    applied across a batch of recordings that don't all carry every tag. If no start resolves, a single
    ``"full"`` phase spanning ``[start_frame, end_frame)`` is returned.

    :param spec: The phase spec (``{"starts": {...}, "end": ...}``).
    :param event_df: ``Recording.event_df`` (columns ``eventtime`` / ``eventmessage``).
    :param start_frame: The recording's first frame (anchor for minute-based boundaries).
    :param end_frame: Frame at which the last phase ends when no ``end`` boundary resolves.
    :param samp_rate: Sampling rate (Hz), used to convert minutes to frames.
    :returns: The resolved phases.
    :rtype: Phases
    """
    starts = spec.get("starts") or {}

    found: list[tuple[str, int]] = []
    for name, boundary in starts.items():
        frame = _boundary_frame(boundary, event_df, start_frame=start_frame, samp_rate=samp_rate)
        if frame is not None:
            found.append((str(name), frame))

    if not found:
        return Phases.full(start_frame, end_frame)

    # Order phases by start frame so contiguous intervals are well-formed even if the spec order and
    # the resolved frames disagree.
    found.sort(key=lambda nf: nf[1])

    last_end = end_frame
    end_boundary = spec.get("end")
    if end_boundary is not None:
        resolved_end = _boundary_frame(
            end_boundary, event_df, start_frame=start_frame, samp_rate=samp_rate
        )
        if resolved_end is not None:
            last_end = resolved_end

    phases: list[Phase] = []
    for i, (name, start) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else last_end
        phases.append(Phase(name, int(start), int(end)))

    return Phases(tuple(phases))
