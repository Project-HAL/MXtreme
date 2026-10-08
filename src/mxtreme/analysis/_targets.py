"""Normalise what a user asks to analyse into cultures, for the group-level analysis functions.

The group-level functions (e.g. :func:`mxtreme.analysis.activity.plot_spike_activity_summary`) accept
any of four targets, and what they draw depends on which was passed:

=================================== ============ ==============================================
target                              level        meaning
=================================== ============ ==============================================
``RecordingID``                     recording    one recording
``CultureID``                       culture      one culture over its DIVs
``CultureSelector``                 selection    a group of cultures, individually and on average
``dict[str, CultureSelector]`` /    groups       several groups compared against one another
``list[CultureSelector]``
=================================== ============ ==============================================
"""

from __future__ import annotations

from dataclasses import dataclass

from mxtreme.config import Config
from mxtreme.identity import CultureID, CultureSelector, RecordingID
from mxtreme.paths import CulturePaths, RecordingSet, resolve_paths

RECORDING, CULTURE, SELECTION, GROUPS = "recording", "culture", "selection", "groups"


@dataclass(frozen=True)
class Target:
    """A resolved analysis target.

    :param level: One of ``"recording"``, ``"culture"``, ``"selection"``, ``"groups"``.
    :param groups: ``{group label: [CulturePaths, ...]}``. Every level but ``"groups"`` has a single
        group labelled ``""``. A culture's ``recordings`` hold only the recordings selected (one for
        a ``RecordingID``; those passing a selector's ``divs`` / ``experiment`` filters).
    :param name: Short display name for figure titles.
    """

    level: str
    groups: dict[str, list[CulturePaths]]
    name: str

    @property
    def cultures(self) -> list[CulturePaths]:
        """Every culture across all groups, in order."""
        return [c for cultures in self.groups.values() for c in cultures]


def _selector_cultures(selector, config: Config) -> list[CulturePaths]:
    if isinstance(selector, CultureID):
        return [resolve_paths(selector, config)]
    if not isinstance(selector, CultureSelector):
        raise TypeError(f"Expected a CultureSelector (or CultureID), got {type(selector).__name__}.")
    return [c for bpath in resolve_paths(selector, config).values() for c in bpath.cultures.values()]


def _selection_name(cultures: list[CulturePaths]) -> str:
    batches = sorted({c.culture_id.batch_id for c in cultures})
    return f"{len(cultures)} cultures ({', '.join(batches)})"


def resolve_target(target, config: Config) -> Target:
    """Resolve ``target`` (see the module docstring) into a :class:`Target`.

    :raises TypeError: For anything that is not one of the four accepted shapes.
    :raises ValueError: For an empty group of selections.
    """
    if isinstance(target, RecordingID):
        rp = resolve_paths(target, config)
        rid = rp.recording_id
        cpath = CulturePaths(culture_id=rid.culture, recordings=RecordingSet((rp,)))
        return Target(RECORDING, {"": [cpath]}, str(rid))

    if isinstance(target, CultureID):
        return Target(CULTURE, {"": [resolve_paths(target, config)]}, str(target))

    if isinstance(target, CultureSelector):
        cultures = _selector_cultures(target, config)
        return Target(SELECTION, {"": cultures}, _selection_name(cultures))

    if isinstance(target, dict):
        named = {str(k): v for k, v in target.items()}
    elif isinstance(target, (list, tuple)):
        named = {f"Group {i}": v for i, v in enumerate(target, start=1)}
    else:
        raise TypeError(
            "target must be a RecordingID, CultureID, CultureSelector, or a dict/list of "
            f"CultureSelectors -- got {type(target).__name__}."
        )
    if not named:
        raise ValueError("No selections given to compare.")

    groups = {label: _selector_cultures(sel, config) for label, sel in named.items()}
    return Target(GROUPS, groups, " vs ".join(groups))
