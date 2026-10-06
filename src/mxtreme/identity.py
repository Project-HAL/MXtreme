"""
``Identity`` holds a set of object used to define groupings over recordings. 
"""
from dataclasses import dataclass, field
from typing import Optional


def _normalize(obj, **casts) -> None:
    """Coerce a frozen dataclass's fields in place, e.g. ``well=str``.

    Identities are often built straight from registry rows, where pandas hands back ``np.int64`` /
    ``np.str_`` values. Normalizing here means ``CultureID("e", "c", np.int64(0))`` and
    ``CultureID("e", "c", "0")`` are the same culture -- equal, same hash, same path.
    """
    for name, cast in casts.items():
        object.__setattr__(obj, name, cast(getattr(obj, name)))


@dataclass(frozen=True)
class CultureID:
    """
    Defines a single culture which may have multiple recordings over multiple DIV. 
    """
    batch_id: str
    chip: str
    well: str

    def __post_init__(self):
        _normalize(self, batch_id=str, chip=str, well=str)

    def __str__(self):
        return f"{self.batch_id}_{self.chip}_well{self.well}"

@dataclass(frozen=True)
class RecordingID:
    """One preprocessed recording: a culture on one DIV.

    ``experiment`` is the recording's label (``""`` for an unlabelled
    recording). It tells apart two recordings of one
    culture on the same DIV. TODO: describe experiment=None.
    """

    batch_id: str
    chip: str
    well: str
    div: int
    experiment: Optional[str] = None

    def __post_init__(self):
        _normalize(self, batch_id=str, chip=str, well=str, div=int)
        if self.experiment is not None:
            _normalize(self, experiment=str)

    @property
    def culture(self) -> CultureID:
        return CultureID(self.batch_id, self.chip, self.well)

    def __str__(self):
        tail = f"_{self.experiment}" if self.experiment else ""
        return f"{self.batch_id}_{self.chip}_well{self.well}_DIV{self.div}{tail}"

@dataclass
class CultureSelector:
    """
    Allows the user to select a group of recordings from different batches, cultures, and/or DIVs. 
    Cultures can be specified directly by passing a list of CultureIDs or by selecting batches. 
    A set of DIVs or experiment labels can be specified to further define a group of recordings. 
    """
    batch_ids:  Optional[list[str]]       = None  # matches all cultures in these batches
    cultures:   Optional[list[CultureID]] = None  # explicit culture list (takes precedence)
    divs:       Optional[list[int]]       = None  # None = all DIVs
    experiment: Optional[str]             = None  # which label: None = any, "" = unlabelled only, else that label
