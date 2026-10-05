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

    ``experiment`` names an ingested exogenous recording (``""`` for scans). It is not part of the
    culture's identity -- it only tells apart two recordings of one culture on the same DIV.
    """

    batch_id: str
    chip: str
    well: str
    div: int
    experiment: str = ""

    def __post_init__(self):
        _normalize(self, batch_id=str, chip=str, well=str, div=int, experiment=str)

    @property
    def culture(self) -> CultureID:
        return CultureID(self.batch_id, self.chip, self.well)

    def __str__(self):
        tail = f"_{self.experiment}" if self.experiment else ""
        return f"{self.batch_id}_{self.chip}_well{self.well}_DIV{self.div}{tail}"

@dataclass
class CultureSelector:
    batch_ids:  Optional[list[str]]       = None  # matches all cultures in these batches
    cultures:   Optional[list[CultureID]] = None  # explicit culture list (takes precedence)
    divs:       Optional[list[int]]       = None  # None = all DIVs
    experiment: str                       = ""    # which recordings: "" = scans, else an ingested experiment's name
