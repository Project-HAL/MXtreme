from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class CultureID:
    exp_id: str
    chip: str
    well: str

    def __str__(self):
        return f"{self.exp_id}_{self.chip}_well{self.well}"

@dataclass(frozen=True)
class RecordingID:
    exp_id: str
    chip: str
    well: str
    div: int

    @property
    def culture(self) -> CultureID:
        return CultureID(self.exp_id, self.chip, self.well)

    def __str__(self):
        return f"{self.exp_id}_{self.chip}_well{self.well}_DIV{self.div}"
    
@dataclass
class CultureSelector:
    exp_ids:  Optional[list[str]]       = None  # matches all cultures in these experiments
    cultures: Optional[list[CultureID]] = None  # explicit culture list (takes precedence)
    divs:     Optional[list[int]]       = None  # None = all DIVs