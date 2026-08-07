from dataclasses import dataclass
from glob import glob
from pathlib import Path
import pandas as pd
from mxtreme.identity import CultureID, RecordingID, CultureSelector
from mxtreme.constants import PARENT_DIR


@dataclass(frozen=True)
class RecordingPaths:
    recording_id: RecordingID
    npz: Path          # cleaned spike/lfp data
    burst_stats: Path  # per-recording burst CSV

@dataclass(frozen=True)
class CulturePaths:
    culture_id: CultureID
    experimental_conditions: Path           # culture-level preprocessing output
    recordings: dict[int, RecordingPaths]  # keyed by DIV

    def recording(self, div: int) -> RecordingPaths:
        return self.recordings[div]

@dataclass(frozen=True)
class ExperimentPaths:
    exp_id: str
    burst_summary: Path        # experiment-level preprocessing output
    cultures: dict[CultureID, CulturePaths]

    def culture(self, culture_id: CultureID) -> CulturePaths:
        return self.cultures[culture_id]


def _recording_paths(rid: RecordingID, root: Path) -> RecordingPaths:
    base = root / "data" 
    return RecordingPaths(
        recording_id=rid,
        npz=glob(str(base / "preprocessed" / rid.exp_id / rid.chip / f"well{rid.well}" /f"DIV{rid.div}*exp_data.npz"))[0],
        burst_stats=glob(str(base / "burst_data" / rid.exp_id / rid.chip / f"well{rid.well}" /f"DIV{rid.div}*burst_data.csv"))[0],
    )

def _culture_paths(cid: CultureID, divs: list[int]|int, root: Path) -> CulturePaths:
    base = root / "data"
    if isinstance(divs, int):
        divs=[divs]
    return CulturePaths(
        culture_id=cid,
        experimental_conditions=base / "experimental_conditions" / cid.exp_id / f"{cid.chip}_well{cid.well}_exp_conditions.csv",
        recordings={
            div: _recording_paths(RecordingID(cid.exp_id, cid.chip, cid.well, div), root)
            for div in divs
        }
    )

def _experiment_paths(exp_id: str, cultures: list[CulturePaths], root: Path) -> ExperimentPaths:
    base = root / "data"
    return ExperimentPaths(
        exp_id=exp_id,
        burst_summary=base / "burst_data" / f"{exp_id}_burst_log.csv",
        cultures={c.culture_id: c for c in cultures},
    )


def resolve_paths(
    target: CultureSelector | CultureID | RecordingID,
    registry_path: Path = Path(PARENT_DIR) / "data" / "registry.csv",  # your registry/catalog that knows what DIVs exist per culture
    root: Path = Path(PARENT_DIR),
) -> RecordingPaths | CulturePaths | dict[str, ExperimentPaths]:
    """
    RecordingID      -> RecordingPaths
    CultureID        -> CulturePaths  (all available DIVs)
    CultureSelector  -> dict[exp_id, ExperimentPaths]
    """
    df = pd.read_csv(registry_path)
    df = df[df.status == "complete"]

    def get_divs(cid: CultureID, divs: list[int] | int | None = None) -> list[int]:

        mask = (
            (df.exp_id == cid.exp_id) &
            (df.chip == cid.chip) &
            (df.well == cid.well)
        )
        available = df[mask]["div"].tolist()

        if divs:
            if isinstance(divs, int):
                return divs if divs in available else []
            return [d for d in divs if d in available]
        return available

    def get_culture_ids(sel: CultureSelector) -> list[CultureID]:
        if sel.cultures:
            return sel.cultures
        if sel.exp_ids:
            rows = df[df.exp_id.isin(sel.exp_ids)][["exp_id", "chip", "well"]].drop_duplicates()
        else:
            rows = df[["exp_id", "chip", "well"]].drop_duplicates()
        return [CultureID(r.exp_id, r.chip, r.well) for r in rows.itertuples()]

    if isinstance(target, RecordingID):
        return _recording_paths(target, root)

    if isinstance(target, CultureID):
        divs = get_divs(target)
        return _culture_paths(target, divs, root)

    if isinstance(target, CultureSelector):
        culture_ids = get_culture_ids(target)
        by_exp: dict[str, list[CulturePaths]] = {}
        for cid in culture_ids:
            divs = get_divs(cid, divs=target.divs)
            cpaths = _culture_paths(cid, divs, root)
            by_exp.setdefault(cid.exp_id, []).append(cpaths)
        return {
            exp_id: _experiment_paths(exp_id, cultures, root)
            for exp_id, cultures in by_exp.items()
        }
