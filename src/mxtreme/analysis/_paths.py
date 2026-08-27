"""
Shared per-culture/population path helpers for the analysis/ topic modules.

The analysis output root is supplied by the caller as ``analysis_dir`` (typically
``config.analysis_dir`` from :class:`mxtreme.config.Config`) -- nothing here is coupled to a
lab-specific path.
"""


from pathlib import Path
import pandas as pd


def _summary_paths(cpath, analysis_dir: Path, category: str, name: str,
                   ext: str = ".csv") -> tuple[Path, Path]:
    """(save_dir, file_path) for a per-culture artifact, e.g. category='activity', name='burst_activity_summary'.

    ``ext`` covers the artifacts that aren't tidy tables -- the cached distribution arrays are written
    as ``.npz`` -- while keeping every per-culture output under the same predictable layout.
    """
    cid = cpath.culture_id
    save_dir = analysis_dir / category / cid.exp_id / cid.chip / f"well{cid.well}"
    return save_dir, save_dir / f"{cid}_{name}{ext}"


def stamp_identity(df: pd.DataFrame, culture_id) -> pd.DataFrame:
    """Return ``df`` with the identity of the culture it describes attached.

    The topic modules don't all write the same identity columns (``performance_summary`` writes
    chip/well, the activity summaries write culture_id), but every caller that pools summaries knows
    all three, so they are stamped on here. Downstream plots can then group and label per-culture
    series identically whichever summary they read.
    """
    return df.assign(culture_id=str(culture_id), chip=culture_id.chip, well=culture_id.well)


def load_population_summaries(sel_paths, data_dir: Path, suffix: str) -> pd.DataFrame:
    """Concatenate all per-culture CSVs in analysis_dir into one DataFrame.

    Every row is stamped with :func:`stamp_identity`, so pooled frames from any topic module carry
    ``culture_id`` / ``chip`` / ``well`` for grouping and labelling.
    """

    frames = []
    for exp_id in sel_paths:
        epath = sel_paths[exp_id]
        for cid in epath.cultures:
            csv = data_dir / exp_id / cid.chip / f"well{cid.well}" / f"{cid}_{suffix}.csv"
            frames.append(stamp_identity(pd.read_csv(csv), cid))

    if not frames:
        raise FileNotFoundError(f"No {suffix} CSVs found in {data_dir}")

    return pd.concat(frames, ignore_index=True)
