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
    save_dir = analysis_dir / category / cid.batch_id / cid.chip / f"well{cid.well}"
    return save_dir, save_dir / f"{cid}_{name}{ext}"


def read_cached(csv_path: Path, use_existing: bool, required=('experiment',)) -> pd.DataFrame | None:
    """The cached summary at ``csv_path``, or ``None`` when it should be (re)computed.

    ``None`` when ``use_existing`` is off, the file is missing, or it lacks any ``required`` column --
    a summary written before that column existed (e.g. before ``experiment`` told apart several
    recordings on one DIV) can't be trusted, so it is recomputed rather than reused.
    """
    if not (use_existing and csv_path.exists()):
        return None
    cached = pd.read_csv(csv_path)
    missing = [c for c in required if c not in cached.columns]
    if missing:
        print(f"Ignoring {csv_path} (no {', '.join(missing)} column); recomputing.")
        return None
    print(f"Loading existing summary from {csv_path}")
    # An unlabelled recording's "" label reads back as NaN; restore it so it groups and compares
    # the same as a freshly computed frame.
    if 'experiment' in cached.columns:
        cached['experiment'] = cached['experiment'].fillna("").astype(str)
    return cached


def recording_key(div: int, experiment: str | None) -> str:
    """Flat cache key prefix for one recording, e.g. ``"27__NS_ATP1hr"`` (``"27__"`` if unlabelled)."""
    return f"{int(div)}__{experiment or ''}"


def recording_label(div: int, experiment: str | None) -> str:
    """Display label for one recording: ``"DIV27"``, or ``"DIV27 · NS_ATP1hr"`` when labelled."""
    return f"DIV{int(div)} · {experiment}" if experiment else f"DIV{int(div)}"


def unflatten_recording_arrays(npz) -> dict[tuple[int, str], dict] | None:
    """Rebuild ``{(div, experiment): {key: array}}`` from a cache written with :func:`recording_key`.

    Returns ``None`` for a cache in the older ``"<div>__<key>"`` layout (one recording per DIV), so
    the caller recomputes it.
    """
    out: dict[tuple[int, str], dict] = {}
    for flat_key in npz.files:
        parts = flat_key.split("__", 2)
        if len(parts) != 3:
            return None
        div_str, experiment, key = parts
        out.setdefault((int(div_str), experiment), {})[key] = npz[flat_key]
    return out


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
    for batch_id in sel_paths:
        bpath = sel_paths[batch_id]
        for cid in bpath.cultures:
            csv = data_dir / batch_id / cid.chip / f"well{cid.well}" / f"{cid}_{suffix}.csv"
            frames.append(stamp_identity(pd.read_csv(csv), cid))

    if not frames:
        raise FileNotFoundError(f"No {suffix} CSVs found in {data_dir}")

    return pd.concat(frames, ignore_index=True)
