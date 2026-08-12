"""
Shared per-culture/population path helpers for the analysis/ topic modules.

The analysis output root is supplied by the caller as ``analysis_dir`` (typically
``config.analysis_dir`` from :class:`mxtreme.config.Config`) -- nothing here is coupled to a
lab-specific path.
"""


from pathlib import Path
import pandas as pd


def _summary_paths(cpath, analysis_dir: Path, category: str, name: str) -> tuple[Path, Path]:
    """(save_dir, csv_path) for a per-culture summary, e.g. category='activity', name='burst_activity_summary'."""
    cid = cpath.culture_id
    save_dir = analysis_dir / category / cid.exp_id / cid.chip / f"well{cid.well}"
    return save_dir, save_dir / f"{cid}_{name}.csv"


def load_population_summaries(sel_paths, data_dir: Path, suffix: str) -> pd.DataFrame:
    """Concatenate all per-culture CSVs in analysis_dir into one DataFrame."""

    csvs = []
    for exp_id in sel_paths:
        epath = sel_paths[exp_id]
        for cid in epath.cultures:
            csvs.append(data_dir / exp_id / cid.chip / f"well{cid.well}" / f"{cid}_{suffix}.csv")

    if not csvs:
        raise FileNotFoundError(f"No stim summary CSVs found in {data_dir}")

    return pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
