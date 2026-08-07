"""
Shared per-culture/population path helpers for the analysis/ topic modules.

ANALYSIS_DIR lives here for now (matches where it's always defaulted from); it should eventually
move to a user-editable config rather than being sourced from core.constants.PARENT_DIR, once the
package-level config work in claude_configs/PACKAGE_PLAN.md happens.
"""


from pathlib import Path
import pandas as pd
from mxtreme import constants

ANALYSIS_DIR = Path(constants.PARENT_DIR) / "analysis"


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
