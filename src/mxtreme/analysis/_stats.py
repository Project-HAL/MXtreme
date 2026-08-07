"""Shared population-level aggregation helpers for analysis/population.py."""

import pandas as pd


def aggregate_by_div_phase(df: pd.DataFrame, value_cols: list, group_cols=('div', 'phase')) -> pd.DataFrame:
    """
    For each group (default: every (div, phase) pair) compute the across-row mean and SEM
    of each column in value_cols.

    Each row of `df` is one culture's observation at that (div[, phase]), so grouping by
    group_cols and taking mean/sem gives the population-level statistic -- the unit of
    aggregation is always the culture, never the individual burst/spike/channel.

    Returns one row per group, with columns `<col>` (mean) and `sem_<col>` (SEM) for every
    col in value_cols, alongside the group columns.
    """
    group_cols = list(group_cols)

    agg = (
        df
        .dropna(subset=group_cols)
        .groupby(group_cols)[value_cols]
        .agg(['mean', 'sem'])
        .reset_index()
    )

    agg.columns = group_cols + [
        col if stat == 'mean' else f'sem_{col}'
        for col, stat in agg.columns[len(group_cols):]
    ]

    return agg.sort_values(group_cols)
