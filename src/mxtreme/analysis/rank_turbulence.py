"""Rank-turbulence divergence between two rankings of the same elements.

Dodds et al. 2023, *Allotaxonometry and rank-turbulence divergence: a universal instrument for
comparing complex systems* (doi:10.1140/epjds/s13688-023-00400-x). Two systems rank the same types; the divergence
sums, over types, how far each one's rank moved::

    D_alpha(A || B)  ~  sum_i | 1/r_i,A^alpha  -  1/r_i,B^alpha |^(1/(alpha+1))

normalised to ``[0, 1]`` by the divergence of two systems that share no types. ``alpha`` tunes how
much the top of the rankings dominates: ``0`` is the limit that compares log ranks evenly, ``1/3``
(the paper's working value) still lets a type that falls from rank 2 to rank 200 outweigh one that
moves from 2000 to 20000, and large values look only at the very top.

Here the "types" are a recording's channels and the "systems" two intervals of it (or of two
recordings): the ranking is by a per-channel measure such as the firing rate, so the divergence
says how much the channels' relative prominence was reshuffled between the two, and the table
says by which channels.

Ties share the mean of the ranks they span (``scipy.stats.rankdata(method="average")``), which is
what keeps a hundred silent channels from being ordered by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata


@dataclass(frozen=True)
class RankTurbulence:
    """The divergence and its per-element breakdown.

    :param divergence: ``D_alpha`` in ``[0, 1]``.
    :param alpha: The ``alpha`` it was computed with.
    :param n: How many elements were ranked (those with a finite value in both systems).
    :param table: One row per ranked element: ``id, value_a, value_b, rank_a, rank_b,
        contribution`` (its unnormalised term) and ``share`` (its fraction of the total), sorted
        by ``share`` descending.
    """

    divergence: float
    alpha: float
    n: int
    table: pd.DataFrame


def ranks(values) -> np.ndarray:
    """Descending ranks (1 = largest) with ties averaged; ``NaN`` for a non-finite value."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.any():
        out[ok] = rankdata(-v[ok], method="average")
    return out


def _rank_distance(a, b, alpha: float) -> np.ndarray:
    """Dodds et al. equation 7, omitting the factor that cancels in equation 8."""
    if alpha == 0:
        return np.abs(np.log(a / b))
    return np.abs(np.power(a, -alpha) - np.power(b, -alpha)) ** (1 / (1 + alpha))


def rank_turbulence(values_a, values_b, *, alpha: float = 1 / 3, ids=None) -> RankTurbulence:
    """Rank-turbulence divergence between two measures of the same elements.

    :param values_a: The measure in system A, one per element (a larger value ranks higher).
    :param values_b: The same elements' measure in system B, aligned with ``values_a``.
    :param alpha: See the module docstring; ``0`` for the log-rank limit.
    :param ids: Labels for the rows of the table (channel ids, say); indices when ``None``.
    :returns: :class:`RankTurbulence`.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"values_a and values_b must align, got {a.shape} and {b.shape}")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be finite and >= 0, got {alpha}")
    ok = np.isfinite(a) & np.isfinite(b)
    ids = np.arange(len(a)) if ids is None else np.asarray(ids)
    ra, rb = ranks(a[ok]), ranks(b[ok])
    n = len(ra)
    contrib = _rank_distance(ra, rb, alpha)
    total = float(contrib.sum())
    # In the disjoint reference each list retains its ranks (including ties).
    absent_rank = 1.5 * n
    norm = (_rank_distance(ra, absent_rank, alpha).sum()
            + _rank_distance(rb, absent_rank, alpha).sum()) if n else 0.0
    table = pd.DataFrame({
        "id": ids[ok], "value_a": a[ok], "value_b": b[ok], "rank_a": ra, "rank_b": rb,
        "contribution": contrib, "share": contrib / total if total > 0 else 0.0,
    }).sort_values("share", ascending=False, kind="stable").reset_index(drop=True)
    return RankTurbulence(divergence=total / norm if norm > 0 else 0.0, alpha=alpha, n=n, table=table)


def from_table(table: pd.DataFrame, phase_a: str, phase_b: str, *, value: str = "firing_rate_hz",
               alpha: float = 1 / 3, id_column: str = "channel") -> RankTurbulence:
    """Rank-turbulence divergence between two phases of a per-channel, per-phase table.

    Takes the shape :func:`mxtreme.analysis.activity.firing_rate` returns (``channel, phase, ...``):
    the two phases' rows are aligned on ``id_column`` and compared on ``value``.
    """
    a = table[table["phase"] == phase_a].set_index(id_column)[value]
    b = table[table["phase"] == phase_b].set_index(id_column)[value]
    if a.empty or b.empty:
        missing = [p for p, s in ((phase_a, a), (phase_b, b)) if s.empty]
        raise KeyError(f"no rows for phase(s) {missing}; available: {sorted(table['phase'].unique())}")
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    return rank_turbulence(joined["a"].to_numpy(), joined["b"].to_numpy(), alpha=alpha, ids=joined.index.to_numpy())
