"""Regions of the array, for statistics that survive a change of electrode configuration.

Two recordings of one culture on different days rarely route the same electrodes, so a
per-channel comparison between them has nothing to align on. A region is a place on the
array in micrometres -- a rectangle, a polygon, or one cell of a regular grid laid over the
whole array -- and it means the same place whichever electrodes happened to be routed inside
it. Aggregating a per-channel table by region turns a 1020-channel recording and a 980-channel
recording of the same well into two tables over the same rows, which is what a comparison
across days needs.

- :func:`grid_regions` lays a ``cols x rows`` grid over the array (:data:`ARRAY_WIDTH_UM` by
  :data:`ARRAY_HEIGHT_UM`): the regridding to a coarser, spatially uniform layout.
- :class:`Region` is one place; :func:`membership` says which channels of a recording fall in
  each region (a channel can be in several overlapping custom regions; grid cells are disjoint).
- :func:`region_table` aggregates a per-channel, per-phase table (the shape
  :func:`mxtreme.analysis.activity.firing_rate` returns) to one row per region and phase.
- :func:`region_rates` bins each region's spikes over a window into a population rate, and
  :func:`region_correlation` / :func:`region_correlation_course` correlate those rates between
  regions -- over the whole window, with a lag, or in a window sliding through it.

Coordinates are the channel map's ``x`` / ``y`` columns, in µm, with the array's origin at its
top-left electrode. Drawn regions have inclusive edges. Grid cells use half-open bounds.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mxtreme.device import CHIP_HEIGHT, CHIP_WIDTH, ELEC_SIZE

#: The array's extent in µm: 220 x 120 electrodes on a 17.5 µm pitch (:mod:`mxtreme.device`).
ARRAY_WIDTH_UM = CHIP_WIDTH * ELEC_SIZE
ARRAY_HEIGHT_UM = CHIP_HEIGHT * ELEC_SIZE

#: How a region's channels are reduced to one number.
AGGREGATES = ("mean", "median", "sum", "max")


@dataclass(frozen=True)
class Region:
    """One place on the array.

    :param id: Stable identifier (``"g2_5"`` for a grid cell, anything for a drawn region).
    :param label: What to call it on a plot.
    :param kind: ``"rect"`` -- ``points`` is ``((x0, y0), (x1, y1))``, any corner order;
        ``"polygon"`` -- ``points`` are its vertices in order, closed implicitly.
        ``"grid"`` uses the same corner format as a rectangle with exclusive upper edges.
    :param points: Coordinates in µm.
    """

    id: str
    label: str
    kind: str
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.kind not in ("rect", "polygon", "grid"):
            raise ValueError(f"kind must be 'rect', 'polygon' or 'grid', not {self.kind!r}")
        pts = tuple((float(x), float(y)) for x, y in self.points)
        if self.kind in ("rect", "grid") and len(pts) != 2:
            raise ValueError("a rectangle or grid cell has exactly two corner points")
        if self.kind == "polygon" and len(pts) < 3:
            raise ValueError("a polygon has at least three vertices")
        object.__setattr__(self, "points", pts)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` of the region's bounding box."""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def contains(self, x, y) -> np.ndarray:
        """Boolean mask of which ``(x, y)`` points (µm, array-likes) fall inside the region."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.kind != "polygon":
            x0, y0, x1, y1 = self.bounds
            if self.kind == "grid":
                return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        return _in_polygon(x, y, np.asarray(self.points, dtype=float))

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "kind": self.kind, "points": [list(p) for p in self.points]}

    @classmethod
    def from_dict(cls, d: Mapping) -> Region:
        return cls(str(d["id"]), str(d.get("label") or d["id"]), str(d["kind"]), tuple(tuple(p) for p in d["points"]))


def _in_polygon(x: np.ndarray, y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd ray casting, vectorised over points; a point on an edge counts as inside."""
    inside = np.zeros(x.shape, dtype=bool)
    on_edge = np.zeros(x.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        (xa, ya), (xb, yb) = poly[i], poly[(i + 1) % n]
        # On the segment: collinear and within its bounding box.
        cross = (xb - xa) * (y - ya) - (yb - ya) * (x - xa)
        on_edge |= (np.abs(cross) < 1e-6) & (x >= min(xa, xb) - 1e-9) & (x <= max(xa, xb) + 1e-9) & (y >= min(ya, yb) - 1e-9) & (y <= max(ya, yb) + 1e-9)
        crosses = (ya > y) != (yb > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = xa + (y - ya) * (xb - xa) / (yb - ya)
        inside ^= crosses & (x < xint)
    return inside | on_edge


def grid_regions(cols: int, rows: int, *, width: float = ARRAY_WIDTH_UM, height: float = ARRAY_HEIGHT_UM) -> list[Region]:
    """A ``cols x rows`` grid of rectangles over the array, row-major from the top-left.

    Cell ``(r, c)`` is ``g<r>_<c>`` and labelled ``r<r>c<c>``. Shared edges belong to
    the cell to their right or below. The array extent ends beyond the last electrode.
    """
    if cols < 1 or rows < 1:
        raise ValueError("a grid needs at least one column and one row")
    w, h = width / cols, height / rows
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(Region(f"g{r}_{c}", f"r{r}c{c}", "grid", ((c * w, r * h), ((c + 1) * w, (r + 1) * h))))
    return out


def membership(channelmap, regions: Iterable[Region]) -> dict[str, np.ndarray]:
    """Which channels fall in each region: ``{region.id: channel ids}`` (the map's ``channel`` column)."""
    cm = np.asarray(channelmap, dtype=float)
    if cm.size == 0:
        return {r.id: np.zeros(0, dtype=np.int64) for r in regions}
    chans, x, y = cm[:, 1].astype(np.int64), cm[:, 3], cm[:, 4]
    return {r.id: chans[r.contains(x, y)] for r in regions}


def region_table(
    table: pd.DataFrame,
    channelmap,
    regions: Sequence[Region],
    *,
    value: str = "firing_rate_hz",
    how: str = "mean",
    channel_column: str = "channel",
) -> pd.DataFrame:
    """Aggregate a per-channel, per-phase table to one row per region and phase.

    :param table: ``channel, phase, <value>`` rows, as :func:`~mxtreme.analysis.activity.firing_rate`
        returns.
    :param channelmap: The recording's channel map, to place its channels.
    :param regions: The regions, in the order the result lists them.
    :param value: The column to aggregate.
    :param how: One of :data:`AGGREGATES`. ``"mean"`` of per-channel firing rates is the
        region's spikes per second per electrode.
    :returns: Columns ``region, label, phase, n_channels, <value>``; a region with no channel in
        this recording appears with ``n_channels = 0`` and ``NaN``, so two recordings aggregate to
        the same rows whatever they routed.
    """
    if how not in AGGREGATES:
        raise ValueError(f"how must be one of {AGGREGATES}, not {how!r}")
    members = membership(channelmap, regions)
    phases = list(dict.fromkeys(table["phase"]))
    rows = []
    for region in regions:
        chans = members[region.id]
        sub = table[table[channel_column].isin(chans)]
        for phase in phases:
            vals = sub.loc[sub["phase"] == phase, value].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            agg = float(getattr(np, how)(vals)) if vals.size else float("nan")
            rows.append({"region": region.id, "label": region.label, "phase": phase, "n_channels": len(chans), value: agg})
    return pd.DataFrame(rows, columns=["region", "label", "phase", "n_channels", value])


def region_rates(
    frames,
    rows,
    channelmap,
    regions: Sequence[Region],
    samp_rate: float,
    *,
    start: int,
    stop: int,
    bin_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Each region's population rate over ``[start, stop)``: its spikes per bin, per second,
    per routed electrode.

    :param frames: Spike times in frames, ascending.
    :param rows: The channel-map row of each spike (an index into ``channelmap``, not its
        ``channel`` column).
    :param channelmap: The recording's channel map, to place its rows.
    :param start: First frame of the window; ``stop`` is one past its last.
    :param bin_ms: Bin width. Correlations between population rates depend on it: narrow bins
        see synchrony within a burst, wide ones the shared slow drive.
    :returns: ``(rates, n_channels)`` -- ``rates`` is ``(len(regions), n_bins)``, a row of
        ``NaN`` for a region with no channel in this recording. A channel inside two
        overlapping regions counts in both.
    """
    frames = np.asarray(frames)
    rows = np.asarray(rows)
    cm = np.asarray(channelmap, dtype=float)
    bin_frames = float(bin_ms) * samp_rate / 1000.0
    if bin_frames <= 0:
        raise ValueError("bin_ms must be positive")
    if stop <= start:
        raise ValueError("stop must be after start")
    n_bins = max(1, int((stop - start) // bin_frames))
    i0, i1 = np.searchsorted(frames, [start, start + n_bins * bin_frames])
    col = ((frames[i0:i1] - start) / bin_frames).astype(np.int64)
    col = np.minimum(col, n_bins - 1)
    r = rows[i0:i1]
    rates = np.full((len(regions), n_bins), np.nan)
    n_channels = np.zeros(len(regions), dtype=np.int64)
    if cm.size == 0:
        return rates, n_channels
    per_sec = 1000.0 / float(bin_ms)
    for k, region in enumerate(regions):
        inside = region.contains(cm[:, 3], cm[:, 4])
        n_channels[k] = int(inside.sum())
        if not n_channels[k]:
            continue
        mask = inside[r]
        rates[k] = np.bincount(col[mask], minlength=n_bins) * (per_sec / n_channels[k])
    return rates, n_channels


def _zscore(rates: np.ndarray) -> np.ndarray:
    """Rows to zero mean and unit variance; a row that never varies (or is ``NaN``) becomes ``NaN``."""
    x = np.asarray(rates, dtype=float)
    sd = x.std(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x - x.mean(axis=1, keepdims=True)) / sd
    z[~np.isfinite(sd[:, 0]) | (sd[:, 0] == 0)] = np.nan
    return z


def region_correlation(rates, *, max_lag: int = 0) -> dict[str, np.ndarray]:
    """Pearson correlation between every pair of region rates (rows of ``rates``).

    :param rates: ``(n_regions, n_bins)``, as :func:`region_rates` returns.
    :param max_lag: In bins. Above zero, the rates are also correlated at every shift up to
        that many bins either way, which finds where the coupling peaks when one region
        follows the other.
    :returns: ``{"r": (n, n)}`` -- the zero-lag correlation, ``NaN`` for a region that never
        varies -- and with ``max_lag`` also ``"peak_r"`` and ``"peak_lag"`` (bins, ``(n, n)``):
        the largest correlation over the lags and the shift it is at. ``peak_lag[i, j] > 0``
        means region ``j`` follows region ``i`` by that many bins. Each lagged correlation uses
        the whole window's means and variances over the overlapping bins, the usual estimate
        when the lags are short against the window.
    """
    z = _zscore(rates)
    n, t = z.shape
    ok = np.isfinite(z).all(axis=1)
    zz = np.where(ok[:, None], z, 0.0)
    r = (zz @ zz.T) / max(t, 1)
    r[~ok, :] = np.nan
    r[:, ~ok] = np.nan
    out = {"r": r}
    max_lag = int(max_lag)
    if max_lag > 0:
        max_lag = min(max_lag, max(t - 2, 0))
        best = r.copy()
        lag = np.zeros((n, n), dtype=np.int64)
        for L in range(1, max_lag + 1):
            # c[i, j] at +L: region i at time s against region j at s + L (j follows i).
            c = (zz[:, :-L] @ zz[:, L:].T) / (t - L)
            for cc, sign in ((c, 1), (c.T, -1)):
                better = cc > best
                best = np.where(better, cc, best)
                lag = np.where(better, sign * L, lag)
        best[~ok, :] = np.nan
        best[:, ~ok] = np.nan
        out["peak_r"] = best
        out["peak_lag"] = lag
    return out


def region_correlation_course(rates, *, window: int, step: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Zero-lag correlation between region rates in a window sliding through them.

    :param rates: ``(n_regions, n_bins)``, as :func:`region_rates` returns.
    :param window: Window length in bins (at least 3).
    :param step: How far it moves each time, in bins; half the window by default.
    :returns: ``(starts, r)`` -- the first bin of each window, and ``(n_windows, n, n)``
        correlations, ``NaN`` where a region does not vary inside that window. Empty when the
        rates are shorter than one window.
    """
    x = np.asarray(rates, dtype=float)
    window = int(window)
    if window < 3:
        raise ValueError("a window needs at least three bins")
    step = max(1, int(step if step is not None else window // 2))
    n, t = x.shape
    starts = np.arange(0, t - window + 1, step, dtype=np.int64)
    out = np.full((len(starts), n, n), np.nan)
    for k, s in enumerate(starts):
        out[k] = region_correlation(x[:, s : s + window])["r"]
    return starts, out
