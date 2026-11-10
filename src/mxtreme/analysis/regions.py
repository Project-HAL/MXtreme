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

Coordinates are the channel map's ``x`` / ``y`` columns, in µm, with the array's origin at its
top-left electrode as MaxWell lays it out; a region's edges are inclusive.
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
    :param points: Coordinates in µm.
    """

    id: str
    label: str
    kind: str
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.kind not in ("rect", "polygon"):
            raise ValueError(f"kind must be 'rect' or 'polygon', not {self.kind!r}")
        pts = tuple((float(x), float(y)) for x, y in self.points)
        if self.kind == "rect" and len(pts) != 2:
            raise ValueError("a rect has exactly two corner points")
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
        """Boolean mask of which ``(x, y)`` points (µm, array-likes) fall inside, edges included."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.kind == "rect":
            x0, y0, x1, y1 = self.bounds
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

    Cell ``(r, c)`` is ``g<r>_<c>`` and labelled ``r<r>c<c>``. Cells abut: a point on a shared
    edge belongs to both neighbours, which for electrodes on a 17.5 µm pitch happens only when
    a cell edge lands exactly on an electrode column or row.
    """
    if cols < 1 or rows < 1:
        raise ValueError("a grid needs at least one column and one row")
    w, h = width / cols, height / rows
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(Region(f"g{r}_{c}", f"r{r}c{c}", "rect", ((c * w, r * h), ((c + 1) * w, (r + 1) * h))))
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
