"""What one channel does inside bursts.

A channel's *train* in a burst is its spikes between the burst's onset and its offset. Whether
a culture's channels behave like themselves from burst to burst -- join the same share of them,
fire the same number of times, slow down along the train the same way -- is a property the
array-wide measures average away, and one that tells recordings apart that look alike in rate
and burst statistics.

- :func:`peri_burst` -- one channel's raster around every burst: a row per burst, time from the
  burst's onset.
- :func:`train_summary` -- pooled over channels: the interval between a channel's successive
  spikes by position in its train (trains of 4, 6 and 8 spikes), its firing rate against time
  since its own first spike in the burst, and how long a train lasts against how many spikes
  it holds.
- :func:`channel_stats` -- one value per channel of each of :data:`STATS`: how many bursts it
  joins, its count per burst and how steady that is, when in the burst it fires, its intervals,
  and its firing between bursts.
- :func:`pick_channels` -- a few channels from sparse to busy, to look at first.

Inputs are the flat spike arrays of :func:`mxtreme.analysis.propagation.spike_arrays` and frame
spans ``[start, stop)`` from :func:`~mxtreme.analysis.propagation.burst_spans`; times come back
in milliseconds.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Train lengths the spacing along the train is drawn for.
TRAINS = (4, 6, 8)
#: Ranks (of mean spikes per burst) :func:`pick_channels` picks at, sparse to busy.
RANKS = (0.15, 0.4, 0.65, 0.85, 0.97)
#: Joins, intervals or gap spikes a per channel statistic needs to be defined.
MIN_EVENTS = 5
#: Trains (or intervals) a pooled quantile needs.
MIN_POOLED = 20
#: Two spikes between bursts this close are a doublet.
DOUBLET_MS = 20.0

#: The per channel statistics of :func:`channel_stats`, with what each is.
STATS = {
    "log_rate": "log10 of the firing rate (Hz)",
    "particip": "share of bursts the channel joins",
    "count_mean": "spikes per burst it joins, mean",
    "count_cv": "spikes per burst it joins, coefficient of variation",
    "count_max": "spikes per burst it joins, the most",
    "burst_share": "share of its spikes that fall inside bursts",
    "isi_in_p50": "median interval inside bursts (ms)",
    "lat_frac_p50": "first spike latency as a fraction of the burst, median",
    "lat_frac_sd": "first spike latency as a fraction of the burst, spread (sd)",
    "span_frac_p50": "train span (first to last spike) as a fraction of the burst, median",
    "log_gap_rate": "log10 of the rate between bursts (Hz)",
    "gap_rate": "rate between bursts (Hz)",
    "gap_fano": "Fano factor of its spike counts per gap between bursts",
    "gap_doublet": "share of its spikes between bursts that sit within 20 ms of another",
    "isi_cv": "coefficient of variation of its intervals",
    "isi_lv": "local variation of its intervals",
}


def _spans(spans) -> np.ndarray:
    s = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
    return s[np.argsort(s[:, 0], kind="stable")] if len(s) else s


def _trains(frames, rows, spans) -> dict:
    """Every channel's train in every burst, as one sorted table.

    Spikes inside bursts sorted by (burst, channel, time): ``f`` their frames, ``train_of`` the
    train each belongs to; per train its ``burst`` (index into the sorted spans), ``row``,
    ``start`` (index of its first spike in ``f``) and ``n`` (spikes).
    """
    f = np.asarray(frames, dtype=np.int64)
    ch = np.asarray(rows, dtype=np.int64)
    spans = _spans(spans)
    if not len(spans) or not f.size:
        z = np.zeros(0, dtype=np.int64)
        return {"f": z, "train_of": z, "burst": z, "row": z, "start": z, "n": z, "spans": spans}
    b = np.searchsorted(spans[:, 0], f, side="right") - 1
    inside = (b >= 0) & (f < spans[np.clip(b, 0, None), 1])
    idx = np.flatnonzero(inside)
    idx = idx[np.lexsort((f[idx], ch[idx], b[idx]))]
    tb, tc, tf = b[idx], ch[idx], f[idx]
    new = np.ones(idx.size, dtype=bool)
    new[1:] = (tb[1:] != tb[:-1]) | (tc[1:] != tc[:-1])
    start = np.flatnonzero(new)
    return {"f": tf, "train_of": np.cumsum(new) - 1, "burst": tb[start], "row": tc[start], "start": start,
            "n": np.diff(np.r_[start, idx.size]), "spans": spans}


def train_summary(frames, rows, samp_rate: float, spans, *, channels: Sequence[int] | None = None,
                  lengths: Sequence[int] = TRAINS, rate_edges_ms=None, min_pooled: int = MIN_POOLED) -> dict:
    """Three summaries of the channels' trains inside bursts, pooled.

    :param channels: Channel-map rows to pool over; every channel by default.
    :returns: ``{"by_pos": {n: (3, n - 1) quartiles (25, 50, 75%) of the interval at each
        position in trains of exactly ``n`` spikes, or ``None`` with fewer than ``min_pooled``
        such trains}; "n_trains": {n: how many}; "rate_edges_ms": bins of time since the
        channel's own first spike; "rate": (bins, 3) quartiles of ``1000 / interval`` (Hz)
        placed at the interval's midpoint, trains of 3 or more spikes, ``nan`` for a bin with
        fewer than ``min_pooled`` intervals; "span": {n: quartiles of first-to-last spike (ms)
        for trains of ``n`` spikes, 12 standing for 12 or more}; "trains": how many trains in
        all; "bursts": how many bursts}``.
    """
    hz = float(samp_rate)
    tr = _trains(frames, rows, spans)
    edges = np.arange(0.0, 205.0, 10.0) if rate_edges_ms is None else np.asarray(rate_edges_ms, dtype=float)
    t = tr["f"] * 1000.0 / hz
    n, start, train_of = tr["n"], tr["start"], tr["train_of"]
    use = np.ones(n.size, dtype=bool) if channels is None else np.isin(tr["row"], np.asarray(list(channels), dtype=np.int64))
    out = {"by_pos": {}, "n_trains": {}, "rate_edges_ms": edges, "rate": np.full((edges.size - 1, 3), np.nan),
           "span": {}, "trains": int(use.sum()), "bursts": len(tr["spans"])}
    q = lambda a: np.percentile(a, [25, 50, 75]) if len(a) >= min_pooled else np.full(3, np.nan)
    for length in lengths:
        sel = np.flatnonzero(use & (n == length))
        out["n_trains"][int(length)] = int(sel.size)
        if sel.size >= min_pooled:
            d = np.diff(t[start[sel][:, None] + np.arange(length)], axis=1)
            out["by_pos"][int(length)] = np.percentile(d, [25, 50, 75], axis=0)
        else:
            out["by_pos"][int(length)] = None
    if t.size > 1:
        d = np.diff(t)
        left = train_of[:-1]
        ok = (train_of[1:] == left) & use[left] & (n[left] >= 3)
        since = (t[:-1] + t[1:]) / 2 - t[start[left]]
        rate = 1000.0 / np.maximum(d, 1.0)
        which = np.digitize(since[ok], edges) - 1
        v = rate[ok]
        for k in range(edges.size - 1):
            out["rate"][k] = q(v[which == k])
    multi = use & (n >= 2)
    span = t[start[multi] + n[multi] - 1] - t[start[multi]] if multi.any() else np.zeros(0)
    key = np.minimum(n[multi], 12)
    for length in sorted(set(key.tolist())):
        out["span"][int(length)] = q(span[key == length])
    return out


def peri_burst(frames, rows, samp_rate: float, spans, channel: int) -> dict:
    """One channel around every burst: its spikes' times from each burst's onset.

    :returns: ``{"t_ms": time of each of its in-burst spikes from its burst's onset; "burst":
        the burst (index into the spans, in time order) each belongs to; "dur_ms": every
        burst's duration; "onset_frame": every burst's onset; "joined": the share of bursts it
        fires in; "mean_spikes": its mean spikes per burst it joins}``.
    """
    hz = float(samp_rate)
    tr = _trains(frames, rows, spans)
    spans = tr["spans"]
    mine = np.flatnonzero(tr["row"] == int(channel))
    t, burst = [], []
    for k in mine:
        s, n, b = tr["start"][k], tr["n"][k], tr["burst"][k]
        t.append((tr["f"][s:s + n] - spans[b, 0]) * 1000.0 / hz)
        burst.append(np.full(n, b, dtype=np.int64))
    n_spikes = int(tr["n"][mine].sum())
    return {
        "t_ms": np.concatenate(t) if t else np.zeros(0), "burst": np.concatenate(burst) if burst else np.zeros(0, dtype=np.int64),
        "dur_ms": (spans[:, 1] - spans[:, 0]) * 1000.0 / hz, "onset_frame": spans[:, 0].copy(),
        "joined": mine.size / max(len(spans), 1), "mean_spikes": n_spikes / mine.size if mine.size else 0.0,
    }


def pick_channels(frames, rows, n_channels: int, spans, *, ranks: Sequence[float] = RANKS, min_joined: float = 0.3) -> dict:
    """Channels that join at least ``min_joined`` of the bursts, at ``ranks`` of mean spikes per
    burst (sparse to busy), with every channel's share of bursts joined and mean count.

    :returns: ``{"channels": the picks (rows), "joined": per channel, "mean_spikes": per channel}``.
    """
    tr = _trains(frames, rows, spans)
    joins = np.bincount(tr["row"], minlength=n_channels).astype(float)
    spikes = np.bincount(tr["row"], weights=tr["n"], minlength=n_channels).astype(float)
    ok = np.flatnonzero(joins >= min_joined * len(tr["spans"])) if len(tr["spans"]) else np.zeros(0, dtype=np.int64)
    order = ok[np.argsort(spikes[ok] / np.maximum(joins[ok], 1), kind="stable")]
    picks = [int(order[int(r * (order.size - 1))]) for r in ranks] if order.size else []
    return {"channels": picks, "joined": joins / max(len(tr["spans"]), 1), "mean_spikes": spikes / np.maximum(joins, 1)}


def channel_stats(frames, rows, n_channels: int, samp_rate: float, spans, *, start: int | None = None,
                  stop: int | None = None) -> dict[str, np.ndarray]:
    """One value per channel of each statistic in :data:`STATS`, over ``[start, stop)``.

    ``nan`` where a channel has too few events for a statistic (:data:`MIN_EVENTS` joined
    bursts, intervals or spikes between bursts; a channel with no spike at all has ``nan``
    everywhere but the two rates). Bursts are the spans that start inside the window; a gap is
    the stretch before a burst, the last one the stretch after the last burst. ``n_spikes`` (the
    channel's spikes in the window) comes back too.
    """
    f = np.asarray(frames, dtype=np.int64)
    ch = np.asarray(rows, dtype=np.int64)
    hz = float(samp_rate)
    lo = int(f[0]) if start is None and f.size else int(start or 0)
    hi = int(f[-1]) + 1 if stop is None and f.size else int(stop or 0)
    a, b = np.searchsorted(f, [lo, hi])
    f, ch = f[a:b], ch[a:b]
    spans = _spans(spans)
    spans = spans[(spans[:, 0] >= lo) & (spans[:, 0] < hi)] if len(spans) else spans
    if len(spans):
        spans = np.column_stack([spans[:, 0], np.minimum(spans[:, 1], hi)])
    n = int(n_channels)
    t = f * 1000.0 / hz
    on_ms, off_ms = spans[:, 0] * 1000.0 / hz, spans[:, 1] * 1000.0 / hz
    duration_ms = max((hi - lo) * 1000.0 / hz, 1e-9)
    out = {k: np.full(n, np.nan) for k in STATS}
    total = np.bincount(ch, minlength=n).astype(float)
    out["log_rate"] = np.log10(np.maximum(total, 0.5) / (duration_ms / 1000.0))
    if len(spans):
        bi = np.searchsorted(spans[:, 0], f, side="right") - 1
        in_burst = (bi >= 0) & (f < spans[np.clip(bi, 0, None), 1])
    else:
        in_burst = np.zeros(f.size, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["burst_share"] = np.where(total > 0, np.bincount(ch[in_burst], minlength=n) / total, np.nan)
    gap_s = max(duration_ms - float(np.sum(off_ms - on_ms)), 1.0) / 1000.0
    gap_n = np.bincount(ch[~in_burst], minlength=n).astype(float)
    out["log_gap_rate"] = np.log10(np.maximum(gap_n, 0.5) / gap_s)
    out["gap_rate"] = gap_n / gap_s

    # per channel per burst: count, latency and span as fractions of the burst
    tr = _trains(f, ch, spans)
    tt = tr["f"] * 1000.0 / hz
    first = tt[tr["start"]] if tr["start"].size else np.zeros(0)
    last = tt[tr["start"] + tr["n"] - 1] if tr["start"].size else np.zeros(0)
    dur = (off_ms - on_ms)[tr["burst"]] if tr["burst"].size else np.zeros(0)
    lat = (first - on_ms[tr["burst"]]) / dur if dur.size else np.zeros(0)
    span = (last - first) / dur if dur.size else np.zeros(0)
    by_row = np.argsort(tr["row"], kind="stable")
    row_sorted = tr["row"][by_row]

    # intervals, per channel: all of them, and the ones inside one burst
    o = np.lexsort((t, ch))
    t_s, c_s, b_s = t[o], ch[o], in_burst[o]
    burst_id = np.searchsorted(spans[:, 0], f[o], side="right") if len(spans) else np.zeros(t_s.size, dtype=np.int64)
    d_all = np.diff(t_s)
    same = np.diff(c_s) == 0
    inside = same & b_s[1:] & b_s[:-1] & (np.diff(burst_id) == 0)
    n_gaps = len(spans) + 1
    for c in np.flatnonzero(total > 0):
        j0, j1 = np.searchsorted(row_sorted, c), np.searchsorted(row_sorted, c, side="right")
        mine = by_row[j0:j1]
        counts = tr["n"][mine].astype(float)
        out["particip"][c] = mine.size / max(len(spans), 1)
        if mine.size >= MIN_EVENTS:
            out["count_mean"][c] = counts.mean()
            out["count_max"][c] = counts.max()
            out["count_cv"][c] = counts.std() / counts.mean()
            out["lat_frac_p50"][c] = np.median(lat[mine])
            out["lat_frac_sd"][c] = lat[mine].std()
            multi = span[mine][counts >= 2]
            if multi.size >= MIN_EVENTS:
                out["span_frac_p50"][c] = np.median(multi)
        s0, s1 = np.searchsorted(c_s, c), np.searchsorted(c_s, c, side="right")
        d = d_all[s0:s1 - 1]
        if d.size >= 2 * MIN_EVENTS:
            out["isi_cv"][c] = d.std() / d.mean()
            out["isi_lv"][c] = 3.0 * np.mean(((d[1:] - d[:-1]) / np.maximum(d[1:] + d[:-1], 1e-9)) ** 2)
        di = d[inside[s0:s1 - 1]]
        if di.size >= MIN_EVENTS:
            out["isi_in_p50"][c] = np.median(di)
        g = ~b_s[s0:s1]
        if g.sum() >= MIN_EVENTS:
            per_gap = np.bincount(burst_id[s0:s1][g], minlength=n_gaps).astype(float)
            out["gap_fano"][c] = per_gap.var() / per_gap.mean()
            tg = t_s[s0:s1][g]
            dg = np.diff(tg)
            near = np.zeros(tg.size, dtype=bool)
            near[:-1] |= dg <= DOUBLET_MS
            near[1:] |= dg <= DOUBLET_MS
            out["gap_doublet"][c] = near.mean()
    out["n_spikes"] = total
    return out
