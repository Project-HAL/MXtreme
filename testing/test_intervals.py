"""Intervals cut at bursts, region membership, and rank-turbulence divergence."""

import numpy as np
import pandas as pd
import pytest

from mxtreme import transactions as tx
from mxtreme.analysis import activity
from mxtreme.analysis.intervals import (
    burst_phases,
    burst_windows,
    window_phases,
)
from mxtreme.analysis.rank_turbulence import from_table, rank_turbulence, ranks
from mxtreme.bursting import BurstDetector
from mxtreme.config import Config
from mxtreme.params import BurstDetectParams
from mxtreme.recording import Recording


def _bursts(rows):
    """A minimal burst table: ``(id, kind, start, stop[, onset, offset])`` rows."""
    out = []
    for r in rows:
        bid, kind, start, stop = r[:4]
        onset, offset = (r[4], r[5]) if len(r) > 4 else (-1, -1)
        out.append({"id": bid, "kind": kind, "peak_frame": (start + stop) // 2, "peak_amp": 0.3,
                    "group_start_frame": start, "group_stop_frame": stop, "n_spikes": 10,
                    "onset_frame": onset, "offset_frame": offset, "ignore": False})
    return pd.DataFrame(out)


def test_burst_windows_prefer_onset_edges_and_drop_minis_and_ignored():
    df = _bursts([(1, "network", 100, 200, 120, 180), (2, "mini", 300, 400), (3, "network", 500, 600)])
    df.loc[df.id == 3, "ignore"] = True
    w = burst_windows(df)
    assert w["id"].tolist() == [1]
    assert (w["start_frame"][0], w["end_frame"][0]) == (120, 180)
    assert burst_windows(df, edges="group")["start_frame"].tolist() == [100]
    assert burst_windows(df, kinds=None)["id"].tolist() == [1, 2]


def test_burst_phases_inter_burst_and_cycle_naming():
    df = _bursts([(1, "network", 100, 200), (2, "network", 500, 600), (3, "network", 900, 1000)])
    inter = burst_phases(df, kind="inter", start_frame=0, end_frame=1500)
    assert inter.names == ("pre_B1", "post_B1", "post_B2", "post_B3")
    assert (inter.get_phase("post_B1").start_frame, inter.get_phase("post_B1").end_frame) == (201, 499)
    assert inter.get_phase("post_B3").end_frame == 1500
    assert burst_phases(df, kind="inter").names == ("post_B1", "post_B2")  # no recording bounds: gaps only
    bursts = burst_phases(df, kind="burst")
    assert bursts.names == ("B1", "B2", "B3") and bursts.get_phase("B2").n_frames == 101
    cycle = burst_phases(df, kind="cycle", end_frame=1500)
    assert (cycle.get_phase("cycle_B1").start_frame, cycle.get_phase("cycle_B1").end_frame) == (100, 499)
    assert cycle.get_phase("cycle_B3").end_frame == 1500
    assert len(burst_phases(df.iloc[:0], kind="inter", start_frame=0, end_frame=10)) == 0


def test_window_phases_requires_distinct_names():
    ph = window_phases([("a", 0, 10), ("b", 20, 30)])
    assert ph.names == ("a", "b")
    with pytest.raises(ValueError):
        window_phases([("a", 0, 10), ("a", 20, 30)])


def test_historical_burst_annotations_are_read_only(tmp_path):
    config = Config(data_root=tmp_path)
    config.transactions_path.write_text(
        '{"time":"2026-11-17T00:00:00","op":"burst.relabelled","batch_id":"test","well":0,"div":1}\n'
    )
    assert [entry.op for entry in tx.read(config)] == ["burst.relabelled"]
    with pytest.raises(ValueError, match="read only"):
        tx.record(config, "burst.added")


def test_firing_rate_per_inter_burst_phase_from_a_lazy_recording(make_recording_data):
    data = make_recording_data()
    full = Recording(0, data)
    lazy_data = {k: v for k, v in data.items() if k != "spike_bin"}
    lazy = Recording(0, lazy_data)
    assert lazy.spike_bin is None
    assert "asdr" not in vars(full) and "asdr" not in vars(lazy)
    n = min(len(full.asdr), len(lazy.asdr))
    assert np.allclose(full.asdr[:n], lazy.asdr[:n])
    assert lazy.asdr is vars(lazy)["asdr"]
    bursts = BurstDetector(BurstDetectParams()).detect(lazy).to_dataframe()
    assert len(bursts) >= 2
    lazy.phases = burst_phases(bursts, kind="inter", start_frame=0, end_frame=int(lazy.spike_data["frameno"].max()))
    table = activity.firing_rate(lazy, save=False)
    assert set(table["phase"]) == set(lazy.phases.names)
    assert len(table) == len(lazy.phases) * lazy.channelmap.shape[0]
    assert (table["firing_rate_hz"] >= 0).all()


def test_ranks_average_ties_and_skip_nan():
    r = ranks([5.0, 1.0, 5.0, np.nan, 0.0])
    assert r[0] == r[2] == 1.5 and r[1] == 3.0 and r[4] == 4.0 and np.isnan(r[3])


def test_rank_turbulence_identity_is_zero_and_reversal_is_large():
    v = np.arange(1, 101, dtype=float)
    same = rank_turbulence(v, v, alpha=1 / 3)
    assert same.divergence == 0.0 and same.n == 100 and (same.table["share"] == 0).all()
    rev = rank_turbulence(v, v[::-1], alpha=1 / 3)
    assert 0 < rev.divergence <= 1
    assert rev.table.iloc[0]["id"] in (0, 99)  # the ends move the furthest
    assert abs(rev.table["share"].sum() - 1) < 1e-9
    for alpha in (0, 1, 3):
        d = rank_turbulence(v, v[::-1], alpha=alpha).divergence
        assert 0 < d <= 1
    with pytest.raises(ValueError):
        rank_turbulence(v, v[:10])


def test_rank_turbulence_matches_published_distance_for_two_reversed_ranks():
    result = rank_turbulence([2, 1], [1, 2], alpha=1)
    contribution = np.sqrt(1 - 1 / 2)
    normalizer = 2 * (np.sqrt(1 - 1 / 3) + np.sqrt(1 / 2 - 1 / 3))
    assert result.divergence == pytest.approx(2 * contribution / normalizer)


def test_from_table_aligns_phases_on_channel():
    table = pd.DataFrame({
        "channel": [1, 2, 3, 1, 2, 3], "phase": ["a"] * 3 + ["b"] * 3,
        "firing_rate_hz": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
    })
    res = from_table(table, "a", "b")
    assert res.n == 3 and res.table.set_index("id").loc[2, "share"] == 0.0
    with pytest.raises(KeyError):
        from_table(table, "a", "nope")


# --- stimulation episodes and regions -------------------------------------------------------------

from mxtreme.analysis.intervals import stim_windows
from mxtreme.analysis.regions import (
    ARRAY_HEIGHT_UM,
    ARRAY_WIDTH_UM,
    Region,
    grid_regions,
    membership,
    region_table,
)


def test_stim_windows_merges_trains_and_separates_them():
    sr = 20000.0
    train = lambda start: np.arange(start, start + 1000)  # 50 ms of stimulated frames
    frames = np.concatenate([train(0), train(4000), train(100000)])  # two trains 150 ms apart, one far off
    w = stim_windows(frames, sr)
    assert w.tolist() == [[0, 999], [4000, 4999], [100000, 100999]]
    assert stim_windows(frames, sr, gap_sec=0.2).tolist() == [[0, 4999], [100000, 100999]]
    assert stim_windows(np.array([]), sr).shape == (0, 2)
    assert stim_windows(None, sr).shape == (0, 2)


def test_region_membership_rect_polygon_and_grid():
    # channelmap columns: index, channel, electrode, x, y
    cm = np.array([[0, 10, 100, 0.0, 0.0], [1, 11, 101, 100.0, 100.0], [2, 12, 102, 3832.5, 2082.5], [3, 13, 103, 50.0, 10.0]])
    rect = Region("a", "A", "rect", ((120.0, 120.0), (0.0, 0.0)))  # corners in any order
    tri = Region("t", "T", "polygon", ((0.0, 0.0), (200.0, 0.0), (0.0, 200.0)))
    m = membership(cm, [rect, tri])
    assert sorted(m["a"].tolist()) == [10, 11, 13]
    assert sorted(m["t"].tolist()) == [10, 11, 13]  # (100, 100) lies on the hypotenuse: inside
    grid = grid_regions(2, 2)
    assert [g.id for g in grid] == ["g0_0", "g0_1", "g1_0", "g1_1"]
    assert grid[3].bounds == (ARRAY_WIDTH_UM / 2, ARRAY_HEIGHT_UM / 2, ARRAY_WIDTH_UM, ARRAY_HEIGHT_UM)
    boundary = [[4, 14, 104, ARRAY_WIDTH_UM / 2, ARRAY_HEIGHT_UM / 2]]
    gm = membership(np.vstack([cm, boundary]), grid)
    assert sorted(gm["g0_0"].tolist()) == [10, 11, 13]
    assert gm["g1_1"].tolist() == [12, 14]
    assert sorted(np.concatenate(list(gm.values())).tolist()) == [10, 11, 12, 13, 14]
    with pytest.raises(ValueError):
        Region("x", "x", "rect", ((0, 0),))
    assert Region.from_dict(rect.to_dict()) == rect


def test_region_table_aggregates_per_phase_and_keeps_empty_regions():
    cm = np.array([[0, 1, 100, 10.0, 10.0], [1, 2, 101, 20.0, 10.0], [2, 3, 102, 3000.0, 2000.0]])
    table = pd.DataFrame({
        "channel": [1, 2, 3, 1, 2, 3], "phase": ["p", "p", "p", "q", "q", "q"],
        "firing_rate_hz": [1.0, 3.0, 10.0, 0.0, np.nan, 5.0],
    })
    left = Region("L", "left", "rect", ((0, 0), (100, 100)))
    empty = Region("E", "empty", "rect", ((1000, 1000), (1100, 1100)))
    out = region_table(table, cm, [left, empty], how="mean")
    assert out["region"].tolist() == ["L", "L", "E", "E"]
    assert out.loc[0, "firing_rate_hz"] == 2.0 and out.loc[1, "firing_rate_hz"] == 0.0  # NaN channel skipped
    assert out.loc[2, "n_channels"] == 0 and np.isnan(out.loc[2, "firing_rate_hz"])
    assert region_table(table, cm, [left], how="sum").loc[0, "firing_rate_hz"] == 4.0
    with pytest.raises(ValueError):
        region_table(table, cm, [left], how="mode")
