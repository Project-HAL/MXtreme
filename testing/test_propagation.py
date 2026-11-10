"""Propagation measures and per channel burst views, on small synthetic spike arrays."""

import numpy as np
import pandas as pd
import pytest

from mxtreme.analysis import channel_bursts as cb
from mxtreme.analysis import propagation as pr
from mxtreme.io import SIMULATION_BATCH, load_simulation
from mxtreme.recording import Recording

HZ = 10000.0


def _wave(xy, onsets_frames, delay_frames):
    """One spike per channel per burst, at the burst's onset plus the channel's delay."""
    frames = (np.asarray(onsets_frames)[:, None] + np.asarray(delay_frames)[None, :]).ravel()
    rows = np.tile(np.arange(len(xy)), len(onsets_frames))
    order = np.argsort(frames, kind="stable")
    return frames[order].astype(np.int64), rows[order].astype(np.int64)


def test_frame_step_and_first_spikes():
    assert pr.frame_step(np.array([0, 3, 4, 10])) == 1
    assert pr.frame_step(np.array([40, 100, 250, 250, 990])) == 10
    assert pr.frame_step(np.array([7])) == 1
    frames = np.array([5, 6, 6, 9, 20]); rows = np.array([2, 0, 2, 0, 1])
    cc, first = pr.first_spikes(frames, rows, 5, 10)
    assert cc.tolist() == [0, 2] and first.tolist() == [6, 5]


def test_burst_spans_from_a_burst_table():
    df = pd.DataFrame({
        "id": [1, 2, 3], "kind": ["network", "mini", "network"], "peak_frame": [150, 1100, 5050],
        "group_start_frame": [100, 1000, 5000], "group_stop_frame": [399, 1199, 5099],
        "onset_frame": [120, -1, 5010], "offset_frame": [380, -1, 5090], "ignore": [False, False, False],
    })
    ids, spans = pr.burst_spans(df)
    assert ids.tolist() == [1, 3] and spans.tolist() == [[120, 381], [5010, 5091]]
    ids, spans = pr.burst_spans(df, kinds=None, edges="group")
    assert ids.tolist() == [1, 2, 3] and spans[1].tolist() == [1000, 1200]
    ids, _ = pr.burst_spans(df, min_ms=20.0, samp_rate=HZ)
    assert ids.tolist() == [1]  # the third lasts 8.1 ms
    with pytest.raises(ValueError):
        pr.burst_spans(df, min_ms=20.0)


def test_lag_by_distance_finds_a_front_on_a_line():
    # 16 electrodes on a line, 100 um apart; a front at 0.5 m/s is 2 frames per 100 um at 10 kHz.
    xy = np.c_[np.arange(16) * 100.0, np.zeros(16)]
    # onsets one frame further apart each time, so the spikes do not all sit on a 2 frame grid
    frames, rows = _wave(xy, np.arange(20) * 5001 + 1000, np.arange(16) * 2)
    res = pr.lag_by_distance(frames, rows, xy, HZ, step=1)
    assert res["fine"].shape == (15, 30) and res["ms_per_bin"] == pytest.approx(0.1)
    for k in range(1, 15):  # the pair 100*k um apart sits at a lag of 2k frames, and nowhere else
        assert int(np.argmax(res["fine"][k])) == 2 * k
        assert res["fine"][k].sum() == res["fine"][k, 2 * k] == 20 * (16 - k)
    assert res["n_refs"] == frames.size and res["far_per_frame"].sum() == 0
    half = pr.lag_by_distance(frames, rows, xy, HZ, step=1, max_refs=100)
    assert half["n_refs"] <= 100
    # spikes on a 1 ms grid at a nominal 10 kHz: lags are binned on that grid, three columns to 3 ms
    coarse = pr.lag_by_distance(frames * 5, rows, xy, HZ, step=1)  # 0.5 m/s is now 1 ms per 100 um
    assert coarse["ms_per_bin"] == pytest.approx(0.5) and coarse["fine"].shape == (15, 6)
    assert int(np.argmax(coarse["fine"][1])) == 2 and int(np.argmax(coarse["fine"][2])) == 4  # steps of 0.5 ms


def test_per_burst_plane_fit_recovers_a_plane_wave():
    gx, gy = np.meshgrid(np.arange(20) * 100.0, np.arange(10) * 100.0)
    xy = np.c_[gx.ravel(), gy.ravel()]
    delay = np.round(xy[:, 0] / 500.0 * 10).astype(int)  # 0.5 m/s along x
    onsets = np.arange(6) * 10000 + 2000
    frames, rows = _wave(xy, onsets, delay)
    spans = np.c_[onsets, onsets + 600]
    res = pr.per_burst(frames, rows, xy, HZ, spans)
    assert set(res) == set(pr.PER_BURST) and all(len(v) == 6 for v in res.values())
    assert np.all(res["channels"] == 200)
    assert np.all(res["plane_r2"] > 0.99) and np.all(res["plane_r2_shuffled"] < 0.1)
    assert np.allclose(res["speed_m_s"], 0.5, rtol=0.05)
    assert np.all(res["radial_rho"] > 0.9) and np.all(res["t10_90_ms"] > res["t10_50_ms"])
    few = pr.per_burst(frames, rows, xy, HZ, spans, min_channels=500)
    assert np.all(np.isnan(few["plane_r2"])) and np.all(few["channels"] == 200) and np.all(np.isfinite(few["rise_10_90_ms"]))


def test_rise_time():
    # silence, then 10 spikes per ms from 5 ms on: 10% and 90% of the peak are crossed in one bin
    frames = np.repeat(np.arange(50, 150, 10), 10) + np.tile(np.arange(10), 10)
    assert pr.rise_time_ms(frames, HZ, 0, 150, pre_ms=0.0) == 0.0
    assert pr.rise_time_ms(frames, HZ, 50, 150, pre_ms=5.0) == 0.0
    slow = np.sort(np.concatenate([np.full(k + 1, k * 10) for k in range(20)]))  # k+1 spikes in ms k
    assert pr.rise_time_ms(slow, HZ, 0, 200, pre_ms=0.0) == pytest.approx(16.0)  # below 2 until ms 1, 18 of 20 at ms 17
    assert np.isnan(pr.rise_time_ms(np.array([5000]), HZ, 0, 100))


def test_from_leader_and_profile_recover_a_radial_front():
    gx, gy = np.meshgrid(np.arange(40) * 100.0, np.arange(20) * 100.0)
    xy = np.c_[gx.ravel(), gy.ravel()]
    delay = np.round(np.hypot(xy[:, 0], xy[:, 1]) / 500.0 * 10).astype(int)  # 0.5 m/s from the corner
    onsets = np.arange(5) * 20000 + 5000
    frames, rows = _wave(xy, onsets, delay)
    spans = np.c_[onsets, onsets + 1000]
    D, T = pr.from_leader(frames, rows, xy, HZ, spans)
    assert D.size == T.size == 5 * (800 - 5)
    Ds, Ts = pr.from_leader(frames, rows, xy, HZ, spans, shuffle=True)
    prof = pr.leader_profile(D, T, Ds, Ts)
    assert prof["speed"][25.0]["m_per_s"] == pytest.approx(0.5, rel=0.1)
    assert prof["speed"][10.0]["ms_per_mm"] == pytest.approx(2.0, rel=0.15)
    flat = prof["q25_shuffled"][np.isfinite(prof["q25_shuffled"])]
    real = prof["q"][25.0][np.isfinite(prof["q"][25.0])]
    assert np.ptp(flat) < 0.3 * np.ptp(real)  # shuffled positions: no front
    assert len(prof["bands"]) == 4 and prof["bands"][0]["fired"][-1] == 1.0
    near, far = prof["bands"][0]["fired"], prof["bands"][3]["fired"]
    assert near[20] > far[20]  # 5 ms after the leader, the near band is ahead
    assert pr.from_leader(frames, rows, xy, HZ, spans, min_channels=5000)[0].size == 0


# --- one channel inside bursts ---------------------------------------------------------------


def _trains_fixture(n_bursts=25):
    """Bursts of 200 ms every second. Row 0: a train of 4 spikes 10, 20 and 30 ms apart, starting
    10 ms after each onset. Row 1: one spike 50 ms into every other burst. Row 2: between bursts
    only, two spikes 10 ms apart in each gap."""
    onsets = (np.arange(n_bursts) * 1.0 + 0.5) * HZ
    spans = np.c_[onsets, onsets + 0.2 * HZ].astype(np.int64)
    f, r = [], []
    for k, on in enumerate(onsets):
        for ms in (10, 20, 40, 70):
            f.append(on + ms * 10); r.append(0)
        if k % 2 == 0:
            f.append(on + 500); r.append(1)
        for ms in (600, 610):
            f.append(on + ms * 10); r.append(2)
    f, r = np.asarray(f, dtype=np.int64), np.asarray(r, dtype=np.int64)
    o = np.argsort(f, kind="stable")
    return f[o], r[o], spans


def test_train_summary_by_position_rate_and_span():
    f, r, spans = _trains_fixture()
    s = cb.train_summary(f, r, HZ, spans)
    assert s["bursts"] == 25 and s["n_trains"][4] == 25 and s["n_trains"][6] == 0
    assert s["by_pos"][4][1].tolist() == [10.0, 20.0, 30.0] and s["by_pos"][6] is None
    assert s["span"][4].tolist() == [60.0, 60.0, 60.0]
    mid = (s["rate_edges_ms"][:-1] + s["rate_edges_ms"][1:]) / 2
    rate = dict(zip(mid.tolist(), s["rate"][:, 1].tolist()))
    assert rate[5.0] == pytest.approx(100.0) and rate[25.0] == pytest.approx(50.0) and rate[45.0] == pytest.approx(1000 / 30)
    assert np.isnan(rate[65.0])
    only_one = cb.train_summary(f, r, HZ, spans, channels=[1])
    assert only_one["trains"] == 13 and only_one["by_pos"][4] is None


def test_peri_burst_and_pick():
    f, r, spans = _trains_fixture()
    p = cb.peri_burst(f, r, HZ, spans, 0)
    assert p["joined"] == 1.0 and p["mean_spikes"] == 4.0 and p["dur_ms"].tolist() == [200.0] * 25
    assert p["t_ms"][:4].tolist() == [10.0, 20.0, 40.0, 70.0] and p["burst"][:5].tolist() == [0, 0, 0, 0, 1]
    q = cb.peri_burst(f, r, HZ, spans, 1)
    assert q["joined"] == pytest.approx(13 / 25) and q["mean_spikes"] == 1.0
    assert cb.peri_burst(f, r, HZ, spans, 2)["t_ms"].size == 0
    picks = cb.pick_channels(f, r, 3, spans)
    assert set(picks["channels"]) <= {0, 1} and picks["joined"].tolist() == [1.0, 13 / 25, 0.0]
    assert picks["mean_spikes"][0] == 4.0


def test_channel_stats():
    f, r, spans = _trains_fixture()
    st = cb.channel_stats(f, r, 4, HZ, spans, start=0, stop=int(25.5 * HZ))
    assert set(cb.STATS) <= set(st)
    assert st["particip"][:3].tolist() == [1.0, 13 / 25, 0.0] and np.isnan(st["particip"][3])
    assert st["count_mean"][0] == 4.0 and st["count_cv"][0] == 0.0 and st["count_max"][0] == 4.0
    assert st["burst_share"][:3].tolist() == [1.0, 1.0, 0.0]
    assert st["isi_in_p50"][0] == 20.0
    assert st["lat_frac_p50"][0] == pytest.approx(10 / 200) and st["lat_frac_sd"][0] == pytest.approx(0.0, abs=1e-12)
    assert st["span_frac_p50"][0] == pytest.approx(60 / 200) and np.isnan(st["span_frac_p50"][1])
    assert st["gap_doublet"][2] == 1.0 and st["gap_rate"][2] == pytest.approx(50 / (25.5 - 25 * 0.2))
    assert st["gap_rate"][0] == 0.0 and np.isnan(st["gap_fano"][0])
    assert st["gap_fano"][2] == pytest.approx(np.var([0] + [2] * 25) / np.mean([0] + [2] * 25))
    assert np.isfinite(st["isi_lv"][0]) and st["n_spikes"].tolist() == [100.0, 13.0, 50.0, 0.0]
    # a window that holds only some of the bursts counts only those
    part = cb.channel_stats(f, r, 4, HZ, spans, start=0, stop=int(10.4 * HZ))
    assert part["particip"][0] == 1.0 and part["count_mean"][0] == 4.0 and part["n_spikes"][0] == 40.0


# --- simulated recordings ----------------------------------------------------------------------


def test_load_simulation_fills_what_a_simulation_lacks(tmp_path):
    spikes = np.array([(40, 0, -5e-5), (50, 1, -4e-5), (1000, 0, -6e-5)], dtype=[("frameno", "<i8"), ("channel", "<i4"), ("amplitude", "<f4")])
    cm = np.array([[0, 0, 52, 910.0, 0.0], [1, 1, 72, 1260.0, 0.0]])
    path = tmp_path / "es_member_k1.npz"
    np.savez(path, spike_data=spikes, channelmap=cm, samp_rate=np.array([10000.0]), seed=7, run_ms=100.0)
    well = load_simulation(path)
    assert well["simulated"] is True and well["batch_id"] == SIMULATION_BATCH and well["chip"] == "es_member_k1"
    assert well["well"] == 0 and well["DIV"] == 0 and well["seed"] == 7 and "spike_bin" not in well
    assert float(well["rec_t_sec"]) == pytest.approx(0.1)
    rec = Recording(0, well)
    assert rec.spike_bin is None and rec.asdr.size > 0 and rec.batch_id == SIMULATION_BATCH
    frames, rows, xy = pr.spike_arrays(rec)
    assert frames.tolist() == [40, 50, 1000] and rows.tolist() == [0, 1, 0] and xy[1].tolist() == [1260.0, 0.0]
    assert pr.frame_step(frames) == 10
    np.savez(tmp_path / "not_a_sim.npz", something=np.zeros(3))
    with pytest.raises(KeyError):
        load_simulation(tmp_path / "not_a_sim.npz")
    with pytest.raises(FileNotFoundError):
        load_simulation(tmp_path / "missing.npz")
