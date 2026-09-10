"""Region selection: patch choice, routing budget, baseline coupling, and role assignment."""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from mxtreme.scans import region_selection as rs


def _well(active, pitch=17.5):
    """A well dict the way get_active_electrodes leaves it, from a list of active electrodes."""
    electrodes = np.array(sorted(active))
    mapping = pd.DataFrame(
        {
            "channel": np.arange(len(electrodes)),
            "electrode": electrodes,
            "x": (electrodes % rs.COLS) * pitch,
            "y": (electrodes // rs.COLS) * pitch,
        }
    )
    return {"mapping": mapping, "active_electrodes": pd.DataFrame({"electrode": electrodes})}


def _blob(cx_um, cy_um, radius_um):
    """Every electrode inside a circle."""
    out = []
    for e in range(rs.COLS * rs.ROWS):
        x, y = (e % rs.COLS) * rs.PITCH_UM, (e // rs.COLS) * rs.PITCH_UM
        if (x - cx_um) ** 2 + (y - cy_um) ** 2 <= radius_um**2:
            out.append(e)
    return out


def test_choose_centers_keeps_them_apart():
    well = _well(
        _blob(600, 1000, 120) + _blob(700, 1000, 120) + _blob(2400, 600, 120) + _blob(3000, 1600, 120)
    )
    centres = rs.choose_centers(well, n=3, min_separation_um=1000, radius_um=150, min_active=20)
    assert len(centres) == 3
    for i in range(3):
        for j in range(i + 1, 3):
            assert rs.np.hypot(centres[i][0] - centres[j][0], centres[i][1] - centres[j][1]) >= 1000
    # The two overlapping blobs on the left are one patch, not two.
    assert sum(c[0] < 1500 for c in centres) == 1


def test_choose_centers_raises_when_too_few():
    well = _well(_blob(600, 1000, 120))
    with pytest.raises(ValueError):
        rs.choose_centers(well, n=3, min_separation_um=1000, min_active=20)


def test_recording_electrodes_respects_budget_and_covers_patches():
    active = _blob(600, 1000, 120) + _blob(2400, 600, 120) + list(range(0, rs.COLS * rs.ROWS, 37))
    well = _well(active)
    centres = [(600, 1000), (2400, 600)]
    out = rs.recording_electrodes(well, centres, radius_um=150, budget=400, reserve=32)
    patch_0, patch_1, spread = out["patch_0"], out["patch_1"], out["global"]
    assert set(patch_0) >= set(_blob(600, 1000, 120))
    assert not set(patch_0) & set(patch_1)
    assert len(patch_0) + len(patch_1) + len(spread) <= 400 - 32
    assert len(set(spread)) == len(spread)
    assert not set(spread) & (set(patch_0) | set(patch_1))


def test_region_correlations_sees_coupling():
    rng = np.random.default_rng(0)
    fps = 20000
    drive = rng.random(2000) < 0.05  # shared 10 ms bins
    frames, elecs = [], []
    for name, elec, own in (("a", 1, drive), ("b", 2, drive), ("c", 3, rng.random(2000) < 0.05)):
        hits = np.flatnonzero(own)
        frames.append(hits * 200 + 7)
        elecs.append(np.full(len(hits), elec))
    frames, elecs = np.concatenate(frames), np.concatenate(elecs)
    names, matrix = rs.region_correlations(frames, elecs, {"a": [1], "b": [2], "c": [3]}, fps, bin_ms=10)
    assert names == ["a", "b", "c"]
    assert matrix[0, 1] > 0.99
    assert abs(matrix[0, 2]) < 0.2


def test_assign_roles_avoids_the_coupled_pair():
    # A square, so every triple is the same right isosceles shape and geometry does not decide.
    centres = [(0, 0), (1000, 0), (0, 1000), (1000, 1000)]
    matrix = np.eye(4)
    matrix[0, 1] = matrix[1, 0] = 0.9  # 0 and 1 move together
    matrix[0, 2] = matrix[2, 0] = 0.3
    matrix[0, 3] = matrix[3, 0] = 0.1
    density = [50, 40, 30, 20]
    roles = rs.assign_roles(centres, matrix, density)
    assert set(roles) == {"US", "CS", "NS"}
    chosen = set(roles.values())
    assert not ({(0, 0), (1000, 0)} <= chosen)  # never both of the coupled pair
    # {1, 2, 3} is the only triple with no coupling at all, so it wins over the denser {0, 2, 3}.
    assert set(roles.values()) == {(1000, 0), (0, 1000), (1000, 1000)}
    # The right angle is at (1000, 1000): equidistant from the other two, so it is US.
    assert roles["US"] == (1000, 1000)


def test_assign_roles_prefers_density_among_equally_coupled():
    centres = [(0, 0), (1000, 0), (500, 866), (500, -866)]  # two equilateral triangles share a side
    matrix = np.eye(4)
    density = [50, 40, 30, 20]
    roles = rs.assign_roles(centres, matrix, density)
    assert set(roles.values()) == {(0, 0), (1000, 0), (500, 866)}
    assert roles["US"] == (0, 0)  # every vertex is balanced; density breaks the tie


def test_assign_roles_rejects_a_line():
    # Three in a row plus one off to the side: the equilateral-ish triple wins even though the
    # collinear one is denser and equally uncoupled.
    centres = [(0, 0), (1000, 0), (2000, 0), (500, 900)]
    matrix = np.eye(4)
    density = [90, 90, 90, 10]
    roles = rs.assign_roles(centres, matrix, density)
    chosen = set(roles.values())
    assert (500, 900) in chosen
    assert not ({(0, 0), (1000, 0), (2000, 0)} <= chosen)
    # US is the apex, equidistant from CS and NS.
    us = roles["US"]
    d = [np.hypot(us[0] - c[0], us[1] - c[1]) for r, c in roles.items() if r != "US"]
    assert abs(d[0] - d[1]) < 1.0


def test_triangle_ratio():
    assert rs.triangle([(0, 0), (1000, 0), (500, 866)])[0] == pytest.approx(1.0, abs=0.01)
    assert rs.triangle([(0, 0), (1000, 0), (2000, 0)])[0] == pytest.approx(0.5)


def test_correlation_outside_bursts_removes_burst_coupling():
    rng = np.random.default_rng(1)
    fps = 20000
    frames, elecs = [], []
    # Two regions that only ever spike together inside network bursts every 2 s, plus
    # independent background at 5 Hz each.
    for elec in (1, 2):
        background = np.sort(rng.integers(0, 60 * fps, 300))
        frames.append(background)
        elecs.append(np.full(len(background), elec))
    for k in range(30):
        start = k * 2 * fps
        for elec, lag in ((1, 0), (2, 100)):  # region 1 leads by 5 ms
            burst = start + lag + np.arange(0, 2000, 40)
            frames.append(burst)
            elecs.append(np.full(len(burst), elec))
    frames, elecs = np.concatenate(frames), np.concatenate(elecs)
    order = np.argsort(frames)
    frames, elecs = frames[order], elecs[order]
    regions = {"a": [1], "b": [2]}
    bursts = rs.network_bursts(frames, fps)
    assert 25 <= len(bursts) <= 30
    _, raw = rs.region_correlations(frames, elecs, regions, fps)
    _, outside = rs.region_correlations(frames, elecs, regions, fps, exclude=bursts)
    assert raw[0, 1] > 0.8
    assert abs(outside[0, 1]) < 0.2
    _, lag_ms, lead, used = rs.burst_onset_lags(frames, elecs, regions, fps, bursts)
    assert used >= 25
    assert lead[0, 1] > 0.9 and lag_ms[0, 1] == pytest.approx(-5.0, abs=1.0)
    c = rs.coupling(outside, lead)
    assert c[0, 1] > 0.8  # uncorrelated outside bursts, but always the same order into them


def test_assign_roles_needs_enough_candidates():
    with pytest.raises(ValueError):
        rs.assign_roles([(0, 0), (1, 1)], np.eye(2), [1, 1])


def test_rank_patches_explains_the_walk():
    well = _well(_blob(600, 1000, 120) + _blob(700, 1000, 120) + _blob(2400, 600, 120))
    # Asking for three with only two blobs makes the walk run to the end.
    ranked = rs.rank_patches(well, n=3, min_separation_um=1000, radius_um=150, min_active=20)
    statuses = [r["status"] for r in ranked]
    assert statuses.count("chosen") == 2
    assert any(s.startswith("too close to") for s in statuses)
    assert "too few active" in statuses
    assert ranked[0]["status"] == "chosen"
    assert all(a["count"] >= b["count"] for a, b in pairwise(ranked))


def test_well_data_from_npz_round_trips(tmp_path):
    electrodes = np.array([100, 101, 5000])
    channelmap = np.column_stack(
        [
            np.arange(3),
            np.array([7, 8, 9]),
            electrodes,
            (electrodes % rs.COLS) * rs.PITCH_UM,
            (electrodes // rs.COLS) * rs.PITCH_UM,
        ]
    ).astype(float)
    spikes = np.array(
        [(10, 7, -5e-5), (20, 9, -6e-5), (30, 7, -4e-5)],
        dtype=[("frameno", "<i8"), ("channel", "<i4"), ("amplitude", "<f4")],
    )
    path = tmp_path / "well.npz"
    np.savez(
        path,
        spike_data=spikes,
        channelmap=channelmap,
        samp_rate=np.array([20000.0]),
        rec_t_sec=np.array([12.0]),
        lsb=np.array([6e-6]),
    )
    well = rs.well_data_from_npz(str(path))
    assert list(well["mapping"]["electrode"]) == [100, 101, 5000]
    assert set(well["spike_data"]["electrode"]) == {100, 5000}
    assert well["lsb"] == 1.0 and well["rec_length_sec"] == 12.0
    frames, elecs, fps = rs.baseline_spikes(str(path), 0)
    assert list(frames) == [10, 20, 30] and list(elecs) == [100, 5000, 100] and fps == 20000.0


def test_rank_patches_threshold_is_relative_to_recorded():
    # Sparse recording: nine electrodes recorded in a patch, all active. An absolute floor of 30
    # can never pass; the fraction rule with a floor of 5 does.
    sparse = _well(
        [e for e in _blob(600, 1000, 140) if e % 7 == 0] + [e for e in _blob(2400, 600, 140) if e % 7 == 0]
    )
    ranked = rs.rank_patches(sparse, n=2, min_separation_um=1000, radius_um=150)
    assert [r["status"] for r in ranked].count("chosen") == 2
    assert all(r["recorded"] >= r["count"] for r in ranked)
    with pytest.raises(ValueError):
        rs.choose_centers(sparse, n=2, min_separation_um=1000, radius_um=150, min_active=30)


def test_assign_roles_matches_the_two_cs_and_counterbalances():
    # Equilateral, so geometry does not decide. Candidate 3 is coupled to 0 at 0.3; candidates
    # 1 and 2 are both coupled to 0 at 0.1. The triple {0, 1, 2} has matched CS couplings.
    centres = [(0, 0), (1000, 0), (500, 866), (-500, 866)]
    matrix = np.eye(4)
    matrix[0, 1] = matrix[1, 0] = 0.1
    matrix[0, 2] = matrix[2, 0] = 0.1
    matrix[0, 3] = matrix[3, 0] = 0.3
    matrix[1, 2] = matrix[2, 1] = 0.1
    matrix[2, 3] = matrix[3, 2] = 0.1
    density = [10, 10, 10, 10]
    roles = rs.assign_roles(centres, matrix, density)
    assert set(roles.values()) == {(0, 0), (1000, 0), (500, 866)}
    flipped = rs.assign_roles(centres, matrix, density, counterbalance=1)
    assert flipped["US"] == roles["US"]
    assert flipped["CS"] == roles["NS"] and flipped["NS"] == roles["CS"]


def _paired_spikes(fps=20000, lag_frames=100, n_bursts=30):
    """Two regions that fire together in bursts every 2 s, region a leading by ``lag_frames``,
    plus independent background."""
    rng = np.random.default_rng(1)
    frames, elecs = [], []
    for elec in (1, 2):
        background = np.sort(rng.integers(0, 60 * fps, 300))
        frames.append(background)
        elecs.append(np.full(len(background), elec))
    for k in range(n_bursts):
        start = k * 2 * fps
        for elec, lag in ((1, 0), (2, lag_frames)):
            burst = start + lag + np.arange(0, 2000, 40)
            frames.append(burst)
            elecs.append(np.full(len(burst), elec))
    frames, elecs = np.concatenate(frames), np.concatenate(elecs)
    order = np.argsort(frames)
    return frames[order], elecs[order], {"a": [1], "b": [2]}


def test_cross_correlogram_lag_sign_says_who_leads():
    fps = 20000
    frames, elecs, regions = _paired_spikes(fps, lag_frames=100)  # b lags a by 5 ms
    names, lags, corr = rs.cross_correlograms(frames, elecs, regions, fps, bin_ms=1.0, max_lag_ms=20.0)
    assert names == ["a", "b"]
    # a leads b, so the a->b correlogram peaks at a positive lag of about +5 ms.
    assert lags[int(np.argmax(corr[0, 1]))] == pytest.approx(5.0, abs=1.0)
    # and the mirror image peaks at -5 ms.
    assert lags[int(np.argmax(corr[1, 0]))] == pytest.approx(-5.0, abs=1.0)
    # Lag zero agrees with the plain correlation at the same bin width.
    _, plain = rs.region_correlations(frames, elecs, regions, fps, bin_ms=1.0)
    zero = corr[0, 1][len(lags) // 2]
    assert zero == pytest.approx(plain[0, 1], abs=0.02)


def test_cross_correlogram_masking_bursts_removes_the_peak():
    fps = 20000
    frames, elecs, regions = _paired_spikes(fps, lag_frames=100)
    bursts = rs.network_bursts(frames, fps)
    _, _, raw = rs.cross_correlograms(frames, elecs, regions, fps, bin_ms=1.0, max_lag_ms=20.0)
    _, _, masked = rs.cross_correlograms(
        frames, elecs, regions, fps, bin_ms=1.0, max_lag_ms=20.0, exclude=bursts
    )
    assert np.nanmax(raw[0, 1]) > 0.3
    assert np.nanmax(np.abs(masked[0, 1])) < 0.15


def test_burst_onsets_and_rates():
    fps = 20000
    frames, elecs, regions = _paired_spikes(fps, lag_frames=100)
    bursts = rs.network_bursts(frames, fps)
    names, onsets = rs.burst_onsets(frames, elecs, regions, fps, bursts)
    assert names == ["a", "b"] and onsets.shape == (len(bursts), 2)
    lead = onsets[:, 1] - onsets[:, 0]
    assert np.nanmedian(lead) == pytest.approx(100, abs=25)  # b joins 5 ms after a

    names, t, rates = rs.patch_rates(frames, elecs, regions, fps, bin_ms=100.0)
    assert rates.shape[0] == 2 and len(t) == rates.shape[1]
    assert rates.max() > rates.mean()  # the bursts stand out


def test_correlogram_peak_only_claims_a_lead_when_it_stands_out():
    lags = np.arange(-50, 51, 1.0)
    rng = np.random.default_rng(0)
    flat = rng.normal(0, 0.01, len(lags))
    lag, height, stands_out = rs.correlogram_peak(lags, flat)
    assert not stands_out  # wandering noise is not a lead

    peaked = flat + 0.3 * np.exp(-((lags - 5.0) ** 2) / 8.0)
    lag, height, stands_out = rs.correlogram_peak(lags, peaked)
    assert stands_out and lag == pytest.approx(5.0, abs=1.0) and height > 0.25

    lag, height, stands_out = rs.correlogram_peak(lags, np.full(len(lags), np.nan))
    assert not stands_out and np.isnan(height)


def _write_h5(path, recordings, well=0, fps=10000.0):
    """A minimal MaxWell-shaped file with `recordings` rounds of (frames, channels)."""
    import h5py

    with h5py.File(path, "w") as f:
        for k, (frames, channels) in enumerate(recordings):
            rec = f.create_group(f"recordings/rec{k:04d}/well{well:03d}")
            rec.create_dataset(
                "spikes",
                data=np.array(
                    list(zip(frames, channels, [-5e-5] * len(frames))),
                    dtype=[("frameno", "<i8"), ("channel", "<i4"), ("amplitude", "<f4")],
                ),
            )
            rec.create_dataset(
                "settings/mapping",
                data=np.array(
                    [(c, 1000 + c, 0.0, 0.0) for c in sorted(set(channels))],
                    dtype=[("channel", "<i4"), ("electrode", "<i4"), ("x", "<f8"), ("y", "<f8")],
                ),
            )
            rec.create_dataset("settings/sampling", data=np.array([fps]))


def test_baseline_spikes_reads_one_continuous_recording(tmp_path):
    path = tmp_path / "one.raw.h5"
    _write_h5(path, [([10, 20, 30], [1, 2, 1])])
    frames, elecs, fps = rs.baseline_spikes(str(path), 0)
    assert list(frames) == [10, 20, 30]
    assert list(elecs) == [1001, 1002, 1001]
    assert fps == 10000.0


def test_baseline_spikes_refuses_a_multi_round_scan(tmp_path):
    path = tmp_path / "scan.raw.h5"
    _write_h5(path, [([10, 20], [1, 2]), ([30, 40], [3, 4])])
    with pytest.raises(ValueError, match="scan rather than one continuous recording"):
        rs.baseline_spikes(str(path), 0)


def _scan_well(rounds, seconds=60.0, lsb=1.0):
    """A merged scan: `rounds` is a list of {electrode: n_spikes}; the mapping repeats an
    electrode once per round it was recorded in, as load_activity_scan leaves it."""
    spikes, mapping = [], []
    for r, counts in enumerate(rounds):
        for electrode, n in counts.items():
            mapping.append({"channel": electrode, "electrode": electrode, "x": 0.0, "y": 0.0})
            for k in range(n):
                spikes.append(
                    {
                        "frameno": r * 10000 + k,
                        "channel": electrode,
                        "amplitude": -1e-4,
                        "electrode": electrode,
                    }
                )
    return {
        "spike_data": pd.DataFrame(spikes),
        "mapping": pd.DataFrame(mapping),
        "samp_rate": 20000.0,
        "lsb": lsb,
        "rec_length_sec": seconds,
    }


def test_active_electrodes_divides_by_each_electrode_s_own_recorded_time():
    # Both electrodes fire at 5 spikes / 60 s = 0.083 Hz, under the 0.1 Hz threshold. Electrode 2
    # was recorded in two rounds, so summing its rounds without dividing would put it at 0.17 Hz
    # and wrongly mark it active.
    well = _scan_well([{1: 5, 2: 5}, {2: 5}])
    assert int(rs.scan_visits(well)[2]) == 2
    out = rs.active_electrodes(well)
    assert set(out["active_electrodes"]["electrode"]) == set()

    # Genuinely active in every round it appears in: still active.
    well = _scan_well([{1: 5, 2: 60}, {2: 60}])
    out = rs.active_electrodes(well)
    assert set(out["active_electrodes"]["electrode"]) == {2}


def test_active_electrodes_matches_the_plain_rule_on_one_round():
    from mxtreme.scans import electrode_selection

    well = _scan_well([{1: 60, 2: 1}])
    mine = rs.active_electrodes({**well, "spike_data": well["spike_data"].copy()})
    theirs = electrode_selection.get_active_electrodes(
        {0: {**well, "spike_data": well["spike_data"].copy()}}
    )[0]
    assert set(mine["active_electrodes"]["electrode"]) == set(theirs["active_electrodes"]["electrode"])
