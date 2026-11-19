"""Correlation between regions' population rates (mxtreme.analysis.regions)."""

import numpy as np

from mxtreme.analysis.regions import Region, region_correlation, region_correlation_course, region_rates

FS = 20_000.0


def _map(xs):
    # columns: index, channel, electrode, x, y
    return np.array([[i, 100 + i, 1000 + i, x, 10.0] for i, x in enumerate(xs)], dtype=float)


def _regions():
    return [Region("a", "A", "rect", ((0, 0), (50, 50))), Region("b", "B", "rect", ((100, 0), (150, 50))), Region("c", "C", "rect", ((500, 0), (550, 50)))]


def test_rates_are_per_electrode_and_empty_regions_are_nan():
    cm = _map([10, 20, 120])  # two channels in A, one in B, none in C
    frames = np.array([0, 10, 250, 260], dtype=np.int64)  # bins of 10 ms = 200 frames
    rows = np.array([0, 1, 2, 2])
    rates, n = region_rates(frames, rows, cm, _regions(), FS, start=0, stop=400, bin_ms=10)
    assert n.tolist() == [2, 1, 0]
    assert rates.shape == (3, 2)
    np.testing.assert_allclose(rates[0], [100.0, 0.0])  # 2 spikes / 2 electrodes / 10 ms
    np.testing.assert_allclose(rates[1], [0.0, 200.0])
    assert np.isnan(rates[2]).all()


def test_lagged_coupling_is_found_with_its_direction():
    rng = np.random.default_rng(0)
    a = rng.poisson(2.0, 4000).astype(float)
    b = np.roll(a, 3) + rng.normal(0, 0.1, 4000)  # B follows A by 3 bins
    noise = rng.normal(0, 1, 4000)
    res = region_correlation(np.vstack([a, b, noise]), max_lag=5)
    assert res["r"][0, 1] < 0.2
    assert res["peak_r"][0, 1] > 0.95
    assert res["peak_lag"][0, 1] == 3 and res["peak_lag"][1, 0] == -3
    np.testing.assert_allclose(res["r"], res["r"].T)
    assert abs(res["r"][0, 2]) < 0.1


def test_a_flat_region_correlates_with_nothing():
    res = region_correlation(np.vstack([np.arange(10.0), np.zeros(10)]))
    assert np.isnan(res["r"][1]).all() and np.isnan(res["r"][:, 1]).all()


def test_course_follows_a_change_in_coupling():
    rng = np.random.default_rng(1)
    a = rng.normal(size=2000)
    b = np.concatenate([a[:1000], rng.normal(size=1000)])  # coupled, then not
    starts, r = region_correlation_course(np.vstack([a, b]), window=200, step=200)
    assert len(starts) == 10
    assert (r[:5, 0, 1] > 0.99).all() and (np.abs(r[5:, 0, 1]) < 0.3).all()
