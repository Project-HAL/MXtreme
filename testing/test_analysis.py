"""Tests for the analysis module: path resolution against a Config store, per-culture summaries,
the pluggable performance objective, and PDF report generation."""

import matplotlib
matplotlib.use("Agg")  # headless

import numpy as np
import pandas as pd
import pytest

from mxtreme import io
from mxtreme.config import Config
from mxtreme.identity import CultureID, CultureSelector
from mxtreme.recording import Recording
from mxtreme.bursting import BurstDetector
from mxtreme.params import BurstDetectParams, BurstFeatureParams
from mxtreme.paths import CulturePaths, resolve_paths
from mxtreme.analysis import activity, performance, generate_report

# Detection params tuned for the small synthetic fixture (see test_bursting.py).
DETECT = BurstDetectParams(n=50, noise_thresh=0.02, burst_thresh=0.15, min_dist_bins=10)


def _add_recording(config: Config, data: dict, well_no: int) -> None:
    """Write one synthetic recording (npz + burst CSV) into the managed store and register it."""
    exp_id, chip, div = str(data["exp_id"]), str(data["chip"]), int(data["DIV"])

    out_dir = config.preprocessed_dir / exp_id / chip / f"well{well_no}"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"DIV{div}_250101_{chip}_{exp_id}_well{well_no}_exp_data.npz"
    np.savez_compressed(npz, **data)

    rec = Recording(0, io.load_preprocessed(npz))
    bursts = BurstDetector(DETECT).detect(rec)
    bursts.extract_features(rec, BurstFeatureParams())
    io.save_burst_data(bursts, config.burst_data_dir, rec)

    io.register(
        {well_no: {"exp_id": exp_id, "chip": chip, "well": well_no, "DIV": div}},
        config.registry_path,
    )


@pytest.fixture
def store(tmp_path, make_recording_data):
    """A managed store with two cultures: A (exp 'expA', 2 DIVs) and B (exp 'expB', 1 DIV)."""
    config = Config(data_root=tmp_path)

    # Culture A: same chip/well across two DIVs.
    for seed, div in ((1, 7), (2, 8)):
        _add_recording(
            config,
            make_recording_data(seed=seed, exp_id="expA", chip="C0001", well=0, DIV=div),
            well_no=0,
        )
    # Culture B: a different experiment/chip (cross-experiment group case).
    _add_recording(
        config,
        make_recording_data(seed=3, exp_id="expB", chip="C0002", well=0, DIV=7),
        well_no=0,
    )

    cid_a = CultureID("expA", "C0001", "0")
    cid_b = CultureID("expB", "C0002", "0")
    return config, cid_a, cid_b


def test_resolve_single_culture_all_divs(store):
    config, cid_a, _ = store
    cpath = resolve_paths(cid_a, config)
    assert isinstance(cpath, CulturePaths)
    assert sorted(cpath.recordings) == [7, 8]
    assert cpath.recordings[7].npz.exists()
    assert cpath.recordings[7].burst_stats.exists()


def test_resolve_group_across_experiments(store):
    config, cid_a, cid_b = store
    resolved = resolve_paths(CultureSelector(cultures=[cid_a, cid_b]), config)
    assert set(resolved) == {"expA", "expB"}
    assert len(resolved["expA"].cultures) == 1
    assert len(resolved["expB"].cultures) == 1


def test_channel_activity_summary_columns(store):
    config, cid_a, _ = store
    cpath = resolve_paths(cid_a, config)
    df = activity.channel_activity_summary(cpath, config.analysis_dir, show_plot=False)
    for col in ("culture_id", "div", "phase", "mean_fr_hz", "median_isi_sec", "mean_amp_uv", "pct_active_chan"):
        assert col in df.columns
    assert set(df["div"]) == {7, 8}
    assert set(df["phase"]) == {"full"}


def test_burst_activity_summary_columns(store):
    config, cid_a, _ = store
    cpath = resolve_paths(cid_a, config)
    df = activity.burst_activity_summary(cpath, config.analysis_dir, show_plot=False)
    for col in ("culture_id", "div", "phase", "burst_rate_hz", "median_ibi_sec", "median_size_pct", "median_dur_sec"):
        assert col in df.columns
    # Burst rate is bursts / phase-seconds and must be finite & non-negative.
    assert (df["burst_rate_hz"].dropna() >= 0).all()


def test_summaries_honor_embedded_phase_spec(tmp_path, make_recording_data):
    # A recording whose npz carries a phase spec is split into those phases by the analysis with no
    # make_phases / phase_tags anywhere. The fixture has dense bursts near frames 100k and 300k
    # (10 kHz), so a split at ~0.333 min (200k frames) puts a burst in each phase.
    config = Config(data_root=tmp_path)
    spec = {"starts": {"early": 0.0, "late": 0.333}, "end": 0.833}
    data = make_recording_data(
        seed=1, exp_id="expP", chip="C0009", well=0, DIV=7,
        phase_spec=np.asarray(spec, dtype=object),
    )
    _add_recording(config, data, well_no=0)

    cpath = resolve_paths(CultureID("expP", "C0009", "0"), config)

    ch = activity.channel_activity_summary(cpath, config.analysis_dir, show_plot=False)
    assert set(ch["phase"]) == {"early", "late"}

    bu = activity.burst_activity_summary(cpath, config.analysis_dir, show_plot=False)
    assert set(bu["phase"].dropna()) <= {"early", "late"}
    assert (bu["burst_rate_hz"].dropna() >= 0).all()


def test_performance_summary_default_objective(store):
    config, cid_a, _ = store
    cpath = resolve_paths(cid_a, config)
    df = performance.performance_summary(cpath, config.analysis_dir, show_plot=False)
    assert list(df.columns) == ["chip", "well", "div", "phase", "score"]
    scores = df["score"].dropna()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_performance_summary_custom_objective(store):
    config, cid_a, _ = store
    cpath = resolve_paths(cid_a, config)
    df = performance.performance_summary(
        cpath, config.analysis_dir, objective_fn=lambda b: 0.42, show_plot=False, use_existing=False
    )
    assert (df["score"] == 0.42).all()


def test_generate_report_single_culture(store):
    config, cid_a, _ = store
    out = generate_report(cid_a, config, sections=("overview", "activity", "bursting", "performance"))
    assert out.exists() and out.stat().st_size > 0


def test_generate_report_group_across_experiments(store):
    config, cid_a, cid_b = store
    out = generate_report(
        CultureSelector(cultures=[cid_a, cid_b]),
        config,
        sections=("overview", "activity", "bursting"),
    )
    assert out.exists() and out.stat().st_size > 0
