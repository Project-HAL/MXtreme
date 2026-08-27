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
from mxtreme.paths import CulturePaths, resolve_paths, resolve_recordings
from mxtreme.analysis import activity, performance, stimulation, generate_report
from mxtreme.analysis._paths import _summary_paths

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
    """A managed store with two cultures: A (exp 'expA', 2 DIVs) and B (exp 'expB', 1 DIV).

    Culture A is right-trained ([2, 0]); culture B carries no condition pair, so it has no trained side.
    """
    config = Config(data_root=tmp_path)

    # Culture A: same chip/well across two DIVs.
    for seed, div in ((1, 7), (2, 8)):
        _add_recording(
            config,
            make_recording_data(seed=seed, exp_id="expA", chip="C0001", well=0, DIV=div,
                                exp_condition=np.array([2, 0])),
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


def test_resolve_ignores_activity_scan_rows(store):
    """The registry also indexes scans -- raw .h5 files with no .npz behind them.

    Counting those rows would invent DIVs (and whole cultures) that resolve to preprocessed paths
    which do not exist.
    """
    config, cid_a, _ = store

    class _ScanParams:
        exp_id, chip, div = "expA", "C0001", 21  # a DIV culture A has no recording for
        wells, conditions = [0], []

    io.register_scan(_ScanParams(), config.registry_path)
    assert len(pd.read_csv(config.registry_path)) == 4  # the scan row really is there

    assert sorted(resolve_paths(cid_a, config).recordings) == [7, 8]
    assert len(resolve_recordings(CultureSelector(exp_ids=["expA"]), config)) == 2


def test_resolve_reads_a_registry_written_before_scans_were_indexed(store):
    """No `kind` column at all: every row is a recording, and resolution is unchanged."""
    config, cid_a, _ = store
    df = pd.read_csv(config.registry_path).drop(columns=["kind"])
    df.to_csv(config.registry_path, index=False)

    assert sorted(resolve_paths(cid_a, config).recordings) == [7, 8]


def test_resolve_recordings_flattens_a_selection(store):
    config, cid_a, cid_b = store
    recs = resolve_recordings(CultureSelector(cultures=[cid_a, cid_b]), config)

    # Culture A has 2 DIVs, culture B has 1; ordered by exp_id, chip, well, then DIV.
    assert len(recs) == 3
    assert [r.recording_id.div for r in recs] == [7, 8, 7]
    assert [r.recording_id.exp_id for r in recs] == ["expA", "expA", "expB"]
    assert recs.npz == [r.npz for r in recs]
    assert all(p.exists() for p in recs.npz)
    assert all(p.exists() for p in recs.burst_stats)
    assert recs[0] is recs.recordings[0]


def test_resolve_recordings_accepts_any_target(store):
    config, cid_a, _ = store
    assert len(resolve_recordings(cid_a, config)) == 2          # CultureID -> all DIVs
    rid = resolve_recordings(cid_a, config).ids[0]
    assert len(resolve_recordings(rid, config)) == 1            # RecordingID -> just that one


def test_resolve_recordings_to_frame(store):
    config, cid_a, cid_b = store
    df = resolve_recordings(CultureSelector(cultures=[cid_a, cid_b]), config).to_frame()

    assert list(df.columns) == ["exp_id", "chip", "well", "div", "npz", "burst_stats"]
    assert len(df) == 3
    assert df["div"].tolist() == [7, 8, 7]
    assert set(df["exp_id"]) == {"expA", "expB"}


def test_resolve_recordings_respects_div_filter(store):
    config, cid_a, cid_b = store
    recs = resolve_recordings(CultureSelector(cultures=[cid_a, cid_b], divs=[8]), config)
    assert [r.recording_id.div for r in recs] == [8]


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
    assert list(df.columns) == ["chip", "well", "div", "phase", "condition", "trained_side", "score"]
    scores = df["score"].dropna()
    assert ((scores >= 0) & (scores <= 1)).all()
    # Culture A is [2, 0]: every row is scored against the right-hand target.
    assert set(df["trained_side"]) == {"right"}
    assert df["condition"].tolist() == [[2, 0]] * len(df)


@pytest.mark.parametrize("condition, expected", [
    ([0, 2], "left"),                              # lower number on the left
    ([1, 2], "left"),
    ([2, 0], "right"),                             # lower number on the right
    ([1, 0], "right"),
    ([2, 1], "right"),
    ([0, 0], None),                                # no lower number -> untrained control
    ([2, 2], None),
    (None, None),                                  # recording predating conditions
    ({"left_stim": 0, "right_stim": 0}, None),     # legacy dict-shaped condition
    ([0, 1, 2], None),                             # not a pair
])
def test_trained_side_mapping(condition, expected):
    stored = np.asarray(condition) if condition is not None else None
    assert performance.trained_side(stored) is expected


def _directional_bursts(n_rightward, n_leftward):
    """Bursts that propagate left-to-right (origin_x < peak_x) and right-to-left, respectively."""
    rows = [{"origin_x": 0.0, "peak_x": 100.0}] * n_rightward
    rows += [{"origin_x": 100.0, "peak_x": 0.0}] * n_leftward
    return pd.DataFrame(rows)


def test_default_objective_scores_toward_the_trained_side():
    """The same bursts score oppositely for a left- and a right-trained culture."""
    bursts = _directional_bursts(n_rightward=3, n_leftward=1)

    left = performance.default_direction_objective(bursts.assign(trained_side="left"))
    right = performance.default_direction_objective(bursts.assign(trained_side="right"))

    assert left == 0.75      # 3 of 4 originate left of their peak
    assert right == 0.25


def test_default_objective_without_trained_side_column_is_paradigm_free():
    # Column absent means the condition is unknown -- keep the historical left-to-right fraction.
    bursts = _directional_bursts(n_rightward=3, n_leftward=1)
    assert performance.default_direction_objective(bursts) == 0.75


def test_default_objective_with_no_trained_side_is_nan():
    # Column present and None means the culture is known to have no target: nothing to score.
    bursts = _directional_bursts(n_rightward=3, n_leftward=1)
    assert np.isnan(performance.default_direction_objective(bursts.assign(trained_side=None)))


def test_performance_summary_recomputes_pre_condition_cache(store):
    """A cached summary without `trained_side` holds wrong-signed scores and must not be reused."""
    config, cid_a, _ = store
    cpath = resolve_paths(cid_a, config)
    fresh = performance.performance_summary(cpath, config.analysis_dir, show_plot=False)

    # Overwrite the cache with an old-schema one carrying an obviously bogus score.
    _, csv_path = _summary_paths(cpath, config.analysis_dir, "performance", "performance_summary")
    stale = fresh[["chip", "well", "div", "phase"]].copy()
    stale["score"] = -1.0
    stale.to_csv(csv_path, index=False)

    df = performance.performance_summary(cpath, config.analysis_dir, show_plot=False, use_existing=True)
    assert "trained_side" in df.columns
    assert (df["score"] != -1.0).all()


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


# --- stimulation ---------------------------------------------------------------------------------


def _stim_events(n=5, phase_us=200.0):
    """Maxlab-shaped stimulation events: one dict per pulse carrying start_stimulation + phase_us."""
    eventtime = np.array([50000 * (i + 1) for i in range(n)], dtype="<i8")
    messages = np.array(
        [{"start_stimulation": "1", "phase_us": phase_us} for _ in range(n)], dtype=object
    )
    return eventtime, messages


def _stim_store(tmp_path, make_recording_data, *, chip, phase_spec=None, messages=None, eventtime=None):
    """A one-culture store whose single recording carries stimulation events."""
    config = Config(data_root=tmp_path)
    if eventtime is None or messages is None:
        eventtime, messages = _stim_events()
    extra = {} if phase_spec is None else {"phase_spec": np.asarray(phase_spec, dtype=object)}
    data = make_recording_data(
        seed=1, exp_id="expS", chip=chip, well=0, DIV=7,
        eventtime=eventtime, event_messages=messages, **extra,
    )
    _add_recording(config, data, well_no=0)
    return config, resolve_paths(CultureID("expS", chip, "0"), config)


def test_stim_summary_uses_train_phase_window(tmp_path, make_recording_data):
    # The fixture is 500k frames @ 10 kHz = 50 s. A train phase from 1/6 min to 0.5 min spans
    # 1/3 min; the old hardcoded (rec_t_sec - 40*60)/60 would have returned -39.2.
    spec = {"starts": {"pre": 0.0, "train": 1 / 6, "post": 0.5}, "end": 0.833}
    config, cpath = _stim_store(tmp_path, make_recording_data, chip="C0010", phase_spec=spec)

    df = stimulation.stim_summary(cpath, config.analysis_dir, show_plot=False)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["total_stim_ms"] == pytest.approx(5 * 200.0 / 1000)
    assert row["total_train_min"] == pytest.approx(1 / 3, abs=1e-3)


def test_stim_summary_falls_back_to_full_recording_without_train_phase(tmp_path, make_recording_data):
    # No phase spec -> a single "full" phase, so train time is the whole recording (50 s), not a
    # negative number left over from the 20-min pre/post assumption.
    config, cpath = _stim_store(tmp_path, make_recording_data, chip="C0011")

    df = stimulation.stim_summary(cpath, config.analysis_dir, show_plot=False)

    assert df.iloc[0]["total_train_min"] == pytest.approx(50 / 60, abs=1e-3)


def test_stim_summary_tolerates_non_dict_event_messages(tmp_path, make_recording_data):
    # Older stores hold plain strings alongside the dict messages; those must be skipped, not raise.
    eventtime = np.array([1000, 50000, 90000], dtype="<i8")
    messages = np.array(
        ["legacy string", {"start_stimulation": "1", "phase_us": 100.0}, {"other_event": "1"}],
        dtype=object,
    )
    config, cpath = _stim_store(
        tmp_path, make_recording_data, chip="C0012", eventtime=eventtime, messages=messages,
    )

    df = stimulation.stim_summary(cpath, config.analysis_dir, show_plot=False)

    assert df.iloc[0]["total_stim_ms"] == pytest.approx(0.1)


def test_generate_report_with_stimulation_section(tmp_path, make_recording_data):
    spec = {"starts": {"pre": 0.0, "train": 1 / 6, "post": 0.5}, "end": 0.833}
    config, _ = _stim_store(tmp_path, make_recording_data, chip="C0013", phase_spec=spec)

    out = generate_report(
        CultureID("expS", "C0013", "0"), config, sections=("activity", "stimulation")
    )
    assert out.exists() and out.stat().st_size > 0
