"""Tests for the analysis module: path resolution against a Config store, per-culture summaries,
the pluggable performance objective, and PDF report generation."""

import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mxtreme import io
from mxtreme.config import Config
from mxtreme.identity import CultureID, CultureSelector
from mxtreme.recording import Recording
from mxtreme.bursting import BurstDetector
from mxtreme.params import BurstDetectParams, BurstFeatureParams, CommunityParams, NetworkParams
from mxtreme.paths import CulturePaths, resolve_paths, resolve_paths_flat
from mxtreme.analysis import activity, network, performance, spatial, stimulation, generate_report
from mxtreme.analysis._paths import _summary_paths, load_population_summaries
from mxtreme.analysis._stats import aggregate_by_div_phase

# Detection params tuned for the small synthetic fixture (see test_bursting.py).
DETECT = BurstDetectParams(n=50, noise_thresh=0.02, burst_thresh=0.15, min_dist_bins=10)


def _add_recording(config: Config, data: dict, well_no: int) -> None:
    """Write one synthetic recording (npz + burst CSV) into the managed store and register it."""
    batch_id, chip, div = str(data["batch_id"]), str(data["chip"]), int(data["DIV"])

    out_dir = config.preprocessed_dir / batch_id / chip / f"well{well_no}"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"DIV{div}_250101_{chip}_{batch_id}_well{well_no}_exp_data.npz"
    np.savez_compressed(npz, **data)

    rec = Recording(0, io.load_preprocessed(npz))
    bursts = BurstDetector(DETECT).detect(rec)
    bursts.extract_features(rec, BurstFeatureParams())
    io.save_burst_data(bursts, config.burst_data_dir, rec)

    io.register(
        {well_no: {"batch_id": batch_id, "chip": chip, "well": well_no, "DIV": div}},
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
            make_recording_data(seed=seed, batch_id="expA", chip="C0001", well=0, DIV=div,
                                exp_condition=np.array([2, 0])),
            well_no=0,
        )
    # Culture B: a different experiment/chip (cross-experiment group case).
    _add_recording(
        config,
        make_recording_data(seed=3, batch_id="expB", chip="C0002", well=0, DIV=7),
        well_no=0,
    )

    cid_a = CultureID("expA", "C0001", "0")
    cid_b = CultureID("expB", "C0002", "0")
    return config, cid_a, cid_b


def test_resolve_single_culture_all_divs(store):
    config, cid_a, _ = store
    cpath = resolve_paths(cid_a, config)
    assert isinstance(cpath, CulturePaths)
    assert cpath.divs == [7, 8]
    assert cpath.recording(7).npz.exists()
    assert cpath.recording(7).burst_stats.exists()


def test_resolve_without_burst_data(store):
    """A store of freshly preprocessed recordings has no burst CSVs yet: resolution still succeeds,
    with ``burst_stats=None``, and analyses that need bursts say so clearly."""
    import shutil

    config, cid_a, _ = store
    shutil.rmtree(config.burst_data_dir)

    # Built the way a user would, straight from a registry row (numpy-typed well).
    row = pd.read_csv(config.registry_path).iloc[0]
    culture = CultureID(row["batch_id"], row["chip"], row["well"])
    assert culture == cid_a

    cpath = resolve_paths(culture, config)
    assert cpath.divs == [7, 8]
    assert all(r.npz.exists() for r in cpath.recordings)
    assert all(r.burst_stats is None for r in cpath.recordings)
    with pytest.raises(FileNotFoundError, match="run burst detection"):
        cpath.recording(7).require_burst_stats()


def test_resolve_ignores_activity_scan_rows(store):
    """The registry also indexes scans -- raw .h5 files with no .npz behind them.

    Counting those rows would invent DIVs (and whole cultures) that resolve to preprocessed paths
    which do not exist.
    """
    config, cid_a, _ = store

    class _ScanParams:
        batch_id, chip, div = "expA", "C0001", 21  # a DIV culture A has no recording for
        wells, conditions = [0], []

    io.register_scan(_ScanParams(), config.registry_path)
    assert len(pd.read_csv(config.registry_path)) == 4  # the scan row really is there

    assert resolve_paths(cid_a, config).divs == [7, 8]
    assert len(resolve_paths_flat(CultureSelector(batch_ids=["expA"]), config)) == 2


def test_resolve_reads_a_registry_written_before_scans_were_indexed(store):
    """No `kind` column at all: every row is a recording, and resolution is unchanged."""
    config, cid_a, _ = store
    df = pd.read_csv(config.registry_path).drop(columns=["kind"])
    df.to_csv(config.registry_path, index=False)

    assert resolve_paths(cid_a, config).divs == [7, 8]


def _add_labelled(config, make_recording_data, label, div, seed=4, batch="expA", chip="C0001",
                  with_bursts=False):
    """Write and register one labelled recording (an ingested experiment's) of culture ``chip``/well0.

    ``with_bursts`` also runs burst detection, for tests that analyse the recording.
    """
    out_dir = config.preprocessed_dir / batch / chip / "well0"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"DIV{div}_250101_{chip}_{batch}_well0_{label}_exp_data.npz"
    np.savez_compressed(npz, **make_recording_data(seed=seed, batch_id=batch, chip=chip, DIV=div, experiment=label))
    if with_bursts:
        rec = Recording(0, io.load_preprocessed(npz))
        bursts = BurstDetector(DETECT).detect(rec)
        bursts.extract_features(rec, BurstFeatureParams())
        io.save_burst_data(bursts, config.burst_data_dir, rec)
    io.register({0: {"batch_id": batch, "experiment": label, "chip": chip, "DIV": div}}, config.registry_path)
    return npz


def test_resolve_finds_labelled_recordings_without_being_told_the_label(store, make_recording_data):
    """Every recording carries a label and each DIV has one: the default resolves them all."""
    config, _, _ = store
    culture = CultureID("expA", "C0009", "0")
    ns = {div: _add_labelled(config, make_recording_data, "NS", div, chip="C0009") for div in (12, 15)}
    atp = _add_labelled(config, make_recording_data, "NS_ATP1hr", 27, chip="C0009")

    cpath = resolve_paths(culture, config)
    assert cpath.divs == [12, 15, 27]
    assert cpath.recording(15).npz == ns[15] and cpath.recording(27).npz == atp
    assert cpath.recording(27).recording_id.experiment == "NS_ATP1hr"

    from mxtreme.identity import RecordingID

    assert resolve_paths(RecordingID("expA", "C0009", "0", 15), config).npz == ns[15]
    assert resolve_paths(culture, config, experiment="NS").divs == [12, 15]
    assert len(resolve_paths(culture, config, experiment="").recordings) == 0


def test_resolve_paths_keeps_every_recording_when_a_div_has_several(store, make_recording_data):
    """Two labels on one DIV both resolve, side by side; picking one takes a label."""
    config, cid_a, _ = store  # culture A already has an unlabelled recording on DIV 7
    npz = _add_labelled(config, make_recording_data, "stim1", 7)

    cpath = resolve_paths(cid_a, config)
    assert [(r.recording_id.div, r.recording_id.experiment) for r in cpath.recordings] == [
        (7, ""), (7, "stim1"), (8, "")
    ]
    assert cpath.divs == [7, 8] and cpath.experiments == ["", "stim1"]
    assert len(cpath.at_div(7)) == 2
    with pytest.raises(ValueError, match=r"DIV 7 \('' \(unlabelled\), 'stim1'\)"):
        cpath.recording(7)
    assert cpath.recording(7, "stim1").npz == npz
    assert cpath.recording(7, "").npz != npz
    with pytest.raises(KeyError):
        cpath.recording(7, "nope")

    # The same through a selector -- the call that used to raise.
    batch = resolve_paths(CultureSelector(batch_ids=["expA"]), config)
    assert len(batch["expA"].cultures[cid_a].recordings) == 3

    # A bare RecordingID on that DIV is still ambiguous.
    from mxtreme.identity import RecordingID

    with pytest.raises(ValueError, match=r"DIV 7: '' \(unlabelled\), 'stim1'"):
        resolve_paths(RecordingID("expA", "C0001", "0", 7), config)
    assert resolve_paths(RecordingID("expA", "C0001", "0", 7, "stim1"), config).npz == npz

    scans = resolve_paths(cid_a, config, experiment="")
    assert scans.divs == [7, 8] and scans.recording(7).npz != npz
    stim = resolve_paths(cid_a, config, experiment="stim1")
    assert stim.divs == [7] and stim.recording(7).npz == npz
    assert resolve_paths_flat(CultureSelector(experiment="stim1"), config).npz == [npz]

    everything = resolve_paths_flat(cid_a, config)
    assert [(r.div, r.experiment) for r in everything.ids] == [(7, ""), (7, "stim1"), (8, "")]
    assert everything.to_frame()["experiment"].tolist() == ["", "stim1", ""]


ATP_LABELS = ("NS_ATP12hr", "NS_ATP1hr", "NS_ATP6hr")


@pytest.fixture
def replicate_store(store, make_recording_data):
    """Culture ``expA/C0009/well0``: an unlabelled-free culture with DIVs 12 and 27, where DIV 27 has
    three differently labelled recordings -- the shape of a timed-treatment experiment."""
    config, _, _ = store
    _add_labelled(config, make_recording_data, "NS", 12, seed=5, chip="C0009", with_bursts=True)
    for seed, label in enumerate(ATP_LABELS, start=6):
        _add_labelled(config, make_recording_data, label, 27, seed=seed, chip="C0009", with_bursts=True)
    return config, CultureID("expA", "C0009", "0")


def test_selector_over_a_batch_with_replicates_on_a_div_resolves(replicate_store):
    """The reported failure: a batch selector over a culture with several recordings on one DIV."""
    config, culture = replicate_store

    batch = resolve_paths(CultureSelector(batch_ids=["expA"]), config)
    cpath = batch["expA"].cultures[culture]

    assert cpath.divs == [12, 27]
    assert [r.recording_id.experiment for r in cpath.at_div(27)] == list(ATP_LABELS)
    assert cpath.at_div(27).to_frame()["experiment"].tolist() == list(ATP_LABELS)


def test_summaries_keep_replicates_on_a_div_apart(replicate_store):
    config, culture = replicate_store

    for df in (
        activity.summarize_spike_activity(culture, config),
        activity.summarize_burst_activity(culture, config),
        network.summarize_network_metrics(culture, config),
        performance.summarize_performance(culture, config),
    ):
        at_27 = df[df["div"] == 27]
        assert sorted(at_27["experiment"].unique()) == list(ATP_LABELS)
        assert set(df.loc[df["div"] == 12, "experiment"]) == {"NS"}

    # Reloading from the cache gives the same labelled rows back.
    cached = activity.summarize_spike_activity(culture, config)
    assert sorted(cached.loc[cached["div"] == 27, "experiment"].unique()) == list(ATP_LABELS)

    for distributions in (activity.spike_activity_distributions, activity.burst_activity_distributions):
        dists = distributions(culture, config)
        assert sorted(dists) == [(12, "NS")] + [(27, label) for label in ATP_LABELS]
        assert sorted(distributions(culture, config)) == sorted(dists)


def test_aggregation_groups_by_experiment():
    df = pd.DataFrame({
        "culture_id": ["a", "b", "a", "b"],
        "div": [27, 27, 27, 27],
        "experiment": ["NS_ATP1hr", "NS_ATP1hr", "NS_ATP6hr", "NS_ATP6hr"],
        "phase": ["full"] * 4,
        "x": [1.0, 3.0, 10.0, 30.0],
    })
    stats = aggregate_by_div_phase(df, value_cols=["x"]).set_index("experiment")
    assert stats.loc["NS_ATP1hr", "x"] == 2.0 and stats.loc["NS_ATP6hr", "x"] == 20.0

    # A frame with no experiment column (or unlabelled NaNs read back from CSV) is one group.
    legacy = aggregate_by_div_phase(df.drop(columns="experiment"), value_cols=["x"])
    assert len(legacy) == 1 and legacy.iloc[0]["experiment"] == ""
    unlabelled = aggregate_by_div_phase(df.assign(experiment=np.nan), value_cols=["x"])
    assert len(unlabelled) == 1


def test_summaries_recompute_a_cache_without_experiment(store):
    """A summary cached before recordings carried labels can't tell replicates apart: recompute it."""
    config, cid_a, _ = store
    cpath = resolve_paths(cid_a, config)
    fresh = activity.summarize_spike_activity(cid_a, config)

    _, csv_path = _summary_paths(cpath, config.analysis_dir, "activity", "spike_activity_summary")
    stale = fresh.drop(columns="experiment").assign(mean_fr_hz=-1.0)
    stale.to_csv(csv_path, index=False)

    df = activity.summarize_spike_activity(cid_a, config)
    assert "experiment" in df.columns
    assert (df["mean_fr_hz"] != -1.0).all()


def test_distributions_recompute_a_div_keyed_cache(store):
    config, cid_a, _ = store
    cpath = resolve_paths(cid_a, config)
    _, npz_path = _summary_paths(cpath, config.analysis_dir, "distributions",
                                 "default_spike_distributions", ext=".npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **{"7__fr_hz": np.array([1.0])})  # the old "<div>__<key>" layout

    dists = activity.spike_activity_distributions(cid_a, config)
    assert sorted(dists) == [(7, ""), (8, "")]


def test_plots_and_report_render_with_replicates(replicate_store, tmp_path):
    config, culture = replicate_store

    for split_by in ("phase", "experiment"):
        activity.plot_spike_activity_summary(culture, config, split_by=split_by, show_plot=False,
                                             save_path=tmp_path / f"culture_{split_by}.png")
        activity.plot_burst_activity_summary(CultureSelector(cultures=[culture]), config,
                                             split_by=split_by, show_plot=False,
                                             save_path=tmp_path / f"selection_{split_by}.png")
        assert (tmp_path / f"culture_{split_by}.png").exists()
        assert (tmp_path / f"selection_{split_by}.png").exists()

    network.plot_connectivity_grid(culture, config, save_path=tmp_path / "g.png", show_plot=False)
    network.plot_pc_variance_explained(culture, config, save_path=tmp_path / "v.png", show_plot=False)
    assert (tmp_path / "g.png").exists() and (tmp_path / "v.png").exists()

    out = generate_report(culture, config, sections=("activity", "bursting"))
    assert out.exists() and out.stat().st_size > 0


def test_resolve_paths_flat_flattens_a_selection(store):
    config, cid_a, cid_b = store
    recs = resolve_paths_flat(CultureSelector(cultures=[cid_a, cid_b]), config)

    # Culture A has 2 DIVs, culture B has 1; ordered by batch_id, chip, well, then DIV.
    assert len(recs) == 3
    assert [r.recording_id.div for r in recs] == [7, 8, 7]
    assert [r.recording_id.batch_id for r in recs] == ["expA", "expA", "expB"]
    assert recs.npz == [r.npz for r in recs]
    assert all(p.exists() for p in recs.npz)
    assert all(p.exists() for p in recs.burst_stats)
    assert recs[0] is recs.recordings[0]


def test_resolve_paths_flat_accepts_any_target(store):
    config, cid_a, _ = store
    assert len(resolve_paths_flat(cid_a, config)) == 2          # CultureID -> all DIVs
    rid = resolve_paths_flat(cid_a, config).ids[0]
    assert len(resolve_paths_flat(rid, config)) == 1            # RecordingID -> just that one


def test_resolve_paths_flat_to_frame(store):
    config, cid_a, cid_b = store
    df = resolve_paths_flat(CultureSelector(cultures=[cid_a, cid_b]), config).to_frame()

    assert list(df.columns) == ["batch_id", "chip", "well", "div", "experiment", "npz", "burst_stats"]
    assert len(df) == 3
    assert df["div"].tolist() == [7, 8, 7]
    assert set(df["batch_id"]) == {"expA", "expB"}


def test_resolve_paths_flat_respects_div_filter(store):
    config, cid_a, cid_b = store
    recs = resolve_paths_flat(CultureSelector(cultures=[cid_a, cid_b], divs=[8]), config)
    assert [r.recording_id.div for r in recs] == [8]


def test_resolve_group_across_experiments(store):
    config, cid_a, cid_b = store
    resolved = resolve_paths(CultureSelector(cultures=[cid_a, cid_b]), config)
    assert set(resolved) == {"expA", "expB"}
    assert len(resolved["expA"].cultures) == 1
    assert len(resolved["expB"].cultures) == 1


def test_spike_activity_summary_columns(store):
    config, cid_a, _ = store
    df = activity.summarize_spike_activity(cid_a, config)
    for col in ("batch_id", "culture_id", "chip", "well", "div", "experiment", "phase",
                "mean_fr_hz", "median_isi_msec", "mean_amp_uv", "pct_active_chan"):
        assert col in df.columns
    assert set(df["div"]) == {7, 8}
    assert set(df["phase"]) == {"full"}


def test_burst_activity_summary_columns(store):
    config, cid_a, _ = store
    df = activity.summarize_burst_activity(cid_a, config)
    for col in ("culture_id", "div", "phase", "burst_rate_hz", "median_ibi_sec", "median_size_pct", "median_dur_sec"):
        assert col in df.columns
    # Burst rate is bursts / phase-seconds and must be finite & non-negative.
    assert (df["burst_rate_hz"].dropna() >= 0).all()


def test_summaries_honor_embedded_phase_spec(tmp_path, make_recording_data):
    # A recording whose npz carries a phase spec is split into those phases by the analysis with no
    # make_phases / phase_tags anywhere. The fixture has dense bursts near frames 100k and 300k
    # (10 kHz), so a split at ~0.333 min (200k frames) puts a burst in each phase.
    config = Config(data_root=tmp_path)
    spec = {"early": 0.0, "late": {"start": 0.333, "end": 0.833}}
    data = make_recording_data(
        seed=1, batch_id="expP", chip="C0009", well=0, DIV=7,
        phase_spec=np.asarray(spec, dtype=object),
    )
    _add_recording(config, data, well_no=0)

    culture = CultureID("expP", "C0009", "0")

    ch = activity.summarize_spike_activity(culture, config)
    assert set(ch["phase"]) == {"early", "late"}

    bu = activity.summarize_burst_activity(culture, config)
    assert set(bu["phase"].dropna()) <= {"early", "late"}
    assert (bu["burst_rate_hz"].dropna() >= 0).all()


def test_performance_score_default_objective(store):
    config, cid_a, _ = store
    rec, bursts = _fixture_recording(config, cid_a)

    df = performance.performance_score(rec, bursts, config.analysis_dir)
    assert list(df.columns) == ["phase", "n_bursts", "condition", "trained_side", "score"]
    assert 0 <= df.iloc[0]["score"] <= 1
    # Culture A is [2, 0]: it is scored against the right-hand target.
    assert df["trained_side"].tolist() == ["right"] and df["condition"].tolist() == ["[2, 0]"]
    assert list((config.analysis_dir / "performance").rglob("*_performance_default_direction_objective.csv"))

    table = performance.summarize_performance(cid_a, config)
    assert set(table["div"]) == {7, 8} and set(table["trained_side"]) == {"right"}
    scores = table["score"].dropna()
    assert ((scores >= 0) & (scores <= 1)).all()


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
    assert performance._trained_side(stored) is expected


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


def test_performance_caches_each_objective_separately(store, tmp_path):
    config, cid_a, _ = store

    def constant_objective(bursts):
        return 0.42

    default = performance.summarize_performance(cid_a, config)
    custom = performance.summarize_performance(cid_a, config, objective_fn=constant_objective)
    assert (custom["score"] == 0.42).all()
    assert not (default["score"] == 0.42).any()  # the custom run didn't overwrite the default cache
    assert (performance.summarize_performance(cid_a, config)["score"].values == default["score"].values).all()

    performance.plot_performance_summary({"A": CultureSelector(cultures=[cid_a])}, config,
                                         objective_fn=constant_objective, show_plot=False,
                                         save_path=tmp_path / "perf.png")
    assert (tmp_path / "perf.png").exists()


def test_generate_report_single_culture(store):
    config, cid_a, _ = store
    out = generate_report(cid_a, config, sections=("overview", "activity", "bursting", "performance"))
    assert out.exists() and out.stat().st_size > 0


def test_generate_report_group_across_experiments(store):
    config, cid_a, cid_b = store
    out = generate_report(
        CultureSelector(cultures=[cid_a, cid_b]),
        config,
        sections=("overview", "activity", "bursting", "cultures"),
    )
    assert out.exists() and out.stat().st_size > 0


def test_generate_report_name_keeps_the_default_directory(store):
    config, cid_a, _ = store
    out = generate_report(cid_a, config, sections=("activity",), name="ATP dose")
    assert out == config.analysis_dir / "reports" / "ATP dose.pdf"
    assert out.exists()
    assert generate_report(cid_a, config, sections=("activity",), name="again.pdf").name == "again.pdf"


@pytest.mark.parametrize("kwargs", [
    {"name": "a", "output_path": "elsewhere/a.pdf"},
    {"name": "sub/dir"},
    {"name": ""},
])
def test_generate_report_rejects_a_bad_name(store, kwargs):
    config, cid_a, _ = store
    with pytest.raises(ValueError):
        generate_report(cid_a, config, sections=("activity",), **kwargs)


def _fake_culture(batch, chip, well, divs, experiments=("",)):
    from types import SimpleNamespace as NS
    return NS(culture_id=NS(batch_id=batch, chip=chip, well=str(well)),
              recordings=[NS(recording_id=NS(div=d, experiment=e)) for d in divs for e in experiments])


def test_report_overview_compacts_divs_and_labels():
    from mxtreme.analysis import report

    assert report._div_ranges([40, 32, 33, 34, 38, 41, 33]) == "32–34, 38, 40–41"
    (row,) = report._overview_rows([_fake_culture("b", "M07459", 2, [7, 8, 9],
                                                  ["NS_ATP12hr", "NS_ATP1hr", ""])])
    assert row[1:] == ["M07459", "2", "MaxTwo", "9", "7–9", "NS_ATP1hr, NS_ATP12hr"]


def test_report_overview_paginates_without_clipping():
    """A large selection with long labels spills onto continuation pages instead of off the page, and
    each page names the batch on its first row."""
    import matplotlib.text
    from mxtreme.analysis import report

    cultures = [_fake_culture(f"batch_with_a_rather_long_name_{b}", f"M0{chip}", well,
                              [*range(28, 36), 38, 41, 45], ["NS_ATP1hr", "NS_ATP12hr", "NS_ATP24hr_x"])
                for b in range(2) for chip in range(4) for well in range(6)]
    figs = report._overview_figures("Title", ["Generated: today", "Cultures: 48"], cultures)
    try:
        assert len(figs) > 1
        n_rows = 0
        for fig in figs:
            fig.canvas.draw()
            renderer, page = fig.canvas.get_renderer(), fig.bbox
            for text in fig.findobj(matplotlib.text.Text):
                if not text.get_text():
                    continue
                box = text.get_window_extent(renderer)
                assert page.x0 - 1 <= box.x0 and box.x1 <= page.x1 + 1, text.get_text()
                assert page.y0 - 1 <= box.y0 and box.y1 <= page.y1 + 1, text.get_text()
            (table,) = fig.axes[0].tables
            cells = table.get_celld()
            body_rows = sorted({r for r, _c in cells if r > 0})
            assert cells[(1, 0)].get_text().get_text()  # batch named on each page's first row
            n_rows += len(body_rows)
        assert n_rows == len(cultures)
    finally:
        for fig in figs:
            plt.close(fig)


def test_metric_grid_tells_experiment_labels_apart_and_drops_full():
    from mxtreme.analysis._plotting import plot_metric_grid

    rows = [dict(culture_id=f"c{c}", chip="M0", well=c, div=d, experiment=e, phase="full", a=float(c + d))
            for c in range(3) for e in ("NS_ATP12hr", "NS_ATP1hr") for d in (30, 31)]
    df = pd.DataFrame(rows)
    fig = plot_metric_grid(aggregate_by_div_phase(df, ["a"]), [("a", "sem_a", "A", "A")],
                           overlay_df=df, show_plot=False, ncols=1)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "NS_ATP1hr (mean ± SEM)" in labels and "NS_ATP12hr (mean ± SEM)" in labels
    assert not any("Full" in label for label in labels)

    bold = [c for c in ax.containers if c.get_label().endswith("(mean ± SEM)")]
    markers = {c.get_label(): c.lines[0].get_marker() for c in bold}
    colors = {c.get_label(): c.lines[0].get_color() for c in bold}
    assert len(set(markers.values())) == 2 and len(set(colors.values())) == 2
    plt.close(fig)


def test_plot_summary_title_drops_an_unphased_phase(store):
    config, cid_a, cid_b = store
    fig = activity.plot_spike_activity_summary({"A": CultureSelector(cultures=[cid_a]),
                                                "B": CultureSelector(cultures=[cid_b])},
                                               config, show_plot=False)
    assert "phase" not in fig._suptitle.get_text()
    plt.close(fig)


# --- distributions and per-culture pages -----------------------------------------------------------


def test_isi_respects_the_isi_threshold(store):
    config, cid_a, _ = store
    rec = Recording(0, io.load_preprocessed(resolve_paths(cid_a, config).recording(7).npz))

    # The fixture's tonic background sits ~5000 frames (500 ms) apart, so a tight threshold keeps only
    # the dense within-burst intervals while a generous one keeps both populations.
    tight = activity.isi(rec, isi_threshold_ms=100, save=False)["isi_ms"]
    loose = activity.isi(rec, isi_threshold_ms=10_000, save=False)["isi_ms"]

    assert len(tight) < len(loose)
    assert (tight < 100).all()


def test_distributions_keys_and_cache(store):
    config, cid_a, _ = store

    for distributions, keys in ((activity.spike_activity_distributions, activity.SPIKE_DISTRIBUTION_KEYS),
                                (activity.burst_activity_distributions, activity.BURST_DISTRIBUTION_KEYS)):
        dists = distributions(cid_a, config)

        assert sorted(dists) == [(7, ""), (8, "")]
        for per_rec in dists.values():
            assert set(per_rec) == set(keys)
            for key, values in per_rec.items():
                assert np.isfinite(values).all(), key

        # The npz cache round-trips to the same arrays.
        cached = distributions(cid_a, config)
        for rec_key, per_rec in dists.items():
            for key, values in per_rec.items():
                assert np.array_equal(cached[rec_key][key], values)

    # Firing rates are per electrode, so they must not be scalars.
    assert len(activity.spike_activity_distributions(cid_a, config)[(7, "")]["fr_hz"]) > 1

    with pytest.raises(TypeError, match="per culture"):
        activity.spike_activity_distributions(CultureSelector(cultures=[cid_a]), config)


def test_population_plot_helpers_render(store, tmp_path):
    # Every population plot goes through the shared metric grid; this exercises that path.
    config, cid_a, cid_b = store
    sel = CultureSelector(cultures=[cid_a, cid_b])
    activity.plot_spike_activity_summary(sel, config, show_plot=False, save_path=tmp_path / "spike.png")
    activity.plot_burst_activity_summary(sel, config, error="std", show_plot=False,
                                         save_path=tmp_path / "burst.png")
    performance.plot_performance_summary(sel, config, show_plot=False, save_path=tmp_path / "perf.png")

    assert (tmp_path / "spike.png").exists()
    assert (tmp_path / "burst.png").exists()
    assert (tmp_path / "perf.png").exists()


def test_pooled_summaries_carry_culture_identity(store):
    config, cid_a, cid_b = store
    sel = CultureSelector(cultures=[cid_a, cid_b])
    sel_paths = resolve_paths(sel, config)

    summary = activity.summarize_spike_activity(sel, config)
    assert set(summary["culture_id"]) == {str(cid_a), str(cid_b)}

    pop_df = load_population_summaries(
        sel_paths, data_dir=config.analysis_dir / "activity", suffix="spike_activity_summary"
    )

    assert {"culture_id", "chip", "well"} <= set(pop_df.columns)
    assert set(pop_df["chip"]) == {"C0001", "C0002"}
    assert set(pop_df["culture_id"]) == {str(cid_a), str(cid_b)}


# --- activity: recording level ---------------------------------------------------------------------


def _fixture_recording(config, cid):
    rp = resolve_paths(cid, config).recording(7)
    return Recording(0, io.load_preprocessed(rp.npz)), pd.read_csv(rp.require_burst_stats())


def test_firing_rate_is_count_over_phase_duration(store, make_recording_data):
    config, cid_a, _ = store
    rec, _ = _fixture_recording(config, cid_a)

    df = activity.firing_rate(rec, config.analysis_dir)

    phase = rec.phases.phases[0]
    duration = phase.n_frames / rec.samp_rate
    counts = pd.Series(rec.spike_data["channel"]).value_counts()
    assert set(df["phase"]) == {"full"}
    assert len(df) == len(np.unique(rec.channelmap[:, 1]))
    for row in df.itertuples():
        assert row.n_spikes == counts.get(row.channel, 0)
        assert row.firing_rate_hz == pytest.approx(row.n_spikes / duration)

    # Saved next to the culture's summaries, named after the recording.
    saved = list((config.analysis_dir / "activity" / "expA" / "C0001" / "well0").glob("DIV7_*_firing_rate.csv"))
    assert len(saved) == 1

    # A mapped channel that never fires is listed at 0 Hz rather than dropped.
    data = make_recording_data(seed=1)
    data["channelmap"] = np.vstack([data["channelmap"], [[20, 99, 999, 0.0, 0.0]]])
    silent = activity.firing_rate(Recording(0, data), save=False).set_index("channel")
    assert silent.loc[99, "n_spikes"] == 0 and silent.loc[99, "firing_rate_hz"] == 0


def test_instantaneous_firing_rate_conserves_spikes(store):
    config, cid_a, _ = store
    rec, _ = _fixture_recording(config, cid_a)

    ifr = activity.instantaneous_firing_rate(rec, config.analysis_dir, bin_size_sec=2.0)

    n_channels = len(np.unique(rec.channelmap[:, 1]))
    assert ifr["rate"].shape == (n_channels, len(ifr["time_sec"]))
    assert ifr["rate"].sum() * 2.0 == pytest.approx(len(rec.spike_data))
    assert ifr["time_sec"][-1] + 2.0 >= rec.rec_t_sec
    assert [p[0] for p in ifr["phases"]] == ["full"]
    assert list((config.analysis_dir / "activity").rglob("*_instantaneous_firing_rate_2s.npz"))


def test_isi_levels(store):
    config, cid_a, _ = store
    rec, _ = _fixture_recording(config, cid_a)

    per_channel = activity.isi(rec, isi_threshold_ms=np.inf, save=False)
    merged = activity.isi(rec, level="global", isi_threshold_ms=np.inf, save=False)

    assert {"channel", "phase", "isi_ms"} == set(per_channel.columns)
    assert {"phase", "isi_ms"} == set(merged.columns)
    # Within-channel intervals: one fewer than the channel's spikes, for every channel.
    n_spikes = pd.Series(rec.spike_data["channel"]).value_counts()
    assert len(per_channel) == int((n_spikes - 1).sum())
    # Merging every channel into one train leaves one interval per spike (less one), each no longer
    # than the gap to that spike's own channel predecessor -- so the array-wide ISI is shorter.
    assert len(merged) == len(rec.spike_data) - 1
    assert merged["isi_ms"].median() <= per_channel["isi_ms"].median()
    assert (merged["isi_ms"] >= 0).all()

    with pytest.raises(ValueError, match="level"):
        activity.isi(rec, level="array", save=False)


def test_burst_rate_and_ibi(store):
    config, cid_a, _ = store
    rec, bursts = _fixture_recording(config, cid_a)

    rate = activity.burst_rate(rec, bursts, config.analysis_dir)
    n_network = int((bursts["kind"] == "network").sum())
    assert rate.loc[0, "n_bursts"] == n_network
    assert rate.loc[0, "burst_rate_hz"] == pytest.approx(n_network / rate.loc[0, "duration_sec"])

    intervals = activity.ibi(rec, bursts, config.analysis_dir)
    peaks = np.sort(bursts.loc[bursts["kind"] == "network", "peak_frame"].to_numpy()) / rec.samp_rate
    assert np.allclose(intervals["ibi_sec"].to_numpy(), np.diff(peaks))
    assert list((config.analysis_dir / "activity").rglob("*_burst_rate.csv"))
    assert list((config.analysis_dir / "activity").rglob("*_ibi.csv"))


def test_recording_functions_do_not_save_without_an_analysis_dir(store):
    config, cid_a, _ = store
    rec, bursts = _fixture_recording(config, cid_a)
    activity.firing_rate(rec)
    activity.burst_rate(rec, bursts)
    assert not (config.analysis_dir / "activity").exists()


# --- activity: group level --------------------------------------------------------------------------


def test_summarize_at_every_target_level(store):
    config, cid_a, cid_b = store
    rid = resolve_paths(cid_a, config).recording(8).recording_id

    one = activity.summarize_spike_activity(rid, config)
    assert one["div"].tolist() == [8] and "group" not in one.columns

    culture = activity.summarize_spike_activity(cid_a, config)
    assert sorted(culture["div"]) == [7, 8]

    sel = CultureSelector(cultures=[cid_a, cid_b])
    assert activity.summarize_burst_activity(sel, config)["culture_id"].nunique() == 2

    groups = activity.summarize_burst_activity(
        {"A": CultureSelector(cultures=[cid_a]), "B": CultureSelector(cultures=[cid_b])}, config
    )
    assert groups.groupby("group")["culture_id"].unique().to_dict() == {"A": [str(cid_a)], "B": [str(cid_b)]}
    listed = activity.summarize_burst_activity([CultureSelector(cultures=[cid_a])], config)
    assert set(listed["group"]) == {"Group 1"}

    with pytest.raises(ValueError, match="No data for phase"):
        activity.summarize_spike_activity(cid_a, config, phase="nope")
    with pytest.raises(TypeError):
        activity.summarize_spike_activity("expA", config)


def test_summary_cache_fills_in_recordings_incrementally(store):
    """Summarising one recording caches only it; the culture later reuses it and adds the rest."""
    config, cid_a, _ = store
    cpath = resolve_paths(cid_a, config)
    _, csv_path = _summary_paths(cpath, config.analysis_dir, "activity", "spike_activity_summary")

    activity.summarize_spike_activity(cpath.recording(7).recording_id, config)
    assert sorted(pd.read_csv(csv_path)["div"]) == [7]

    activity.summarize_spike_activity(cid_a, config)
    assert sorted(pd.read_csv(csv_path)["div"]) == [7, 8]


def test_plot_summary_at_every_target_level(store, tmp_path):
    config, cid_a, cid_b = store
    rid = resolve_paths(cid_a, config).recording(7).recording_id
    targets = {
        "recording": rid,
        "culture": cid_a,
        "selection": CultureSelector(cultures=[cid_a, cid_b]),
        "groups": {"A": CultureSelector(cultures=[cid_a]), "B": CultureSelector(cultures=[cid_b])},
    }
    for name, target in targets.items():
        for plot in (activity.plot_spike_activity_summary, activity.plot_burst_activity_summary):
            out = tmp_path / f"{name}_{plot.__name__}.png"
            fig = plot(target, config, show_plot=False, save_path=out)
            assert fig is not None and out.exists(), (name, plot.__name__)

    with pytest.raises(ValueError, match="split_by"):
        activity.plot_spike_activity_summary(cid_a, config, split_by="chip", show_plot=False)
    with pytest.raises(ValueError, match="error"):
        activity.plot_spike_activity_summary(cid_a, config, error="ci", show_plot=False)


def test_group_comparison_plots_one_series_per_group(store):
    config, cid_a, cid_b = store
    fig = activity.plot_spike_activity_summary(
        {"Left": CultureSelector(cultures=[cid_a]), "Right": CultureSelector(cultures=[cid_b])},
        config, show_plot=False,
    )
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["Left", "Right"]


def test_distribution_plots_render(store, tmp_path):
    config, cid_a, _ = store
    activity.plot_spike_activity_distributions(cid_a, config, show_plot=False, save_path=tmp_path / "s.png")
    activity.plot_burst_activity_distributions(cid_a, config, show_plot=False, save_path=tmp_path / "b.png")
    assert (tmp_path / "s.png").exists() and (tmp_path / "b.png").exists()


def test_aggregation_can_report_sd():
    df = pd.DataFrame({"div": [7, 7], "phase": ["full"] * 2, "x": [1.0, 3.0]})
    stats = aggregate_by_div_phase(df, value_cols=["x"], stat="std")
    assert stats.loc[0, "std_x"] == pytest.approx(np.std([1.0, 3.0], ddof=1))
    with pytest.raises(ValueError):
        aggregate_by_div_phase(df, value_cols=["x"], stat="var")


# --- phase selection -------------------------------------------------------------------------------


def _phased_store(tmp_path, make_recording_data):
    """A one-culture store whose recording carries an early/late phase spec."""
    config = Config(data_root=tmp_path)
    spec = {"early": 0.0, "late": {"start": 0.333, "end": 0.833}}
    _add_recording(
        config,
        make_recording_data(seed=1, batch_id="expQ", chip="C0014", well=0, DIV=7,
                            phase_spec=np.asarray(spec, dtype=object)),
        well_no=0,
    )
    return config, CultureID("expQ", "C0014", "0")


def test_generate_report_defaults_to_the_first_phase(tmp_path, make_recording_data):
    config, cid = _phased_store(tmp_path, make_recording_data)

    out = generate_report(cid, config, sections=("activity", "cultures"))

    assert out.exists() and out.stat().st_size > 0
    # The default phase is the first of the sequence, so that phase's distributions are what got cached.
    assert (config.analysis_dir / "distributions" / "expQ" / "C0014" / "well0" /
            f"{cid}_early_spike_distributions.npz").exists()


def test_generate_report_honors_an_explicit_phase(tmp_path, make_recording_data):
    config, cid = _phased_store(tmp_path, make_recording_data)

    out = generate_report(cid, config, sections=("activity", "cultures"), phase="late",
                          output_path=config.analysis_dir / "reports" / "late.pdf")

    assert out.exists() and out.stat().st_size > 0
    assert (config.analysis_dir / "distributions" / "expQ" / "C0014" / "well0" /
            f"{cid}_late_spike_distributions.npz").exists()


def test_generate_report_rejects_a_phase_the_data_lacks(store):
    config, cid_a, _ = store
    with pytest.raises(ValueError, match="No data for phase"):
        generate_report(cid_a, config, sections=("activity",), phase="nonexistent")


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
        seed=1, batch_id="expS", chip=chip, well=0, DIV=7,
        eventtime=eventtime, event_messages=messages, **extra,
    )
    _add_recording(config, data, well_no=0)
    return config, CultureID("expS", chip, "0")


def _stim_recording(config, culture):
    return Recording(0, io.load_preprocessed(resolve_paths(culture, config).recording(7).npz))


STIM_SPEC = {"pre": 0.0, "train": 1 / 6, "post": {"start": 0.5, "end": 0.833}}


def test_stim_delivered_attributes_stim_to_the_phase_it_falls_in(tmp_path, make_recording_data):
    # The fixture's spikes start at ~frame 3000, so the phases run pre [3k, 103k), train [103k, 303k),
    # post [303k, ...). The five stim pulses sit at 50k..250k, i.e. two in pre and three in train.
    config, culture = _stim_store(tmp_path, make_recording_data, chip="C0010", phase_spec=STIM_SPEC)
    rec = _stim_recording(config, culture)

    events = stimulation.stim_events(rec, config.analysis_dir)
    assert events["phase"].tolist() == ["pre", "pre", "train", "train", "train"]
    assert (events["pulse_phase_us"] == 200.0).all()

    df = stimulation.stim_delivered(rec, config.analysis_dir).set_index("phase")
    assert df.loc[["pre", "train", "post"], "n_pulses"].tolist() == [2, 3, 0]
    assert df.loc["pre", "total_stim_ms"] == pytest.approx(2 * 200.0 / 1000)
    assert df.loc["train", "total_stim_ms"] == pytest.approx(3 * 200.0 / 1000)
    assert df.loc["post", "total_stim_ms"] == pytest.approx(0.0)
    assert df.loc["train", "phase_dur_min"] == pytest.approx(1 / 3, abs=1e-3)
    assert list((config.analysis_dir / "stimulation").rglob("*_stim_delivered.csv"))

    # The group-level table agrees, and carries identity columns.
    table = stimulation.summarize_stimulation(culture, config).set_index("phase")
    assert table.loc["train", "total_stim_ms"] == pytest.approx(0.6)
    assert table["culture_id"].unique().tolist() == [str(culture)]


def test_stim_delivered_unphased_recording_is_a_single_full_row(tmp_path, make_recording_data):
    # No phase spec -> a single "full" phase spanning the recording.
    config, culture = _stim_store(tmp_path, make_recording_data, chip="C0011")
    rec = _stim_recording(config, culture)
    span_min = rec.phases.phases[0].n_frames / rec.samp_rate / 60

    df = stimulation.stim_delivered(rec, save=False)

    assert len(df) == 1
    assert df.iloc[0]["phase"] == "full"
    assert df.iloc[0]["phase_dur_min"] == pytest.approx(span_min)
    assert df.iloc[0]["total_stim_ms"] == pytest.approx(5 * 200.0 / 1000)


def test_stim_events_tolerate_non_dict_event_messages(tmp_path, make_recording_data):
    # Older stores hold plain strings alongside the dict messages; those must be skipped, not raise.
    eventtime = np.array([1000, 50000, 90000], dtype="<i8")
    messages = np.array(
        ["legacy string", {"start_stimulation": "1", "phase_us": 100.0}, {"other_event": "1"}],
        dtype=object,
    )
    config, culture = _stim_store(tmp_path, make_recording_data, chip="C0012",
                                  eventtime=eventtime, messages=messages)
    rec = _stim_recording(config, culture)

    assert stimulation.stim_events(rec, save=False)["frame"].tolist() == [50000]
    assert stimulation.stim_delivered(rec, save=False).iloc[0]["total_stim_ms"] == pytest.approx(0.1)


def test_generate_report_with_stimulation_section(tmp_path, make_recording_data):
    config, culture = _stim_store(tmp_path, make_recording_data, chip="C0013", phase_spec=STIM_SPEC)
    out = generate_report(culture, config, sections=("activity", "stimulation"))
    assert out.exists() and out.stat().st_size > 0


def test_stimulation_summary_plot_renders(tmp_path, make_recording_data):
    config, culture = _stim_store(tmp_path, make_recording_data, chip="C0015", phase_spec=STIM_SPEC)
    out = tmp_path / "stim.png"
    stimulation.plot_stimulation_summary(CultureSelector(cultures=[culture]), config, phase="train",
                                         show_plot=False, save_path=out)
    assert out.exists()


# --- network: functional connectivity and PCA ----------------------------------------------------


def test_smoothed_rate_is_causal_and_peaks_after_the_spike():
    """A lone spike produces a silent prefix and a double-exponential response peaking at the
    analytic maximum of the difference of the two exponentials."""
    params = NetworkParams(sample_bin_sec=0.01)  # no decimation, so bins map 1:1 to samples
    bin_size = 0.01
    spike_at = 100

    spike_bin = np.zeros((1, 2000), dtype=np.uint8)
    spike_bin[0, spike_at] = 1

    rates = activity._double_exponential(spike_bin, bin_size, params.tau_rise_sec,
                                         params.tau_decay_sec, params.sample_bin_sec)[0]

    assert rates.shape == (2000,)
    assert np.allclose(rates[:spike_at + 1], 0.0)  # causal: nothing before the spike
    assert rates[spike_at + 1:].min() >= 0

    tau_r, tau_d = params.tau_rise_sec, params.tau_decay_sec
    peak_sec = np.log(tau_d / tau_r) * tau_r * tau_d / (tau_d - tau_r)
    assert np.argmax(rates) == pytest.approx(spike_at + peak_sec / bin_size, abs=3)


def test_smoothed_firing_rate_of_a_recording(store):
    config, cid_a, _ = store
    rec, _ = _fixture_recording(config, cid_a)

    smooth = activity.smoothed_firing_rate(rec, config.analysis_dir, sample_bin_sec=0.05)

    n_bins = rec.spike_bin.shape[1]
    assert smooth["rate"].shape == (rec.spike_bin.shape[0], -(-n_bins // 5))  # 10 ms bins -> 50 ms
    assert smooth["rate"].dtype == np.float32
    assert smooth["sample_bin_sec"] == pytest.approx(0.05)
    assert smooth["channels"].tolist() == rec.channelmap[:, 1].astype(int).tolist()
    assert list((config.analysis_dir / "activity").rglob("*_smoothed_firing_rate.npz"))


def test_correlation_recovers_known_correlations_and_drops_silent_electrodes():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=500)

    corr, keep = network._correlation(np.vstack([signal, signal, -signal, np.zeros(500)]))

    # The constant row has no variance and is dropped; the other three survive, in order.
    assert keep.tolist() == [0, 1, 2]
    assert corr.shape == (3, 3)
    assert corr[0, 1] == pytest.approx(1.0, abs=1e-5)
    assert corr[0, 2] == pytest.approx(-1.0, abs=1e-5)


def test_spectrum_and_effective_rank_bracket_the_extremes():
    rng = np.random.default_rng(1)

    # Rank 1: every electrode is a scaled copy of one signal, so one component carries everything.
    p_one = network._spectrum(np.outer(np.arange(1, 9), rng.normal(size=400)))
    assert p_one.sum() == pytest.approx(1.0)
    assert p_one[0] == pytest.approx(1.0, abs=1e-6)
    assert network._effective_rank(p_one) == pytest.approx(1.0, abs=1e-3)

    # Independent equal-variance electrodes spread variance evenly: effective rank approaches d.
    independent = rng.normal(size=(8, 20_000))
    assert network._effective_rank(network._spectrum(independent)) == pytest.approx(8, abs=0.5)


def test_recording_level_network_functions(store):
    config, cid_a, _ = store
    rec, _ = _fixture_recording(config, cid_a)

    fc = network.connectivity(rec, config.analysis_dir)
    n = fc["corr"].shape[0]
    assert fc["corr"].shape == (n, n) and len(fc["channels"]) == n == fc["elec_index"].size
    assert fc["phase"] == "full"

    pcs = network.pca(rec, config.analysis_dir, n_components=4)
    assert pcs["var_frac"].sum() == pytest.approx(1.0)
    assert np.all(np.diff(pcs["cumvar"]) >= -1e-6) and pcs["cumvar"][-1] == pytest.approx(1.0)
    assert pcs["components"].shape == (n, 4) and pcs["scores"].shape[0] == 4
    assert pcs["scores"].shape[1] == pcs["time_sec"].size
    assert np.all(pcs["components"].sum(axis=0) > 0)  # signs fixed
    assert np.array_equal(pcs["channels"], fc["channels"])

    metrics = network.network_metrics(rec, config.analysis_dir)
    assert metrics["phase"].tolist() == ["full"]
    row = metrics.iloc[0]
    assert row["n_electrodes_used"] == n
    assert -1 <= row["mean_corr"] <= 1
    assert row["effective_rank"] == pytest.approx(pcs["effective_rank"])
    assert row["n_pc_80"] <= row["n_pc_90"] <= n
    # The fixture's bursts recruit every channel, so the array moves as one: PC1 dominates.
    assert row["pc1_var_frac"] > 0.5
    assert row["n_communities"] >= 1 and np.isfinite(row["modularity"])

    saved = {p.name.split("well0_")[-1] for p in (config.analysis_dir / "network").rglob("*")}
    assert {"connectivity_full.npz", "pca_full.npz", "network_metrics.csv"} <= saved


def test_summarize_network_metrics(store):
    config, cid_a, _ = store
    df = network.summarize_network_metrics(cid_a, config)

    assert set(df["div"]) == {7, 8} and set(df["phase"]) == {"full"}
    assert (df["effective_rank"] >= 1).all()
    # Settings are stamped on every row ...
    assert set(df["sample_bin_sec"]) == {NetworkParams().sample_bin_sec}

    # ... and changing them recomputes instead of reusing rows made under the old ones.
    coarse = network.summarize_network_metrics(cid_a, config, params=NetworkParams(sample_bin_sec=0.5))
    assert set(coarse["sample_bin_sec"]) == {0.5}

    # Community settings are stamped and honoured the same way.
    assert {"modularity", "n_communities", "community_method"} <= set(df.columns)
    fine = network.summarize_network_metrics(
        cid_a, config, params=NetworkParams(communities=CommunityParams(resolution=2.0)))
    assert set(fine["community_resolution"]) == {2.0}


def test_network_summary_honors_embedded_phase_spec(tmp_path, make_recording_data):
    config, cid = _phased_store(tmp_path, make_recording_data)
    assert set(network.summarize_network_metrics(cid, config)["phase"]) == {"early", "late"}


def test_culture_connectivity_cache(store):
    config, cid_a, _ = store
    cpath = resolve_paths(cid_a, config)

    conn = network._culture_connectivity(cpath, config.analysis_dir)
    assert sorted(conn) == [(7, ""), (8, "")]
    _save_path, npz_path = _summary_paths(cpath, config.analysis_dir, "network",
                                          "default_connectivity", ext=".npz")
    assert npz_path.exists()

    cached = network._culture_connectivity(cpath, config.analysis_dir)
    assert np.array_equal(cached[(7, "")]["corr"], conn[(7, "")]["corr"])
    assert np.array_equal(cached[(7, "")]["cumvar"], conn[(7, "")]["cumvar"])
    assert np.array_equal(cached[(7, "")]["scores"], conn[(7, "")]["scores"])
    assert str(cached[(7, "")]["phase"]) == "full"


def test_culture_connectivity_recomputes_a_cache_without_trajectories(store):
    """A cache written before the trajectory arrays existed is recomputed, not half-used."""
    config, cid_a, _ = store
    cpath = resolve_paths(cid_a, config)
    conn = network._culture_connectivity(cpath, config.analysis_dir)
    _save_dir, npz_path = _summary_paths(cpath, config.analysis_dir, "network",
                                         "default_connectivity", ext=".npz")
    old = {k: v for k, v in np.load(npz_path).items() if k.rsplit("__", 1)[-1] in ("corr", "cumvar", "elec_index")}
    np.savez_compressed(npz_path, **old)

    redone = network._culture_connectivity(cpath, config.analysis_dir)
    assert np.array_equal(redone[(8, "")]["scores"], conn[(8, "")]["scores"])


def test_network_plots_render(store, tmp_path):
    config, cid_a, cid_b = store
    network.plot_connectivity_grid(cid_a, config, save_path=tmp_path / "grid.png", show_plot=False)
    network.plot_pc_variance_explained(cid_a, config, save_path=tmp_path / "pcs.png", show_plot=False)
    network.plot_network_summary(CultureSelector(cultures=[cid_a, cid_b]), config, phase="full",
                                 show_plot=False, save_path=tmp_path / "summary.png")
    fig = network.plot_connectivity_grid(cid_a, config, order="communities", show_plot=False,
                                         save_path=tmp_path / "grid_comm.png")
    assert "Q =" in fig.axes[0].get_title()
    fig = network.plot_pc_trajectory_grid(cid_a, config, show_plot=False, save_path=tmp_path / "traj.png")
    # Unphased data: the phase is left out of the culture-level titles.
    assert "phase" not in fig._suptitle.get_text()
    assert all((tmp_path / f).exists()
               for f in ("grid.png", "pcs.png", "summary.png", "grid_comm.png", "traj.png"))
    with pytest.raises(ValueError):
        network.plot_connectivity_grid(cid_a, config, order="nope", show_plot=False)
    with pytest.raises(ValueError):
        network.plot_pc_trajectory_grid(cid_a, config, pcs=(1, 4), show_plot=False)


# --- network: PCA trajectories and communities ----------------------------------------------------


def test_pca_scores_project_the_centred_rates_with_fixed_signs():
    rng = np.random.default_rng(2)
    rates = rng.normal(size=(6, 300)) + np.outer(np.arange(6), rng.normal(size=300))
    p, components, scores = network._pca(rates, n_components=3)

    assert p == pytest.approx(network._spectrum(rates))
    centred = rates - rates.mean(axis=1, keepdims=True)
    assert scores == pytest.approx(components.T @ centred, abs=1e-4)
    assert np.all(components.sum(axis=0) > 0)
    assert components.T @ components == pytest.approx(np.eye(3), abs=1e-8)
    # The flip is a convention, not a fit: negating the data negates nothing in the output.
    _p, comp_neg, _scores = network._pca(-rates, n_components=3)
    assert comp_neg == pytest.approx(components, abs=1e-8)


def _planted_corr(sizes=(30, 20, 10), noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(len(sizes)), sizes)
    signal = rng.normal(size=(len(sizes), 3000))[truth] + noise * rng.normal(size=(truth.size, 3000))
    order = rng.permutation(truth.size)
    return np.corrcoef(signal[order]), truth[order]


def test_communities_recover_a_planted_partition():
    corr, truth = _planted_corr()
    found = network.communities(corr)

    assert found["n_communities"] == 3
    # Each planted block is exactly one community, numbered largest first.
    for block, size in zip(range(3), (30, 20, 10)):
        assert set(found["labels"][truth == block]) == {block}
        assert (found["labels"] == block).sum() == size
    assert sorted(found["order"]) == list(range(truth.size))
    assert found["boundaries"].tolist() == [30, 50, 60]
    # The order groups each community into one contiguous block.
    assert np.all(np.diff(found["labels"][found["order"]]) >= 0)
    assert 0 < found["modularity"] < 1

    signed = network.communities(corr, params=CommunityParams(negative="signed"))
    assert signed["n_communities"] == 3


def test_communities_find_one_community_when_everything_is_synchronized():
    corr = np.full((40, 40), 0.7)
    np.fill_diagonal(corr, 1.0)
    found = network.communities(corr)
    assert found["n_communities"] == 1
    assert found["modularity"] == pytest.approx(0.0, abs=1e-9)


def test_communities_are_reproducible_and_modularity_is_consistent():
    from mxtreme.analysis import _communities

    corr, _truth = _planted_corr(sizes=(25, 25, 25, 25), noise=3.0, seed=4)
    params = CommunityParams(seed=7)
    a, b = network.communities(corr, params=params), network.communities(corr, params=params)
    assert np.array_equal(a["labels"], b["labels"])
    # The reported Q is the modularity of the reported partition.
    B = _communities.modularity_matrix(corr, params.resolution, params.negative)
    assert a["modularity"] == pytest.approx(_communities.modularity(B, a["labels"]))
    # More resolution never yields fewer communities here.
    finer = network.communities(corr, params=CommunityParams(resolution=1.5))
    assert finer["n_communities"] >= a["n_communities"]


def test_communities_reject_an_unknown_method():
    with pytest.raises(ValueError, match="Available"):
        network.communities(np.eye(3), params=CommunityParams(method="nope"))


def test_communities_of_a_recording_map_channels_and_save(store, tmp_path):
    config, cid_a, _ = store
    rec, _ = _fixture_recording(config, cid_a)
    conn = network.connectivity(rec)
    found = network.communities(conn, save_path=tmp_path / "comm.csv")

    assert np.array_equal(found["channels"], conn["channels"]) and found["phase"] == "full"
    table = network.community_table(found)
    assert list(table.columns) == ["channel", "elec_index", "community"]
    assert len(table) == len(conn["channels"])
    saved = pd.read_csv(tmp_path / "comm.csv")
    pd.testing.assert_frame_equal(saved[["channel", "elec_index", "community"]], table, check_dtype=False)
    assert {"modularity", "method", "resolution"} <= set(saved.columns)


def test_plot_connectivity_and_trajectory(store, tmp_path):
    from matplotlib.collections import LineCollection

    config, cid_a, _ = store
    rec, _ = _fixture_recording(config, cid_a)
    conn = network.connectivity(rec)

    fig = network.plot_connectivity(conn, communities=True, show_plot=False,
                                    save_path=tmp_path / "fc.png")
    assert "communities" in fig.axes[0].get_title()
    with pytest.raises(ValueError):
        network.plot_connectivity(conn, communities=network.communities(np.eye(3)), show_plot=False)

    fig = network.plot_pc_trajectory(rec, show_plot=False, save_path=tmp_path / "traj.png")
    assert any(isinstance(c, LineCollection) for c in fig.axes[0].collections)
    assert fig.axes[0].get_xlabel().startswith("PC1 (")

    # On a caller's axes: draws there, makes no figure of its own.
    fig, ax = plt.subplots()
    assert network.plot_pc_trajectory(rec, ax=ax, pcs=(1, 3), colorbar=False) is fig
    assert ax.get_ylabel().startswith("PC3 (")
    plt.close(fig)
    assert (tmp_path / "fc.png").exists() and (tmp_path / "traj.png").exists()


def test_animate_pc_trajectory_writes_a_gif(store, tmp_path):
    from PIL import Image

    config, cid_a, _ = store
    rec, _ = _fixture_recording(config, cid_a)
    out = network.animate_pc_trajectory(rec, tmp_path / "anim.gif", duration_sec=1.0, step_sec=0.25,
                                        raster_window_sec=2.0, dpi=40, order="communities")
    with Image.open(out) as gif:
        assert gif.n_frames == 4

    with pytest.raises(ValueError, match="max_frames"):
        network.animate_pc_trajectory(rec, tmp_path / "long.gif", duration_sec=10.0, step_sec=0.25,
                                      max_frames=5)
    with pytest.raises(ValueError, match=".gif or .mp4"):
        network.animate_pc_trajectory(rec, tmp_path / "anim.avi")
    assert not (tmp_path / "long.gif").exists()


# --- spatial -------------------------------------------------------------------------------------


def test_spatial_metrics_and_summary(store, tmp_path):
    config, cid_a, cid_b = store
    rec, _ = _fixture_recording(config, cid_a)

    one = spatial.spatial_metrics(rec, config.analysis_dir)
    assert len(one) == 1 and one.iloc[0]["phase"] == "full"
    assert one.iloc[0]["n_electrodes"] == rec.channelmap.shape[0]
    # The fixture lays electrodes on a 350 µm grid, so each one's nearest neighbour is 350 µm away.
    assert one.iloc[0]["mean_nn_distance_um"] == pytest.approx(350.0)

    table = spatial.summarize_spatial(CultureSelector(cultures=[cid_a, cid_b]), config)
    assert table["culture_id"].nunique() == 2 and len(table) == 3

    spatial.plot_spatial_summary(cid_a, config, show_plot=False, save_path=tmp_path / "spatial.png")
    spatial.plot_mea_layouts(cid_a, config, show_plot=False, save_path=tmp_path / "mea.png")
    assert (tmp_path / "spatial.png").exists() and (tmp_path / "mea.png").exists()
