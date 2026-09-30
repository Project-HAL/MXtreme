"""Tests for the bursting module: phases, the Burst record, detection, and feature extraction."""

import numpy as np
import pandas as pd
import pytest

from mxtreme import io, utils
from mxtreme.recording import Recording
from mxtreme.bursting import Burst, BurstDetector, BurstSet
from mxtreme.bursting.burst import _COLUMNS
from mxtreme.params import BurstDetectParams, BurstFeatureParams
from mxtreme.phases import Phase, Phases, phases_from_spec


# Detection params tuned for the small synthetic fixture (N=300 needs millions of spikes).
DETECT = BurstDetectParams(n=50, noise_thresh=0.02, burst_thresh=0.15, min_dist_bins=10)


def _peak_in_group(burst, rec):
    """Peak sits within its group interval, compared in bin space.

    ``peak_frame`` is a binned quantity and the group bounds are raw spike frames, so an exact frame
    comparison can be off by up to one bin at the edges; comparing bins is the meaningful check.
    """
    fb = lambda f: utils.frame_to_bin(f, rec.samp_rate, rec.bin_size)
    return fb(burst.group_start_frame) <= fb(burst.peak_frame) <= fb(burst.group_stop_frame)


# --- phases ---------------------------------------------------------------------------------------


def test_phases_full_default_labels_everything():
    phases = Phases.full(0, 1000)
    assert phases.names == ("full",)
    assert phases.label_for(0) == "full"
    assert phases.label_for(1000) == "full"  # both bounds inclusive
    assert phases.label_for(1001) is None
    assert phases.phases[0].n_frames == 1001


def test_phases_get_phase_by_name():
    phases = Phases((Phase("pre", 0, 100), Phase("train_1", 100, 200), Phase("train_2", 300, 400)))
    assert phases.get_phase("train_2") == Phase("train_2", 300, 400)
    with pytest.raises(KeyError, match="train_1"):  # the error lists the available names
        phases.get_phase("train")


_EMPTY_EVENTS = pd.DataFrame({"eventtime": [], "eventmessage": []})


def _events(*rows):
    """Build an event_df from ``(frame, message)`` pairs."""
    return pd.DataFrame({"eventtime": [f for f, _ in rows], "eventmessage": [m for _, m in rows]})


def _bounds(phases):
    return [(p.name, p.start_frame, p.end_frame) for p in phases]


def test_phases_from_spec_tags_run_to_next_start():
    events = _events(
        (10, {"pre_recording_start": "5"}), (100, {"closed_loop_start": "5"}),
        (1000, {"post_recording_start": "5"}),
    )
    spec = {"pre": "pre_recording_start", "train": "closed_loop_start", "post": "post_recording_start"}
    phases = phases_from_spec(spec, events, start_frame=0, end_frame=2000, samp_rate=100.0)
    assert _bounds(phases) == [("pre", 10, 99), ("train", 100, 999), ("post", 1000, 2000)]
    assert phases.label_for(5) is None  # before the first tag


def test_phases_from_spec_orders_by_start_frame():
    events = _events((10, {"a": "1"}), (100, {"b": "1"}))
    phases = phases_from_spec({"b": "b", "a": "a"}, events, start_frame=0, end_frame=500, samp_rate=100.0)
    assert _bounds(phases) == [("a", 10, 99), ("b", 100, 500)]


def test_phases_from_spec_minutes_anchored_to_first_frame():
    # 20 min @ 100 Hz = 120000 frames.
    spec = {"pre": 0, "train": 20, "post": {"start": 40, "end": 60}}
    phases = phases_from_spec(spec, _EMPTY_EVENTS, start_frame=0, end_frame=999999, samp_rate=100.0)
    assert [(p.start_frame, p.end_frame) for p in phases] == [
        (0, 119999), (120000, 239999), (240000, 359999)
    ]


def test_phases_from_spec_minutes_offset_by_first_frame():
    spec = {"pre": 0, "train": {"start": 10, "end": 20}}
    phases = phases_from_spec(spec, _EMPTY_EVENTS, start_frame=3000, end_frame=999999, samp_rate=100.0)
    # 10 min @ 100 Hz = 60000, anchored at first frame 3000.
    assert [(p.start_frame, p.end_frame) for p in phases] == [(3000, 62999), (63000, 122999)]


def test_phases_from_spec_mixed_boundaries():
    events = _events((50000, {"closed_loop_start": "5"}))
    spec = {"pre": 0, "train": {"start": "closed_loop_start", "end": 10}}
    phases = phases_from_spec(spec, events, start_frame=0, end_frame=999999, samp_rate=100.0)
    # pre at minute 0 -> frame 0; train at the tag frame 50000; end at 10 min -> last frame 59999.
    assert _bounds(phases) == [("pre", 0, 49999), ("train", 50000, 59999)]


def test_phases_from_spec_multi_key_matcher():
    events = _events(
        (10, {"closed_loop_start": "5", "side": "right"}),
        (20, {"closed_loop_start": "5"}),
        (30, {"closed_loop_start": "5", "side": "left", "stim_electrode": "1234"}),
    )
    spec = {"train": {"closed_loop_start": None, "side": "left", "stim_electrode": 1234}}
    # A bare matcher mapping isn't a definition; it must sit under "start".
    with pytest.raises(ValueError):
        phases_from_spec(spec, events, start_frame=0, end_frame=100, samp_rate=100.0)
    phases = phases_from_spec(
        {"train": {"start": spec["train"]}}, events, start_frame=0, end_frame=100, samp_rate=100.0
    )
    # Only the event carrying every pair matches; None is a wildcard and 1234 matches '1234'.
    assert _bounds(phases) == [("train", 30, 100)]


def test_phases_from_spec_explicit_end_leaves_gap():
    events = _events(
        (10, {"pre_recording_start": "5"}), (50, {"pre_recording_end": "5"}),
        (100, {"closed_loop_start": "5"}),
    )
    spec = {
        "pre": {"start": "pre_recording_start", "end": {"pre_recording_end": None}},
        "train": "closed_loop_start",
    }
    phases = phases_from_spec(spec, events, start_frame=0, end_frame=500, samp_rate=100.0)
    assert _bounds(phases) == [("pre", 10, 50), ("train", 100, 500)]
    assert phases.label_for(75) is None


def test_phases_from_spec_replicates_are_suffixed():
    events = _events(
        (10, {"pre_recording_start": "5"}),
        (100, {"closed_loop_start": "5"}), (200, {"closed_loop_end": "5"}),
        (300, {"closed_loop_start": "5"}), (400, {"closed_loop_end": "5"}),
    )
    spec = {"pre": "pre_recording_start", "train": {"start": "closed_loop_start", "end": "closed_loop_end"}}
    phases = phases_from_spec(spec, events, start_frame=0, end_frame=500, samp_rate=100.0)
    assert _bounds(phases) == [("pre", 10, 99), ("train_1", 100, 200), ("train_2", 300, 400)]


def test_phases_from_spec_missing_end_falls_back():
    events = _events((10, {"a": "1"}), (100, {"b": "1"}))
    spec = {"a": {"start": "a", "end": "never_seen"}, "b": {"start": "b", "end": "never_seen"}}
    phases = phases_from_spec(spec, events, start_frame=0, end_frame=500, samp_rate=100.0)
    # No end event -> the frame before the next phase start, else the end of the recording.
    assert _bounds(phases) == [("a", 10, 99), ("b", 100, 500)]


def test_phases_from_spec_empty_falls_back_to_full():
    phases = phases_from_spec({}, _EMPTY_EVENTS, start_frame=0, end_frame=500, samp_rate=100.0)
    assert phases.names == ("full",)
    # A tag-only spec whose tags are all absent also collapses to a single full phase.
    phases2 = phases_from_spec(
        {"pre": "never_seen"}, _EMPTY_EVENTS, start_frame=0, end_frame=500, samp_rate=100.0
    )
    assert phases2.names == ("full",)


@pytest.mark.parametrize("spec", [
    {"starts": {"pre": 0, "post": 20}, "end": 40},  # the old format
    {"pre": {"start": 0, "stop": 20}},              # unknown definition key
    {"pre": {"end": 20}},                           # no start
    {"pre": None},
    {"pre": {"start": {}}},
])
def test_phases_from_spec_rejects_malformed_spec(spec):
    with pytest.raises(ValueError):
        phases_from_spec(spec, _EMPTY_EVENTS, start_frame=0, end_frame=500, samp_rate=100.0)


def test_recording_resolves_phases_from_tag_spec(make_recording_data):
    # A spec passed straight to Recording is resolved internally against the recording's own
    # events -- no provisional Recording needed to read event_df / end_frame first.
    data = make_recording_data(
        eventtime=np.array([10, 100], dtype="<i8"),
        event_messages=[{"pre_recording_start": "5"}, {"closed_loop_start": "5"}],
    )
    rec = Recording(
        0, data,
        phase_tags={
            "pre": "pre_recording_start",
            # end absent from events -> final phase runs to the last frame
            "train": {"start": "closed_loop_start", "end": "end_experiment"},
        },
    )
    assert rec.phases.names == ("pre", "train")
    assert rec.phases.label_for(50) == "pre"
    assert rec.phases.label_for(int(rec.spike_data["frameno"].max()) - 1) == "train"


def test_recording_phases_and_tags_are_mutually_exclusive(make_recording_data):
    with pytest.raises(ValueError):
        Recording(0, make_recording_data(), phases=Phases.full(0, 10),
                   phase_tags={"pre": "pre_recording_start"})


def test_recording_resolves_phases_from_embedded_spec(make_recording_data):
    # samp_rate defaults to 10000 Hz in the fixture; recording spans ~[0, 500000). Minutes must be small.
    spec = {"pre": 0.0, "train": 0.1, "post": {"start": 0.2, "end": 0.5}}
    rec = Recording(0, make_recording_data(phase_spec=np.asarray(spec, dtype=object)))
    assert rec.phases.names == ("pre", "train", "post")


def test_explicit_phases_override_embedded_spec(make_recording_data):
    spec = {"pre": 0.0, "train": {"start": 0.1, "end": 0.5}}
    data = make_recording_data(phase_spec=np.asarray(spec, dtype=object))
    rec = Recording(0, data, phases=Phases.full(0, 10))
    assert rec.phases.names == ("full",)


def test_absent_embedded_spec_defaults_to_full(make_recording_data):
    # The fixture carries no phase_spec key at all -> single "full" phase (backward compatible).
    rec = Recording(0, make_recording_data())
    assert rec.phases.names == ("full",)


# --- Burst record ---------------------------------------------------------------------------------


def test_burst_row_roundtrip_parses_dict_columns():
    import io as _io

    burst = Burst(
        id=1, peak_frame=1000, peak_amp=0.4, kind="network",
        group_start_frame=900, group_stop_frame=1100, n_spikes=60, phase="pre",
        origin_chan_counts={1: 2, 3: 1}, constituent_chan_counts={1: 5},
    )
    row = burst.to_row()
    assert list(row.keys()) == _COLUMNS

    # Simulate a CSV round-trip, where dict cells come back as strings.
    buf = _io.StringIO()
    pd.DataFrame([row]).to_csv(buf, index=False)
    buf.seek(0)
    reloaded = Burst.from_row(pd.read_csv(buf).to_dict(orient="records")[0])
    assert reloaded.origin_chan_counts == {1: 2, 3: 1}
    assert reloaded.constituent_chan_counts == {1: 5}
    assert reloaded.kind == "network"
    assert reloaded.peak_frame == 1000


def test_empty_burstset_has_canonical_columns():
    df = BurstSet(bursts=[]).to_dataframe()
    assert list(df.columns) == _COLUMNS
    assert len(df) == 0


def test_schema_has_no_seconds_or_bin_columns():
    # Every temporal column is in frames; guard against reintroducing _sec/_bin columns.
    assert not [c for c in _COLUMNS if c.endswith("_sec") or c.endswith("_bin")]


# --- detection ------------------------------------------------------------------------------------


def test_detect_finds_network_bursts(make_recording_data):
    rec = Recording(0, make_recording_data())
    bursts = BurstDetector(DETECT).detect(rec)
    assert len(bursts) >= 2
    df = bursts.to_dataframe()
    assert set(df["kind"]).issubset({"network", "mini"})
    assert (df["kind"] == "network").sum() >= 2
    assert set(df["phase"]) == {"full"}  # default phases


def test_detect_labels_injected_phases(make_recording_data):
    rec = Recording(0, make_recording_data(), phases=Phases((
        Phase("early", 0, 200000),
        Phase("late", 200000, 500000),
    )))
    bursts = BurstDetector(DETECT).detect(rec)
    labels = {b.phase for b in bursts}
    assert labels == {"early", "late"}  # one burst cluster in each interval


def test_unknown_method_raises(make_recording_data):
    rec = Recording(0, make_recording_data())
    with pytest.raises(ValueError):
        BurstDetector(DETECT, method="nope").detect(rec)


def test_isi_n_method_one_burst_per_group(make_recording_data):
    rec = Recording(0, make_recording_data())
    bursts = BurstDetector(DETECT, method="isi_n").detect(rec)
    assert len(bursts) >= 2
    # isi_n yields exactly one burst per ISI-N group (unique group intervals), all kind "burst".
    assert {b.kind for b in bursts} == {"burst"}
    intervals = [(b.group_start_frame, b.group_stop_frame) for b in bursts]
    assert len(intervals) == len(set(intervals))
    # each burst's peak sits within its own group interval
    assert all(_peak_in_group(b, rec) for b in bursts)


def test_isi_rate_peaks_within_group_intervals(make_recording_data):
    rec = Recording(0, make_recording_data())
    bursts = BurstDetector(DETECT).detect(rec)
    assert all(_peak_in_group(b, rec) for b in bursts)


# --- feature extraction ---------------------------------------------------------------------------


def test_extract_features_populates_frame_columns(make_recording_data):
    rec = Recording(0, make_recording_data())
    bursts = BurstDetector(DETECT).detect(rec)
    bursts.extract_features(rec, BurstFeatureParams())

    assert isinstance(bursts.n_ignored, int)
    df = bursts.to_dataframe()
    clean = df[~df["ignore"]]
    assert len(clean) > 0
    # Frame-unit invariants for successfully featurized bursts.
    assert (clean["onset_frame"] <= clean["offset_frame"]).all()
    assert (clean["duration_frames"] == clean["offset_frame"] - clean["onset_frame"]).all()
    assert (clean["duration_frames"] >= 0).all()
    assert (clean["size_frac_elec"] > 0).all()


def test_extract_features_rejects_unsorted_spike_data(make_recording_data):
    """Feature windows are found by binary search, which returns a wrong index rather than raising on
    unsorted input -- so the guard must fire before any burst is featurized."""
    data = make_recording_data()
    bursts = BurstDetector(DETECT).detect(Recording(0, data))

    unsorted = make_recording_data(spike_data=data["spike_data"][::-1])
    with pytest.raises(ValueError, match="sorted by frameno"):
        bursts.extract_features(Recording(0, unsorted), BurstFeatureParams())

    # ...and sorting it is all that's needed to get through.
    repaired = make_recording_data(spike_data=utils.sort_spike_data(data["spike_data"][::-1]))
    bursts.extract_features(Recording(0, repaired), BurstFeatureParams())


# --- recording data-access ------------------------------------------------------------------------


def test_recording_default_phase_is_full(make_recording_data):
    rec = Recording(0, make_recording_data())
    assert list(rec.phases.names) == ["full"]
    assert isinstance(rec.samp_rate, float)
    assert rec.asdr.shape[0] == rec.spike_bin.shape[1]


def test_recording_injected_phases(make_recording_data):
    phases = Phases((Phase("a", 0, 250000), Phase("b", 250000, 500000)))
    rec = Recording(0, make_recording_data(), phases=phases)
    assert [(p.name, p.start_frame, p.end_frame) for p in rec.phases] == [
        ("a", 0, 250000), ("b", 250000, 500000)
    ]


# --- two-stage recompute + step timestamps --------------------------------------------------------


def test_recompute_features_from_reloaded_burstset(make_recording_data):
    rec = Recording(0, make_recording_data())
    bursts = BurstDetector(DETECT).detect(rec)
    bursts.extract_features(rec, BurstFeatureParams())
    assert bursts.n_ignored == 0

    # Simulate a CSV round-trip (detection fields survive), then seed a stale ignore flag.
    reloaded = BurstSet.from_dataframe(bursts.to_dataframe())
    reloaded.bursts[0].ignore = True
    before = [(b.peak_frame, b.kind, b.group_start_frame, b.group_stop_frame) for b in reloaded.bursts]

    reloaded.extract_features(rec, BurstFeatureParams())

    after = [(b.peak_frame, b.kind, b.group_start_frame, b.group_stop_frame) for b in reloaded.bursts]
    assert before == after                        # detection-set fields unchanged by recompute
    assert reloaded.n_ignored == 0
    assert reloaded.bursts[0].ignore is False      # stale ignore cleared on successful recompute


def test_detect_and_extract_set_timestamps(make_recording_data):
    rec = Recording(0, make_recording_data())
    bursts = BurstDetector(DETECT).detect(rec)
    assert bursts.detected_at is not None
    assert bursts.features_computed_at is None
    bursts.extract_features(rec, BurstFeatureParams())
    assert bursts.features_computed_at is not None


def test_update_burst_log_preserves_step_fields(make_recording_data, tmp_path):
    rec = Recording(0, make_recording_data())

    # Stage A: detection only.
    det = BurstDetector(DETECT).detect(rec)
    io.update_burst_log(tmp_path, rec, det)
    log1 = pd.read_csv(tmp_path / f"{rec.exp_id}_burst_log.csv")
    assert len(log1) == 1
    assert log1.loc[0, "detection_completed_at"] == det.detected_at
    assert pd.isna(log1.loc[0, "features_computed_at"])
    detect_params_logged = log1.loc[0, "detect_params"]
    assert isinstance(detect_params_logged, str) and "burst_thresh" in detect_params_logged

    # Stage B: features only, from a reloaded BurstSet (no detect_params, no detection timestamp).
    reloaded = BurstSet.from_dataframe(det.to_dataframe())
    assert reloaded.detect_params is None and reloaded.detected_at is None
    reloaded.extract_features(rec, BurstFeatureParams())
    io.update_burst_log(tmp_path, rec, reloaded)

    log2 = pd.read_csv(tmp_path / f"{rec.exp_id}_burst_log.csv")
    assert len(log2) == 1                                              # same row updated in place
    assert log2.loc[0, "detection_completed_at"] == det.detected_at    # preserved
    assert log2.loc[0, "detect_params"] == detect_params_logged        # preserved
    assert isinstance(log2.loc[0, "features_computed_at"], str)        # now set
    assert "onset_thresh_pct" in log2.loc[0, "feature_params"]


def test_detect_auto_updates_burst_log(make_recording_data, tmp_path):
    rec = Recording(0, make_recording_data())

    # Detection alone, with burst_data_dir given, writes the log automatically (no manual call).
    det = BurstDetector(DETECT).detect(rec, burst_data_dir=tmp_path)

    log = pd.read_csv(tmp_path / f"{rec.exp_id}_burst_log.csv")
    assert len(log) == 1
    assert log.loc[0, "detection_completed_at"] == det.detected_at
    assert pd.isna(log.loc[0, "features_computed_at"])         # features haven't run yet


def test_extract_features_auto_updates_log_independently(make_recording_data, tmp_path):
    rec = Recording(0, make_recording_data())

    # Stage A: detection auto-logs its timestamp.
    det = BurstDetector(DETECT).detect(rec, burst_data_dir=tmp_path)
    detected_at = det.detected_at

    # Stage B: features run apart from detection (reloaded set has no detection timestamp/params) and
    # auto-log -- features_computed_at is set while the earlier detection stamp is preserved.
    reloaded = BurstSet.from_dataframe(det.to_dataframe())
    assert reloaded.detected_at is None
    reloaded.extract_features(rec, BurstFeatureParams(), burst_data_dir=tmp_path)

    log = pd.read_csv(tmp_path / f"{rec.exp_id}_burst_log.csv")
    assert len(log) == 1                                       # same row updated in place
    assert log.loc[0, "detection_completed_at"] == detected_at  # preserved across the features-only write
    assert isinstance(log.loc[0, "features_computed_at"], str)  # now set
