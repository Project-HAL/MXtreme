"""Tests for the bursting module: phases, the Burst record, detection, and feature extraction."""

import numpy as np
import pandas as pd
import pytest

from mxtreme import io, utils
from mxtreme.recording import Recording
from mxtreme.bursting import Burst, BurstDetector, BurstSet
from mxtreme.bursting.burst import _COLUMNS
from mxtreme.params import BurstDetectParams, BurstFeatureParams
from mxtreme.phases import Phase, Phases, phases_from_event_tags


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
    assert phases.label_for(999) == "full"
    assert phases.label_for(1000) is None  # half-open interval


def test_phases_from_event_tags_orders_and_bounds():
    event_df = pd.DataFrame({
        "eventtime": [10, 100, 1000],
        "eventmessage": [
            {"pre_recording_start": "5"},
            {"closed_loop_start": "5"},
            {"post_recording_start": "5"},
        ],
    })
    phases = phases_from_event_tags(
        event_df,
        [("pre", "pre_recording_start"), ("train", "closed_loop_start"), ("post", "post_recording_start")],
        end_frame=2000,
    )
    assert phases.names == ("pre", "train", "post")
    assert phases.label_for(50) == "pre"
    assert phases.label_for(500) == "train"
    assert phases.label_for(1500) == "post"
    assert phases.label_for(5) is None  # before the first tag


def test_phases_from_event_tags_missing_tags_fall_back_to_full():
    event_df = pd.DataFrame({"eventtime": [], "eventmessage": []})
    phases = phases_from_event_tags(event_df, [("pre", "pre_recording_start")], end_frame=500)
    assert phases.names == ("full",)


def test_recording_resolves_phases_from_tag_spec(make_recording_data):
    # A tag spec passed straight to Recording is resolved internally against the recording's own
    # events -- no provisional Recording needed to read event_df / end_frame first.
    data = make_recording_data(
        eventtime=np.array([10, 100], dtype="<i8"),
        event_messages=[{"pre_recording_start": "5"}, {"closed_loop_start": "5"}],
    )
    rec = Recording(
        0, data,
        phase_tags=[("pre", "pre_recording_start"), ("train", "closed_loop_start")],
        end_tag="end_experiment",  # absent from events -> final phase runs to the last frame
    )
    assert rec.phases.names == ("pre", "train")
    assert rec.phases.label_for(50) == "pre"
    assert rec.phases.label_for(int(rec.spike_data["frameno"].max()) - 1) == "train"


def test_recording_phases_and_tags_are_mutually_exclusive(make_recording_data):
    with pytest.raises(ValueError):
        Recording(0, make_recording_data(), phases=Phases.full(0, 10),
                   phase_tags=[("pre", "pre_recording_start")])


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
