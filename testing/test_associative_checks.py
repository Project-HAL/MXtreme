"""The calibration verdicts and the baseline gate: the arithmetic and the advice, without a recording."""

import numpy as np
import pytest

from mxtreme.experiments.associative import checks


def _spikes(events):
    """events: [(frame, electrode)] -> sorted (frames, electrodes)."""
    frames = np.array([f for f, _ in events], dtype=np.int64)
    elecs = np.array([e for _, e in events], dtype=int)
    order = np.argsort(frames, kind="stable")
    return frames[order], elecs[order]


REGIONS = {"US": [10], "CS": [20], "NS": [30]}
FPS = 20000.0


def test_pulse_responses_subtracts_the_mirror_window():
    # One pulse at frame 100000. US fires 4 times in the window after it and once before;
    # CS fires the same before and after, so its evoked response is zero.
    after = int(0.010 * FPS)
    before = int(-0.010 * FPS)
    events = [(100000 + after, 10)] * 4 + [(100000 + before, 10)]
    events += [(100000 + after, 20), (100000 + before, 20)]
    frames, elecs = _spikes(events)
    out = checks.pulse_responses(frames, elecs, REGIONS, FPS, [("US", 100000)])
    assert out["US"]["US"][0] == pytest.approx(3.0)
    assert out["US"]["CS"][0] == pytest.approx(0.0)
    assert out["US"]["NS"][0] == pytest.approx(0.0)


def test_crosstalk_is_relative_to_the_local_response():
    responses = {
        "US": {"US": np.array([10.0, 10.0]), "CS": np.array([1.0, 1.0]), "NS": np.array([0.0, 0.0])},
        "CS": {"US": np.array([5.0, 5.0]), "CS": np.array([10.0, 10.0]), "NS": np.array([0.0, 0.0])},
        "NS": {"US": np.array([0.0, 0.0]), "CS": np.array([0.0, 0.0]), "NS": np.array([0.0, 0.0])},
    }
    names, evoked, ratio = checks.crosstalk(responses)
    assert evoked[0, 0] == pytest.approx(10.0)
    assert names[:3] == ["US", "CS", "NS"]
    assert ratio[0, 0] == pytest.approx(1.0)
    assert ratio[0, 1] == pytest.approx(0.1)
    assert ratio[1, 0] == pytest.approx(0.5)
    # NS evokes nothing in itself, so no ratio can be taken from it.
    assert np.all(np.isnan(ratio[2]))


def test_independence_verdicts_flag_a_dead_site_and_a_bursting_one():
    responses = {
        "US": {"US": np.array([8.0]), "CS": np.array([0.1]), "NS": np.array([0.0])},
        "CS": {"US": np.array([0.2]), "CS": np.array([0.05]), "NS": np.array([0.0])},
        "NS": {"US": np.array([0.0]), "CS": np.array([0.0]), "NS": np.array([6.0])},
    }
    names, evoked, ratio = checks.crosstalk(responses)
    out = checks.independence_verdicts(names, evoked, ratio, {"US": 0.5, "CS": 0.0, "NS": 0.0})
    text = " ".join(v.message for v in out)
    assert any(v.level == "stop" for v in out)
    assert "stimulating CS evokes" in text  # dead site
    assert "50% of US pulses" in text  # bursting site
    assert [v.level for v in out].count("stop") == 2


def test_independence_verdicts_flag_crosstalk_and_a_missing_path():
    coupled = {
        "US": {"US": np.array([10.0]), "CS": np.array([4.0]), "NS": np.array([0.1])},
        "CS": {"US": np.array([0.5]), "CS": np.array([10.0]), "NS": np.array([0.1])},
        "NS": {"US": np.array([0.1]), "CS": np.array([0.1]), "NS": np.array([10.0])},
    }
    names, evoked, ratio = checks.crosstalk(coupled)
    out = checks.independence_verdicts(names, evoked, ratio, dict.fromkeys(names, 0.0))
    assert any("not independent" in v.message for v in out)

    isolated = {
        "US": {"US": np.array([10.0]), "CS": np.array([0.0]), "NS": np.array([0.0])},
        "CS": {"US": np.array([0.0]), "CS": np.array([10.0]), "NS": np.array([0.0])},
        "NS": {"US": np.array([0.0]), "CS": np.array([0.0]), "NS": np.array([10.0])},
    }
    names, evoked, ratio = checks.crosstalk(isolated)
    out = checks.independence_verdicts(names, evoked, ratio, dict.fromkeys(names, 0.0))
    assert any("may not be connected at all" in v.message for v in out)


def test_independence_verdicts_pass_a_good_culture():
    good = {
        "US": {"US": np.array([10.0]), "CS": np.array([0.5]), "NS": np.array([0.4])},
        "CS": {"US": np.array([0.6]), "CS": np.array([10.0]), "NS": np.array([0.5])},
        "NS": {"US": np.array([0.4]), "CS": np.array([0.5]), "NS": np.array([10.0])},
    }
    names, evoked, ratio = checks.crosstalk(good)
    out = checks.independence_verdicts(names, evoked, ratio, dict.fromkeys(names, 0.05))
    assert [v.level for v in out] == ["ok"]


def test_burst_rate_counts_bursts_starting_after_a_pulse():
    pulses = [("US", 1000), ("US", 100000), ("CS", 200000)]
    bursts = [(1500, 5000)]  # 25 ms after the first US pulse at 20 kHz
    rates = checks.burst_rate(pulses, bursts, FPS, within_ms=500)
    assert rates["US"] == pytest.approx(0.5)
    assert rates["CS"] == pytest.approx(0.0)


def test_calibration_picks_the_largest_safe_amplitude():
    tokens = {
        "stim_us_a50_ac": ("US", 50.0, "anodic-first"),
        "stim_us_a100_ac": ("US", 100.0, "anodic-first"),
        "stim_us_a200_ac": ("US", 200.0, "anodic-first"),
    }
    responses = {
        "stim_us_a50_ac": {"US": np.array([0.1, 0.0, 0.2, 0.1])},
        "stim_us_a100_ac": {"US": np.array([3.0, 2.0, 4.0, 3.0])},
        "stim_us_a200_ac": {"US": np.array([9.0, 8.0, 10.0, 9.0])},
    }
    bursts = {"stim_us_a50_ac": 0.0, "stim_us_a100_ac": 0.05, "stim_us_a200_ac": 0.8}
    rows = checks.calibration_table(responses, bursts, tokens)
    assert [r["amplitude_mv"] for r in rows] == [50.0, 100.0, 200.0]
    chosen, out = checks.calibration_verdicts(rows, ["US"])
    assert chosen["US"]["amplitude_mv"] == 100.0  # 200 responds but bursts
    assert out[0].level == "ok"


def test_calibration_says_why_when_nothing_works():
    tokens = {
        "stim_cs_a50_ac": ("CS", 50.0, "anodic-first"),
        "stim_cs_a100_ac": ("CS", 100.0, "anodic-first"),
    }
    quiet = {"stim_cs_a50_ac": {"CS": np.zeros(4)}, "stim_cs_a100_ac": {"CS": np.array([0.1, 0.0, 0.2, 0.1])}}
    chosen, out = checks.calibration_verdicts(
        checks.calibration_table(quiet, dict.fromkeys(tokens, 0.0), tokens), ["CS"]
    )
    assert chosen["CS"] is None and out[0].level == "stop" and "Extend the ladder" in out[0].message

    loud = {
        "stim_cs_a50_ac": {"CS": np.array([5.0, 4.0, 6.0, 5.0])},
        "stim_cs_a100_ac": {"CS": np.full(4, 9.0)},
    }
    chosen, out = checks.calibration_verdicts(
        checks.calibration_table(loud, dict.fromkeys(tokens, 0.9), tokens), ["CS"]
    )
    assert chosen["CS"] is None and "starts network" in out[0].message


def test_silent_recording_is_named_rather_than_reported_as_dead_sites():
    fps = 20000.0
    assert checks.silent_recording(np.array([]), fps).level == "stop"
    quiet = np.arange(0, 300 * fps, 2 * fps)  # 0.5 spikes/s over five minutes
    verdict = checks.silent_recording(quiet, fps)
    assert verdict is not None and "the plate is silent" in verdict.message
    busy = np.arange(0, 300 * fps, fps / 50)  # 50 spikes/s
    assert checks.silent_recording(busy, fps) is None


def test_calibration_does_not_trust_a_handful_of_chance_detections():
    """Two pulses with five noise crossings between them made a 'usable' row on the saline dry
    run; a usable row needs enough pulses and a mean clear of its own error."""
    tokens = {"stim_us_a45_ac": ("US", 45.0, "anodic-first"), "stim_us_a60_ac": ("US", 60.0, "anodic-first")}
    few = {"stim_us_a45_ac": {"US": np.array([5.0, 0.0])}, "stim_us_a60_ac": {"US": np.array([1.0, 0.0])}}
    chosen, out = checks.calibration_verdicts(
        checks.calibration_table(few, dict.fromkeys(tokens, 0.0), tokens), ["US"]
    )
    assert chosen["US"] is None and out[0].level == "stop" and "not reliably" in out[0].message
    # The same means over six consistent pulses are believed.
    many = {
        "stim_us_a45_ac": {"US": np.full(6, 2.5)},
        "stim_us_a60_ac": {"US": np.array([0.4, 0.6, 0.5, 0.5, 0.4, 0.6])},
    }
    chosen, out = checks.calibration_verdicts(
        checks.calibration_table(many, dict.fromkeys(tokens, 0.0), tokens), ["US"]
    )
    assert chosen["US"]["amplitude_mv"] == 60.0 and out[0].level == "ok"


def test_calibration_prefers_an_amplitude_that_stays_in_its_region():
    """The ideal amplitude drives the site without driving the other two: an amplitude whose
    remote response is over 30% of its local one is not usable, so the pick stops below it."""
    tokens = {
        "stim_us_a60_ac": ("US", 60.0, "anodic-first"),
        "stim_us_a80_ac": ("US", 80.0, "anodic-first"),
        "stim_us_a100_ac": ("US", 100.0, "anodic-first"),
    }
    responses = {
        "stim_us_a60_ac": {"US": np.full(6, 2.0), "CS": np.full(6, 0.1), "NS": np.full(6, 0.1)},
        "stim_us_a80_ac": {"US": np.full(6, 3.0), "CS": np.full(6, 0.5), "NS": np.full(6, 0.3)},
        "stim_us_a100_ac": {"US": np.full(6, 4.0), "CS": np.full(6, 2.0), "NS": np.full(6, 2.0)},
    }
    rows = checks.calibration_table(responses, dict.fromkeys(tokens, 0.0), tokens)
    assert [round(r["spread"], 2) for r in rows] == [0.05, 0.13, 0.5]
    chosen, out = checks.calibration_verdicts(rows, ["US"])
    assert chosen["US"]["amplitude_mv"] == 80.0
    # If every responsive amplitude spreads, the verdict names that rather than "no response".
    spreading = {t: {"US": np.full(6, 3.0), "CS": np.full(6, 2.0), "NS": np.full(6, 2.0)} for t in tokens}
    chosen, out = checks.calibration_verdicts(
        checks.calibration_table(spreading, dict.fromkeys(tokens, 0.0), tokens), ["US"]
    )
    assert chosen["US"] is None and "drives the other regions" in out[0].message
