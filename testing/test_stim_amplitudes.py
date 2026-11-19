import numpy as np

from mxtreme.analysis.intervals import stim_amplitudes, stim_electrodes, stim_roles, stim_windows


def test_each_episode_takes_its_pulses_amplitude():
    # two trains at 20 kHz: 100 mV pulses at 1 s, a 50 -> 300 mV ramp at 5 s
    starts = [20_000, 20_200, 100_000, 100_200, 100_400]
    amps = [100, 100, 50, 150, -300]
    frames, times, msgs = [], [], [{"pre_recording_start": "0"}]
    times.append(0)
    for s, a in zip(starts, amps):
        frames.extend(range(s, s + 8))
        times += [s, s + 8]
        msgs += [{"amp_mV": str(a), "start_stimulation": "pulse0"}, repr({"amp_mV": str(a), "end_stimulation": "pulse0"})]
    w = stim_windows(frames, 20_000.0)
    assert len(w) == 2
    out = stim_amplitudes(w, np.array(times), np.array(msgs, dtype=object))
    np.testing.assert_allclose(out, [100.0, 300.0])


def test_no_amplitude_is_nan():
    w = np.array([[10, 20]])
    assert np.isnan(stim_amplitudes(w, np.array([10]), np.array([{"start_stimulation": "x"}], dtype=object))).all()
    assert stim_amplitudes(np.zeros((0, 2)), None, None).shape == (0,)


def test_each_episode_names_its_electrodes():
    # one region at a time, then a paired presentation through both
    frames = list(range(1000, 1010)) + list(range(70_000, 70_010)) + list(range(140_000, 140_010))
    times = [1000, 70_000, 140_000, 140_002]
    msgs = [
        {"start_stimulation": "cs", "amp_mV": "170", "elecs": "10,11"},
        {"start_stimulation": "us", "amp_mV": "240", "elecs": "20, 21"},
        {"start_stimulation": "pair", "amp_mV": "240", "elecs": "20,21"},
        {"start_stimulation": "pair", "amp_mV": "170", "elecs": "10,11"},
    ]
    w = stim_windows(frames, 10_000.0)
    assert stim_electrodes(w, np.array(times), np.array(msgs, dtype=object)) == [[10, 11], [20, 21], [20, 21, 10, 11]]
    assert stim_electrodes(w, np.array([1000]), np.array([{"start_stimulation": "x"}], dtype=object)) == [None, None, None]
    for m, r in zip(msgs, ["CS", "US", "US", "CS"]):
        m["role"] = r
    assert stim_roles(w, np.array(times), np.array(msgs, dtype=object)) == ["CS", "US", "US+CS"]
