"""Shared fixtures for the preprocessing test suite.

`make_well` builds a small, synthetic per-well data dict shaped exactly like the output of
`mxtreme.extract.extract`, so the cleaning steps / pipeline / io can be tested without reading a real
`.h5` file.
"""

import numpy as np
import pytest

SPIKE_DTYPE = np.dtype([("frameno", "<i8"), ("channel", "<i4"), ("amplitude", "<f8")])
MAPPING_DTYPE = np.dtype([("channel", "<i4"), ("electrode", "<i4"), ("x", "<f8"), ("y", "<f8")])


def _spikes(rows):
    return np.array(rows, dtype=SPIKE_DTYPE)


BURST_SPIKE_DTYPE = np.dtype([("frameno", "<i8"), ("channel", "<i4"), ("amplitude", "<f4")])


@pytest.fixture
def make_recording_data():
    """Return a factory building a synthetic preprocessed-``.npz``-shaped dict for burst tests.

    Shaped like :func:`mxtreme.io.load_preprocessed` output (keys ``spike_data``, ``spike_bin``,
    ``channelmap``, ``samp_rate``, ``bin_size``, identity fields, ...). The default data has sparse
    tonic background plus two dense spike clusters (around frames 100k and 300k) across all channels,
    so the ``isi_rate`` detector reliably finds network bursts. Pass ``seed`` / ``n_chan`` to vary.
    """
    def _make(seed=0, n_chan=20, samp_rate=10000.0, bin_size=0.01, total=500000, **overrides):
        rng = np.random.default_rng(seed)
        rows = []
        # sparse tonic background (large inter-spike intervals)
        for ch in range(n_chan):
            for f in range(3000, total, 5000):
                rows.append((f + int(rng.integers(0, 50)), ch, -5e-5))
        # two dense bursts (small inter-spike intervals) across every channel
        for center in (100000, 300000):
            for ch in range(n_chan):
                for _ in range(15):
                    rows.append((center + int(rng.integers(0, 4000)), ch, -8e-5))

        spikes = np.array(sorted(rows), dtype=BURST_SPIKE_DTYPE)

        frames_per_bin = int(samp_rate * bin_size)
        n_bins = total // frames_per_bin + 1
        spike_bin = np.zeros((n_chan, n_bins))
        bins = (spikes["frameno"] / frames_per_bin).astype(int)
        spike_bin[spikes["channel"], bins] = 1.0

        # channelmap columns: index, channel, electrode, x(µm), y(µm)
        channelmap = np.array(
            [[i, i, i + 100, (i % 10) * 350.0, (i // 10) * 350.0] for i in range(n_chan)], dtype=float
        )

        data = {
            "spike_data": spikes,
            "channelmap": channelmap,
            "stim_elecs": np.array([]),
            "rec_t_sec": np.array([total / samp_rate]),
            "samp_rate": np.array([samp_rate]),
            "lsb": np.array([1.0]),
            "eventtime": np.array([], dtype="<i8"),
            "event_messages": [],
            "spike_bin": spike_bin,
            "bin_size": np.array(bin_size),
            "exp_condition": np.asarray({"left_stim": 0, "right_stim": 0}),
            "DIV": np.array(7),
            "plate_date": np.array("250101"),
            "well": np.array(0),
            "chip": np.array("C0001"),
            "exp_id": np.array("testExp"),
            "stim_frames": np.array([]),
            "raw_start": np.array(0),
            "path_to_h5": np.array("/tmp/fake.raw.h5"),
        }
        data.update(overrides)
        return data

    return _make


@pytest.fixture
def make_well():
    """Return a factory that builds a fresh synthetic well dict on each call.

    Defaults: sample rate 20 kHz, ``lsb=1.0`` (so amplitudes are already 'volts'), three mapped channels
    (0, 1, 2), and a handful of spikes. Pass ``spikes`` / ``mapping`` / other keys to override.
    """
    def _make(**overrides):
        # frameno, channel, amplitude(V). Includes: sub-threshold, positive, a refractory pair on ch0,
        # and a spike on unmapped channel 9.
        spikes = _spikes([
            (100, 0, -5e-5),
            (120, 0, -3e-5),   # 20 frames after the first ch0 spike (1 ms @ 20 kHz -> within 2 ms refractory)
            (500, 1, -8e-5),
            (900, 2, -1e-5),   # sub-threshold (|amp| < 20 µV)
            (950, 2,  6e-5),   # positive deflection
            (700, 9, -9e-5),   # channel not in mapping
        ])
        mapping = np.array([(0, 10, 0.0, 0.0), (1, 11, 17.5, 0.0), (2, 12, 35.0, 0.0)], dtype=MAPPING_DTYPE)
        well = {
            "well": 0,
            "exp_id": "testExp",
            "chip": "C0001",
            "plate_date": 250101,
            "DIV": 7,
            "path_to_h5": "/tmp/fake.raw.h5",
            "data": spikes,
            "samp_rate": np.float64(20000.0),
            "mapping": mapping,
            "lsb": np.array([1.0]),
            "stim_elecs": None,
            "raw_start": 50,
            "eventtime": np.array([], dtype="<i8"),
            "event_messages": [],
            "experimental_condition": np.asarray({"left_stim": 0, "right_stim": 0}),
        }
        well.update(overrides)
        return well

    return _make
