"""Tests for the helpers in ``mxtreme.utils``."""

import logging

import numpy as np

from mxtreme import utils

SPIKE_DTYPE = np.dtype([("frameno", "<i8"), ("channel", "<i4"), ("amplitude", "<f4")])


def _spikes(rows):
    return np.array(rows, dtype=SPIKE_DTYPE)


def _per_channel_blocks(n_chan):
    """Frames ascending within each channel but resetting at the boundary (n_chan - 1 descents).

    What a per-channel concatenation looks like -- the pattern a drop-the-descents repair would gut.
    """
    return _spikes([(f, ch, -5e-5) for ch in range(n_chan) for f in range(0, 200, 10)])


def test_sorted_input_returned_unchanged():
    # The fast path returns the input array itself, so already-clean data costs nothing to check.
    spikes = _spikes([(100, 0, -5e-5), (200, 1, -6e-5), (300, 2, -7e-5)])
    assert utils.sort_spike_data(spikes) is spikes


def test_out_of_order_tail_spike_is_repaired():
    # The shape of the real bug: one spike from the final buffer flush landing after its successor.
    spikes = _spikes([(100, 0, -5e-5), (200, 1, -6e-5), (300, 2, -7e-5), (250, 3, -8e-5)])
    ordered = utils.sort_spike_data(spikes)

    assert ordered["frameno"].tolist() == [100, 200, 250, 300]
    assert len(ordered) == len(spikes)                       # repaired, not filtered
    assert sorted(ordered["channel"].tolist()) == [0, 1, 2, 3]


def test_ties_keep_recorded_order():
    # Spikes sharing a frame must keep their recorded channel order. np.sort(order="frameno") would
    # break these ties with the remaining dtype fields and reshuffle them.
    spikes = _spikes([(100, 7, -5e-5), (100, 3, -6e-5), (100, 5, -7e-5), (50, 9, -8e-5)])
    ordered = utils.sort_spike_data(spikes)

    assert ordered["frameno"].tolist() == [50, 100, 100, 100]
    assert ordered["channel"][1:].tolist() == [7, 3, 5]


def test_empty_and_single_element_are_noops():
    empty = _spikes([])
    single = _spikes([(100, 0, -5e-5)])
    assert utils.sort_spike_data(empty) is empty
    assert utils.sort_spike_data(single) is single


def test_isolated_descent_logs_at_info(caplog):
    spikes = _spikes([(100, 0, -5e-5), (300, 1, -6e-5), (250, 2, -7e-5)])
    with caplog.at_level(logging.INFO, logger="mxtreme.utils"):
        utils.sort_spike_data(spikes)

    assert [r.levelno for r in caplog.records] == [logging.INFO]
    assert "1 descent(s)" in caplog.records[0].getMessage()


def test_structurally_misordered_data_logs_at_warning(caplog):
    spikes = _per_channel_blocks(12)  # 11 descents, above the default threshold of 10
    with caplog.at_level(logging.INFO, logger="mxtreme.utils"):
        ordered = utils.sort_spike_data(spikes)

    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "11 descent(s)" in caplog.records[0].getMessage()
    assert len(ordered) == len(spikes)                       # every spike survives the repair
    frameno = ordered["frameno"]
    assert np.all(frameno[:-1] <= frameno[1:])


def test_warn_threshold_is_tunable(caplog):
    spikes = _per_channel_blocks(12)
    with caplog.at_level(logging.INFO, logger="mxtreme.utils"):
        utils.sort_spike_data(spikes, warn_threshold=20)

    assert [r.levelno for r in caplog.records] == [logging.INFO]
