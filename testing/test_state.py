"""State vectors, their PCA trajectory, and burst loops."""

import numpy as np

from mxtreme.analysis.regions import grid_regions, membership
from mxtreme.analysis.state import burst_loops, loop_clusters, loop_distances, state_vectors, trajectory
from mxtreme.recording import Recording


def test_state_vectors_by_channel_and_by_region(make_recording_data):
    rec = Recording(0, make_recording_data())
    end = int(rec.spike_data["frameno"].max())
    sv = state_vectors(rec, start_frame=0, end_frame=end, bin_sec=0.01)
    n_chan = rec.channelmap.shape[0]
    assert sv["X"].shape[0] == n_chan and sv["X"].shape[1] == len(sv["time_sec"])
    assert sv["units"] == [int(c) for c in rec.channelmap[:, 1]]
    assert abs(sv["bin_sec"] - 0.01) < 1e-12
    # standardised: each unit that fired has mean ~0 and unit variance
    live = sv["X"].std(axis=1) > 0
    assert np.allclose(sv["X"][live].mean(axis=1), 0, atol=1e-4) and np.allclose(sv["X"][live].std(axis=1), 1, atol=1e-3)
    # the two dense clusters of the fixture (around 100k and 300k frames) are the two peaks of the mean state
    mean = sv["X"].mean(axis=0)
    t = sv["time_sec"]
    assert abs(t[np.argmax(mean)] - 10.0) < 0.6 or abs(t[np.argmax(mean)] - 30.0) < 0.6
    regions = grid_regions(2, 1)
    groups = membership(rec.channelmap, regions)
    sr = state_vectors(rec, start_frame=0, end_frame=end, bin_sec=0.02, groups=groups, max_bins=1000)
    assert sr["units"] == ["g0_0", "g0_1"] and sr["X"].shape[1] <= 1000 and sr["bin_sec"] >= 0.02
    assert sum(sr["n_channels"]) == n_chan


def test_trajectory_and_loops(make_recording_data):
    rec = Recording(0, make_recording_data())
    end = int(rec.spike_data["frameno"].max())
    sv = state_vectors(rec, start_frame=0, end_frame=end, bin_sec=0.01)
    tr = trajectory(sv["X"], n_components=3)
    assert tr["scores"].shape == (sv["X"].shape[1], 3)
    assert 0 < tr["var_frac"][0] <= 1 and tr["effective_rank"] >= 1
    loops = burst_loops(tr["scores"], sv["time_sec"], [10.0, 30.0, 49.99], pre_sec=0.05, post_sec=0.5, n_points=20)
    assert loops["kept"] == [0, 1]  # the last window runs past the data
    assert loops["loops"].shape == (2, 20, 3) and loops["t_rel"][0] == -0.05 and loops["t_rel"][-1] == 0.5
    D = loop_distances(loops["loops"])
    assert D.shape == (2, 2) and D[0, 0] == 0 and D[0, 1] == D[1, 0] >= 0
    assert loop_clusters(D, 2).tolist() in ([1, 2], [2, 1])
    assert loop_clusters(D, 1).tolist() == [1, 1]
    assert loop_clusters(np.zeros((0, 0)), 3).size == 0
    same = np.stack([loops["loops"][0], loops["loops"][0]])
    assert loop_distances(same)[0, 1] < 1e-6
