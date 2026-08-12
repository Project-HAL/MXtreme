"""Smoke tests for the data-level visualizations (``mxtreme.visualizations``).

Uses the non-interactive Agg backend; asserts the plots build without error, not their appearance.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mxtreme import visualizations as viz


def _spike_bin():
    rng = np.random.default_rng(0)
    return (rng.random((32, 500)) > 0.9).astype(float)


def _channelmap(n=32):
    cm = np.zeros((n, 5))
    cm[:, 0] = np.arange(n)
    cm[:, 1] = np.arange(n)
    cm[:, 2] = np.arange(n)
    cm[:, 3] = np.linspace(0, 3000, n)   # x (µm)
    cm[:, 4] = np.linspace(0, 1800, n)   # y (µm)
    return cm


def test_plot_asdr_on_ax():
    fig, ax = plt.subplots()
    out = viz.plot_asdr(_spike_bin(), ax=ax)
    assert out is ax
    plt.close(fig)


def test_raster_on_ax():
    fig, ax = plt.subplots()
    out = viz.raster(_spike_bin(), zoom=(0, 200), ax=ax)
    assert out is ax
    plt.close(fig)


def test_mea_runs_with_and_without_stim():
    fig, ax = plt.subplots()
    viz.MEA(ax, _channelmap(), stim_elecs=None)
    viz.MEA(ax, _channelmap(), stim_elecs=np.array([0, 5]))
    plt.close(fig)
