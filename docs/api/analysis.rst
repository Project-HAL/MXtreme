Analysis
========

.. automodule:: mxtreme.analysis

Activity
--------

.. automodule:: mxtreme.analysis.activity
   :no-members:

Spiking
^^^^^^^

Recording level: one :class:`~mxtreme.recording.Recording` in, the data out.

.. autofunction:: mxtreme.analysis.activity.firing_rate
.. autofunction:: mxtreme.analysis.activity.instantaneous_firing_rate
.. autofunction:: mxtreme.analysis.activity.smoothed_firing_rate
.. autofunction:: mxtreme.analysis.activity.isi

Group level: a ``RecordingID``, ``CultureID``, ``CultureSelector``, or several selectors.

.. autofunction:: mxtreme.analysis.activity.summarize_spike_activity
.. autofunction:: mxtreme.analysis.activity.plot_spike_activity_summary

Culture level: the distributions behind the summary.

.. autofunction:: mxtreme.analysis.activity.spike_activity_distributions
.. autofunction:: mxtreme.analysis.activity.plot_spike_activity_distributions

Bursting
^^^^^^^^

Recording level: one :class:`~mxtreme.recording.Recording` and its bursts in, the data out.

.. autofunction:: mxtreme.analysis.activity.burst_rate
.. autofunction:: mxtreme.analysis.activity.ibi

Group level: a ``RecordingID``, ``CultureID``, ``CultureSelector``, or several selectors.

.. autofunction:: mxtreme.analysis.activity.summarize_burst_activity
.. autofunction:: mxtreme.analysis.activity.plot_burst_activity_summary

Culture level: the distributions behind the summary.

.. autofunction:: mxtreme.analysis.activity.burst_activity_distributions
.. autofunction:: mxtreme.analysis.activity.plot_burst_activity_distributions

Network
-------

.. automodule:: mxtreme.analysis.network
   :no-members:

Recording level: one :class:`~mxtreme.recording.Recording` in, the data out.

.. autofunction:: mxtreme.analysis.network.connectivity
.. autofunction:: mxtreme.analysis.network.pca
.. autofunction:: mxtreme.analysis.network.network_metrics
.. autofunction:: mxtreme.analysis.network.plot_pc_trajectory
.. autofunction:: mxtreme.analysis.network.animate_pc_trajectory

Community structure of a connectivity matrix (the output of :func:`~mxtreme.analysis.network.connectivity`).

.. autofunction:: mxtreme.analysis.network.communities
.. autofunction:: mxtreme.analysis.network.community_table
.. autofunction:: mxtreme.analysis.network.plot_connectivity

Group level: a ``RecordingID``, ``CultureID``, ``CultureSelector``, or several selectors.

.. autofunction:: mxtreme.analysis.network.summarize_network_metrics
.. autofunction:: mxtreme.analysis.network.plot_network_summary

Culture level: one panel or line per recording.

.. autofunction:: mxtreme.analysis.network.plot_connectivity_grid
.. autofunction:: mxtreme.analysis.network.plot_pc_variance_explained
.. autofunction:: mxtreme.analysis.network.plot_pc_trajectory_grid

Spatial
-------

.. automodule:: mxtreme.analysis.spatial
   :no-members:

Recording level: one :class:`~mxtreme.recording.Recording` in, the data out.

.. autofunction:: mxtreme.analysis.spatial.spatial_metrics

Group level: a ``RecordingID``, ``CultureID``, ``CultureSelector``, or several selectors.

.. autofunction:: mxtreme.analysis.spatial.summarize_spatial
.. autofunction:: mxtreme.analysis.spatial.plot_spatial_summary

Culture level: each distinct electrode configuration.

.. autofunction:: mxtreme.analysis.spatial.plot_mea_layouts

Stimulation
-----------

.. automodule:: mxtreme.analysis.stimulation
   :no-members:

Recording level: one :class:`~mxtreme.recording.Recording` in, the data out.

.. autofunction:: mxtreme.analysis.stimulation.stim_events
.. autofunction:: mxtreme.analysis.stimulation.stim_delivered

Group level: a ``RecordingID``, ``CultureID``, ``CultureSelector``, or several selectors.

.. autofunction:: mxtreme.analysis.stimulation.summarize_stimulation
.. autofunction:: mxtreme.analysis.stimulation.plot_stimulation_summary

Performance
-----------

.. automodule:: mxtreme.analysis.performance
   :no-members:

Recording level: one :class:`~mxtreme.recording.Recording` and its bursts in, the data out.

.. autofunction:: mxtreme.analysis.performance.performance_score
.. autofunction:: mxtreme.analysis.performance.default_direction_objective

Group level: a ``RecordingID``, ``CultureID``, ``CultureSelector``, or several selectors.

.. autofunction:: mxtreme.analysis.performance.summarize_performance
.. autofunction:: mxtreme.analysis.performance.plot_performance_summary

Reports
-------

.. automodule:: mxtreme.analysis.report
   :members:
