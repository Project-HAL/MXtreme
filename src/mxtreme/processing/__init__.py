"""Thin drivers for the burst stages of the pipeline: npz -> bursts -> burst features.

These wrap `mxtreme.recording` and establish the on-disk layout that `mxtreme.paths` and
`mxtreme.analysis` expect.

The raw `.h5` -> npz stage now lives in the composable `mxtreme.extract` / `mxtreme.clean` /
`mxtreme.pipeline` modules (see `examples/preprocess_pipeline.ipynb`).
"""
