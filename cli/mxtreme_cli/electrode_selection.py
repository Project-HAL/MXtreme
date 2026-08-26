"""Electrode Selection screen: turn an activity scan into a network-scan electrode list.

Wraps :mod:`mxtreme.scans.electrode_selection`. The pipeline's stages are driven individually rather
than through :func:`~mxtreme.scans.electrode_selection.select_electrodes` for two reasons: the
selection parameters (distance threshold, routing limit) become editable, and both the activity-scan
figures and the network results land in one output directory the user chose, whatever the input file
happens to be named.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # every figure is written to disk; nothing is ever shown interactively

import numpy as np  # noqa: E402  -- after the backend is fixed

from mxtreme.scans import electrode_selection as es  # noqa: E402
from mxtreme_cli import prompts, ui  # noqa: E402
from mxtreme_cli.prompts import Param  # noqa: E402


@dataclass
class SelectionParams:
    """User-facing settings for one run of the selection pipeline.

    :param h5_path: The activity scan to read.
    :param output_dir: Directory for the figures and the ``.npz`` electrode lists.
    :param file_stem: Base name for every output file.
    :param dist_thresh: Minimum spacing in µm between consecutively selected electrodes. Relaxed
        automatically, per well, when no candidate is left that far away.
    :param max_electrodes: Most electrodes to select per well -- the chip's routing limit.
    :param seed: Seed for the random selection, or ``None`` for fresh randomness.
    """

    h5_path: Path
    output_dir: str
    file_stem: str
    dist_thresh: int = 100
    max_electrodes: int = 1020
    seed: int | None = None

    def validate(self) -> None:
        """:raises ValueError: If a setting would fail once the pipeline is running."""
        if self.dist_thresh < 0:
            raise ValueError("Distance threshold must be non-negative.")
        if not 1 <= self.max_electrodes <= 1020:
            raise ValueError("Electrodes per well must be in 1..1020 (the chip's routing limit).")
        if not self.file_stem:
            raise ValueError("Output file prefix cannot be empty.")


PARAM_FIELDS = [
    Param("output_dir", "Output directory", prompts.parse_dir, help="Figures and .npz lists go here"),
    Param("file_stem", "Output file prefix", prompts.parse_text, help="Files are <prefix>_well<N>.*"),
    Param(
        "dist_thresh",
        "Min. electrode spacing",
        prompts.parse_int(0, 4000),
        help="µm between consecutive picks; relaxed automatically when unreachable",
    ),
    Param(
        "max_electrodes",
        "Electrodes per well",
        prompts.parse_int(1, 1020),
        help="Routing limit for the network scan",
    ),
    Param(
        "seed",
        "Random seed",
        prompts.parse_optional_int,
        format=lambda v: "none" if v is None else str(v),
        help="Set for a reproducible selection; blank for none",
        allow_blank=True,
    ),
]


def default_params(h5_path: Path) -> SelectionParams:
    """Build sensible defaults for a given scan file: outputs beside the file, named after it."""
    stem = h5_path.name
    for suffix in (".raw.h5", ".h5"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    # MaxLab Live names its own recordings 'data.raw.h5', which makes a useless output prefix.
    if stem in ("data", ""):
        stem = h5_path.parent.name or "activity_scan"

    return SelectionParams(
        h5_path=h5_path,
        output_dir=str(h5_path.parent / "electrode_selection"),
        file_stem=stem,
    )


def screen(h5_path: Path | None = None) -> None:
    """Run the Electrode Selection screen.

    :param h5_path: Scan to work on. When ``None`` the user is asked for one -- this is the normal
        path from the main menu; a scan that has just finished passes its own file straight in.
    """
    ui.banner(
        "Electrode Selection",
        "Pick network-scan recording electrodes from an activity scan",
    )

    try:
        if h5_path is None:
            ui.hint("Point at the activity scan's .h5 file (MaxLab writes these as *.raw.h5).")
            h5_path = prompts.ask(
                "Activity scan .h5", parse=prompts.parse_existing_file(".h5")
            )

        params = default_params(h5_path)
        ui.info(f"\nActivity scan: {params.h5_path}")

        if not prompts.edit_params(
            params,
            PARAM_FIELDS,
            title="Selection parameters",
            validate=lambda p: p.validate(),
        ):
            ui.warn("Cancelled.")
            return

        if not prompts.confirm("\nRun electrode selection?", default=True):
            ui.warn("Cancelled.")
            return

        _run(params)

    except prompts.Cancelled:
        ui.warn("Cancelled.")
        return

    prompts.pause()


def _run(params: SelectionParams) -> None:
    """Drive the selection pipeline and report what it produced."""
    output_dir = Path(params.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if params.seed is not None:
        # network_selection() draws from the global numpy RNG, so seeding it is what makes a run
        # reproducible.
        np.random.seed(params.seed)

    ui.section("Running")
    try:
        ui.info("Loading activity scan (this takes a moment for a long scan)...")
        data = es.load_activity_scan(str(params.h5_path))
        ui.success(f"  Loaded {len(data)} well(s): {sorted(data)}")

        ui.info("Filtering to active electrodes...")
        data = es.get_active_electrodes(data)
        for well in sorted(data):
            n_active = len(np.unique(data[well]["active_electrodes"]["electrode"]))
            n_total = len(np.unique(data[well]["spike_data"]["electrode"]))
            ui.bullet(f"Well {well}: {n_active} active of {n_total} electrodes with spikes")

        ui.info("Writing activity-scan summary figures...")
        es.activity_scan_results(data, savepath=str(output_dir), savefilename=f"{params.file_stem}_AS")

        ui.info("Selecting recording electrodes...")
        data = es.network_selection(
            data,
            dist_thresh=params.dist_thresh,
            max_num_electrodes=params.max_electrodes,
        )

        ui.info("Writing network figures and electrode lists...")
        es.network_scan_results(data, savepath=str(output_dir), savefilename=f"{params.file_stem}_network")
        es.save_network(data, savepath=str(output_dir), savefilename=f"{params.file_stem}_network")

    except (OSError, KeyError, ValueError, IndexError) as exc:
        ui.error(f"Electrode selection failed: {type(exc).__name__}: {exc}")
        ui.hint("Check that the file is an activity scan (a multi-recording sweep), not a single recording.")
        return

    ui.section("Results")
    ui.key_values(
        [
            (f"Well {well}", f"{len(data[well]['recording_electrodes'])} electrodes selected")
            for well in sorted(data)
        ]
    )
    ui.info("")
    ui.success(f"Saved to {output_dir}")
    ui.bullet(f"{params.file_stem}_AS_well<N>.png -- activity summary per well")
    ui.bullet(f"{params.file_stem}_network_well<N>.png -- selected electrodes per well")
    ui.bullet(f"{params.file_stem}_network_well<N>.npz -- electrode lists ('recording_electrodes')")
