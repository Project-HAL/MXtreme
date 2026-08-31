"""The scans subpackage: import boundaries, packaging, and per-well selection independence.

`mxtreme.scans.mx_setup` needs MaxWell's proprietary ``maxlab``, which ships with MaxLab Live and is
not installable from any index. Nothing else in the package may depend on it, so these tests pin the
boundary: everything except ``mx_setup`` must import on a machine that has never seen a rig.
"""

import importlib
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from mxtreme import device

MAXLAB_PRESENT = importlib.util.find_spec("maxlab") is not None


def test_scans_is_a_real_package():
    """``scans`` must be a regular package, not an implicit namespace package.

    ``[tool.setuptools.packages.find]`` only collects directories containing an ``__init__.py``. A
    namespace package imports fine from an editable install (``src/`` is on ``sys.path``) but is
    silently omitted from the built wheel, so this guards the packaging, not the import.
    """
    import mxtreme.scans

    assert mxtreme.scans.__file__ is not None
    assert mxtreme.scans.__file__.endswith("__init__.py")


def test_maxlab_free_modules_import_without_maxlab():
    """Importing the subpackage and its offline modules must not pull in maxlab."""
    import mxtreme.scans
    from mxtreme.scans import electrode_selection, mx_config

    # Reachable as attributes of the package as well as by direct import.
    assert mxtreme.scans.electrode_selection is electrode_selection
    assert mxtreme.scans.mx_config is mx_config
    # And none of that dragged the rig library in behind it.
    assert "maxlab" not in sys.modules


@pytest.mark.skipif(MAXLAB_PRESENT, reason="maxlab installed; this asserts the no-maxlab path")
def test_mx_setup_raises_actionable_error_without_maxlab():
    """The failure must name MaxLab Live and point at the module that works without it."""
    sys.modules.pop("mxtreme.scans.mx_setup", None)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module("mxtreme.scans.mx_setup")

    message = str(excinfo.value)
    assert "MaxLab Live" in message
    assert "electrode_selection" in message


def test_make_config_args_uses_device_geometry():
    """Config coordinates come from ``mxtreme.device``, not duplicated literals.

    Electrode 221 sits one row down and one column across on a ``CHIP_WIDTH``-wide array, so both of
    its coordinates are exactly one electrode pitch.
    """
    from mxtreme.scans import mx_config

    (channel, electrode, x, y), = mx_config.make_config_args([device.CHIP_WIDTH + 1])

    assert channel == -1
    assert electrode == str(device.CHIP_WIDTH + 1)
    assert float(x) == pytest.approx(device.ELEC_SIZE)
    assert float(y) == pytest.approx(device.ELEC_SIZE)

    # Electrode 0 is the origin.
    (_, _, x0, y0), = mx_config.make_config_args([0])
    assert (float(x0), float(y0)) == (0.0, 0.0)


def _packed_well(n_elec=6, spacing=1.0, first_electrode=0):
    """A well whose electrodes are packed far closer together than any sane distance threshold.

    Forces `network_selection` into its threshold-relaxation loop, which is the code path that used
    to leak state into the next well.
    """
    electrodes = list(range(first_electrode, first_electrode + n_elec))
    mapping = pd.DataFrame({
        "channel": range(n_elec),
        "electrode": electrodes,
        "x": [i * spacing for i in range(n_elec)],
        "y": [0.0] * n_elec,
    })
    active = pd.DataFrame({
        "frameno": range(n_elec),
        "channel": range(n_elec),
        "amplitude": [-5e-5] * n_elec,
        "electrode": electrodes,
    })
    return {"mapping": mapping, "active_electrodes": active}


def test_dist_thresh_relaxation_restarts_for_each_well(capsys):
    """Each well must begin its threshold search from the caller's value, not a neighbour's leftovers.

    The relaxation loop steps the threshold down until candidates appear. It used to run on the
    function-local ``dist_thresh`` shared by every iteration of the ``for well in data`` loop, so a
    well that exhausted its candidates left the threshold floored for every well after it.

    Two identical wells packed tighter than the threshold must therefore each walk the same sequence
    100 -> 83 -> 66 -> 49 -> 32 -> 15 and print it. With the leak, the second well inherits 15,
    stops immediately, and prints nothing -- so the number of relaxation lines halves.

    Asserting on the printed trace rather than the selected electrodes is deliberate: with the
    threshold floored, both wells select a single electrode either way, so the selection itself
    cannot distinguish the two behaviours.
    """
    from mxtreme.scans import electrode_selection

    np.random.seed(0)
    data = {0: _packed_well(), 1: _packed_well(first_electrode=100)}

    electrode_selection.network_selection(data, dist_thresh=100, max_num_electrodes=3)

    relaxations = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("New distance threshold:")
    ]

    # Five steps per well (83, 66, 49, 32, 15), for two wells.
    assert len(relaxations) == 10, relaxations
    assert relaxations[:5] == relaxations[5:], "second well did not restart from the caller's value"


# --- where a scan is written --------------------------------------------------------------------


def _params(**overrides):
    from mxtreme.scans.activity_scan import ActivityScanParams

    defaults = {"exp_id": "expA", "chip": "C0001", "plate_date": 250512, "div": 14, "wells": [0]}
    return ActivityScanParams(**{**defaults, **overrides})


def test_scan_defaults_into_the_managed_store(tmp_path):
    """No save_path means the store: <scans_dir>/<exp_id>/<chip>/, mirroring the other trees."""
    from mxtreme.config import Config

    config = Config(data_root=tmp_path)
    resolved = _params().resolved(config)

    assert resolved.h5_path.parent == config.scans_dir / "expA" / "C0001"
    assert resolved.h5_path.name == "DIV14_250512_C0001_expA_activity_scan.raw.h5"


def test_resolved_returns_a_copy_and_leaves_the_caller_s_params_alone(tmp_path):
    from mxtreme.config import Config

    params = _params()
    resolved = params.resolved(Config(data_root=tmp_path))

    assert params.save_path is None
    assert resolved is not params and resolved.save_path is not None


def test_an_explicit_save_path_wins(tmp_path):
    from mxtreme.config import Config

    params = _params(save_path=str(tmp_path / "scratch"))
    resolved = params.resolved(Config(data_root=tmp_path))

    assert resolved is params
    assert resolved.h5_path.parent == tmp_path / "scratch"


def test_no_config_and_no_save_path_is_an_error():
    params = _params()
    with pytest.raises(ValueError, match="needs somewhere to write"):
        params.resolved()
    with pytest.raises(ValueError, match="save_path is unset"):
        _ = params.h5_path


def test_file_name_separates_scans_of_one_chip_by_div():
    """Without the DIV in the name, two scans of a chip collide and MaxLab writes the second as _1."""
    assert _params(div=14).file_name != _params(div=21).file_name


def test_describe_reports_an_unresolved_destination_rather_than_raising():
    """Planning happens before a Config is necessarily in hand."""
    from mxtreme.scans.activity_scan import describe

    text = describe(_params())
    assert "the managed store" in text
    assert "DIV14_250512_C0001_expA_activity_scan.raw.h5" in text


def test_run_activity_scan_rejects_a_stale_positional_seed():
    """`config` took the position `seed` used to hold; a stale call must fail loudly, not silently."""
    from mxtreme.scans.activity_scan import run_activity_scan

    with pytest.raises(TypeError, match="expects a Config"):
        run_activity_scan(_params(), 42)


def test_register_scan_only_for_a_scan_that_landed_in_the_store(tmp_path):
    """A registry row for a file outside the store could never be resolved back to it."""
    import pandas as pd

    from mxtreme.config import Config
    from mxtreme.scans.activity_scan import ActivityScanResult, _register_scan

    config = Config(data_root=tmp_path)
    params = _params(save_path=str(tmp_path / "scratch"))
    result = ActivityScanResult(
        h5_path=params.h5_path, params=params, scan_electrodes={0: [[1, 2]]},
        completed_scans=1, duration_sec=1.0,
    )

    _register_scan(result, None, lambda _: None)  # registry_path=None: written outside the store
    assert not config.registry_path.exists()

    _register_scan(result, config.registry_path, lambda _: None)
    assert len(pd.read_csv(config.registry_path)) == 1


def test_register_scan_survives_a_registry_it_cannot_write(tmp_path):
    """Bookkeeping must never cost a scan that is already safely on disk."""
    from mxtreme.scans.activity_scan import ActivityScanResult, _register_scan

    params = _params(save_path=str(tmp_path))
    result = ActivityScanResult(
        h5_path=params.h5_path, params=params, scan_electrodes={}, completed_scans=1,
        duration_sec=1.0,
    )

    messages = []
    # A directory where the registry CSV should be: writing it raises OSError.
    blocked = tmp_path / "registry.csv"
    blocked.mkdir()
    _register_scan(result, blocked, messages.append)

    assert any("could not register" in m for m in messages)
