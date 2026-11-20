"""``extract`` on a file with an empty well, built by hand so it runs anywhere.

A MaxTwo file can hold a well with no frames beside wells that recorded (seen on mx2,
2026-11-13: one well of six in a CAIT recording). Asked for plainly, that well is an error;
given ``on_empty``, it is reported and left out, and the other wells are still extracted.
"""

import json

import numpy as np
import pytest

from mxtreme import extract

_SPIKES = np.dtype([("frameno", "<i8"), ("channel", "<i4"), ("amplitude", "<f4")])
_BLOB = {"Batch ID": "fall2026_batch3_E18", "Chip ID": "M09473", "Plate date": 261013, "DIV": 24}


def _h5(path, frames):
    """A wells-format file with one well per entry of ``frames``: how many raw frames it holds
    (``None`` for a well with no raw group at all, which then has no spikes either)."""
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset("assay/metadata", data=np.array([json.dumps(_BLOB).encode()]))
        for well, n in enumerate(frames):
            rec = f"recordings/rec0000/well00{well}"
            spikes = np.zeros(3 if n else 0, dtype=_SPIKES)
            spikes["frameno"] = 1000 + np.arange(len(spikes)) * 7
            f.create_dataset(f"{rec}/spikes", data=spikes)
            f.create_dataset(f"{rec}/settings/sampling", data=np.array([10000.0]))
            f.create_dataset(f"{rec}/settings/mapping", data=np.zeros(2, dtype=[("channel", "<i4"), ("electrode", "<i4"), ("x", "<f8"), ("y", "<f8")]))
            f.create_dataset(f"{rec}/settings/lsb", data=np.array([6.3e-6]))
            groups = f.create_group(f"wells/well{well:03d}/rec0000/groups")
            if n is not None:
                groups.create_dataset("routed/frame_nos", data=1000 + np.arange(n))
    return path


def test_an_empty_well_raises_unless_the_caller_takes_it(tmp_path):
    path = _h5(tmp_path / "six.raw.h5", [50, 50, 0, 50])

    with pytest.raises(ValueError, match="holds no data for well 2: the recording has 0 frames"):
        extract.extract(str(path))

    left_out = []
    data = extract.extract(str(path), on_empty=lambda well, why: left_out.append((well, why)))
    assert sorted(data) == [0, 1, 3] and left_out == [(2, "the recording has 0 frames")]
    assert data[3]["raw_start"] == 1000 and len(data[3]["spike_data"]) == 3

    # asked for alone, the empty well is still the caller's to take or refuse
    assert extract.extract(str(path), wells=2, on_empty=lambda *_: None) == {}
    assert sorted(extract.extract(str(path), wells=[0, 1])) == [0, 1]


def test_a_well_with_no_raw_group_and_no_spikes_is_empty_too(tmp_path):
    path = _h5(tmp_path / "bare.raw.h5", [None, 50])

    with pytest.raises(ValueError, match="well 0: no raw frames and no spikes"):
        extract.extract(str(path))
    why = {}
    assert sorted(extract.extract(str(path), on_empty=why.__setitem__)) == [1] and why == {0: "no raw frames and no spikes"}


def test_every_well_empty_is_an_empty_answer(tmp_path):
    path = _h5(tmp_path / "none.raw.h5", [0, 0])
    assert extract.extract(str(path), on_empty=lambda *_: None) == {}
