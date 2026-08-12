"""Integration test for ``mxtreme.extract`` against a real raw file.

Skipped automatically when the example data isn't present, so the fast unit suite still runs anywhere.
"""

from pathlib import Path

import pytest

from mxtreme import extract

_P004722 = (
    Path(__file__).resolve().parents[2]
    / "mxtreme-claude/claude_data/raw/Fall_BurstTrainer/P004722/111825_P004722_DIV48_burstTrain.raw.h5"
)

pytestmark = pytest.mark.skipif(not _P004722.exists(), reason="example raw .h5 not available")


def test_extract_reads_embedded_metadata():
    data = extract.extract(str(_P004722))
    assert list(data) == [0]                     # MaxOne -> single well
    well = data[0]
    assert well["exp_id"] == "m1BurstTrainer"
    assert well["chip"] == "P004722"
    assert well["DIV"] == 48
    assert well["well"] == 0
    for key in ("data", "samp_rate", "mapping", "lsb", "raw_start"):
        assert key in well
    assert well["data"].shape[0] > 0


def test_extract_missing_metadata_requires_manual_dict():
    # A file with embedded metadata still works; the manual-dict path is exercised by the old-file case,
    # here we just assert extract raises clearly when asked for a nonexistent file.
    with pytest.raises(FileNotFoundError):
        extract.extract("/does/not/exist.raw.h5")
