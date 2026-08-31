"""Tests for TOML-backed configuration (``mxtreme.config``)."""

import pytest

from mxtreme.config import Config


def _write_toml(tmp_path, body):
    p = tmp_path / "mxtreme.toml"
    p.write_text(body)
    return p


def test_from_toml_resolves_directories(tmp_path):
    toml = _write_toml(tmp_path, '[data]\nroot = "/data/store"\n')
    cfg = Config.from_toml(toml)
    assert str(cfg.data_root) == "/data/store"
    assert cfg.scans_dir == cfg.data_root / "scans"
    assert cfg.preprocessed_dir == cfg.data_root / "preprocessed"
    assert cfg.burst_data_dir == cfg.data_root / "burst_data"
    assert cfg.analysis_dir == cfg.data_root / "analysis"
    assert cfg.registry_path == cfg.data_root / "registry.csv"


def test_from_toml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.from_toml(tmp_path / "nope.toml")


def test_from_toml_missing_root_raises(tmp_path):
    toml = _write_toml(tmp_path, "[data]\n")
    with pytest.raises(KeyError):
        Config.from_toml(toml)
