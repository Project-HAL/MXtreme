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
    assert cfg.recordings_dir == cfg.data_root / "recordings"
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


def test_from_toml_simulations_dir_is_optional(tmp_path):
    assert Config.from_toml(_write_toml(tmp_path, '[data]\nroot = "/data/store"\n')).simulations_dir is None
    cfg = Config.from_toml(_write_toml(tmp_path, '[data]\nroot = "/data/store"\n\n[simulations]\ndir = "/local/sims"\n'))
    assert str(cfg.simulations_dir) == "/local/sims" and str(cfg.data_root) == "/data/store"
