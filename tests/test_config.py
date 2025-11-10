from pathlib import Path

# When running tests, ensure PYTHONPATH includes the project 'src' directory.
from usa_econ.config import load_config


def test_load_config_finds_project_root():
    cfg = load_config()
    assert isinstance(cfg.project_root, Path)
    assert cfg.project_root.exists()