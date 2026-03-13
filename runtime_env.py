"""
Runtime environment guards for local DeepShield entry scripts.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path


def _noop(*args, **kwargs):
    return None


class _DisabledRun:
    def __init__(self):
        self.config = {}
        self.summary = {}
        self.name = "disabled"
        self.id = "disabled"
        self.dir = None

    def log(self, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        return None


class _DisabledMedia:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _install_wandb_stub():
    if "wandb" in sys.modules:
        return

    module = types.ModuleType("wandb")
    module.__dict__.update(
        {
            "__all__": [],
            "__version__": "disabled-stub",
            "run": None,
            "config": {},
            "Settings": _DisabledMedia,
            "Artifact": _DisabledMedia,
            "Image": _DisabledMedia,
            "Audio": _DisabledMedia,
            "Video": _DisabledMedia,
            "Table": _DisabledMedia,
            "init": lambda *args, **kwargs: _DisabledRun(),
            "finish": _noop,
            "log": _noop,
            "watch": _noop,
            "login": _noop,
            "define_metric": _noop,
        }
    )
    sys.modules["wandb"] = module


def configure_wandb_environment(project_root: str | Path | None = None) -> dict[str, str]:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent
    root = root.resolve()

    directories = {
        "WANDB_DIR": root / ".wandb-run",
        "WANDB_CACHE_DIR": root / ".wandb-cache",
        "WANDB_DATA_DIR": root / ".wandb-data",
        "WANDB_ARTIFACT_DIR": root / ".wandb-artifacts",
    }
    temp_dir = root / ".tmp"

    for env_name, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_name, str(directory))

    temp_dir.mkdir(parents=True, exist_ok=True)
    for env_name in ("TMP", "TEMP", "TMPDIR"):
        os.environ.setdefault(env_name, str(temp_dir))
    tempfile.tempdir = str(temp_dir)

    # This repo does not use W&B directly. Users can override these externally.
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")
    if os.environ.get("WANDB_MODE") == "disabled":
        _install_wandb_stub()

    return {name: str(path) for name, path in directories.items()}
