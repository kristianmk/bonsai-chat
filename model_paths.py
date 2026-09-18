"""Where the Bonsai model pack lives. Standard library only.

Shared by app.py and install_model.py so both agree on the location.
"""
from __future__ import annotations

import os
from pathlib import Path

MODEL_REPO = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
MODEL_FOLDER = MODEL_REPO.split("/", 1)[1]

# Written by the installer when a non-default models directory is chosen.
POINTER_FILE = Path(__file__).resolve().parent / ".bonsai-model-path"

# Location recommended by earlier READMEs. Still honoured if it is the only copy.
LEGACY_MODEL_SUBPATH = "projects/ternary-bonsai-2/bonsai2-27b-mlx"


def models_dir(environ=None, home: Path | None = None) -> Path:
    """The shared models directory: $BONSAI_MODELS_DIR, else ~/models."""
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else home
    raw = environ.get("BONSAI_MODELS_DIR", "").strip()
    return Path(raw).expanduser() if raw else home / "models"


def default_model_path(environ=None, home: Path | None = None) -> Path:
    """The model's own folder inside the models directory."""
    return models_dir(environ, home) / MODEL_FOLDER


def legacy_model_path(home: Path | None = None) -> Path:
    home = Path.home() if home is None else home
    return home / LEGACY_MODEL_SUBPATH


def read_pointer(pointer: Path | None = None) -> Path | None:
    pointer = POINTER_FILE if pointer is None else pointer
    try:
        raw = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return Path(raw).expanduser() if raw else None


def resolve_model_path(environ=None, home: Path | None = None,
                       pointer: Path | None = None) -> tuple[Path, str]:
    """Return (path, source). The path is not required to exist.

    Order: BONSAI_MODEL_PATH, the installer's pointer file, the models
    directory, then the legacy location if only that one exists.
    """
    environ = os.environ if environ is None else environ
    raw = environ.get("BONSAI_MODEL_PATH", "").strip()
    if raw:
        return Path(raw).expanduser().resolve(), "BONSAI_MODEL_PATH"

    pointed = read_pointer(pointer)
    if pointed is not None:
        return pointed.resolve(), "installer"

    default = default_model_path(environ, home)
    legacy = legacy_model_path(home)
    if not default.is_dir() and legacy.is_dir():
        return legacy.resolve(), "legacy"
    return default.resolve(), "default"
