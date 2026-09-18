from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
FakePathBase = type(Path())


class FakePath(FakePathBase):
    @classmethod
    def home(cls):
        return cls("/tmp/fake-home")


def load_path_config(environ: dict[str, str] | None = None, *, path_class=FakePath, os_module=None):
    source = ast.parse((ROOT / "app.py").read_text())
    assignments = [
        node for node in source.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in {"DEFAULT_MODEL_PATH", "MODEL_PATH"}
            for target in node.targets
        )
    ]
    if os_module is None:
        os_module = SimpleNamespace(environ={} if environ is None else environ)
    namespace = {"Path": path_class, "os": os_module}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "app.py", "exec"), namespace)
    return namespace["DEFAULT_MODEL_PATH"], namespace["MODEL_PATH"]


class AppConfigTests(unittest.TestCase):
    def test_default_model_path_uses_current_home_directory(self):
        default_model_path, model_path = load_path_config({})
        expected = FakePath.home() / "projects/ternary-bonsai-2/bonsai2-27b-mlx"
        self.assertEqual(default_model_path, str(expected))
        self.assertEqual(model_path, expected.resolve())

    def test_environment_override_is_preserved(self):
        override = "/tmp/custom-bonsai-model"
        _, model_path = load_path_config({"BONSAI_MODEL_PATH": override})
        self.assertEqual(model_path, FakePath(override).resolve())

    def test_environment_override_expands_user_home(self):
        with patch.dict(
            os.environ,
            {"HOME": str(FakePath.home()), "BONSAI_MODEL_PATH": "~/custom-bonsai-model"},
        ):
            _, model_path = load_path_config(path_class=Path, os_module=os)
        self.assertEqual(model_path, (FakePath.home() / "custom-bonsai-model").resolve())


if __name__ == "__main__":
    unittest.main()
