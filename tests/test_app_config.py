from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model_paths
from model_paths import MODEL_FOLDER, resolve_model_path


class ModelPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name).resolve() / "home"
        self.home.mkdir()
        self.pointer = self.home / "no-pointer-file"

    def resolve(self, environ=None):
        return resolve_model_path(environ or {}, self.home, self.pointer)

    def test_default_is_own_folder_in_home_models_directory(self):
        path, source = self.resolve()
        self.assertEqual(path, self.home / "models" / MODEL_FOLDER)
        self.assertEqual(source, "default")
        self.assertEqual(MODEL_FOLDER, "Ternary-Bonsai-2-27B-mlx-2bit")

    def test_models_directory_override(self):
        path, _ = self.resolve({"BONSAI_MODELS_DIR": str(self.home / "big-disk")})
        self.assertEqual(path, self.home / "big-disk" / MODEL_FOLDER)

    def test_explicit_model_path_wins_over_everything(self):
        self.pointer.write_text(str(self.home / "pointed"))
        override = self.home / "custom-bonsai-model"
        path, source = self.resolve({
            "BONSAI_MODEL_PATH": str(override), "BONSAI_MODELS_DIR": str(self.home / "other"),
        })
        self.assertEqual((path, source), (override, "BONSAI_MODEL_PATH"))

    def test_explicit_model_path_expands_user_home(self):
        path, _ = self.resolve({"BONSAI_MODEL_PATH": "~/custom-bonsai-model"})
        self.assertEqual(path, (Path.home() / "custom-bonsai-model").resolve())

    def test_installer_pointer_is_used(self):
        self.pointer.write_text(f"{self.home / 'elsewhere' / MODEL_FOLDER}\n")
        path, source = self.resolve()
        self.assertEqual((path, source), (self.home / "elsewhere" / MODEL_FOLDER, "installer"))

    def test_empty_pointer_is_ignored(self):
        self.pointer.write_text("\n")
        self.assertEqual(self.resolve()[1], "default")

    def test_legacy_location_is_used_only_when_it_is_the_only_copy(self):
        legacy = self.home / model_paths.LEGACY_MODEL_SUBPATH
        legacy.mkdir(parents=True)
        self.assertEqual(self.resolve(), (legacy, "legacy"))
        (self.home / "models" / MODEL_FOLDER).mkdir(parents=True)
        self.assertEqual(self.resolve(), (self.home / "models" / MODEL_FOLDER, "default"))

    def test_app_uses_the_shared_resolver(self):
        source = (ROOT / "app.py").read_text()
        self.assertIn("MODEL_PATH, MODEL_PATH_SOURCE = resolve_model_path()", source)


if __name__ == "__main__":
    unittest.main()
