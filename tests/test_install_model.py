"""Installer logic with a small fake model pack. No network, no huggingface_hub."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import install_model
from model_paths import LEGACY_MODEL_SUBPATH, MODEL_FOLDER

PACK_FILES = {
    "config.json": b"{}",
    "model.safetensors": b"weights" * 100,
    "runtime/vision_artifact.py": b"# loader\n",
    "runtime/requirements.txt": b"mlx==0.32.0\n",
    "README.md": b"# Model card\n",
}


def write_pack(directory: Path) -> None:
    manifest = {}
    for relative, data in PACK_FILES.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest[relative] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    (directory / "files.json").write_text(json.dumps(manifest))


class InstallModelTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()
        self.models = self.tmp / "models"
        self.target = self.models / MODEL_FOLDER
        self.pointer = self.tmp / "pointer"
        self.downloads: list[Path] = []
        for name, value in {
            "POINTER_FILE": self.pointer,
            "download": self.downloads.append,
            "legacy_model_path": lambda: self.tmp / LEGACY_MODEL_SUBPATH,
            "check_free_space": lambda _target: None,
        }.items():
            patcher = patch.object(install_model, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_main(self, *argv: str) -> tuple[int, str]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = install_model.main(list(argv))
        return code, output.getvalue()

    def test_intact_pack_verifies(self):
        write_pack(self.target)
        self.assertEqual(install_model.verify(self.target, log=lambda _: None), ([], []))

    def test_damaged_weights_fail_but_changed_readme_only_warns(self):
        write_pack(self.target)
        (self.target / "README.md").write_bytes(b"# Updated model card\n")
        errors, warnings = install_model.verify(self.target, log=lambda _: None)
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)

        (self.target / "model.safetensors").write_bytes(b"WEIGHTS" * 100)  # Same size.
        errors, _ = install_model.verify(self.target, log=lambda _: None)
        self.assertEqual(errors, ["model.safetensors: sha256 mismatch"])

    def test_missing_required_file_is_an_error_even_without_manifest(self):
        write_pack(self.target)
        (self.target / "files.json").unlink()
        (self.target / "runtime/vision_artifact.py").unlink()
        errors, warnings = install_model.verify(self.target, log=lambda _: None)
        self.assertEqual(errors, ["missing required file: runtime/vision_artifact.py"])
        self.assertIn("checksums were not verified", warnings[0])

    def test_install_downloads_into_own_folder_and_records_custom_location(self):
        def fake_download(target: Path) -> None:
            self.downloads.append(target)
            write_pack(target)

        with patch.object(install_model, "download", fake_download):
            code, output = self.run_main("--models-dir", str(self.models))
        self.assertEqual(code, 0, output)
        self.assertEqual(self.downloads, [self.target])
        self.assertEqual(self.pointer.read_text().strip(), str(self.target))

    def test_existing_legacy_copy_stops_before_downloading(self):
        write_pack(self.tmp / LEGACY_MODEL_SUBPATH)
        code, output = self.run_main("--models-dir", str(self.models))
        self.assertEqual(code, 2)
        self.assertIn("--move-from", output)
        self.assertEqual(self.downloads, [])
        self.assertFalse(self.target.exists())

    def test_move_from_adopts_existing_copy(self):
        legacy = self.tmp / LEGACY_MODEL_SUBPATH
        write_pack(legacy)
        code, output = self.run_main("--models-dir", str(self.models), "--move-from", str(legacy))
        self.assertEqual(code, 0, output)
        self.assertFalse(legacy.exists())
        self.assertTrue((self.target / "model.safetensors").is_file())
        self.assertEqual(self.downloads, [self.target])  # Fills in anything missing.

    def test_move_from_never_overwrites_an_installed_pack(self):
        write_pack(self.target)
        other = self.tmp / "other-copy"
        write_pack(other)
        with self.assertRaises(SystemExit):
            self.run_main("--models-dir", str(self.models), "--move-from", str(other))
        self.assertTrue(other.is_dir())

    def test_verify_only_reports_damage_without_downloading(self):
        write_pack(self.target)
        (self.target / "config.json").write_bytes(b"[]")
        code, output = self.run_main("--models-dir", str(self.models), "--verify-only")
        self.assertEqual(code, 1)
        self.assertIn("config.json: sha256 mismatch", output)
        self.assertEqual(self.downloads, [])

    def test_print_path(self):
        code, output = self.run_main("--models-dir", str(self.models), "--print-path")
        self.assertEqual((code, output.strip()), (0, str(self.target)))


if __name__ == "__main__":
    unittest.main()
