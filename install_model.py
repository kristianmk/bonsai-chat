"""Download the Bonsai model pack into a models directory and verify it.

Normally called by install.sh. It can also be run directly from an environment
that has huggingface_hub installed:

    python install_model.py --models-dir ~/models

Only the download step needs huggingface_hub; verification is standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from model_paths import (
    MODEL_FOLDER, MODEL_REPO, POINTER_FILE, default_model_path, legacy_model_path, models_dir,
)

MANIFEST = "files.json"
REQUIRED = ("config.json", "model.safetensors", "runtime/vision_artifact.py", "runtime/requirements.txt")
# Used for the free-space check when the manifest is not available yet.
APPROXIMATE_BYTES = 8_700_000_000


def is_documentation(relative: str) -> bool:
    """Files the runtime never loads. A checksum mismatch here is only a warning."""
    name = relative.rsplit("/", 1)[-1]
    return (
        relative.startswith("assets/") or name.endswith(".md")
        or name in {"LICENSE", "NOTICE.txt", ".gitattributes"}
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify(model_dir: Path, *, log=print) -> tuple[list[str], list[str]]:
    """Check the pack against its own files.json. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    for relative in REQUIRED:
        if not (model_dir / relative).is_file():
            errors.append(f"missing required file: {relative}")
    manifest_path = model_dir / MANIFEST
    if not manifest_path.is_file():
        warnings.append(f"{MANIFEST} not found; checksums were not verified")
        return errors, warnings
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = sorted(manifest.items())
    except (ValueError, AttributeError) as exc:
        return errors + [f"{MANIFEST} is unreadable: {exc}"], warnings

    for relative, expected in entries:
        path = model_dir / relative
        problem = None
        if not path.is_file():
            problem = "missing"
        elif path.stat().st_size != expected.get("size"):
            problem = f"size {path.stat().st_size} != {expected.get('size')}"
        else:
            if path.stat().st_size > 256 * 1024 * 1024:
                log(f"  hashing {relative} ({path.stat().st_size / 1e9:.1f} GB)...")
            if sha256_of(path) != expected.get("sha256"):
                problem = "sha256 mismatch"
        if problem is None:
            continue
        target = warnings if is_documentation(relative) else errors
        target.append(f"{relative}: {problem}")
    return errors, warnings


def check_free_space(target: Path) -> None:
    present = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) if target.is_dir() else 0
    needed = APPROXIMATE_BYTES - present
    probe = target
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if needed > 0 and free < needed + 1_000_000_000:
        raise SystemExit(
            f"Not enough free space under {probe}: about {needed / 1e9:.1f} GB still to download, "
            f"{free / 1e9:.1f} GB free."
        )


def adopt(source: Path, target: Path) -> None:
    """Move an existing copy into the models directory instead of downloading again."""
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"--move-from: not a directory: {source}")
    if source == target.resolve():
        return
    if target.exists():
        raise SystemExit(f"--move-from: target already exists, not overwriting: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Moving {source} -> {target}")
    shutil.move(str(source), str(target))


def download(target: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed in this Python environment. "
            "Run ./install.sh, or: python -m pip install huggingface_hub"
        ) from None
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_REPO} -> {target}")
    print("Files that are already complete are skipped, so an interrupted download can be resumed.")
    snapshot_download(repo_id=MODEL_REPO, local_dir=str(target))


def write_pointer(target: Path, pointer: Path | None = None) -> None:
    """Record a non-default location so app.py finds it without environment variables."""
    pointer = POINTER_FILE if pointer is None else pointer
    if target.resolve() == default_model_path({}).resolve():
        pointer.unlink(missing_ok=True)
        return
    pointer.write_text(str(target.resolve()) + "\n", encoding="utf-8")
    print(f"Recorded model location in {pointer.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="Models directory (default: $BONSAI_MODELS_DIR or ~/models). "
                             f"The pack goes into its own '{MODEL_FOLDER}' folder inside it.")
    parser.add_argument("--move-from", type=Path, default=None, metavar="DIR",
                        help="Move an existing downloaded pack into the models directory first.")
    parser.add_argument("--download", action="store_true",
                        help="Download even if a copy exists at the legacy location.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify an installed pack.")
    parser.add_argument("--no-verify", action="store_true", help="Skip checksum verification.")
    parser.add_argument("--print-path", action="store_true",
                        help="Print the resolved model folder and exit.")
    args = parser.parse_args(argv)

    base = (args.models_dir.expanduser() if args.models_dir else models_dir()).resolve()
    target = base / MODEL_FOLDER
    if args.print_path:
        print(target)
        return 0

    if not args.verify_only:
        legacy = legacy_model_path()
        if args.move_from is not None:
            adopt(args.move_from, target)
        elif not target.exists() and legacy.is_dir() and not args.download:
            print(
                f"An existing model pack was found at the old location:\n  {legacy}\n"
                "Choose one:\n"
                f"  ./install.sh --move-from \"{legacy}\"   # move it to {target}\n"
                "  ./install.sh --download              # leave it there and download a new copy",
                file=sys.stderr,
            )
            return 2
        check_free_space(target)
        download(target)

    if not target.is_dir():
        print(f"Model folder does not exist: {target}", file=sys.stderr)
        return 1

    if args.no_verify:
        errors = [f"missing required file: {r}" for r in REQUIRED if not (target / r).is_file()]
        warnings: list[str] = []
    else:
        print(f"Verifying {target} against {MANIFEST}...")
        errors, warnings = verify(target)
    for line in warnings:
        print(f"  warning: {line}")
    for line in errors:
        print(f"  ERROR: {line}", file=sys.stderr)
    if errors:
        print("Verification failed. Delete the files listed above, then re-run the installer; "
              "it downloads whatever is missing.", file=sys.stderr)
        return 1

    if not args.verify_only:
        write_pointer(target)
    print(f"Model pack OK: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
