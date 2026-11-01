"""Prepare only the inference files for Spaces. Never upload or alter the originals."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def export(destination):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory; this already exists: {destination}")
    checkpoint = ROOT / "models/fine-tuned/best.pt"
    info = json.loads((ROOT / "oral_app/assets/model_info.json").read_text(encoding="utf-8"))
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != info["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum does not match the evaluated model.")
    destination.mkdir(parents=True)
    for filename in ["requirements.txt", "LICENSE"]:
        shutil.copy2(ROOT / filename, destination / filename)
    shutil.copytree(ROOT / "oral_app", destination / "oral_app", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "app.py").write_text("from oral_app.app import launch_app\n\nif __name__ == '__main__':\n    launch_app()\n", encoding="utf-8")
    shutil.copy2(ROOT / "docs/GRADIO_APP.md", destination / "README.md")
    models = destination / "models"
    models.mkdir()
    shutil.copy2(checkpoint.parent / "class_to_idx.json", models / "class_to_idx.json")
    # On the same disk, a hard link shares the 94 MB file's storage. Treat weights
    # as immutable: changing a hard-linked file also changes its original.
    try:
        os.link(checkpoint, models / "best.pt")
        storage = "hard link (no second allocation for weights)"
    except OSError:
        shutil.copy2(checkpoint, models / "best.pt")
        storage = "copy"
    (destination / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    (destination / ".gitignore").write_text("__pycache__/\n*.pyc\n.env\n.venv/\n", encoding="utf-8")
    print(f"Prepared: {destination}")
    print(f"Checkpoint storage: {storage}")
    print("This command performs no upload. Review example-image rights before public redistribution.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "local/space-build")
    export(parser.parse_args().output)
