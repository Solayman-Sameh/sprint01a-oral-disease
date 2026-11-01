"""Reconstruct the selected ResNet50 and mirror its evaluation transform."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(__file__).resolve().parent / "assets"
CLASSES = ("Calculus", "Caries", "Gingivitis", "Hypodontia", "Mouth Ulcer", "Tooth Discoloration")
MAX_IMAGE_PIXELS = 20_000_000


class ImageInputError(ValueError):
    """An image cannot safely be passed through the evaluation transform."""


class PadSquare:
    def __call__(self, image):
        width, height = image.size
        side = max(width, height)
        left, top = (side - width) // 2, (side - height) // 2
        return ImageOps.expand(image, (left, top, side - width - left, side - height - top), fill=0)


eval_transform = transforms.Compose([
    PadSquare(),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def checkpoint_path():
    """Use existing local weights, or the flatter layout of an exported Space."""
    configured = os.environ.get("ORAL_MODEL_PATH")
    candidates = [Path(configured).expanduser()] if configured else [
        ROOT / "models/fine-tuned/best.pt", ROOT / "models/best.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("Selected checkpoint missing. Place best.pt in models/ or set ORAL_MODEL_PATH.")


@lru_cache(maxsize=1)
def load_model(checkpoint_file=None, device="cpu"):
    """Load once per process; fail clearly if weights or their metadata differ."""
    path = Path(checkpoint_file) if checkpoint_file else checkpoint_path()
    info = json.loads((ASSETS / "model_info.json").read_text(encoding="utf-8"))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != info["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 differs from the selected, evaluated ResNet50.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected_mapping = dict(zip(CLASSES, range(6)))
    mapping = json.loads((path.parent / "class_to_idx.json").read_text(encoding="utf-8"))
    if checkpoint["class_to_idx"] != expected_mapping or mapping != expected_mapping:
        raise ValueError("Class mapping differs from the trained six-class model.")
    expected_preprocessing = {
        "size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "interpolation": "InterpolationMode.BILINEAR", "geometry": "pad_square_black_then_resize",
    }
    if checkpoint["preprocessing"] != expected_preprocessing:
        raise ValueError("Checkpoint preprocessing differs from the app's evaluation transform.")
    if checkpoint["manifest_sha256"] != info["manifest_sha256"]:
        raise ValueError("Checkpoint data split differs from the documented evaluation.")
    # Two CPU threads avoid oversubscription on a free Space. CUDA is unnecessary.
    torch.set_num_threads(max(1, min(4, int(os.environ.get("ORAL_CPU_THREADS", "2")))))
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 6))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, {idx: name for name, idx in mapping.items()}


def prepare_image(image):
    if not isinstance(image, Image.Image):
        raise ImageInputError("invalid_image")
    width, height = image.size
    # Padding also allocates side² pixels, even for a narrow panoramic image.
    if width < 1 or height < 1 or max(width, height) ** 2 > MAX_IMAGE_PIXELS:
        raise ImageInputError("image_too_large")
    return eval_transform(ImageOps.exif_transpose(image).convert("RGB"))


@torch.inference_mode()
def predict(model, image, idx_to_class, device="cpu"):
    tensor = prepare_image(image).unsqueeze(0).to(device)
    logits = model(tensor)
    if logits.shape != (1, 6) or not torch.isfinite(logits).all():
        raise RuntimeError("The model returned invalid scores.")
    scores = logits.float().softmax(dim=1)[0].cpu().tolist()
    return sorted([
        {"class": idx_to_class[idx], "probability": float(score)}
        for idx, score in enumerate(scores)
    ], key=lambda result: result["probability"], reverse=True)
