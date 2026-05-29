from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from .config import MODEL_SPECS
from .models import DenseNet121Scratch


def load_class_index_map(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {str(k): int(v) for k, v in data.items()}


def invert_mapping(class_to_idx: dict[str, int]) -> list[str]:
    max_idx = max(class_to_idx.values())
    result = [""] * (max_idx + 1)
    for name, idx in class_to_idx.items():
        result[idx] = name
    return result


def build_model(arch: str, num_classes: int, checkpoint: dict | None = None) -> nn.Module:
    if arch == "densenet" and isinstance(checkpoint, dict):
        return DenseNet121Scratch(
            num_classes=num_classes,
            growth_rate=int(checkpoint.get("growth_rate", 32)),
            compression=float(checkpoint.get("compression", 0.5)),
            block_layers=tuple(int(v) for v in checkpoint.get("block_layers", (6, 12, 24, 16))),
        )
    return MODEL_SPECS[arch].builder(num_classes)


def load_model(
    arch: str,
    checkpoint_path: Path,
    class_to_idx: dict[str, int],
    device: torch.device,
) -> tuple[nn.Module, list[str]]:
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        num_classes = int(raw.get("num_classes", len(class_to_idx)))
    elif isinstance(raw, dict):
        state_dict = raw
        num_classes = len(class_to_idx)
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(raw)}")

    model = build_model(arch, num_classes, checkpoint=raw if isinstance(raw, dict) else None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(
            "Checkpoint is missing required keys — wrong architecture?\n"
            f"Missing: {missing[:8]}"
        )
    if unexpected:
        print(f"[WARN] Checkpoint has {len(unexpected)} unused key(s) (ignored): {unexpected[:4]}")

    model.to(device).eval()
    return model, invert_mapping(class_to_idx)


def resolve_paths(arch: str, ckpt: Path | None, ci: Path | None) -> tuple[Path, Path]:
    spec = MODEL_SPECS[arch]
    checkpoint_path = ckpt or spec.checkpoint_path
    class_indices_path = ci or spec.class_indices_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not class_indices_path.exists():
        raise FileNotFoundError(f"Class indices not found: {class_indices_path}")
    return checkpoint_path, class_indices_path


def preprocess_face(
    face_bgr: np.ndarray,
    image_size: int,
    normalize_input: bool,
    device: torch.device,
) -> Tensor:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    arr = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    if normalize_input:
        tensor = (tensor - 0.5) / 0.5
    return tensor


@torch.no_grad()
def predict_face(
    model: nn.Module,
    face_bgr: np.ndarray,
    image_size: int,
    normalize_input: bool,
    device: torch.device,
) -> Tensor:
    inputs = preprocess_face(face_bgr, image_size, normalize_input, device)
    logits = model(inputs)
    return torch.softmax(logits, dim=1)[0].detach().cpu()
