from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from torch import nn

from .models import DenseNet121Scratch, ResNet50Scratch, VGG16Scratch, ViTScratch


ROOT = Path(__file__).resolve().parent.parent

EMOTIONS: list[str] = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]

EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    "Anger": (50, 50, 230),
    "Contempt": (30, 140, 255),
    "Disgust": (30, 180, 90),
    "Fear": (180, 30, 200),
    "Happiness": (30, 220, 220),
    "Neutral": (200, 200, 200),
    "Sadness": (220, 100, 30),
    "Surprise": (30, 200, 200),
}

EMOTION_EMOJI: dict[str, str] = {
    "Anger": "😠",
    "Contempt": "😒",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happiness": "😄",
    "Neutral": "😐",
    "Sadness": "😢",
    "Surprise": "😲",
}

COLOR_BG_PANEL = (25, 25, 35)
COLOR_ACCENT = (80, 200, 120)
COLOR_ACCENT2 = (220, 160, 60)
COLOR_TEXT_WHITE = (240, 240, 240)
COLOR_TEXT_DIM = (150, 150, 160)
COLOR_HUD_BG = (15, 15, 20)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint_path: Path
    class_indices_path: Path
    builder: Callable[[int], nn.Module]
    normalize_input: bool = False
    image_size: int = 48


MODEL_SPECS: dict[str, ModelSpec] = {
    "vgg": ModelSpec(
        name="VGG16",
        checkpoint_path=ROOT / "vgg" / "best_vgg16_emotion.pt",
        class_indices_path=ROOT / "vgg" / "class_indices.json",
        builder=lambda nc: VGG16Scratch(num_classes=nc),
    ),
    "resnet": ModelSpec(
        name="ResNet50",
        checkpoint_path=ROOT / "resnet" / "best_resnet50_emotion.pt",
        class_indices_path=ROOT / "resnet" / "class_indices.json",
        builder=lambda nc: ResNet50Scratch(num_classes=nc),
    ),
    "densenet": ModelSpec(
        name="DenseNet121",
        checkpoint_path=ROOT / "densenet" / "best_densenet_emotion.pt",
        class_indices_path=ROOT / "densenet" / "class_indices.json",
        builder=lambda nc: DenseNet121Scratch(num_classes=nc),
    ),
    "vit": ModelSpec(
        name="ViT",
        checkpoint_path=ROOT / "vit" / "best_vit_emotion.pt",
        class_indices_path=ROOT / "vit" / "class_indices.json",
        builder=lambda nc: ViTScratch(num_classes=nc),
        normalize_input=True,
    ),
}

ARCH_ORDER = ["resnet", "vgg", "densenet", "vit"]
