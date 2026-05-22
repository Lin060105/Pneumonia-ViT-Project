"""Shared model, preprocessing, checkpoint, and decision helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


CLASS_NAMES: Tuple[str, str] = ("NORMAL", "PNEUMONIA")
PNEUMONIA_INDEX = CLASS_NAMES.index("PNEUMONIA")
DEFAULT_MODEL_NAME = "vit_base_patch16_224"
DEFAULT_IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(
    model_name: str = DEFAULT_MODEL_NAME,
    num_classes: int = len(CLASS_NAMES),
    pretrained: bool = False,
) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained)
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes)
    elif hasattr(model, "head") and hasattr(model.head, "in_features"):
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported classifier head for model: {model_name}")
    return model


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(DEFAULT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(DEFAULT_IMAGE_SIZE),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _torch_load(path: str | Path, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def unpack_checkpoint(payload: Any) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        metadata = dict(payload.get("metadata", {}))
    elif isinstance(payload, Mapping):
        state_dict = payload
        metadata = {}
    else:
        raise ValueError("Checkpoint must be a state_dict or a metadata checkpoint.")

    return _strip_module_prefix(state_dict), metadata


def load_model_checkpoint(
    path: str | Path,
    device: torch.device | str | None = None,
    strict: bool = True,
) -> tuple[nn.Module, Dict[str, Any]]:
    device = device or get_device()
    payload = _torch_load(path, map_location=device)
    state_dict, metadata = unpack_checkpoint(payload)

    class_names = tuple(metadata.get("class_names", CLASS_NAMES))
    model_name = metadata.get("model_name", DEFAULT_MODEL_NAME)
    model = create_model(model_name=model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()

    metadata.setdefault("class_names", class_names)
    metadata.setdefault("model_name", model_name)
    metadata.setdefault("image_size", DEFAULT_IMAGE_SIZE)
    metadata.setdefault("mean", IMAGENET_MEAN)
    metadata.setdefault("std", IMAGENET_STD)
    return model, metadata


def save_model_checkpoint(
    path: str | Path,
    model: nn.Module,
    metadata: Mapping[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": dict(metadata),
        },
        path,
    )


def image_to_tensor(image: Image.Image, device: torch.device | str) -> torch.Tensor:
    return get_eval_transform()(image.convert("RGB")).unsqueeze(0).to(device)


def predict_probabilities(
    model: nn.Module,
    image_tensor: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
    return probabilities.squeeze(0).detach().cpu().numpy()


def decide_screening_status(
    probabilities: Sequence[float],
    threshold: float = 0.5,
    uncertainty_margin: float = 0.05,
) -> Dict[str, Any]:
    pneumonia_probability = float(probabilities[PNEUMONIA_INDEX])
    normal_probability = float(probabilities[0])

    if abs(pneumonia_probability - threshold) <= uncertainty_margin:
        decision = "REVIEW"
    elif pneumonia_probability >= threshold:
        decision = "PNEUMONIA"
    else:
        decision = "NORMAL"

    return {
        "decision": decision,
        "normal_probability": normal_probability,
        "pneumonia_probability": pneumonia_probability,
        "threshold": threshold,
        "uncertainty_margin": uncertainty_margin,
    }
