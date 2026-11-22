"""Safe CPU model loading and PlantVillage image preprocessing."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from plant_disease_model import CNN, IDX_TO_CLASSES

IMAGE_SIZE = 224
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Match the original training transform: Resize(255), CenterCrop(224), ToTensor."""
    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image has invalid dimensions.")

    short_edge = 255
    if width <= height:
        resized_width = short_edge
        resized_height = int(round(short_edge * height / width))
    else:
        resized_height = short_edge
        resized_width = int(round(short_edge * width / height))

    image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    left = (resized_width - IMAGE_SIZE) // 2
    top = (resized_height - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.unsqueeze(0)


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Model checkpoint must contain a state dictionary.")
    for key in ("state_dict", "model_state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    if not checkpoint or not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise ValueError("Checkpoint does not look like a PyTorch state dictionary.")
    if any(key.startswith("module.") for key in checkpoint):
        checkpoint = {
            key.removeprefix("module."): value
            for key, value in checkpoint.items()
        }
    return checkpoint


def load_model(model_path: str | Path, device: str = "cpu") -> CNN:
    """Load a trusted state-dict checkpoint using CPU mapping and safe weights-only mode."""
    checkpoint_path = Path(model_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {checkpoint_path}. Download the trained .pt file and set PLANT_DISEASE_MODEL_PATH."
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        # Compatibility fallback for older supported PyTorch versions.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = _extract_state_dict(checkpoint)
    model = CNN(num_classes=len(IDX_TO_CLASSES))
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "The model checkpoint does not match the included 39-class CNN architecture."
        ) from exc
    model.to(device)
    model.eval()
    return model


def predict_tensor(model: torch.nn.Module, batch: torch.Tensor, top_k: int = 3) -> list[dict[str, Any]]:
    """Run inference and return class labels with softmax scores for ranking."""
    if batch.ndim != 4 or batch.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("Expected an image tensor with shape (1, 3, 224, 224).")
    if not 1 <= top_k <= len(IDX_TO_CLASSES):
        raise ValueError("top_k is outside the supported class range.")
    with torch.inference_mode():
        logits = model(batch)
        if logits.ndim != 2 or logits.shape[1] != len(IDX_TO_CLASSES):
            raise ValueError("Model output does not contain 39 class scores.")
        probabilities = torch.softmax(logits, dim=1)[0]
        scores, indices = torch.topk(probabilities, k=top_k)
    return [
        {
            "index": int(index.item()),
            "class_name": IDX_TO_CLASSES[int(index.item())],
            "score": float(score.item()),
        }
        for score, index in zip(scores, indices)
    ]


def predict_image(model: torch.nn.Module, image: Image.Image, top_k: int = 3) -> list[dict[str, Any]]:
    batch = preprocess_image(image)
    device = next(model.parameters()).device
    return predict_tensor(model, batch.to(device), top_k=top_k)
