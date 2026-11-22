from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import torch
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError

import CNN

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("PLANT_DISEASE_MODEL_PATH", str(BASE_DIR / "plant_disease_model_1_latest.pt"))).expanduser()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

disease_info = pd.read_csv(BASE_DIR / "disease_info.csv", encoding="cp1252").set_index("index")
supplement_info = pd.read_csv(BASE_DIR / "supplement_info.csv", encoding="cp1252").set_index("index")
_model_cache = None

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def load_model():
    """Load the trusted model checkpoint once per worker process."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. Download the trained checkpoint and set PLANT_DISEASE_MODEL_PATH."
        )
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a PyTorch state dictionary.")
    if any(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}

    network = CNN.CNN(len(CNN.idx_to_classes))
    network.load_state_dict(checkpoint, strict=True)
    network.to("cpu")
    network.eval()
    _model_cache = network
    return network


def preprocess(image: Image.Image) -> torch.Tensor:
    """Mirror the original training transform: Resize(255), CenterCrop(224), ToTensor."""
    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image has invalid dimensions.")

    if width <= height:
        resized = (255, int(round(255 * height / width)))
    else:
        resized = (int(round(255 * width / height)), 255)
    image = image.resize(resized, Image.Resampling.BILINEAR)
    left = (resized[0] - 224) // 2
    top = (resized[1] - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    import numpy as np
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.unsqueeze(0)


def prediction(image: Image.Image) -> int:
    network = load_model()
    batch = preprocess(image)
    with torch.inference_mode():
        logits = network(batch)
        if logits.ndim != 2 or logits.shape[1] != len(CNN.idx_to_classes):
            raise ValueError("Model output does not match the 39-class label mapping.")
        return int(logits.argmax(dim=1).item())


@app.errorhandler(413)
def upload_too_large(_error):
    return render_template("index.html", error="Image is larger than 10 MB. Please upload a smaller file."), 413


@app.route("/")
def home_page():
    return render_template("home.html")


@app.route("/contact")
def contact():
    return render_template("contact-us.html")


@app.route("/index")
def ai_engine_page():
    return render_template("index.html")


@app.route("/mobile-device")
def mobile_device_detected_page():
    return redirect(url_for("ai_engine_page"))


@app.route("/submit", methods=["GET", "POST"])
def submit():
    if request.method == "GET":
        return redirect(url_for("ai_engine_page"))

    image_file = request.files.get("image")
    if image_file is None or not image_file.filename:
        return render_template("index.html", error="Choose a leaf image before submitting."), 400

    extension = Path(image_file.filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        return render_template("index.html", error="Upload a JPG, JPEG or PNG image."), 400

    raw = image_file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return render_template("index.html", error="Image is larger than 10 MB."), 413

    try:
        with Image.open(io.BytesIO(raw)) as check:
            check.verify()
        with Image.open(io.BytesIO(raw)) as opened:
            image = opened.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return render_template("index.html", error="The uploaded file is not a valid image."), 400

    try:
        pred = prediction(image)
    except FileNotFoundError as exc:
        return render_template("index.html", error=str(exc)), 503
    except (RuntimeError, ValueError) as exc:
        app.logger.exception("Plant disease model inference failed")
        return render_template("index.html", error=f"Model inference failed: {exc}"), 500

    if pred not in disease_info.index or pred not in supplement_info.index:
        return render_template("index.html", error="The model returned a class with no matching information row."), 500

    disease = disease_info.loc[pred]
    supplement = supplement_info.loc[pred]
    return render_template(
        "submit.html",
        title=disease["disease_name"],
        desc=disease["description"],
        prevent=disease["Possible Steps"],
        image_url=disease["image_url"],
        pred=pred,
        sname=supplement["supplement name"],
        simage=supplement["supplement image"],
        buy_link=supplement["buy link"],
    )


@app.route("/market")
def market():
    return render_template(
        "market.html",
        supplement_image=supplement_info["supplement image"].tolist(),
        supplement_name=supplement_info["supplement name"].tolist(),
        disease=disease_info["disease_name"].tolist(),
        buy=supplement_info["buy link"].tolist(),
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
