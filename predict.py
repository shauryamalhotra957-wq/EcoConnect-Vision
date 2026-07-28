"""Command-line image inference for the EcoConnect Vision waste model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence


CLASS_NAMES = ("biodegradable", "non_biodegradable")


class InferenceError(RuntimeError):
    """Raised when local model inference cannot be completed."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    raw_score: float
    threshold: float


def threshold_value(value: str) -> float:
    threshold = float(value)
    if not 0 < threshold < 1:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return threshold


def interpret_score(score: float, threshold: float = 0.5) -> Prediction:
    """Map the model's sigmoid output to a label and class confidence."""

    if not 0 <= score <= 1:
        raise ValueError("model score must be between 0 and 1")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")

    is_non_biodegradable = score >= threshold
    return Prediction(
        label=CLASS_NAMES[1 if is_non_biodegradable else 0],
        confidence=score if is_non_biodegradable else 1 - score,
        raw_score=score,
        threshold=threshold,
    )


def _load_runtime() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        from keras.models import load_model
        from keras.utils import img_to_array, load_img
    except ImportError as exc:
        raise InferenceError(
            "TensorFlow image dependencies are missing. "
            "Install them with: python -m pip install -r requirements.txt"
        ) from exc
    return np, load_model, (load_img, img_to_array)


def predict_image(
    image_path: str | Path,
    model_path: str | Path = "waste_scanner_model.keras",
    threshold: float = 0.5,
) -> Prediction:
    """Load an image and return a confidence-aware binary prediction."""

    image_path = Path(image_path)
    model_path = Path(model_path)
    if not image_path.is_file():
        raise InferenceError(f"Image not found: {image_path}")
    if not model_path.is_file():
        raise InferenceError(f"Model not found: {model_path}")

    np, load_model, image_runtime = _load_runtime()
    load_img, img_to_array = image_runtime
    try:
        model = load_model(model_path, compile=False)
        input_shape = model.input_shape
        if not isinstance(input_shape, tuple) or len(input_shape) != 4:
            raise InferenceError(f"Expected a 4D image model input, received {input_shape!r}")
        height = int(input_shape[1] or 224)
        width = int(input_shape[2] or 224)
        image = load_img(image_path, target_size=(height, width), color_mode="rgb")
        batch = np.expand_dims(img_to_array(image), axis=0)
        output = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError(f"Inference failed: {exc}") from exc

    if output.size != 1:
        raise InferenceError(f"Expected one sigmoid score, received {output.size}")
    return interpret_score(float(output[0]), threshold)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify one waste image with the bundled EcoConnect Vision model.",
    )
    parser.add_argument("image", help="Path to a JPG, PNG, or other TensorFlow-supported image")
    parser.add_argument(
        "--model",
        default="waste_scanner_model.keras",
        help="Keras model path (default: waste_scanner_model.keras)",
    )
    parser.add_argument(
        "--threshold",
        type=threshold_value,
        default=0.5,
        help="Non-biodegradable decision threshold between 0 and 1 (default: 0.5)",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prediction = predict_image(args.image, model_path=args.model, threshold=args.threshold)
    except InferenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = {
        **asdict(prediction),
        "image": str(Path(args.image)),
        "model": str(Path(args.model)),
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        readable_label = prediction.label.replace("_", " ")
        print(f"{readable_label}: {prediction.confidence:.1%} confidence")
        print(f"raw score {prediction.raw_score:.4f} | threshold {prediction.threshold:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
