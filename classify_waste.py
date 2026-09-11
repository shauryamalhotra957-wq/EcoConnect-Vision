"""Real-time and single-image waste classification with MobileNetV2 and OpenCV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CLASSES = ["biodegradable", "non_biodegradable"]
IMAGE_SIZE = (224, 224)


def load_model(model_path: str = "waste_scanner_model.keras"):
    import tensorflow as tf

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(str(path))


def classify_image(model, image_path: str) -> dict[str, object]:
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
    img_batch = np.expand_dims(img_resized, axis=0)

    prediction = float(model.predict(img_batch, verbose=0)[0][0])
    is_non_bio = prediction >= 0.5
    label = "non_biodegradable" if is_non_bio else "biodegradable"
    confidence = prediction if is_non_bio else (1.0 - prediction)

    return {
        "label": label,
        "confidence": confidence,
        "raw_score": prediction,
    }


def run_webcam(model, camera_index: int = 0) -> None:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}", file=sys.stderr)
        return

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
    print("Starting EcoConnect Vision (press 'q' to quit)...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fg_mask = bg_subtractor.apply(frame)
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            best_contour = None
            max_area = 0.0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 3000 and area > max_area:
                    max_area = area
                    best_contour = cnt

            status_text = "Scanning..."
            box_color = (200, 200, 200)

            if best_contour is not None:
                x, y, bw, bh = cv2.boundingRect(best_contour)
                crop = frame[y : y + bh, x : x + bw]
                if crop.size > 0:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_resized = cv2.resize(crop_rgb, IMAGE_SIZE)
                    crop_batch = np.expand_dims(crop_resized, axis=0)

                    score = float(model.predict(crop_batch, verbose=0)[0][0])
                    if score < 0.5:
                        label = "Biodegradable"
                        conf = (1.0 - score) * 100
                        box_color = (0, 200, 0)
                    else:
                        label = "Non-Biodegradable"
                        conf = score * 100
                        box_color = (0, 0, 255)

                    status_text = f"{label} ({conf:.1f}%)"
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_color, 2)

            cv2.putText(
                frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                box_color,
                2,
            )
            cv2.imshow("EcoConnect Vision", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="EcoConnect Vision Waste Classifier")
    parser.add_argument("--image", type=str, help="Path to an image to classify")
    parser.add_argument(
        "--webcam", action="store_true", help="Launch live webcam inference"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="waste_scanner_model.keras",
        help="Path to .keras model",
    )

    args = parser.parse_args()

    if not args.image and not args.webcam:
        parser.print_help()
        return 0

    model = load_model(args.model)

    if args.image:
        result = classify_image(model, args.image)
        print(
            f"Classification: {result['label']} (Confidence: {result['confidence']:.2%})"
        )
    elif args.webcam:
        run_webcam(model, args.camera)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
