# EcoConnect Vision

[![CI](https://github.com/shauryamalhotra957-wq/EcoConnect-Vision/actions/workflows/ci.yml/badge.svg)](https://github.com/shauryamalhotra957-wq/EcoConnect-Vision/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

> Real-time computer-vision waste classification using MobileNetV2 transfer learning and OpenCV motion tracking.

---

## Features

- **Motion-Gated Scanning**: Uses OpenCV background subtraction (`createBackgroundSubtractorMOG2`) and contour bounding to isolate moving objects on conveyers or in front of cameras.
- **Deep Learning Classifier**: MobileNetV2 fine-tuned on biodegradable vs. non-biodegradable waste categories (`waste_scanner_model.keras`).
- **Flexible Modes**:
  - Live webcam feed (`--webcam`)
  - Single-image inference (`--image <path>`)
  - Batch directory scanning (`--batch <folder>`)
  - Structured JSON export (`--output-json <report.json>`)

## Quickstart

```bash
# Clone the repository
git clone https://github.com/shauryamalhotra957-wq/EcoConnect-Vision.git
cd EcoConnect-Vision

# Install dependencies
pip install -r requirements.txt

# Classify a single image
python classify_waste.py --image sample.jpg

# Batch classify a directory of waste images
python classify_waste.py --batch ./test_images/ --output-json results.json

# Launch live webcam detection
python classify_waste.py --webcam
```

## Running Unit Tests

```bash
python -m unittest discover -s tests
```

## License

MIT © [Shaurya Malhotra](https://github.com/shauryamalhotra957-wq)
