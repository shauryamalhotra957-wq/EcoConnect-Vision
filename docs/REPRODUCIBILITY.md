# Reproducibility and evaluation notes

EcoConnect Vision is a prototype classifier. A reported accuracy is meaningful only when its data split, preprocessing, and operating conditions are recorded alongside it.

## Record each training run

For every model export, capture:

- Dataset source, class definitions, licensing/consent status, and collection date.
- Exact train, validation, and test split identifiers; do not evaluate on images used for tuning.
- Image preprocessing, augmentation, MobileNetV2 base version, optimizer, learning rate, batch size, epochs, and random seed.
- Per-class precision, recall, F1 score, confusion matrix, and the chosen confidence threshold.
- Camera, lighting, background, and motion conditions for any webcam evaluation.

The README's approximate training and validation results should not be treated as a deployment guarantee.

## Verify a released artifact

Store the model hash with the evaluation record so a later notebook run can be tied to a specific file.

```powershell
Get-FileHash .\waste_scanner_model.keras -Algorithm SHA256
```

On macOS or Linux:

```bash
sha256sum waste_scanner_model.keras
```

When replacing the model, rerun a fixed holdout set and record the expected label, confidence, timestamp, and model hash. Review any regressions before publishing the new artifact.

## Deployment boundary

The current motion-isolation pipeline is sensitive to lighting, background movement, class imbalance, and framing. Low-confidence or ambiguous results should be surfaced as uncertain rather than used as an automated recycling or compliance decision. A production deployment needs a larger representative dataset, monitored performance, and a defined human-review path.
