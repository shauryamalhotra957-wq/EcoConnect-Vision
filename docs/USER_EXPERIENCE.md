# Waste Scanner Experience

This guide defines a product-quality path through the current notebook and a future scanner interface without overstating model confidence.

## Primary journey

1. Confirm runtime, model file, and dependency readiness.
2. Upload or capture one clearly framed item.
3. Show preprocessing progress and an image preview.
4. Present the predicted class, confidence, and disposal guidance together.
5. Offer a retake path when confidence or image quality is low.

## Required states

- `ready`: model loaded and camera or upload is available.
- `processing`: the image is being decoded and classified.
- `result`: prediction and confidence are visible.
- `needs another image`: the model cannot make a reliable recommendation.
- `error`: the failed requirement and corrective action are named.

## Result hierarchy

The primary result is the disposal action. The class label and confidence are supporting evidence. Low confidence must never be hidden behind a confident-looking color treatment.

## Accessibility and trust

- Use text and icons in addition to recycling-bin colors.
- Give all example and output images meaningful captions or alt text.
- Keep controls at least 44px and preserve visible keyboard focus.
- Do not animate the result continuously; one short transition is enough.
- State dataset and model limitations beside results in any deployed interface.
