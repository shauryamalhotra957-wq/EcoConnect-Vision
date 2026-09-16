import os
import tempfile
import unittest
from unittest.mock import MagicMock
import numpy as np

# Test without requiring heavy GPU or physical webcam
from classify_waste import CLASSES, IMAGE_SIZE, classify_image, classify_batch

class MockKerasModel:
    def __init__(self, score=0.2):
        self.score = score

    def predict(self, batch, verbose=0):
        # Return 2D array [[score]]
        return np.array([[self.score]])

class TestWasteClassifier(unittest.TestCase):
    def test_classes_definition(self):
        self.assertIn("biodegradable", CLASSES)
        self.assertIn("non_biodegradable", CLASSES)
        self.assertEqual(IMAGE_SIZE, (224, 224))

    def test_classify_image_biodegradable(self):
        # Create a synthetic test image using cv2
        import cv2
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name

        try:
            synthetic_img = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, synthetic_img)

            model = MockKerasModel(score=0.1) # low score -> biodegradable
            result = classify_image(model, temp_path)

            self.assertEqual(result["label"], "biodegradable")
            self.assertAlmostEqual(result["confidence"], 0.9, places=2)
            self.assertAlmostEqual(result["raw_score"], 0.1, places=2)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_classify_image_non_biodegradable(self):
        import cv2
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            synthetic_img = np.ones((250, 250, 3), dtype=np.uint8) * 200
            cv2.imwrite(temp_path, synthetic_img)

            model = MockKerasModel(score=0.85) # high score -> non_biodegradable
            result = classify_image(model, temp_path)

            self.assertEqual(result["label"], "non_biodegradable")
            self.assertAlmostEqual(result["confidence"], 0.85, places=2)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_classify_batch_directory(self):
        import cv2
        with tempfile.TemporaryDirectory() as temp_dir:
            img1 = os.path.join(temp_dir, "item1.jpg")
            img2 = os.path.join(temp_dir, "item2.png")
            cv2.imwrite(img1, np.zeros((100, 100, 3), dtype=np.uint8))
            cv2.imwrite(img2, np.zeros((100, 100, 3), dtype=np.uint8))

            model = MockKerasModel(score=0.3)
            results = classify_batch(model, temp_dir)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["label"], "biodegradable")

if __name__ == "__main__":
    unittest.main()
