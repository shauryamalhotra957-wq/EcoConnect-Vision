from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from predict import InferenceError, interpret_score, main, predict_image, threshold_value


class PredictionTests(unittest.TestCase):
    def test_interpret_score_reports_the_winning_class_confidence(self) -> None:
        compostable = interpret_score(0.18)
        landfill = interpret_score(0.87)

        self.assertEqual(compostable.label, "biodegradable")
        self.assertAlmostEqual(compostable.confidence, 0.82)
        self.assertEqual(landfill.label, "non_biodegradable")
        self.assertAlmostEqual(landfill.confidence, 0.87)

    def test_interpret_score_honors_a_custom_threshold(self) -> None:
        prediction = interpret_score(0.61, threshold=0.7)

        self.assertEqual(prediction.label, "biodegradable")
        self.assertAlmostEqual(prediction.confidence, 0.39)

    def test_threshold_parser_rejects_boundary_values(self) -> None:
        for value in ("0", "1", "-0.2", "1.2"):
            with self.subTest(value=value), self.assertRaises(Exception):
                threshold_value(value)

    def test_missing_assets_fail_before_loading_tensorflow(self) -> None:
        with self.assertRaisesRegex(InferenceError, "Image not found"):
            predict_image("not-a-real-image.jpg")

    def test_cli_returns_a_clear_nonzero_status_for_a_missing_image(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), tempfile.TemporaryDirectory() as temp_dir:
            exit_code = main([str(Path(temp_dir) / "missing.png")])

        self.assertEqual(exit_code, 2)
        self.assertIn("Image not found", stderr.getvalue())

    def test_bundled_model_records_the_supported_keras_version(self) -> None:
        with ZipFile("waste_scanner_model.keras") as archive:
            metadata = json.loads(archive.read("metadata.json"))

        self.assertEqual(metadata["keras_version"], "3.14.1")

    @unittest.skipUnless(
        importlib.util.find_spec("tensorflow") and importlib.util.find_spec("keras"),
        "TensorFlow/Keras inference dependencies are not installed",
    )
    def test_bundled_model_loads_with_the_supported_runtime(self) -> None:
        from keras.models import load_model

        model = load_model("waste_scanner_model.keras", compile=False)

        self.assertEqual(len(model.input_shape), 4)
        self.assertEqual(model.output_shape[-1], 1)


if __name__ == "__main__":
    unittest.main()
