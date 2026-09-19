import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from model_contract import ModelContractError, validate_model_artifact

ROOT = Path(__file__).resolve().parents[1]


def write_artifact(
    path: Path,
    *,
    input_shape: list[int | None] | None = None,
    output_units: int = 1,
    activation: str = "sigmoid",
    include_weights: bool = True,
) -> None:
    model_config = {
        "class_name": "Functional",
        "config": {
            "layers": [
                {
                    "name": "scanner_input",
                    "class_name": "InputLayer",
                    "config": {
                        "batch_shape": input_shape or [None, 224, 224, 3],
                    },
                },
                {
                    "name": "scanner_output",
                    "class_name": "Dense",
                    "config": {
                        "units": output_units,
                        "activation": activation,
                    },
                },
            ],
            "input_layers": ["scanner_input", 0, 0],
            "output_layers": ["scanner_output", 0, 0],
        },
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.json", json.dumps({"keras_version": "3.14.1"}))
        archive.writestr("config.json", json.dumps(model_config))
        if include_weights:
            archive.writestr("model.weights.h5", b"test-weights")


class ModelContractTests(unittest.TestCase):
    def test_bundled_model_matches_scanner_contract(self) -> None:
        contract = validate_model_artifact(ROOT / "waste_scanner_model.keras")

        self.assertEqual(contract.input_shape, (None, 224, 224, 3))
        self.assertEqual(contract.output_units, 1)
        self.assertEqual(contract.output_activation, "sigmoid")

    def test_rejects_an_incompatible_classifier_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "multiclass.keras"
            write_artifact(artifact, output_units=4, activation="softmax")

            with self.assertRaisesRegex(ModelContractError, "expected 1 output unit"):
                validate_model_artifact(artifact)

    def test_rejects_an_artifact_without_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "missing-weights.keras"
            write_artifact(artifact, include_weights=False)

            with self.assertRaisesRegex(ModelContractError, "model.weights.h5"):
                validate_model_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
