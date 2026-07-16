"""Validate the bundled Keras model without importing TensorFlow."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

EXPECTED_INPUT_SHAPE = (None, 224, 224, 3)
EXPECTED_OUTPUT_UNITS = 1
EXPECTED_OUTPUT_ACTIVATION = "sigmoid"
REQUIRED_MEMBERS = {"metadata.json", "config.json", "model.weights.h5"}


class ModelContractError(ValueError):
    """Raised when a Keras artifact does not match the scanner contract."""


@dataclass(frozen=True)
class ModelContract:
    path: Path
    keras_version: str
    input_shape: tuple[int | None, ...]
    output_units: int
    output_activation: str


def _read_json(archive: zipfile.ZipFile, member: str) -> dict[str, Any]:
    try:
        document = json.loads(archive.read(member))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelContractError(f"{member} is missing or invalid JSON") from exc
    if not isinstance(document, dict):
        raise ModelContractError(f"{member} must contain a JSON object")
    return document


def _endpoint_name(reference: object, label: str) -> str:
    if isinstance(reference, list) and reference and isinstance(reference[0], str):
        return reference[0]
    raise ModelContractError(f"model config has no valid {label} layer reference")


def _find_layer(model_config: dict[str, Any], name: str, label: str) -> dict[str, Any]:
    layers = model_config.get("layers")
    if not isinstance(layers, list):
        raise ModelContractError("model config has no layer list")
    for layer in layers:
        if isinstance(layer, dict) and layer.get("name") == name:
            return layer
    raise ModelContractError(f"referenced {label} layer {name!r} is missing")


def validate_model_artifact(path: str | Path) -> ModelContract:
    """Return the discovered contract or raise ``ModelContractError``."""

    model_path = Path(path)
    try:
        with zipfile.ZipFile(model_path) as archive:
            missing = REQUIRED_MEMBERS.difference(archive.namelist())
            if missing:
                raise ModelContractError(
                    f"artifact is missing required member(s): {', '.join(sorted(missing))}"
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ModelContractError(f"artifact contains a corrupt member: {bad_member}")
            if archive.getinfo("model.weights.h5").file_size == 0:
                raise ModelContractError("model.weights.h5 is empty")

            metadata = _read_json(archive, "metadata.json")
            document = _read_json(archive, "config.json")
    except (FileNotFoundError, IsADirectoryError) as exc:
        raise ModelContractError(f"model artifact not found: {model_path}") from exc
    except zipfile.BadZipFile as exc:
        raise ModelContractError(f"model artifact is not a valid Keras archive: {model_path}") from exc

    keras_version = metadata.get("keras_version")
    if not isinstance(keras_version, str) or not keras_version.strip():
        raise ModelContractError("metadata.json has no Keras version")

    model_config = document.get("config")
    if not isinstance(model_config, dict):
        raise ModelContractError("config.json has no model config object")

    input_name = _endpoint_name(model_config.get("input_layers"), "input")
    output_name = _endpoint_name(model_config.get("output_layers"), "output")
    input_layer = _find_layer(model_config, input_name, "input")
    output_layer = _find_layer(model_config, output_name, "output")

    input_config = input_layer.get("config")
    output_config = output_layer.get("config")
    if not isinstance(input_config, dict) or not isinstance(output_config, dict):
        raise ModelContractError("input or output layer config is invalid")

    raw_shape = input_config.get("batch_shape", input_config.get("batch_input_shape"))
    if not isinstance(raw_shape, list) or not all(
        dimension is None or isinstance(dimension, int) for dimension in raw_shape
    ):
        raise ModelContractError("input layer has no valid batch shape")
    input_shape = tuple(raw_shape)
    if input_shape != EXPECTED_INPUT_SHAPE:
        raise ModelContractError(
            f"expected input shape {EXPECTED_INPUT_SHAPE}, found {input_shape}"
        )

    output_units = output_config.get("units")
    output_activation = output_config.get("activation")
    if output_units != EXPECTED_OUTPUT_UNITS:
        raise ModelContractError(
            f"expected {EXPECTED_OUTPUT_UNITS} output unit, found {output_units!r}"
        )
    if output_activation != EXPECTED_OUTPUT_ACTIVATION:
        raise ModelContractError(
            f"expected {EXPECTED_OUTPUT_ACTIVATION!r} output activation, "
            f"found {output_activation!r}"
        )

    return ModelContract(
        path=model_path,
        keras_version=keras_version,
        input_shape=input_shape,
        output_units=output_units,
        output_activation=output_activation,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        default="waste_scanner_model.keras",
        help="path to a .keras artifact (default: waste_scanner_model.keras)",
    )
    args = parser.parse_args(argv)

    try:
        contract = validate_model_artifact(args.model)
    except ModelContractError as exc:
        print(f"Model contract failed: {exc}", file=sys.stderr)
        return 1

    print(
        "Model contract valid: "
        f"Keras {contract.keras_version}, input {contract.input_shape}, "
        f"output {contract.output_units} {contract.output_activation} unit"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
