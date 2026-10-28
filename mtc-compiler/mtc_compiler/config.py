"""Model configuration types and JSON loading."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a model configuration is malformed or unsupported."""


@dataclass(frozen=True)
class LayerConfig:
    type: str
    in_features: int | None = None
    out_features: int | None = None


@dataclass(frozen=True)
class OptimizerConfig:
    type: str
    learning_rate: float


@dataclass(frozen=True)
class TrainingSample:
    inputs: tuple[float, ...]
    targets: tuple[float, ...]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    precision: str
    layers: tuple[LayerConfig, ...]
    loss: str
    optimizer: OptimizerConfig
    input_range: tuple[float, float]
    target_range: tuple[float, float]
    training_data: tuple[TrainingSample, ...]
    epochs: int
    seed: int


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be an object")
    return value


def _require_positive_int(value: Any, location: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{location} must be a positive integer")
    return value


def _reject_unknown_keys(value: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{location} has unsupported field(s): {', '.join(unknown)}")


def _require_range(value: Any, location: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{location} must contain [min, max]")
    bounds = tuple(value)
    if any(isinstance(bound, bool) or not isinstance(bound, (int, float)) for bound in bounds):
        raise ConfigError(f"{location} bounds must be finite numbers")
    lower, upper = (float(bound) for bound in bounds)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ConfigError(f"{location} must have finite bounds with min <= max")
    if lower < -1.0 or upper > 1.0:
        raise ConfigError(f"{location} must fit the Q15 range [-1.0, 1.0]")
    return lower, upper


def _require_vector(value: Any, location: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{location} must be a non-empty array")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ConfigError(f"{location} values must be finite numbers")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item < -1.0 or item > 1.0 for item in result):
        raise ConfigError(f"{location} values must fit the Q15 range [-1.0, 1.0]")
    return result


def model_from_dict(data: Any) -> ModelConfig:
    """Convert decoded JSON data into a validated, typed model configuration."""
    root = _require_mapping(data, "model")
    _reject_unknown_keys(
        root,
        {
            "name", "precision", "layers", "loss", "optimizer", "input_range",
            "target_range", "training_data", "epochs", "seed",
        },
        "Model",
    )
    try:
        name = root["name"]
        precision = root["precision"]
        raw_layers = root["layers"]
        loss = root["loss"]
        raw_optimizer = root["optimizer"]
    except KeyError as error:
        raise ConfigError(f"Missing required model field: {error.args[0]}") from error

    if not isinstance(name, str) or not name:
        raise ConfigError("'name' must be a non-empty string")
    if precision != "int16":
        raise ConfigError(f"Unsupported precision: {precision!r}; expected 'int16'")
    if loss != "mse":
        raise ConfigError(f"Unsupported loss: {loss!r}; expected 'mse'")
    if not isinstance(raw_layers, list):
        raise ConfigError("'layers' must be an array")

    layers = []
    for index, raw_layer in enumerate(raw_layers):
        layer = _require_mapping(raw_layer, f"Layer {index}")
        layer_type = layer.get("type")
        if layer_type == "dense":
            _reject_unknown_keys(layer, {"type", "in_features", "out_features"}, f"Layer {index}")
            layers.append(LayerConfig(
                type=layer_type,
                in_features=_require_positive_int(
                    layer.get("in_features"), f"Layer {index} in_features"
                ),
                out_features=_require_positive_int(
                    layer.get("out_features"), f"Layer {index} out_features"
                ),
            ))
        elif layer_type in {"relu", "sigmoid"}:
            _reject_unknown_keys(layer, {"type"}, f"Layer {index}")
            layers.append(LayerConfig(type=layer_type))
        else:
            raise ConfigError(f"Layer {index}: unsupported layer type {layer_type!r}")

    optimizer = _require_mapping(raw_optimizer, "optimizer")
    _reject_unknown_keys(optimizer, {"type", "learning_rate"}, "optimizer")
    if optimizer.get("type") != "sgd":
        raise ConfigError(
            f"Unsupported optimizer: {optimizer.get('type')!r}; expected 'sgd'"
        )
    learning_rate = optimizer.get("learning_rate")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ConfigError("Optimizer learning_rate must be a positive number")
    if learning_rate <= 0:
        raise ConfigError("Optimizer learning_rate must be a positive number")
    if not math.isfinite(float(learning_rate)) or learning_rate > 1.0:
        raise ConfigError("Optimizer learning_rate must fit the Q15 range (0, 1]")
    if round(float(learning_rate) * 32768) == 0:
        raise ConfigError("Optimizer learning_rate rounds to zero in Q15")

    input_range = _require_range(root.get("input_range", [-1.0, 1.0]), "input_range")
    target_range = _require_range(root.get("target_range", [-1.0, 1.0]), "target_range")
    epochs = _require_positive_int(root.get("epochs", 100), "epochs")
    seed = root.get("seed", 1)
    if type(seed) is not int or seed < 0:
        raise ConfigError("seed must be a non-negative integer")

    raw_training_data = root.get("training_data", [])
    if not isinstance(raw_training_data, list):
        raise ConfigError("'training_data' must be an array")
    training_data = []
    for index, raw_sample in enumerate(raw_training_data):
        sample = _require_mapping(raw_sample, f"Training sample {index}")
        _reject_unknown_keys(sample, {"input", "target"}, f"Training sample {index}")
        try:
            inputs = _require_vector(sample["input"], f"Training sample {index} input")
            targets = _require_vector(sample["target"], f"Training sample {index} target")
        except KeyError as error:
            raise ConfigError(
                f"Training sample {index} is missing required field: {error.args[0]}"
            ) from error
        if any(not input_range[0] <= item <= input_range[1] for item in inputs):
            raise ConfigError(f"Training sample {index} input is outside input_range")
        if any(not target_range[0] <= item <= target_range[1] for item in targets):
            raise ConfigError(f"Training sample {index} target is outside target_range")
        training_data.append(TrainingSample(inputs, targets))

    return ModelConfig(
        name=name,
        precision=precision,
        layers=tuple(layers),
        loss=loss,
        optimizer=OptimizerConfig("sgd", float(learning_rate)),
        input_range=input_range,
        target_range=target_range,
        training_data=tuple(training_data),
        epochs=epochs,
        seed=seed,
    )


def load_model(path: str | Path) -> ModelConfig:
    """Load a model configuration from a JSON file."""
    model_path = Path(path)
    try:
        data = json.loads(model_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"Cannot read model file '{model_path}': {error}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSON in '{model_path}': {error}") from error
    return model_from_dict(data)