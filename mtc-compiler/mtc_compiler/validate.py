"""Semantic validation for model configurations."""

from .config import ConfigError, ModelConfig


def validate_model(config: ModelConfig) -> None:
    """Reject empty networks, leading activations, and disconnected dense layers."""
    if not config.layers:
        raise ConfigError("Model validation failed: 'layers' list cannot be empty.")

    current_features: int | None = None
    has_dense_layer = False
    previous_was_activation = False

    for index, layer in enumerate(config.layers):
        if layer.type == "dense":
            has_dense_layer = True
            if current_features is not None and layer.in_features != current_features:
                raise ConfigError(
                    f"Layer {index}: Input features ({layer.in_features}) do not match "
                    f"previous layer output ({current_features})"
                )
            current_features = layer.out_features
            previous_was_activation = False
        elif current_features is None:
            raise ConfigError(
                f"Layer {index}: Activation function cannot be used as the first layer."
            )
        elif previous_was_activation:
            raise ConfigError(
                f"Layer {index}: Consecutive activation layers are not supported."
            )
        else:
            previous_was_activation = True

    if not has_dense_layer:
        raise ConfigError(
            "Model validation failed: Model must contain at least one Dense layer"
        )

    input_features = next(
        layer.in_features for layer in config.layers if layer.type == "dense"
    )
    output_features = next(
        layer.out_features for layer in reversed(config.layers) if layer.type == "dense"
    )
    final_activation = config.layers[-1].type if config.layers[-1].type != "dense" else None
    if final_activation in {"relu", "sigmoid"} and config.target_range[0] < 0.0:
        raise ConfigError(
            f"target_range minimum must be non-negative for final {final_activation} activation"
        )
    for index, sample in enumerate(config.training_data):
        if len(sample.inputs) != input_features:
            raise ConfigError(
                f"Training sample {index}: expected {input_features} inputs, "
                f"got {len(sample.inputs)}"
            )
        if len(sample.targets) != output_features:
            raise ConfigError(
                f"Training sample {index}: expected {output_features} targets, "
                f"got {len(sample.targets)}"
            )