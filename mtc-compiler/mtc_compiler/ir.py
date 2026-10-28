"""Small typed structures shared by compiler passes."""

from dataclasses import dataclass, replace

from .config import ConfigError, ModelConfig


@dataclass(frozen=True)
class TensorShape:
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if self.rows < 0 or self.cols < 0:
            raise ValueError("Tensor dimensions cannot be negative")

    @property
    def numel(self) -> int:
        return self.rows * self.cols

    @property
    def size_bytes_q15(self) -> int:
        return self.numel * 2


@dataclass(frozen=True)
class DenseStage:
    index: int
    config_index: int
    in_features: int
    out_features: int
    activation: str | None


def build_stages(config: ModelConfig) -> tuple[DenseStage, ...]:
    """Group each dense operation with its optional following activation."""
    stages: list[DenseStage] = []
    for config_index, layer in enumerate(config.layers):
        if layer.type == "dense":
            stages.append(DenseStage(
                index=len(stages),
                config_index=config_index,
                in_features=layer.in_features,
                out_features=layer.out_features,
                activation=None,
            ))
            continue

        if not stages or stages[-1].activation is not None:
            raise ConfigError(
                f"Layer {config_index}: consecutive activation layers are not supported"
            )
        stages[-1] = replace(stages[-1], activation=layer.type)

    return tuple(stages)