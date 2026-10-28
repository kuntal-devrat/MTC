"""Compile-time value and gradient range propagation."""

from dataclasses import dataclass

from .config import ModelConfig
from .ir import DenseStage
from .range_analysis import Interval, Q15_MAX, Q15_MIN, WARNING_THRESHOLD


@dataclass(frozen=True)
class RangeRecord:
    stage: int
    operation: str
    interval: Interval

    @property
    def overflows_q15(self) -> bool:
        return self.interval.min < Q15_MIN or self.interval.max > Q15_MAX

    @property
    def near_overflow(self) -> bool:
        magnitude = max(abs(self.interval.min), abs(self.interval.max))
        return not self.overflows_q15 and magnitude > WARNING_THRESHOLD

    @property
    def may_underflow(self) -> bool:
        return max(abs(self.interval.min), abs(self.interval.max)) < 1.0 / 32768.0


@dataclass(frozen=True)
class StageRanges:
    output: Interval
    delta: Interval
    weight_gradient: Interval
    updated_weight: Interval


@dataclass(frozen=True)
class AnalysisResult:
    stages: tuple[StageRanges, ...]
    records: tuple[RangeRecord, ...]

    @property
    def overflows(self) -> tuple[RangeRecord, ...]:
        return tuple(record for record in self.records if record.overflows_q15)

    @property
    def warnings(self) -> tuple[RangeRecord, ...]:
        return tuple(
            record for record in self.records
            if record.near_overflow or record.may_underflow
        )

    @property
    def stability_score(self) -> float:
        return max(
            (max(abs(record.interval.min), abs(record.interval.max)) / Q15_MAX
             for record in self.records),
            default=0.0,
        )


def _activation(interval: Interval, activation: str | None) -> Interval:
    if activation == "relu":
        return interval.relu()
    if activation == "sigmoid":
        return interval.sigmoid()
    return interval


def _activation_derivative(output: Interval, activation: str | None) -> Interval:
    if activation == "relu":
        if output.min > 0.0:
            return Interval(1.0, 1.0)
        if output.max <= 0.0:
            return Interval(0.0, 0.0)
        return Interval(0.0, 1.0)
    if activation == "sigmoid":
        def derivative(value: float) -> float:
            return value * (1.0 - value)

        endpoint_values = (derivative(output.min), derivative(output.max))
        maximum = 0.25 if output.min <= 0.5 <= output.max else max(endpoint_values)
        return Interval(min(endpoint_values), maximum)
    return Interval(1.0, 1.0)


def analyze_ranges(config: ModelConfig, stages: tuple[DenseStage, ...]) -> AnalysisResult:
    """Propagate initialization-time ranges through forward and one backward step."""
    if not stages:
        raise ValueError("At least one dense stage is required")

    input_range = Interval(*config.input_range)
    target_range = Interval(*config.target_range)
    records: list[RangeRecord] = []
    outputs: list[Interval] = []
    inputs: list[Interval] = []
    weight_ranges: list[Interval] = []

    current = input_range
    for stage in stages:
        inputs.append(current)
        xavier_range = Interval.xavier_init(stage.in_features, stage.out_features)
        weights = Interval(
            max(Q15_MIN, xavier_range.min),
            min(Q15_MAX, xavier_range.max),
        )
        weight_ranges.append(weights)
        records.append(RangeRecord(stage.index, "Q15 initial weights", weights))
        dense_output = current.mul(weights).scale(stage.in_features)
        records.append(RangeRecord(stage.index, "dense accumulator output", dense_output))
        current = _activation(dense_output, stage.activation)
        records.append(RangeRecord(stage.index, "activation output", current))
        outputs.append(current)

    gradient = outputs[-1].sub(target_range).scale(2.0 / stages[-1].out_features)
    stages_reversed: list[StageRanges | None] = [None] * len(stages)
    for stage_index in range(len(stages) - 1, -1, -1):
        stage = stages[stage_index]
        gradient = gradient.mul(_activation_derivative(outputs[stage_index], stage.activation))
        records.append(RangeRecord(stage.index, "backward delta", gradient))
        weight_gradient = inputs[stage_index].mul(gradient)
        records.append(RangeRecord(stage.index, "weight gradient", weight_gradient))
        updated_weight = weight_ranges[stage_index].sub(
            weight_gradient.scale(config.optimizer.learning_rate)
        )
        records.append(RangeRecord(stage.index, "updated weights", updated_weight))
        stages_reversed[stage_index] = StageRanges(
            output=outputs[stage_index],
            delta=gradient,
            weight_gradient=weight_gradient,
            updated_weight=updated_weight,
        )
        if stage_index > 0:
            gradient = gradient.mul(weight_ranges[stage_index]).scale(stage.out_features)
            records.append(RangeRecord(stage.index, "input gradient", gradient))

    return AnalysisResult(tuple(stages_reversed), tuple(records))