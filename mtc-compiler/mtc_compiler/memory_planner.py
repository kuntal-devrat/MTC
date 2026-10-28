"""Static tensor and training-scratch memory planning."""

from dataclasses import dataclass

from .config import ModelConfig
from .ir import DenseStage


@dataclass(frozen=True)
class Allocation:
    name: str
    offset_elements: int
    elements: int
    first_use: int
    last_use: int

    @property
    def end_elements(self) -> int:
        return self.offset_elements + self.elements


@dataclass(frozen=True)
class MemoryPlan:
    allocations: tuple[Allocation, ...]
    pool_elements: int
    persistent_bytes: int
    total_static_ram_bytes: int
    budget_bytes: int

    def offset(self, name: str) -> int:
        for allocation in self.allocations:
            if allocation.name == name:
                return allocation.offset_elements
        raise KeyError(name)


def _allocate_first_fit(requests: list[tuple[str, int, int, int]]) -> tuple[Allocation, ...]:
    allocations: list[Allocation] = []
    for name, elements, first_use, last_use in sorted(
        requests, key=lambda request: (request[2], -request[1], request[0])
    ):
        offset = 0
        while True:
            conflicts = [
                allocation for allocation in allocations
                if not (last_use < allocation.first_use or first_use > allocation.last_use)
                and offset < allocation.end_elements
                and allocation.offset_elements < offset + elements
            ]
            if not conflicts:
                break
            offset = min(conflict.end_elements for conflict in conflicts)
        allocations.append(Allocation(name, offset, elements, first_use, last_use))
    return tuple(allocations)


def plan_memory(
    config: ModelConfig,
    stages: tuple[DenseStage, ...],
    budget_bytes: int = 64 * 1024,
) -> MemoryPlan:
    """Allocate retained activations and two reusable delta buffers statically."""
    if not stages:
        raise ValueError("At least one dense stage is required")

    stage_count = len(stages)
    forward_end = stage_count - 1
    backward_start = stage_count
    backward_end = 2 * stage_count - 1
    requests = [
        (
            f"activation_{stage.index}",
            stage.out_features,
            stage.index,
            backward_end - stage.index,
        )
        for stage in stages
    ]
    weight_elements = sum(stage.in_features * stage.out_features for stage in stages)
    bias_elements = sum(stage.out_features for stage in stages)
    requests.extend((
        ("weights", weight_elements, 0, backward_end),
        ("biases", bias_elements, 0, backward_end),
    ))
    max_width = max(stage.out_features for stage in stages)
    requests.extend((
        ("delta_a", max_width, backward_start, backward_end),
        ("delta_b", max_width, backward_start, backward_end),
    ))
    allocations = _allocate_first_fit(requests)
    pool_elements = max(allocation.end_elements for allocation in allocations)
    weights_and_biases = sum(
        stage.in_features * stage.out_features + stage.out_features for stage in stages
    )
    persistent_bytes = weights_and_biases * 2
    total_static_ram_bytes = pool_elements * 2
    if total_static_ram_bytes > budget_bytes:
        raise ValueError(
            f"Static RAM requirement {total_static_ram_bytes} bytes exceeds "
            f"the {budget_bytes}-byte target budget"
        )

    return MemoryPlan(
        allocations=allocations,
        pool_elements=pool_elements,
        persistent_bytes=persistent_bytes,
        total_static_ram_bytes=total_static_ram_bytes,
        budget_bytes=budget_bytes,
    )