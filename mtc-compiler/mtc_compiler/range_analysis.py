"""Conservative interval arithmetic primitives for Q15 analysis."""

from dataclasses import dataclass
import math

Q15_MIN = -1.0
Q15_MAX = 1.0 - (1.0 / 32768.0)
WARNING_THRESHOLD = 0.8


@dataclass(frozen=True)
class Interval:
    min: float
    max: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.min) or not math.isfinite(self.max):
            raise ValueError("Interval bounds must be finite")
        if self.min > self.max:
            raise ValueError(f"Invalid interval: min ({self.min}) > max ({self.max})")

    @property
    def fits_q15(self) -> bool:
        return self.min >= Q15_MIN and self.max <= Q15_MAX

    @property
    def near_overflow(self) -> bool:
        usage = max(abs(self.max), abs(self.min))
        return usage > WARNING_THRESHOLD and self.fits_q15

    @property
    def overflows_q15(self) -> bool:
        return self.min < Q15_MIN or self.max > Q15_MAX

    @property
    def stability_score(self) -> float:
        return max(abs(self.max), abs(self.min)) / Q15_MAX

    def add(self, other: "Interval") -> "Interval":
        return Interval(self.min + other.min, self.max + other.max)

    def sub(self, other: "Interval") -> "Interval":
        return Interval(self.min - other.max, self.max - other.min)

    def mul(self, other: "Interval") -> "Interval":
        products = (
            self.min * other.min,
            self.min * other.max,
            self.max * other.min,
            self.max * other.max,
        )
        return Interval(min(products), max(products))

    def scale(self, scalar: float) -> "Interval":
        bounds = (self.min * scalar, self.max * scalar)
        return Interval(min(bounds), max(bounds))

    def relu(self) -> "Interval":
        return Interval(max(0.0, self.min), max(0.0, self.max))

    def sigmoid(self) -> "Interval":
        def logistic(value: float) -> float:
            if value >= 0:
                return 1.0 / (1.0 + math.exp(-value))
            exp_value = math.exp(value)
            return exp_value / (1.0 + exp_value)

        return Interval(logistic(self.min), logistic(self.max))

    def sigmoid_derivative(self) -> "Interval":
        def derivative(value: float) -> float:
            sigmoid_value = 1.0 / (1.0 + math.exp(-value))
            return sigmoid_value * (1.0 - sigmoid_value)

        endpoints = (derivative(self.min), derivative(self.max))
        maximum = 0.25 if self.min <= 0.0 <= self.max else max(endpoints)
        return Interval(max(0.0, min(endpoints)), maximum)

    @staticmethod
    def dot_product(left: list["Interval"], right: list["Interval"]) -> "Interval":
        if len(left) != len(right):
            raise ValueError("Dot product dimension mismatch")
        result = Interval(0.0, 0.0)
        for left_value, right_value in zip(left, right):
            result = result.add(left_value.mul(right_value))
        return result

    @staticmethod
    def xavier_init(fan_in: int, fan_out: int) -> "Interval":
        if fan_in < 0 or fan_out < 0 or fan_in + fan_out == 0:
            raise ValueError("fan_in and fan_out must have a positive sum")
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        return Interval(-bound, bound)

    def clamp_q15(self) -> "Interval":
        clamp = lambda value: min(Q15_MAX, max(Q15_MIN, value))
        return Interval(clamp(self.min), clamp(self.max))