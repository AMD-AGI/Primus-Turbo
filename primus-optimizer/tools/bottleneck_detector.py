"""Multi-dimensional bottleneck detection for optimization loops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BottleneckResult:
    is_bottleneck: bool
    reason: str | None = None  # "diminishing_returns" | "near_roofline" | "near_sota"
    details: str | None = None


class BottleneckDetector:
    def __init__(
        self,
        threshold: float = 0.02,
        patience: int = 3,
        roofline_ceiling: float = 0.80,
        sota_gap: float = 0.05,
    ):
        self.threshold = threshold
        self.patience = patience
        self.roofline_ceiling = roofline_ceiling
        self.sota_gap = sota_gap

    def check(
        self,
        rounds: list[dict],
        current_utilization: float | None = None,
        current_tflops: float | None = None,
        sota_tflops: float | None = None,
    ) -> BottleneckResult:
        # Check roofline first (highest confidence signal)
        if current_utilization is not None and current_utilization > self.roofline_ceiling:
            return BottleneckResult(
                is_bottleneck=True,
                reason="near_roofline",
                details=f"Utilization {current_utilization:.1%} > {self.roofline_ceiling:.0%} ceiling",
            )

        # Check SOTA gap
        if current_tflops is not None and sota_tflops is not None and sota_tflops > 0:
            gap = (sota_tflops - current_tflops) / sota_tflops
            if gap < self.sota_gap:
                return BottleneckResult(
                    is_bottleneck=True,
                    reason="near_sota",
                    details=f"Gap to SOTA: {gap:.1%} < {self.sota_gap:.0%} threshold",
                )

        # Check diminishing returns
        if len(rounds) >= self.patience:
            recent = rounds[-self.patience:]
            all_low = all(
                abs(r.get("improvement_pct", 0)) / 100 < self.threshold
                for r in recent
                if r.get("status") == "success"
            )
            successful = [r for r in recent if r.get("status") == "success"]
            if len(successful) >= self.patience and all_low:
                return BottleneckResult(
                    is_bottleneck=True,
                    reason="diminishing_returns",
                    details=f"Last {self.patience} rounds all < {self.threshold:.0%} improvement",
                )

        return BottleneckResult(is_bottleneck=False)
