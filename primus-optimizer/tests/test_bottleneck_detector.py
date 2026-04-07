import pytest
from tools.bottleneck_detector import BottleneckDetector, BottleneckResult


def _make_rounds(gains: list[float]) -> list[dict]:
    """Helper: create round dicts with given improvement percentages."""
    return [
        {"round": i + 1, "improvement_pct": g, "status": "success"}
        for i, g in enumerate(gains)
    ]


def test_no_bottleneck():
    det = BottleneckDetector(threshold=0.02, patience=3)
    rounds = _make_rounds([25.0, 15.0, 8.0])
    result = det.check(rounds)
    assert result.is_bottleneck is False


def test_diminishing_returns_triggers():
    det = BottleneckDetector(threshold=0.02, patience=3)
    rounds = _make_rounds([25.0, 15.0, 8.0, 1.5, 0.8, 1.2])
    result = det.check(rounds)
    assert result.is_bottleneck is True
    assert result.reason == "diminishing_returns"


def test_near_roofline():
    det = BottleneckDetector(threshold=0.02, patience=3)
    rounds = _make_rounds([25.0, 15.0])
    result = det.check(rounds, current_utilization=0.85)
    assert result.is_bottleneck is True
    assert result.reason == "near_roofline"


def test_near_sota():
    det = BottleneckDetector(threshold=0.02, patience=3)
    rounds = _make_rounds([25.0])
    result = det.check(rounds, current_tflops=950, sota_tflops=980)
    assert result.is_bottleneck is True
    assert result.reason == "near_sota"


def test_not_near_sota():
    det = BottleneckDetector(threshold=0.02, patience=3)
    rounds = _make_rounds([25.0])
    result = det.check(rounds, current_tflops=500, sota_tflops=980)
    assert result.is_bottleneck is False


def test_empty_rounds():
    det = BottleneckDetector(threshold=0.02, patience=3)
    result = det.check([])
    assert result.is_bottleneck is False
