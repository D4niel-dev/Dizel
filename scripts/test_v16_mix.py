"""Regression tests for the v1.6 family/source allocation water-fill."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipelines.prepare_v16_data import FAMILY_MIX, water_fill


def test_no_caps_keeps_weights():
    alloc = water_fill(
        {"general": 0.65, "code": 0.20, "math": 0.10, "reasoning": 0.05},
        {
            "general": 10**12,
            "code": 10**12,
            "math": 10**12,
            "reasoning": 10**12,
        },
        1_000,
    )
    assert abs(sum(alloc.values()) - 1_000) < 1e-6
    for key, share in FAMILY_MIX.items():
        assert abs(alloc[key] - 1_000 * share) < 1e-6, (key, alloc)


def test_capped_allocation_is_redistributed():
    alloc = water_fill({"a": 0.5, "b": 0.5}, {"a": 20, "b": 100}, 100)
    assert abs(alloc["a"] - 20) < 1e-6, alloc
    assert abs(alloc["b"] - 80) < 1e-6, alloc
    assert abs(sum(alloc.values()) - 100) < 1e-6, alloc


def test_target_above_total_capacity_is_fully_capped():
    alloc = water_fill({"a": 0.5, "b": 0.5}, {"a": 20, "b": 30}, 100)
    assert abs(alloc["a"] - 20) < 1e-6, alloc
    assert abs(alloc["b"] - 30) < 1e-6, alloc
    assert abs(sum(alloc.values()) - 50) < 1e-6, alloc


def test_phase_a_family_targets_fit_caps():
    caps = {
        "general": 2_145_263_304,
        "code": 661_122_623,
        "math": 314_934_778,
        "reasoning": 152_100_624,
    }
    target = 3_014_999_999
    alloc = water_fill(FAMILY_MIX, caps, target)
    assert abs(sum(alloc.values()) - target) < 1e-3, alloc
    for key, share in FAMILY_MIX.items():
        assert abs(alloc[key] / target - share) < 1e-9, (key, alloc)


if __name__ == "__main__":
    test_no_caps_keeps_weights()
    test_capped_allocation_is_redistributed()
    test_target_above_total_capacity_is_fully_capped()
    test_phase_a_family_targets_fit_caps()
    print("water_fill tests passed")
