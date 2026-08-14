"""
Unit tests for the stateless communication-round LR scheduler.
"""

import math
import sys
from pathlib import Path

# Ensure we can import from the experiments directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from base import ExperimentConfig, compute_round_lr, validate_config


def _make_cosine_config(
    lr_max: float = 0.001,
    lr_min: float = 0.00005,
    warmup_rounds: int = 5,
    decay_end_round: int = 100,
) -> ExperimentConfig:
    return ExperimentConfig(
        lr_schedule="cosine",
        learning_rate=lr_max,
        lr_min=lr_min,
        warmup_rounds=warmup_rounds,
        decay_end_round=decay_end_round,
    )


# ============================================================================
# Constant schedule
# ============================================================================

def test_constant_lr_is_config_value():
    config = ExperimentConfig(lr_schedule="constant", learning_rate=0.001)
    for round_idx in [0, 10, 49, 99]:
        lr = compute_round_lr(round_idx, config)
        assert lr == 0.001, f"Round {round_idx}: expected 0.001, got {lr}"


def test_constant_lr_independent_of_round():
    config = ExperimentConfig(lr_schedule="constant", learning_rate=0.01)
    values = [compute_round_lr(r, config) for r in range(0, 200)]
    assert all(v == 0.01 for v in values), "Constant LR must not change with round"


# ============================================================================
# Cosine schedule — warmup
# ============================================================================

def test_cosine_warmup_first_round():
    """R1 must be 0.1 * lr_max (new corrected warmup)."""
    config = _make_cosine_config(lr_max=0.001, warmup_rounds=5)
    lr = compute_round_lr(0, config)  # R1
    expected = 0.001 * 0.1  # 0.0001
    assert math.isclose(lr, expected, rel_tol=1e-12), f"R1: expected {expected}, got {lr}"


def test_cosine_warmup_exact_values():
    """For W=5, lr_max=0.002, verify exact R1-R5 values."""
    lr_max = 0.002
    config = _make_cosine_config(lr_max=lr_max, warmup_rounds=5)
    expected = [
        lr_max * 0.100,    # R1: 0.0002
        lr_max * 0.325,    # R2: 0.00065
        lr_max * 0.550,    # R3: 0.00110
        lr_max * 0.775,    # R4: 0.00155
        lr_max * 1.000,    # R5: 0.00200
    ]
    for r_idx, exp in enumerate(expected):
        lr = compute_round_lr(r_idx, config)  # r_idx 0→R1, 1→R2, ...
        assert math.isclose(lr, exp, rel_tol=1e-12), (
            f"R{r_idx+1}: expected {exp}, got {lr}"
        )


def test_cosine_warmup_final_round():
    config = _make_cosine_config(lr_max=0.001, warmup_rounds=5)
    lr = compute_round_lr(4, config)  # R5
    expected = 0.001  # warmup reaches lr_max
    assert math.isclose(lr, expected, rel_tol=1e-12), f"R5: expected {expected}, got {lr}"


def test_cosine_warmup_monotonic_increasing():
    """Warmup must be strictly increasing for W=10."""
    config = _make_cosine_config(lr_max=0.002, warmup_rounds=10)
    values = [compute_round_lr(r, config) for r in range(0, 10)]
    for i in range(1, len(values)):
        assert values[i] > values[i - 1], (
            f"Warmup should be monotonic increasing: "
            f"R{i+1}={values[i]} <= R{i}={values[i-1]}"
        )


def test_cosine_warmup_boundary_no_jump():
    """R5 (last warmup) == lr_max, and R6 (first post-warmup) < lr_max."""
    lr_max = 0.002
    config = _make_cosine_config(lr_max=lr_max, lr_min=0.00005, warmup_rounds=5, decay_end_round=80)
    lr_r5 = compute_round_lr(4, config)   # R5
    lr_r6 = compute_round_lr(5, config)   # R6
    assert math.isclose(lr_r5, lr_max, rel_tol=1e-12), (
        f"R5 should equal lr_max, got {lr_r5}"
    )
    assert lr_r6 < lr_max, (
        f"R6 post-warmup should be < lr_max, got {lr_r6}"
    )
    assert lr_r6 > 0.0, f"R6 should be positive, got {lr_r6}"


# ============================================================================
# Cosine schedule — post-warmup / decay
# ============================================================================

def test_cosine_first_post_warmup_round():
    config = _make_cosine_config(lr_max=0.001, lr_min=0.0001, warmup_rounds=5, decay_end_round=70)
    lr = compute_round_lr(5, config)  # R6, first post-warmup
    # decay_progress = (6 - 5) / (70 - 5) = 1/65
    progress = 1.0 / 65.0
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    expected = 0.0001 + (0.001 - 0.0001) * cosine
    assert math.isclose(lr, expected, rel_tol=1e-12), f"R6: expected {expected}, got {lr}"


def test_cosine_decay_end_round():
    config = _make_cosine_config(lr_max=0.001, lr_min=0.00005, warmup_rounds=5, decay_end_round=70)
    lr = compute_round_lr(69, config)  # R70
    expected = 0.00005
    assert math.isclose(lr, expected, rel_tol=1e-12), f"R70: expected {expected}, got {lr}"


def test_cosine_after_decay_end_round():
    config = _make_cosine_config(lr_max=0.001, lr_min=0.00005, warmup_rounds=5, decay_end_round=70)
    for r in range(70, 200):
        lr = compute_round_lr(r, config)
        assert math.isclose(lr, 0.00005, rel_tol=1e-12), f"R{r+1}: expected 0.00005, got {lr}"


def test_cosine_decay_monotonic_non_increasing():
    config = _make_cosine_config(lr_max=0.002, lr_min=0.00005, warmup_rounds=5, decay_end_round=80)
    values = [compute_round_lr(r, config) for r in range(5, 80)]  # R6 to R80
    for i in range(1, len(values)):
        assert values[i] <= values[i - 1] + 1e-12, (
            f"Decay should be monotonic non-increasing: "
            f"R{i+6}={values[i]} > R{i+5}={values[i-1]}"
        )


# ============================================================================
# Stateless property
# ============================================================================

def test_compute_round_lr_is_stateless():
    config = _make_cosine_config(lr_max=0.001, lr_min=0.00005, warmup_rounds=5, decay_end_round=70)
    from copy import deepcopy
    config_copy = deepcopy(config)

    for round_idx in [0, 1, 5, 10, 49, 69, 99]:
        lr1 = compute_round_lr(round_idx, config)
        lr2 = compute_round_lr(round_idx, config_copy)
        assert math.isclose(lr1, lr2, rel_tol=1e-12), (
            f"Round {round_idx}: stateless property violated"
        )


def test_different_configs_produce_different_lrs():
    config_a = _make_cosine_config(lr_max=0.0015, lr_min=0.00005, warmup_rounds=5, decay_end_round=70)
    config_b = _make_cosine_config(lr_max=0.0020, lr_min=0.00005, warmup_rounds=5, decay_end_round=70)
    for round_idx in [10, 20, 30, 50]:
        a = compute_round_lr(round_idx, config_a)
        b = compute_round_lr(round_idx, config_b)
        assert not math.isclose(a, b, rel_tol=1e-8), (
            f"Round {round_idx}: different lr_max should produce different LRs: "
            f"{a} vs {b}"
        )


def test_mode_constant_produces_expected_lr_after_backward_compat():
    config = ExperimentConfig(lr_schedule="constant", learning_rate=0.001)
    for round_idx in range(0, 100):
        lr = compute_round_lr(round_idx, config)
        assert lr == 0.001, f"Round {round_idx}: expected 0.001, got {lr}"


# ============================================================================
# Edge cases
# ============================================================================

def test_warmup_rounds_zero():
    """When warmup_rounds=0, the schedule starts cosine decay from lr_max."""
    config = _make_cosine_config(lr_max=0.001, lr_min=0.00005, warmup_rounds=0, decay_end_round=70)
    lr = compute_round_lr(0, config)  # R1, no warmup
    # With warmup=0, R1 is the first decay step: progress = (1-0)/(70-0) = 1/70
    # This gives a value very close to but slightly less than lr_max
    expected = 0.001  # Should be slightly < lr_max
    assert lr < 0.001, f"R1 with warmup=0 should be < lr_max, got {lr}"
    assert lr > 0.0009, f"R1 with warmup=0 should be close to lr_max, got {lr}"


def test_warmup_rounds_one():
    """When warmup_rounds=1, R1 must be lr_max (only endpoint)."""
    lr_max = 0.001
    config = _make_cosine_config(lr_max=lr_max, lr_min=0.00005, warmup_rounds=1, decay_end_round=70)
    # R1: warmup round, should be lr_max
    lr_r1 = compute_round_lr(0, config)
    assert math.isclose(lr_r1, lr_max, rel_tol=1e-12), (
        f"R1 with warmup=1 should be lr_max, got {lr_r1}"
    )
    # R2: first post-warmup, should be slightly less than lr_max
    lr_r2 = compute_round_lr(1, config)
    assert lr_r2 < lr_max, f"R2 with warmup=1 should be < lr_max, got {lr_r2}"
    # Verify R2 matches expected cosine decay
    progress = 1.0 / 69.0  # (2 - 1) / (70 - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    expected = 0.00005 + (lr_max - 0.00005) * cosine
    assert math.isclose(lr_r2, expected, rel_tol=1e-12), (
        f"R2 with warmup=1: expected {expected}, got {lr_r2}"
    )


def test_decay_end_round_equals_warmup_rounds():
    """When decay_end_round == warmup_rounds, validate_config should reject."""
    config = ExperimentConfig(
        lr_schedule="cosine",
        learning_rate=0.001,
        lr_min=0.00005,
        warmup_rounds=10,
        decay_end_round=10,
    )
    try:
        validate_config(config)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_same_round_same_lr_across_calls():
    config = _make_cosine_config(lr_max=0.002, lr_min=0.00005, warmup_rounds=5, decay_end_round=80)
    results = []
    for _ in range(100):
        results.append(compute_round_lr(50, config))
    assert all(math.isclose(r, results[0], rel_tol=1e-15) for r in results)


# ============================================================================
# S1-S6 specific values (spot checks)
# ============================================================================

def test_s1_lr_values():
    """S1: lr_max=0.0015, decay_end_round=70, warmup=5, lr_min=0.00005
    (corrected warmup_v2 values)."""
    config = _make_cosine_config(lr_max=0.0015, lr_min=0.00005, warmup_rounds=5, decay_end_round=70)
    # R1 warmup (corrected: 0.1 * lr_max)
    lr_r1 = compute_round_lr(0, config)
    expected_r1 = 0.0015 * 0.1
    assert math.isclose(lr_r1, expected_r1, rel_tol=1e-12)
    # R5 end of warmup
    lr_r5 = compute_round_lr(4, config)
    assert math.isclose(lr_r5, 0.0015, rel_tol=1e-12)
    # R70 post-decay
    lr_r70 = compute_round_lr(69, config)
    assert math.isclose(lr_r70, 0.00005, rel_tol=1e-12)
    # R100 post-decay
    lr_r100 = compute_round_lr(99, config)
    assert math.isclose(lr_r100, 0.00005, rel_tol=1e-12)


def test_s4_lr_values():
    """S4: lr_max=0.0020, decay_end_round=80, warmup=5, lr_min=0.00005
    (corrected warmup_v2 values)."""
    config = _make_cosine_config(lr_max=0.0020, lr_min=0.00005, warmup_rounds=5, decay_end_round=80)
    # R1 warmup (corrected: 0.1 * lr_max)
    lr_r1 = compute_round_lr(0, config)
    expected_r1 = 0.0020 * 0.1
    assert math.isclose(lr_r1, expected_r1, rel_tol=1e-12)
    # R5 end of warmup
    lr_r5 = compute_round_lr(4, config)
    assert math.isclose(lr_r5, 0.0020, rel_tol=1e-12)
    # R80 post-decay
    lr_r80 = compute_round_lr(79, config)
    assert math.isclose(lr_r80, 0.00005, rel_tol=1e-12)
    # R100 post-decay
    lr_r100 = compute_round_lr(99, config)
    assert math.isclose(lr_r100, 0.00005, rel_tol=1e-12)


def test_s6_lr_values():
    """S6: lr_max=0.0025, decay_end_round=80, warmup=5, lr_min=0.00005
    (corrected warmup_v2 values)."""
    config = _make_cosine_config(lr_max=0.0025, lr_min=0.00005, warmup_rounds=5, decay_end_round=80)
    # R1 warmup (corrected: 0.1 * lr_max)
    lr_r1 = compute_round_lr(0, config)
    expected_r1 = 0.0025 * 0.1
    assert math.isclose(lr_r1, expected_r1, rel_tol=1e-12)
    # R5 end of warmup
    lr_r5 = compute_round_lr(4, config)
    assert math.isclose(lr_r5, 0.0025, rel_tol=1e-12)
    # R80 post-decay
    lr_r80 = compute_round_lr(79, config)
    assert math.isclose(lr_r80, 0.00005, rel_tol=1e-12)
    # R100 post-decay
    lr_r100 = compute_round_lr(99, config)
    assert math.isclose(lr_r100, 0.00005, rel_tol=1e-12)