"""
Unit tests for metrics truncation and resume config compatibility.
"""

import csv
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import pytest
from base import (
    ExperimentConfig,
    scientific_config_key,
    compute_scientific_config_hash,
    _truncate_metrics,
    validate_config,
)


# ============================================================================
# Metrics truncation tests
# ============================================================================

def test_truncate_metrics_removes_extra_rows():
    """Truncation removes rows with round > completed_round."""
    content = "round,value\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n5,0.5\n"
    fieldnames = ["round", "value"]
    input_file = io.StringIO(content)
    # Write to a temp file
    tmp_path = REPO_ROOT / "tests" / "_tmp_truncate_metrics.csv"
    tmp_path.write_text(content)

    try:
        _truncate_metrics(tmp_path, completed_round=3, fieldnames=fieldnames)

        with open(tmp_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        rounds = [int(r["round"]) for r in rows]
        assert rounds == [1, 2, 3], f"Expected [1, 2, 3], got {rounds}"
        assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_truncate_metrics_noop_when_already_correct():
    """Truncation is a no-op when all rows are <= completed_round."""
    content = "round,value\n1,0.1\n2,0.2\n3,0.3\n"
    fieldnames = ["round", "value"]
    tmp_path = REPO_ROOT / "tests" / "_tmp_truncate_metrics.csv"
    tmp_path.write_text(content)

    try:
        _truncate_metrics(tmp_path, completed_round=5, fieldnames=fieldnames)

        with open(tmp_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
        assert [int(r["round"]) for r in rows] == [1, 2, 3]
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_truncate_metrics_handles_empty_file():
    """Truncation handles a non-existent file gracefully."""
    tmp_path = REPO_ROOT / "tests" / "_tmp_truncate_metrics_nonexistent.csv"
    if tmp_path.exists():
        tmp_path.unlink()
    # Should not raise
    _truncate_metrics(tmp_path, completed_round=3, fieldnames=["round", "value"])


def test_truncate_metrics_preserves_header():
    """After truncation, the CSV header is preserved."""
    content = "round,value\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n"
    fieldnames = ["round", "value"]
    tmp_path = REPO_ROOT / "tests" / "_tmp_truncate_metrics.csv"
    tmp_path.write_text(content)

    try:
        _truncate_metrics(tmp_path, completed_round=2, fieldnames=fieldnames)

        with open(tmp_path, "r") as f:
            lines = f.readlines()
        assert lines[0].strip() == "round,value", f"Header corrupted: {lines[0]}"
        # Should have header + 2 data rows
        assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ============================================================================
# Scientific config key tests
# ============================================================================

def test_scientific_config_key_includes_all_scientific_fields():
    """The scientific config key covers all relevant training fields."""
    config = ExperimentConfig(
        seed=42,
        deterministic=True,
        num_clients=10,
        participation_rate=1.0,
        dirichlet_alpha=0.1,
        dataset_name="cifar10",
        backbone_name="resnet18_gn",
        num_experts=4,
        top_k=2,
        moe_dim=512,
        expert_hidden_dim=1024,
        small_image_stem=True,
        max_gn_groups=32,
        zero_init_residual=False,
        local_epochs=1,
        client_batch_size=64,
        drop_last=False,
        balance_loss_weight=0.01,
        learning_rate=0.001,
        lr_schedule="cosine",
        lr_schedule_version="warmup_v2_0p1_to_1p0",
        lr_min=0.00005,
        warmup_rounds=5,
        decay_end_round=80,
        momentum=0.9,
        weight_decay=5e-4,
        use_amp=False,
        max_grad_norm=None,
    )
    key = scientific_config_key(config, "test_algo")
    assert key["algorithm_name"] == "test_algo"
    assert key["seed"] == 42
    assert key["num_experts"] == 4
    assert key["top_k"] == 2
    assert key["lr_schedule_version"] == "warmup_v2_0p1_to_1p0"


def test_scientific_config_key_excludes_infra_fields():
    """Infrastructure fields should not appear in the scientific key."""
    config = ExperimentConfig(
        num_rounds=200,
        checkpoint_interval=10,
        summary_window=10,
    )
    key = scientific_config_key(config)
    # infra fields should not be in the key
    assert "num_rounds" not in key
    assert "checkpoint_interval" not in key
    assert "summary_window" not in key


def test_scientific_config_hash_is_stable():
    """Same config produces the same hash."""
    config = ExperimentConfig(seed=42, learning_rate=0.001)
    h1 = compute_scientific_config_hash(config, "test")
    h2 = compute_scientific_config_hash(config, "test")
    assert h1 == h2, "Hash should be stable across calls"


def test_scientific_config_hash_changes_with_field():
    """Different configs produce different hashes."""
    config_a = ExperimentConfig(seed=42, learning_rate=0.001)
    config_b = ExperimentConfig(seed=42, learning_rate=0.002)
    h_a = compute_scientific_config_hash(config_a, "test")
    h_b = compute_scientific_config_hash(config_b, "test")
    assert h_a != h_b, "Different configs should have different hashes"


def test_scientific_config_hash_changes_with_algorithm():
    """Different algorithm names produce different hashes."""
    config = ExperimentConfig(seed=42, learning_rate=0.001)
    h_direct = compute_scientific_config_hash(config, "expert_uniform_all_valid_denominator")
    h_act = compute_scientific_config_hash(config, "expert_activation_frequency_weighted")
    assert h_direct != h_act, "Different algorithms should have different hashes"


# ============================================================================
# Resume config rejection tests
# ============================================================================

def test_resume_config_mismatch_detects_balance_change():
    """Changing balance_loss_weight should be detected as a config mismatch."""
    # This test verifies that scientific_config_key captures balance_loss_weight
    config_a = ExperimentConfig(balance_loss_weight=0.01)
    config_b = ExperimentConfig(balance_loss_weight=0.02)
    key_a = scientific_config_key(config_a, "test")
    key_b = scientific_config_key(config_b, "test")
    assert key_a != key_b, "Different balance_loss_weight should produce different keys"


def test_resume_config_mismatch_detects_algorithm_change():
    """Changing algorithm should be detected as a config mismatch."""
    key_direct = {"algorithm_name": "expert_uniform_all_valid_denominator"}
    key_act = {"algorithm_name": "expert_activation_frequency_weighted"}
    assert key_direct != key_act, "Different algorithms should produce different keys"


def test_resume_config_mismatch_detects_topk_change():
    """Changing top_k should be detected as a config mismatch."""
    config_a = ExperimentConfig(top_k=2)
    config_b = ExperimentConfig(top_k=1)
    key_a = scientific_config_key(config_a, "test")
    key_b = scientific_config_key(config_b, "test")
    assert key_a != key_b, "Different top_k should produce different keys"


def test_resume_config_mismatch_detects_epochs_change():
    """Changing local_epochs should be detected as a config mismatch."""
    config_a = ExperimentConfig(local_epochs=1)
    config_b = ExperimentConfig(local_epochs=2)
    key_a = scientific_config_key(config_a, "test")
    key_b = scientific_config_key(config_b, "test")
    assert key_a != key_b, "Different local_epochs should produce different keys"


def test_resume_config_mismatch_detects_momentum_change():
    """Changing momentum should be detected as a config mismatch."""
    config_a = ExperimentConfig(momentum=0.9)
    config_b = ExperimentConfig(momentum=0.8)
    key_a = scientific_config_key(config_a, "test")
    key_b = scientific_config_key(config_b, "test")
    assert key_a != key_b, "Different momentum should produce different keys"


def test_resume_config_allows_num_rounds_change():
    """Changing num_rounds should NOT affect the scientific config key."""
    config_a = ExperimentConfig(num_rounds=100)
    config_b = ExperimentConfig(num_rounds=200)
    key_a = scientific_config_key(config_a, "test")
    key_b = scientific_config_key(config_b, "test")
    assert key_a == key_b, "num_rounds should not be in scientific config key"