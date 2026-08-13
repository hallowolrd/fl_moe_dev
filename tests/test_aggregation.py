"""
Unit tests for Direct and Activation expert aggregation.

These tests verify the exact algebraic formulas required by the protocol.
"""

import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import torch
import torch.nn as nn

from base import (
    ExperimentConfig,
    ClientUpdate,
    SparseMoEClassifier,
    build_resnet18_gn,
    build_sparse_moe,
    aggregate_shared_uniform,
    state_delta,
    validate_config,
)


def _make_dummy_config() -> ExperimentConfig:
    return ExperimentConfig(num_experts=4, top_k=2)


def _make_dummy_model(num_experts: int = 4, num_classes: int = 10) -> SparseMoEClassifier:
    backbone = build_resnet18_gn(in_channels=3, small_image_stem=True)
    model = build_sparse_moe(
        backbone=backbone,
        num_classes=num_classes,
        num_experts=num_experts,
        top_k=2,
        moe_dim=512,
        expert_hidden_dim=1024,
    )
    model.cpu()
    model.eval()
    return model


def _make_dummy_update(
    client_id: int,
    route_counts: list[int],
    num_experts: int = 4,
    num_classes: int = 10,
) -> ClientUpdate:
    """Create a ClientUpdate with zero-valued deltas for testing weight formulas."""
    model = _make_dummy_model(num_experts, num_classes)
    shared_delta = {k: torch.zeros_like(v) for k, v in model.get_shared_state_dict().items()}
    expert_deltas = [
        {k: torch.zeros_like(v) for k, v in model.get_expert_state_dict(e).items()}
        for e in range(num_experts)
    ]
    return ClientUpdate(
        client_id=client_id,
        num_examples=5000,
        num_processed_examples=5000,
        shared_delta=shared_delta,
        expert_deltas=expert_deltas,
        route_counts=torch.tensor(route_counts, dtype=torch.long),
        train_loss=0.5,
        standard_classification_loss=0.5,
        balance_loss=0.0,
        accuracy=0.5,
    )


# ============================================================================
# Direct — all-valid denominator
# ============================================================================

def test_direct_all_valid_denominator_k3():
    """Direct synthetic test: 3 clients, expert e active on 2 of them."""
    from expert_uniform_all_valid_denominator import aggregate_experts_uniform

    model = _make_dummy_model()
    old_state = model.get_expert_state_dict(0, clone=True, to_cpu=True)

    # 3 clients, expert 0 active on client 0 and 1, inactive on client 2
    updates = [
        _make_dummy_update(client_id=0, route_counts=[5, 10, 3, 2]),
        _make_dummy_update(client_id=1, route_counts=[8, 0, 7, 5]),
        _make_dummy_update(client_id=2, route_counts=[0, 12, 8, 0]),
    ]

    # Set non-zero deltas for expert 0 so we can verify weights
    for upd in updates:
        for key in upd.expert_deltas[0]:
            upd.expert_deltas[0][key] = torch.ones_like(upd.expert_deltas[0][key]) * upd.client_id

    participants = aggregate_experts_uniform(model, updates, 4)

    # Expert 0: active on client 0 and 1
    # K_valid = 3
    # Expected weight: 1/3 for clients 0 and 1, 0 for client 2
    assert participants[0] == 2, f"Expected 2 participants for expert 0, got {participants[0]}"

    # Verify the delta was correctly weighted
    new_state = model.get_expert_state_dict(0, clone=True, to_cpu=True)
    for key in old_state:
        old_val = old_state[key]
        new_val = new_state[key]
        if torch.is_floating_point(old_val):
            expected_delta = (torch.ones_like(old_val) * 0 + torch.ones_like(old_val) * 1) / 3.0
            expected = old_val + expected_delta
            assert torch.allclose(new_val, expected, atol=1e-6), (
                f"Expert 0 key {key}: expected {expected}, got {new_val}"
            )


def test_direct_denominator_is_all_valid_not_active():
    """
    Verify that the denominator is K_valid, NOT |A_e|.
    If |A_e| < K_valid, the weight sum < 1.
    """
    from expert_uniform_all_valid_denominator import aggregate_experts_uniform

    model = _make_dummy_model()
    old_state = model.get_expert_state_dict(0, clone=True, to_cpu=True)

    # 3 clients, expert 0 active on only client 0
    updates = [
        _make_dummy_update(client_id=0, route_counts=[5, 10, 3, 2]),
        _make_dummy_update(client_id=1, route_counts=[0, 8, 7, 5]),
        _make_dummy_update(client_id=2, route_counts=[0, 12, 8, 0]),
    ]

    # Set non-zero deltas
    for upd in updates:
        for key in upd.expert_deltas[0]:
            upd.expert_deltas[0][key] = torch.ones_like(upd.expert_deltas[0][key]) * 2.0

    # Expert 0: only client 0 active
    # K_valid = 3
    # Expected: client 0 weight = 1/3, not 1/1
    # If incorrectly renormalized: weight = 1/1 = 1.0
    # Correct: weight = 1/3
    participants = aggregate_experts_uniform(model, updates, 4)
    assert participants[0] == 1, f"Expected 1 participant for expert 0, got {participants[0]}"

    new_state = model.get_expert_state_dict(0, clone=True, to_cpu=True)
    for key in old_state:
        old_val = old_state[key]
        new_val = new_state[key]
        if torch.is_floating_point(old_val):
            # Correct: delta = 2.0 / 3 = 0.666...
            expected_delta = torch.ones_like(old_val) * 2.0 / 3.0
            expected = old_val + expected_delta
            assert torch.allclose(new_val, expected, atol=1e-6), (
                f"Direct denominator test failed for {key}: "
                f"expected {expected}, got {new_val}. "
                "This may indicate renormalization over active clients."
            )


def test_direct_zero_expert_preserves_parameters():
    """If an expert has no active clients, its parameters must remain unchanged."""
    from expert_uniform_all_valid_denominator import aggregate_experts_uniform

    model = _make_dummy_model()
    old_states = [model.get_expert_state_dict(e, clone=True, to_cpu=True) for e in range(4)]

    updates = [
        _make_dummy_update(client_id=0, route_counts=[5, 0, 3, 0]),
        _make_dummy_update(client_id=1, route_counts=[8, 0, 7, 0]),
    ]

    # Expert 1 and 3 have zero route counts
    aggregate_experts_uniform(model, updates, 4)

    for e in [1, 3]:
        new_state = model.get_expert_state_dict(e, clone=True, to_cpu=True)
        for key in old_states[e]:
            old_val = old_states[e][key]
            new_val = new_state[key]
            assert torch.equal(old_val, new_val), (
                f"Expert {e} key {key} changed despite zero active clients"
            )


# ============================================================================
# Activation-frequency weighting
# ============================================================================

def test_activation_frequency_synthetic():
    """
    Activation-frequency synthetic test from the protocol.

    client0: expert e count = 10, total routes = 20, frequency = 0.5
    client1: expert e count = 30, total routes = 100, frequency = 0.3

    Expected normalized weights:
        client0: 0.5 / 0.8 = 0.625
        client1: 0.3 / 0.8 = 0.375

    NOT raw-count weights: 0.25 and 0.75
    """
    from expert_activation_frequency_weighted import (
        aggregate_experts_activation_frequency_weighted,
    )

    model = _make_dummy_model()
    old_state = model.get_expert_state_dict(0, clone=True, to_cpu=True)

    # client0: 10 routes for expert 0, total 20 routes
    # client1: 30 routes for expert 0, total 100 routes
    updates = [
        _make_dummy_update(client_id=0, route_counts=[10, 5, 3, 2]),  # total=20
        _make_dummy_update(client_id=1, route_counts=[30, 20, 25, 25]),  # total=100
    ]

    # Set non-zero deltas for expert 0
    for upd in updates:
        for key in upd.expert_deltas[0]:
            upd.expert_deltas[0][key] = torch.ones_like(upd.expert_deltas[0][key]) * float(upd.client_id)

    participants, client_weights = aggregate_experts_activation_frequency_weighted(
        model, updates, 4
    )

    assert participants[0] == 2, f"Expected 2 participants for expert 0, got {participants[0]}"

    # Check weights
    assert 0 in client_weights[0], "Client 0 missing from expert 0 weights"
    assert 1 in client_weights[0], "Client 1 missing from expert 0 weights"

    w0 = client_weights[0][0]
    w1 = client_weights[0][1]

    assert abs(w0 - 0.625) < 1e-10, f"Expected client 0 weight ≈ 0.625, got {w0}"
    assert abs(w1 - 0.375) < 1e-10, f"Expected client 1 weight ≈ 0.375, got {w1}"

    # Verify weight sum ≈ 1
    assert abs(w0 + w1 - 1.0) < 1e-10, f"Weight sum {w0 + w1} != 1"

    # Verify the delta was correctly weighted
    new_state = model.get_expert_state_dict(0, clone=True, to_cpu=True)
    for key in old_state:
        old_val = old_state[key]
        new_val = new_state[key]
        if torch.is_floating_point(old_val):
            # Expected: 0.625 * client0_delta + 0.375 * client1_delta
            # client0_delta = 0, client1_delta = 1
            expected_delta = 0.625 * torch.ones_like(old_val) * 0.0 + 0.375 * torch.ones_like(old_val) * 1.0
            expected = old_val + expected_delta
            assert torch.allclose(new_val, expected, atol=1e-6), (
                f"Expert 0 key {key}: expected {expected}, got {new_val}"
            )


def test_activation_frequency_not_raw_count():
    """
    Verify that activation-frequency weighting is used, not raw-count weighting.

    If raw counts were used:
        client0 weight = 10 / (10+30) = 0.25
        client1 weight = 30 / (10+30) = 0.75

    With frequency weighting:
        client0 frequency = 10/20 = 0.5
        client1 frequency = 30/100 = 0.3
        client0 weight = 0.5/0.8 = 0.625
        client1 weight = 0.375/0.8 = 0.375
    """
    from expert_activation_frequency_weighted import (
        aggregate_experts_activation_frequency_weighted,
    )

    model = _make_dummy_model()
    updates = [
        _make_dummy_update(client_id=0, route_counts=[10, 10, 0, 0]),  # total=20
        _make_dummy_update(client_id=1, route_counts=[30, 30, 20, 20]),  # total=100
    ]

    participants, client_weights = aggregate_experts_activation_frequency_weighted(
        model, updates, 4
    )

    w0 = client_weights[0][0]
    w1 = client_weights[0][1]

    # Frequency weights, not raw counts
    assert abs(w0 - 0.625) < 1e-10, (
        f"Expected frequency weight 0.625, got {w0}. "
        "Raw-count weight would be 0.25."
    )
    assert abs(w1 - 0.375) < 1e-10, (
        f"Expected frequency weight 0.375, got {w1}. "
        "Raw-count weight would be 0.75."
    )


def test_activation_zero_expert_preserves_parameters():
    """If an expert has no active clients, its parameters must remain unchanged."""
    from expert_activation_frequency_weighted import (
        aggregate_experts_activation_frequency_weighted,
    )

    model = _make_dummy_model()
    old_states = [model.get_expert_state_dict(e, clone=True, to_cpu=True) for e in range(4)]

    updates = [
        _make_dummy_update(client_id=0, route_counts=[5, 0, 3, 0]),
        _make_dummy_update(client_id=1, route_counts=[8, 0, 7, 0]),
    ]

    aggregate_experts_activation_frequency_weighted(model, updates, 4)

    for e in [1, 3]:
        new_state = model.get_expert_state_dict(e, clone=True, to_cpu=True)
        for key in old_states[e]:
            old_val = old_states[e][key]
            new_val = new_state[key]
            assert torch.equal(old_val, new_val), (
                f"Expert {e} key {key} changed despite zero active clients"
            )


def test_activation_weight_sum_close_to_one():
    """Per-expert activation-frequency weights should sum to 1."""
    from expert_activation_frequency_weighted import (
        aggregate_experts_activation_frequency_weighted,
    )

    model = _make_dummy_model()
    updates = [
        _make_dummy_update(client_id=0, route_counts=[5, 10, 3, 2]),
        _make_dummy_update(client_id=1, route_counts=[8, 5, 7, 0]),
        _make_dummy_update(client_id=2, route_counts=[3, 12, 8, 0]),
    ]

    participants, client_weights = aggregate_experts_activation_frequency_weighted(
        model, updates, 4
    )

    for e in range(4):
        if participants[e] > 0:
            wsum = sum(client_weights[e].values())
            assert abs(wsum - 1.0) < 1e-10, f"Expert {e} weight sum = {wsum}, expected 1.0"


# ============================================================================
# Shared aggregation — must be identical for both methods
# ============================================================================

def test_shared_aggregation_is_uniform():
    """Shared aggregation must be uniform over all valid clients."""
    model = _make_dummy_model()
    old_shared = model.get_shared_state_dict(clone=True, to_cpu=True)

    updates = [
        _make_dummy_update(client_id=0, route_counts=[5, 5, 5, 5]),
        _make_dummy_update(client_id=1, route_counts=[5, 5, 5, 5]),
        _make_dummy_update(client_id=2, route_counts=[5, 5, 5, 5]),
    ]

    # Set different shared deltas per client
    for upd in updates:
        for key in upd.shared_delta:
            if torch.is_floating_point(upd.shared_delta[key]):
                upd.shared_delta[key] = torch.ones_like(upd.shared_delta[key]) * float(upd.client_id)

    aggregate_shared_uniform(model, updates)

    new_shared = model.get_shared_state_dict(clone=True, to_cpu=True)
    for key in old_shared:
        old_val = old_shared[key]
        new_val = new_shared[key]
        if torch.is_floating_point(old_val):
            # Expected delta = (0 + 1 + 2) / 3 = 1.0
            expected = old_val + torch.ones_like(old_val) * 1.0
            assert torch.allclose(new_val, expected, atol=1e-6), (
                f"Shared key {key}: expected {expected}, got {new_val}"
            )


# ============================================================================
# Top-2 routing
# ============================================================================

def test_top2_routing_shape():
    """Verify Top-2 routing produces the correct shapes."""
    config = _make_dummy_config()
    model = _make_dummy_model(num_experts=4, num_classes=10)

    batch_size = 16
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    output = model(dummy_input)

    assert output.logits.shape == (batch_size, 10), (
        f"Expected logits shape ({batch_size}, 10), got {output.logits.shape}"
    )
    assert output.topk_indices.shape == (batch_size, 2), (
        f"Expected topk_indices shape ({batch_size}, 2), got {output.topk_indices.shape}"
    )
    assert output.route_counts.sum().item() == batch_size * 2, (
        f"Expected route count sum = {batch_size * 2}, "
        f"got {output.route_counts.sum().item()}"
    )


def test_top2_route_counts_nonzero():
    """With Top-2 routing, at least some experts should get dispatches."""
    model = _make_dummy_model(num_experts=4, num_classes=10)
    batch_size = 64
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    output = model(dummy_input)

    assert output.route_counts.sum().item() == batch_size * 2, (
        f"Route count sum mismatch: {output.route_counts.sum().item()} vs {batch_size * 2}"
    )


# ============================================================================
# Config validation
# ============================================================================

def test_lr_schedule_validation():
    """Verify that invalid lr_schedule values are rejected by validate_config."""
    try:
        config = ExperimentConfig(lr_schedule="invalid")
        validate_config(config)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_cosine_config_validation():
    """Verify that cosine schedule requires decay_end_round > warmup_rounds."""
    try:
        config = ExperimentConfig(
            lr_schedule="cosine",
            learning_rate=0.001,
            lr_min=0.00005,
            warmup_rounds=10,
            decay_end_round=5,
        )
        validate_config(config)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_checkpoint_interval_validation():
    """Verify that negative checkpoint_interval is rejected."""
    try:
        config = ExperimentConfig(checkpoint_interval=-1)
        validate_config(config)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass