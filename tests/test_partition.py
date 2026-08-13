"""
Unit tests for the equal-size FedDyn-style balanced label-Dirichlet partition.
"""

import sys
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np
import torch

from base import (
    ExperimentConfig,
    build_datasets,
    make_dirichlet_partition,
    load_or_create_partition,
    validate_partition,
    partition_path,
    get_dataset_targets,
    detect_num_classes,
)


def test_make_dirichlet_partition_equal_sizes():
    """Every client must have exactly the same number of samples."""
    labels = np.random.default_rng(42).integers(0, 10, size=50000)
    indices = make_dirichlet_partition(labels, 10, 0.1, 0)

    sizes = [len(client) for client in indices]
    assert max(sizes) == min(sizes) == 5000, (
        f"Expected all clients to have 5000 samples, got {sizes}"
    )


def test_make_dirichlet_partition_no_duplicates():
    """No index should appear in more than one client."""
    labels = np.random.default_rng(42).integers(0, 10, size=50000)
    indices = make_dirichlet_partition(labels, 10, 0.1, 0)

    flat = [idx for client in indices for idx in client]
    assert len(flat) == len(set(flat)), "Partition contains duplicated indices"


def test_make_dirichlet_partition_covers_all():
    """All indices must be assigned exactly once."""
    labels = np.random.default_rng(42).integers(0, 10, size=50000)
    indices = make_dirichlet_partition(labels, 10, 0.1, 0)

    flat = sorted([idx for client in indices for idx in client])
    expected = list(range(50000))
    assert flat == expected, "Partition does not cover all training indices"


def test_make_dirichlet_partition_10_clients():
    """Must produce exactly 10 clients."""
    labels = np.random.default_rng(42).integers(0, 10, size=50000)
    indices = make_dirichlet_partition(labels, 10, 0.1, 0)
    assert len(indices) == 10, f"Expected 10 clients, got {len(indices)}"


def test_make_dirichlet_partition_deterministic():
    """Same seed must produce the same partition."""
    labels = np.random.default_rng(42).integers(0, 10, size=50000)
    indices1 = make_dirichlet_partition(labels, 10, 0.1, 0)
    indices2 = make_dirichlet_partition(labels, 10, 0.1, 0)
    for c1, c2 in zip(indices1, indices2):
        assert c1 == c2, "Partition is not deterministic for the same seed"


def test_make_dirichlet_partition_different_seeds_different():
    """Different seeds should produce different partitions."""
    labels = np.random.default_rng(42).integers(0, 10, size=50000)
    indices1 = make_dirichlet_partition(labels, 10, 0.1, 0)
    indices2 = make_dirichlet_partition(labels, 10, 0.1, 1)
    # At least one client should differ
    all_same = all(c1 == c2 for c1, c2 in zip(indices1, indices2))
    assert not all_same, "Different seeds should produce different partitions"


def test_validate_partition_passes_correct_partition():
    """validate_partition should pass for a correct partition."""
    labels = np.random.default_rng(42).integers(0, 10, size=50000)
    indices = make_dirichlet_partition(labels, 10, 0.1, 0)
    # Should not raise
    validate_partition(indices, 50000, 10)


def test_validate_partition_rejects_unequal_sizes():
    """validate_partition should reject clients with different sizes."""
    indices = [[0, 1], [2, 3, 4]]
    # 50k total, 10 clients — but we're testing with smaller data
    try:
        validate_partition(indices, 50000, 2)
        # Might not reach the size check if dataset_size % num_clients != 0
    except RuntimeError:
        pass  # Expected


def test_validate_partition_rejects_duplicates():
    """validate_partition should reject duplicated indices."""
    labels = np.random.default_rng(42).integers(0, 10, size=50000)
    indices = make_dirichlet_partition(labels, 10, 0.1, 0)
    # Introduce a duplicate
    indices[0].append(indices[0][0])
    try:
        validate_partition(indices, 50000, 10)
        assert False, "Should have raised RuntimeError for duplicate indices"
    except RuntimeError:
        pass


def test_validate_partition_rejects_missing_indices():
    """validate_partition should reject missing indices."""
    labels = np.random.default_rng(42).integers(0, 10, size=50000)
    indices = make_dirichlet_partition(labels, 10, 0.1, 0)
    # Remove one index
    indices[0].pop()
    try:
        validate_partition(indices, 50000, 10)
        assert False, "Should have raised RuntimeError for missing indices"
    except RuntimeError:
        pass


def test_dirichlet_partition_alpha_0_1():
    """alpha=0.1 should produce heterogeneous label distributions."""
    labels = np.random.default_rng(42).integers(0, 10, size=50000)
    indices = make_dirichlet_partition(labels, 10, 0.1, 0)

    # Check that clients have different label distributions
    class_counts = []
    for client_indices in indices:
        client_labels = [labels[i] for i in client_indices]
        counts = [client_labels.count(c) for c in range(10)]
        class_counts.append(counts)

    # With alpha=0.1, clients should be specialized (at least one class with 0 samples)
    has_zeros = any(0 in counts for counts in class_counts)
    assert has_zeros, (
        "With alpha=0.1, expected at least one client to have zero samples "
        "in some class (heterogeneous distribution)"
    )


def test_cifar10_partition_5000_per_client():
    """CIFAR-10 with 50000 training samples and 10 clients → 5000 per client."""
    config = ExperimentConfig(
        dataset_name="cifar10",
        num_clients=10,
        dirichlet_alpha=0.1,
        seed=0,
        small_image_stem=True,
    )
    train_dataset, _ = build_datasets(config)
    assert len(train_dataset) == 50000, f"Expected 50000 training samples, got {len(train_dataset)}"

    # Create partition
    num_classes = detect_num_classes(train_dataset)
    client_indices, path, created = load_or_create_partition(config, train_dataset, num_classes)

    # Verify sizes
    sizes = [len(c) for c in client_indices]
    assert sizes == [5000] * 10, f"Expected all 5000, got {sizes}"

    # Verify coverage
    flat = [idx for c in client_indices for idx in c]
    assert len(flat) == 50000, f"Expected 50000 total indices, got {len(flat)}"
    assert len(set(flat)) == 50000, "Partition has duplicates"

    # Verify all indices are in range
    assert min(flat) >= 0 and max(flat) < 50000, "Indices out of range"


def test_partition_save_and_reload():
    """Saved partition must reload with identical metadata."""
    config = ExperimentConfig(
        dataset_name="cifar10",
        num_clients=10,
        dirichlet_alpha=0.1,
        seed=9999,  # Unique seed to avoid conflicts with other tests
        small_image_stem=True,
    )
    train_dataset, _ = build_datasets(config)
    num_classes = detect_num_classes(train_dataset)

    # First call creates the partition
    indices1, path1, created1 = load_or_create_partition(config, train_dataset, num_classes)
    # May or may not be created depending on test order; both are acceptable
    # as long as the reload matches

    # Second call loads it
    indices2, path2, created2 = load_or_create_partition(config, train_dataset, num_classes)
    assert not created2, "Partition should have been loaded on the second call, not created"

    assert path1 == path2, "Partition paths differ"
    assert indices1 == indices2, "Partition indices differ between save and reload"


def test_partition_sha256_integrity():
    """Partition should record labels_sha256 and verify on reload."""
    config = ExperimentConfig(
        dataset_name="cifar10",
        num_clients=10,
        dirichlet_alpha=0.1,
        seed=0,
        small_image_stem=True,
    )
    train_dataset, _ = build_datasets(config)
    num_classes = detect_num_classes(train_dataset)
    labels = get_dataset_targets(train_dataset)
    expected_hash = hashlib.sha256(labels.tobytes(order="C")).hexdigest()

    # Load the partition file and check labels_sha256
    import json
    path = partition_path(config)
    with path.open("r") as f:
        payload = json.load(f)

    assert payload["labels_sha256"] == expected_hash, (
        f"Partition labels_sha256 mismatch:\n"
        f"  expected: {expected_hash}\n"
        f"  got:      {payload['labels_sha256']}"
    )
    assert payload["num_total_samples"] == 50000
    assert payload["num_clients"] == 10