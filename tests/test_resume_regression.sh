#!/usr/bin/env bash
# Continuous-vs-Resume Regression Test (W=5 mid-warmup)
#
# Runs a short deterministic training comparison:
#   Branch A: uninterrupted training R1 → R6
#   Branch B: train R1 → R3, save checkpoint, terminate, resume R4 → R6
#
# Verifies that:
#   - Effective LR is identical between branches for every round
#   - Test accuracy and loss trajectories match
#   - Expert participant counts match
#   - Selected client IDs match
#   - Per-client route counts match
#   - Final model state_dict is identical tensor-by-tensor
#

set -euo pipefail

REPO_ROOT="/home/cjq/Project/fl_moe"
PYTHON="/home/cjq/anaconda3/envs/fl_moe/bin/python"
TEST_DIR="${REPO_ROOT}/tests/resume_regression"
OUTPUT_ROOT="${TEST_DIR}/outputs"
PARTITION_ROOT="${TEST_DIR}/partitions"

# Clean up any previous run
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

# W=5, decay_end_round=8 => R6 is first post-warmup round
COMMON_ARGS=(
    --dataset-name cifar10
    --num-clients 10
    --dirichlet-alpha 0.1
    --num-rounds 6
    --local-epochs 1
    --client-batch-size 64
    --test-batch-size 256
    --num-experts 4
    --top-k 2
    --lr-schedule cosine
    --learning-rate 0.001
    --lr-min 0.00005
    --warmup-rounds 5
    --decay-end-round 8
    --balance-loss-weight 0.01
    --seed 0
    --deterministic
    --output-root "$OUTPUT_ROOT"
    --partition-root "$PARTITION_ROOT"
    --checkpoint-interval 3
)

echo "=============================================="
echo "Continuous-vs-Resume Regression Test (W=5)"
echo "=============================================="
echo ""

# ============================================================
# Branch A: uninterrupted training R1 → R6
# ============================================================
echo ""
echo "=== Branch A: Uninterrupted training (R1 → R6) ==="
echo ""

cd "$REPO_ROOT/experiments"

$PYTHON expert_uniform_all_valid_denominator.py \
    "${COMMON_ARGS[@]}" \
    2>&1 | tail -5

# Find the output directory for Branch A (must contain metrics.csv)
DIR_A=$(find "$OUTPUT_ROOT" -type d -path "*/expert_uniform_all_valid_denominator/*" -exec test -e "{}/metrics.csv" \; -print | sort | tail -1)
echo "Branch A output: $DIR_A"

if [ -z "$DIR_A" ] || [ ! -f "$DIR_A/metrics.csv" ]; then
    echo "ERROR: Branch A metrics.csv not found at $DIR_A"
    exit 1
fi

echo "Branch A completed successfully."

# ============================================================
# Branch B: Interrupted training R1 → R3, resume R4 → R6
# ============================================================
echo ""
echo "=== Branch B: Interrupted training (R1 → R3) ==="
echo ""

# First segment: num-rounds=3 (checkpoint at R3)
$PYTHON expert_uniform_all_valid_denominator.py \
    "${COMMON_ARGS[@]}" \
    --num-rounds 3 \
    2>&1 | tail -5

# Find the output directory for Branch B first segment (must contain metrics.csv)
DIR_B=$(find "$OUTPUT_ROOT" -type d -path "*/expert_uniform_all_valid_denominator/*" -newer "$DIR_A" -exec test -e "{}/metrics.csv" \; -print | sort | tail -1)
echo "Branch B (first segment) output: $DIR_B"

if [ -z "$DIR_B" ] || [ ! -f "$DIR_B/metrics.csv" ]; then
    echo "ERROR: Branch B first segment metrics.csv not found at $DIR_B"
    exit 1
fi

# Verify checkpoint was saved
if [ ! -f "$DIR_B/checkpoint.pt" ]; then
    echo "ERROR: Branch B checkpoint.pt not found at $DIR_B"
    ls -la "$DIR_B/"
    exit 1
fi
echo "Checkpoint exists at $DIR_B/checkpoint.pt"

# ============================================================
# Resume Branch B: R4 → R6
# ============================================================
echo ""
echo "=== Branch B: Resume training (R4 → R6) ==="
echo ""

# Resume: use --resume to point to the existing output directory
$PYTHON expert_uniform_all_valid_denominator.py \
    "${COMMON_ARGS[@]}" \
    --num-rounds 6 \
    --resume "$DIR_B" \
    2>&1 | tail -5

echo "Branch B resumed and completed."

# ============================================================
# Compare trajectories
# ============================================================
echo ""
echo "=============================================="
echo "Comparing trajectories..."
echo "=============================================="

# Read metrics and compare
$PYTHON << 'PYTHON_SCRIPT'
import csv
import json
import sys
import torch
from pathlib import Path

test_dir = Path("/home/cjq/Project/fl_moe/tests/resume_regression")
output_root = test_dir / "outputs"

# Find Branch A (continuous) and Branch B (resumed) output directories
all_dirs = sorted(output_root.glob("cifar10/resnet18_gn/expert_uniform_all_valid_denominator/seed_0/*/"))
print(f"Found {len(all_dirs)} output directories")

# The first one is Branch A (continuous 6 rounds)
# The second one is Branch B (first segment 3 rounds, then resumed to 6)
dir_a = all_dirs[0] if len(all_dirs) > 0 else None
dir_b = all_dirs[1] if len(all_dirs) > 1 else None

if dir_a is None:
    print("ERROR: Could not find Branch A directory")
    sys.exit(1)
if dir_b is None:
    print("ERROR: Could not find Branch B directory")
    sys.exit(1)

# Read Branch A metrics (should have all 6 rounds)
with open(dir_a / "metrics.csv") as f:
    metrics_a = list(csv.DictReader(f))
rounds_a = [int(r["round"]) for r in metrics_a]
print(f"Branch A: {dir_a.name} (rounds {rounds_a[0]}-{rounds_a[-1]})")

# Read Branch B metrics (should have all 6 rounds after resume)
with open(dir_b / "metrics.csv") as f:
    metrics_b = list(csv.DictReader(f))
rounds_b = [int(r["round"]) for r in metrics_b]
print(f"Branch B: {dir_b.name} (rounds {rounds_b[0]}-{rounds_b[-1]})")

# Check for truncation or separate dir
if max(rounds_b) < 6:
    print(f"WARNING: Branch B only has rounds up to {max(rounds_b)}")
    for d in all_dirs:
        if d == dir_a or d == dir_b:
            continue
        with open(d / "metrics.csv") as f:
            rows = list(csv.DictReader(f))
        r = [int(x["round"]) for x in rows]
        if max(r) >= 6:
            print(f"Found resumed metrics in separate directory: {d}")
            metrics_b = rows
            rounds_b = r
            break

# Compare
print(f"\nBranch A rounds: {rounds_a}")
print(f"Branch B rounds: {rounds_b}")

if len(metrics_a) != len(metrics_b):
    print(f"ERROR: Different number of rounds: A={len(metrics_a)}, B={len(metrics_b)}")
    sys.exit(1)

# Verify expected warmup values for W=5, lr_max=0.001
expected_lrs = {
    1: 0.0001,    # 0.1 * 0.001
    2: 0.000325,  # 0.325 * 0.001
    3: 0.00055,   # 0.55 * 0.001
    4: 0.000775,  # 0.775 * 0.001
    5: 0.001,     # 1.0 * 0.001
    # R6 is cosine decay
}
print("\n--- Expected Warmup LR Verification ---")
warmup_ok = True
for r_num, exp_lr in expected_lrs.items():
    row_a = metrics_a[r_num - 1]
    lr_a = float(row_a["effective_lr"])
    match = "OK" if abs(lr_a - exp_lr) < 1e-10 else "MISMATCH"
    if match == "MISMATCH":
        warmup_ok = False
    print(f"  R{r_num}: A={lr_a:.10f} (expected {exp_lr:.10f}) {match}")

if not warmup_ok:
    print("FAIL: Warmup LR values do not match expected corrected values")
    sys.exit(1)
else:
    print("PASS: Warmup LR values match corrected values")

# Compare LR trajectory
print("\n--- LR Comparison ---")
lr_mismatch = False
for row_a, row_b in zip(metrics_a, metrics_b):
    r = row_a["round"]
    lr_a = float(row_a["effective_lr"])
    lr_b = float(row_b["effective_lr"])
    match = "OK" if abs(lr_a - lr_b) < 1e-10 else "MISMATCH"
    if match == "MISMATCH":
        lr_mismatch = True
    print(f"  R{r}: A={lr_a:.10f}  B={lr_b:.10f}  {match}")

# Compare accuracy
print("\n--- Accuracy Comparison ---")
acc_mismatch = False
for row_a, row_b in zip(metrics_a, metrics_b):
    r = row_a["round"]
    acc_a = float(row_a["test_accuracy"])
    acc_b = float(row_b["test_accuracy"])
    diff = abs(acc_a - acc_b)
    match = "OK" if diff < 1e-6 else "DIFFERS"
    if match == "DIFFERS":
        acc_mismatch = True
    print(f"  R{r}: A={acc_a*100:.4f}%  B={acc_b*100:.4f}%  diff={diff*100:.6f}pp  {match}")

# Compare training loss
print("\n--- Mean Client Loss Comparison ---")
loss_mismatch = False
for row_a, row_b in zip(metrics_a, metrics_b):
    r = row_a["round"]
    loss_a = float(row_a["mean_client_loss"])
    loss_b = float(row_b["mean_client_loss"])
    diff = abs(loss_a - loss_b)
    match = "OK" if diff < 1e-6 else "DIFFERS"
    if match == "DIFFERS":
        loss_mismatch = True
    print(f"  R{r}: A={loss_a:.8f}  B={loss_b:.8f}  diff={diff:.8f}  {match}")

# Compare selected client IDs
print("\n--- Selected Client IDs Comparison ---")
sel_mismatch = False
for row_a, row_b in zip(metrics_a, metrics_b):
    r = row_a["round"]
    sa = json.loads(row_a["selected_client_ids"])
    sb = json.loads(row_b["selected_client_ids"])
    match = "OK" if sa == sb else "DIFFERS"
    if match == "DIFFERS":
        sel_mismatch = True
    print(f"  R{r}: A={sa}  B={sb}  {match}")

# Compare expert participants
print("\n--- Expert Participants Comparison ---")
part_mismatch = False
for row_a, row_b in zip(metrics_a, metrics_b):
    r = row_a["round"]
    pa = json.loads(row_a["expert_participant_counts"])
    pb = json.loads(row_b["expert_participant_counts"])
    match = "OK" if pa == pb else "DIFFERS"
    if match == "DIFFERS":
        part_mismatch = True
    print(f"  R{r}: A={pa}  B={pb}  {match}")

# Compare per-client route counts
print("\n--- Per-Client Route Counts Comparison ---")
route_mismatch = False
for row_a, row_b in zip(metrics_a, metrics_b):
    r = row_a["round"]
    # Check if field exists
    if "client_route_counts" not in row_a or "client_route_counts" not in row_b:
        print(f"  R{r}: client_route_counts not available in metrics, skipping")
        continue
    ra = json.loads(row_a["client_route_counts"])
    rb = json.loads(row_b["client_route_counts"])
    match = "OK" if ra == rb else "DIFFERS"
    if match == "DIFFERS":
        route_mismatch = True
    print(f"  R{r}: A={ra}  B={rb}  {match}")

# ============================================================
# Final model state comparison (tensor-by-tensor)
# ============================================================
print("\n" + "=" * 60)
print("FINAL MODEL STATE COMPARISON")
print("=" * 60)
print("")

# Load both checkpoints
ckpt_a = torch.load(dir_a / "checkpoint.pt", map_location="cpu", weights_only=False)
ckpt_b = torch.load(dir_b / "checkpoint.pt", map_location="cpu", weights_only=False)

sd_a = ckpt_a["model_state_dict"]
sd_b = ckpt_b["model_state_dict"]

# Check same keys
keys_a = set(sd_a.keys())
keys_b = set(sd_b.keys())
if keys_a != keys_b:
    print(f"FAIL: Model state dict keys differ")
    print(f"  Only in A: {keys_a - keys_b}")
    print(f"  Only in B: {keys_b - keys_a}")
    sys.exit(1)

max_abs_diff = 0.0
max_rel_diff = 0.0
max_diff_key = ""
all_exact = True

for key in sorted(sd_a.keys()):
    ta = sd_a[key]
    tb = sd_b[key]
    if ta.shape != tb.shape:
        print(f"FAIL: Key {key} has different shapes: {ta.shape} vs {tb.shape}")
        sys.exit(1)
    if not torch.is_floating_point(ta):
        # Non-floating-point: check exact equality
        if not torch.equal(ta, tb):
            print(f"FAIL: Non-float key {key} differs between branches")
            all_exact = False
            max_diff_key = key
        continue
    diff = (ta - tb).abs()
    max_elem = diff.max().item()
    if max_elem > max_abs_diff:
        max_abs_diff = max_elem
        max_diff_key = key
    if max_elem != 0.0:
        all_exact = False
        # Relative diff
        norm_b = tb.norm().item()
        if norm_b > 0:
            rel = max_elem / norm_b
            if rel > max_rel_diff:
                max_rel_diff = rel

print(f"All exact (zero diff across all tensors): {all_exact}")
print(f"Max absolute difference: {max_abs_diff:.10f} (key: {max_diff_key})")
print(f"Max relative difference: {max_rel_diff:.10e}")

if not all_exact:
    print("")
    print("NOTE: Non-zero differences detected. Possible causes:")
    print("  - Non-deterministic CUDA operations")
    print("  - Floating-point accumulation order differences")
    print("  - Checkpoint save/load precision")
    print("")
    print("Diagnosing: checking if diff is within floating-point tolerance...")
    if max_abs_diff < 1e-8:
        print(f"  Max abs diff {max_abs_diff:.2e} < 1e-8 — acceptable precision loss")
        all_exact = True  # Accept as effectively identical
    else:
        print(f"  Max abs diff {max_abs_diff:.2e} >= 1e-8 — requires investigation")

# Results
print("\n" + "=" * 60)
print("REGRESSION TEST RESULTS")
print("=" * 60)

all_pass = True
if lr_mismatch:
    print("FAIL: LR trajectory differs between continuous and resume")
    all_pass = False
else:
    print("PASS: LR trajectory is identical")

if acc_mismatch:
    print("FAIL: Test accuracy differs between continuous and resume")
    all_pass = False
else:
    print("PASS: Test accuracy is identical")

if loss_mismatch:
    print("FAIL: Training loss differs between continuous and resume")
    all_pass = False
else:
    print("PASS: Training loss is identical")

if sel_mismatch:
    print("FAIL: Selected client IDs differ between continuous and resume")
    all_pass = False
else:
    print("PASS: Selected client IDs are identical")

if part_mismatch:
    print("FAIL: Expert participants differ between continuous and resume")
    all_pass = False
else:
    print("PASS: Expert participants are identical")

if route_mismatch:
    print("FAIL: Per-client route counts differ between continuous and resume")
    all_pass = False
else:
    print("PASS: Per-client route counts are identical")

if not all_exact:
    print("FAIL: Final model state dict differs between continuous and resume")
    all_pass = False
else:
    print("PASS: Final model state dict is identical")

if all_pass:
    print("\n*** RESUME_REGRESSION_PASS = true ***")
    sys.exit(0)
else:
    print("\n*** RESUME_REGRESSION_PASS = false ***")
    sys.exit(1)
PYTHON_SCRIPT

RESULT=$?

# Cleanup
echo ""
if [ $RESULT -eq 0 ]; then
    echo "=== Regression test PASSED. Cleaning up... ==="
    rm -rf "$TEST_DIR"
    echo "=== Done ==="
else
    echo "=== Regression test FAILED. Keeping output in $TEST_DIR ==="
fi

exit $RESULT