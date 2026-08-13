#!/usr/bin/env bash
# Continuous-vs-Resume Regression Test
#
# Runs a short deterministic training comparison:
#   Branch A: uninterrupted training R1 → R6
#   Branch B: train R1 → R3, save checkpoint, terminate, resume R4 → R6
#
# Verifies that:
#   - Effective LR is identical between branches for every round
#   - Test accuracy and loss trajectories match
#   - Expert participant counts match
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
    --warmup-rounds 2
    --decay-end-round 6
    --balance-loss-weight 0.01
    --seed 0
    --deterministic
    --output-root "$OUTPUT_ROOT"
    --partition-root "$PARTITION_ROOT"
    --checkpoint-interval 3
)

echo "=============================================="
echo "Continuous-vs-Resume Regression Test"
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

# Find the output directory for Branch A
DIR_A=$(find "$OUTPUT_ROOT" -type d -path "*/expert_uniform_all_valid_denominator/*" | sort | tail -1)
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

# First segment: num-rounds=3
$PYTHON expert_uniform_all_valid_denominator.py \
    "${COMMON_ARGS[@]}" \
    --num-rounds 3 \
    2>&1 | tail -5

# Find the output directory for Branch B first segment
DIR_B=$(find "$OUTPUT_ROOT" -type d -path "*/expert_uniform_all_valid_denominator/*" -newer "$DIR_A" | sort | tail -1)
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

# Read metrics
$PYTHON << 'PYTHON_SCRIPT'
import csv
import json
import sys
from pathlib import Path

test_dir = Path("/home/cjq/Project/fl_moe/tests/resume_regression")
output_root = test_dir / "outputs"

# Find Branch A (continuous) and Branch B (resumed) output directories
all_dirs = sorted(output_root.glob("cifar10/resnet18_gn/expert_uniform_all_valid_denominator/seed_0/*/"))
print(f"Found {len(all_dirs)} output directories")

# The first one is Branch A (continuous 6 rounds)
# The second one is Branch B (first segment 3 rounds)
# The resumed run writes to the same directory as Branch B
dir_a = all_dirs[0] if len(all_dirs) > 0 else None
dir_b = all_dirs[1] if len(all_dirs) > 1 else None

if dir_a is None:
    print("ERROR: Could not find Branch A directory")
    sys.exit(1)
if dir_b is None:
    print("ERROR: Could not find Branch B directory")
    sys.exit(1)

# For Branch B, the metrics.csv should have been updated by the resume
# If the resume wrote to a separate directory, handle that
dir_b_resumed = all_dirs[2] if len(all_dirs) > 2 else None
if dir_b_resumed is not None:
    # Check if the resumed run created a separate directory
    b_resumed_metrics = dir_b_resumed / "metrics.csv"
    with open(b_resumed_metrics) as f:
        rows = list(csv.DictReader(f))
    rounds = [int(r["round"]) for r in rows]
    if max(rounds) >= 6:
        print(f"Note: Resume created separate directory. Using {dir_b_resumed}")
        # The resumed run has its own metrics, but we need the checkpoint-based
        # resume to have worked. Let's use dir_b (which has rounds 1-3)
        # and the resumed dir (which should have rounds 4-6)
        # But actually, the resume should write to the SAME directory (dir_b)
        pass

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

# Check if Branch B has all 6 rounds, or if we need to find the resumed metrics
if max(rounds_b) < 6:
    # The resume may have created a separate directory or the metrics weren't appended
    # Check if the checkpoint was loaded and resumed correctly
    print(f"WARNING: Branch B only has rounds up to {max(rounds_b)}")
    # Check if the resumed run is in a separate dir
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

# Compare LR
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

if part_mismatch:
    print("FAIL: Expert participants differ between continuous and resume")
    all_pass = False
else:
    print("PASS: Expert participants are identical")

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