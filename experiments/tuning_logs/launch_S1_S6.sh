#!/usr/bin/env bash
# Launch S1-S6 formal 100-round scheduler experiments
#
# Each GPU runs one FAIR_AB pair (Direct + Activation) concurrently.
# All experiments use detached tmux sessions for disconnect safety.
#
set -euo pipefail

REPO_ROOT="/home/cjq/Project/fl_moe"
EXPERIMENTS_DIR="${REPO_ROOT}/experiments"
PYTHON="/home/cjq/anaconda3/envs/fl_moe/bin/python"
LOGDIR="${REPO_ROOT}/experiments/tuning_logs/detached"
mkdir -p "$LOGDIR"

# ============================================================
# Common config (as a flat string, not array, to avoid word-splitting
# issues inside tmux command strings)
# ============================================================
COMMON="--dataset-name cifar10 \
 --num-clients 10 \
 --dirichlet-alpha 0.1 \
 --num-rounds 100 \
 --local-epochs 1 \
 --client-batch-size 64 \
 --test-batch-size 256 \
 --num-experts 4 \
 --top-k 2 \
 --lr-schedule cosine \
 --lr-min 0.00005 \
 --warmup-rounds 5 \
 --balance-loss-weight 0.01 \
 --seed 0 \
 --deterministic \
 --checkpoint-interval 20"

# ============================================================
# Scheduler candidates
# ============================================================
declare -A SCHEDULER
SCHEDULER["S1"]="--learning-rate 0.0015 --decay-end-round 70"
SCHEDULER["S2"]="--learning-rate 0.0015 --decay-end-round 80"
SCHEDULER["S3"]="--learning-rate 0.0020 --decay-end-round 70"
SCHEDULER["S4"]="--learning-rate 0.0020 --decay-end-round 80"
SCHEDULER["S5"]="--learning-rate 0.0025 --decay-end-round 70"
SCHEDULER["S6"]="--learning-rate 0.0025 --decay-end-round 80"

# GPU assignment: one FAIR_AB pair per GPU
declare -A GPU_MAP
GPU_MAP["S1"]=0
GPU_MAP["S2"]=1
GPU_MAP["S3"]=2
GPU_MAP["S4"]=3
GPU_MAP["S5"]=4
GPU_MAP["S6"]=5

# ============================================================
# Launch function
# ============================================================
launch_pair() {
    local SCHED_ID="$1"
    local GPU="${GPU_MAP[$SCHED_ID]}"
    local SCHED_ARGS="${SCHEDULER[$SCHED_ID]}"
    local SESSION_DIR="${LOGDIR}/${SCHED_ID}"

    echo "=============================================="
    echo "Launching $SCHED_ID on GPU $GPU"
    echo "  Args: $SCHED_ARGS"
    echo "=============================================="

    mkdir -p "$SESSION_DIR"

    # --- Direct method ---
    local DIRECT_SESSION="${SCHED_ID}_direct"
    local DIRECT_LOG="${SESSION_DIR}/${DIRECT_SESSION}.launcher.log"
    local DIRECT_PIDFILE="${SESSION_DIR}/${DIRECT_SESSION}.pid"
    local DIRECT_EXITFILE="${SESSION_DIR}/${DIRECT_SESSION}.exit_code"

    # Build command string carefully to avoid word-splitting issues
    local DIRECT_CMD
    DIRECT_CMD="cd ${EXPERIMENTS_DIR} && "
    DIRECT_CMD+="echo \$\$ > ${DIRECT_PIDFILE} && "
    DIRECT_CMD+="CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} "
    DIRECT_CMD+="expert_uniform_all_valid_denominator.py "
    DIRECT_CMD+="${COMMON} ${SCHED_ARGS} "
    DIRECT_CMD+=">> ${DIRECT_LOG} 2>&1 ; "
    DIRECT_CMD+="echo \$? > ${DIRECT_EXITFILE}"

    tmux new-session -d -s "$DIRECT_SESSION" "$DIRECT_CMD"

    # --- Activation method ---
    local ACTIVATION_SESSION="${SCHED_ID}_activation"
    local ACTIVATION_LOG="${SESSION_DIR}/${ACTIVATION_SESSION}.launcher.log"
    local ACTIVATION_PIDFILE="${SESSION_DIR}/${ACTIVATION_SESSION}.pid"
    local ACTIVATION_EXITFILE="${SESSION_DIR}/${ACTIVATION_SESSION}.exit_code"

    local ACTIVATION_CMD
    ACTIVATION_CMD="cd ${EXPERIMENTS_DIR} && "
    ACTIVATION_CMD+="echo \$\$ > ${ACTIVATION_PIDFILE} && "
    ACTIVATION_CMD+="CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} "
    ACTIVATION_CMD+="expert_activation_frequency_weighted.py "
    ACTIVATION_CMD+="${COMMON} ${SCHED_ARGS} "
    ACTIVATION_CMD+=">> ${ACTIVATION_LOG} 2>&1 ; "
    ACTIVATION_CMD+="echo \$? > ${ACTIVATION_EXITFILE}"

    tmux new-session -d -s "$ACTIVATION_SESSION" "$ACTIVATION_CMD"

    echo "Launched tmux sessions: $DIRECT_SESSION, $ACTIVATION_SESSION"
    echo "Logs: ${SESSION_DIR}/"
    echo ""
}

# ============================================================
# Launch all 6 pairs
# ============================================================
echo "=============================================="
echo "S1-S6 SCHEDULER SWEEP LAUNCH"
echo "Date: $(date)"
echo "Python: ${PYTHON}"
echo "=============================================="
echo ""

echo "Preflight: checking runtime..."
${PYTHON} - <<'PY' || { echo "RUNTIME PREFLIGHT FAILED"; exit 1; }
import sys, torch
print(f"sys.executable: {sys.executable}")
print(f"torch: {torch.__version__}, CUDA: {torch.version.cuda}, available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA unavailable"
PY
echo "Preflight passed."
echo ""

for S in S1 S2 S3 S4 S5 S6; do
    launch_pair "$S"
done

# ============================================================
# Verify all sessions launched
# ============================================================
echo ""
echo "Waiting 15 seconds for sessions to initialize..."
sleep 15
echo "Verifying tmux sessions..."
ALL_OK=true
for S in S1 S2 S3 S4 S5 S6; do
    for METHOD in direct activation; do
        SESSION="${S}_${METHOD}"
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "  OK: $SESSION"
        else
            echo "  MISSING: $SESSION"
            ALL_OK=false
        fi
    done
done

# Quick check gpu processes
echo ""
nvidia-smi --query-gpu=index,used_memory --format=csv,noheader 2>/dev/null || true

echo ""
if $ALL_OK; then
    echo "All 12 sessions verified OK."
    echo ""
    echo "Monitor with:"
    echo "  tail -f ${LOGDIR}/S1/S1_direct.launcher.log"
    echo "  tail -f ${LOGDIR}/S1/S1_activation.launcher.log"
    echo ""
    echo "Attach:"
    echo "  tmux attach -t S1_direct"
    echo ""
    echo "List all:"
    echo "  tmux list-sessions"
else
    echo "Some sessions are missing. Check logs in ${LOGDIR}/"
fi