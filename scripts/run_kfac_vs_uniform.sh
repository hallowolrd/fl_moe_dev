#!/usr/bin/env bash

set -Eeuo pipefail


# ============================================================
# Usage
#
#   bash scripts/run_kfac_vs_uniform.sh <KFAC_GPU> <UNIFORM_GPU> <SEED>
#
# Example:
#
#   bash scripts/run_kfac_vs_uniform.sh 0 1 42
#
# 表示:
#   KFAC    -> 物理 GPU 0
#   Uniform -> 物理 GPU 1
#   Seed    -> 42
#
# 注意：
# Seed 请传你之前实验使用的那个 seed。
# ============================================================

if [[ $# -lt 3 ]]; then
    echo "Usage:"
    echo "  $0 <KFAC_GPU> <UNIFORM_GPU> <SEED>"
    echo
    echo "Example:"
    echo "  $0 0 1 42"
    exit 2
fi

GPU_KFAC="$1"
GPU_UNIFORM="$2"
SEED="$3"


# ============================================================
# 项目路径
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"


# ============================================================
# 实验公共参数
#
# 后续绝大部分实验只需要修改这里。
# ============================================================

DATASET="cifar10"

NUM_CLIENTS=10
ALPHA=0.1

NUM_ROUNDS=200
LOCAL_EPOCHS=1

CLIENT_BATCH_SIZE=32
TEST_BATCH_SIZE=256

NUM_EXPERTS=8
TOP_K=2

MOE_DIM=512
EXPERT_HIDDEN_DIM=1024

BALANCE_LOSS_WEIGHT=0.0

LEARNING_RATE=0.001
MOMENTUM=0.9
WEIGHT_DECAY=0.0005

BACKBONE="resnet18_gn"
MAX_GN_GROUPS=32

PARTICIPATION_RATE=1.0

SUMMARY_WINDOW=10

PARTITION_ROOT="partitions"

USE_AMP=false
DETERMINISTIC=true


# ============================================================
# KFAC 独有参数
# ============================================================

FISHER_BATCH_SIZE=128

MINIMUM_KFAC_SAMPLES=8

RELATIVE_DAMPING=0.01

MAX_WHITENING_GAIN=5.0

KFAC_SERVER_DEVICE="training"


# ============================================================
# 实验名称
#
# 自动把关键超参数写入目录名。
# ============================================================

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

RUN_NAME="$(
    printf \
    "%s_a%s_seed%s_%s_e%s_top%s_bs%s_lr%s_lb%s_%s" \
    "${DATASET}" \
    "${ALPHA}" \
    "${SEED}" \
    "${BACKBONE}" \
    "${NUM_EXPERTS}" \
    "${TOP_K}" \
    "${CLIENT_BATCH_SIZE}" \
    "${LEARNING_RATE}" \
    "${BALANCE_LOSS_WEIGHT}" \
    "${TIMESTAMP}"
)"

RUN_ROOT="outputs/pair_runs/${RUN_NAME}"

KFAC_OUTPUT="${RUN_ROOT}/kfac"
UNIFORM_OUTPUT="${RUN_ROOT}/uniform"

LOG_DIR="${RUN_ROOT}/logs"
META_DIR="${RUN_ROOT}/meta"

mkdir -p \
    "${KFAC_OUTPUT}" \
    "${UNIFORM_OUTPUT}" \
    "${LOG_DIR}" \
    "${META_DIR}"


# ============================================================
# 公共参数
#
# KFAC / Uniform 必须完全一致。
# ============================================================

COMMON_ARGS=(
    --seed "${SEED}"

    --dataset-name "${DATASET}"

    --num-clients "${NUM_CLIENTS}"
    --participation-rate "${PARTICIPATION_RATE}"
    --num-rounds "${NUM_ROUNDS}"
    --local-epochs "${LOCAL_EPOCHS}"
    --dirichlet-alpha "${ALPHA}"

    --client-batch-size "${CLIENT_BATCH_SIZE}"
    --test-batch-size "${TEST_BATCH_SIZE}"
    --no-drop-last

    --backbone-name "${BACKBONE}"

    --num-experts "${NUM_EXPERTS}"
    --top-k "${TOP_K}"

    --moe-dim "${MOE_DIM}"
    --expert-hidden-dim "${EXPERT_HIDDEN_DIM}"

    --small-image-stem
    --max-gn-groups "${MAX_GN_GROUPS}"

    --balance-loss-weight "${BALANCE_LOSS_WEIGHT}"

    --learning-rate "${LEARNING_RATE}"
    --momentum "${MOMENTUM}"
    --weight-decay "${WEIGHT_DECAY}"

    --summary-window "${SUMMARY_WINDOW}"

    --partition-root "${PARTITION_ROOT}"
)


# ============================================================
# Boolean 参数
# ============================================================

if [[ "${DETERMINISTIC}" == "true" ]]; then
    COMMON_ARGS+=(--deterministic)
fi

if [[ "${USE_AMP}" == "true" ]]; then
    COMMON_ARGS+=(--use-amp)
else
    COMMON_ARGS+=(--no-use-amp)
fi


# ============================================================
# KFAC 参数
# ============================================================

KFAC_ARGS=(
    --fisher-batch-size "${FISHER_BATCH_SIZE}"
    --minimum-kfac-samples "${MINIMUM_KFAC_SAMPLES}"
    --relative-damping "${RELATIVE_DAMPING}"
    --max-whitening-gain "${MAX_WHITENING_GAIN}"
    --kfac-server-device "${KFAC_SERVER_DEVICE}"
)


# ============================================================
# 最终命令
# ============================================================

KFAC_CMD=(
    "${PYTHON_BIN}"
    -u
    experiments/expert_local_kfac_whiten_layer_projection.py
    "${COMMON_ARGS[@]}"
    --device cuda:0
    --output-root "${KFAC_OUTPUT}"
    "${KFAC_ARGS[@]}"
)

UNIFORM_CMD=(
    "${PYTHON_BIN}"
    -u
    experiments/expert_uniform_all_valid_denominator.py
    "${COMMON_ARGS[@]}"
    --device cuda:0
    --output-root "${UNIFORM_OUTPUT}"
)


# ============================================================
# 日志
# ============================================================

KFAC_LOG="${LOG_DIR}/kfac.log"
UNIFORM_LOG="${LOG_DIR}/uniform.log"


# ============================================================
# 保存 Git commit
# ============================================================

if git rev-parse HEAD >/dev/null 2>&1; then
    git rev-parse HEAD > "${META_DIR}/git_commit.txt"

    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "WORKTREE_DIRTY=true" >> "${META_DIR}/git_commit.txt"
        git status --short > "${META_DIR}/git_status.txt"
    else
        echo "WORKTREE_DIRTY=false" >> "${META_DIR}/git_commit.txt"
    fi
fi


# ============================================================
# 保存实验参数
# ============================================================

cat > "${META_DIR}/config.txt" <<EOF
RUN_NAME=${RUN_NAME}

GPU_KFAC=${GPU_KFAC}
GPU_UNIFORM=${GPU_UNIFORM}

SEED=${SEED}

DATASET=${DATASET}

NUM_CLIENTS=${NUM_CLIENTS}
ALPHA=${ALPHA}

NUM_ROUNDS=${NUM_ROUNDS}
LOCAL_EPOCHS=${LOCAL_EPOCHS}

CLIENT_BATCH_SIZE=${CLIENT_BATCH_SIZE}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE}

BACKBONE=${BACKBONE}

NUM_EXPERTS=${NUM_EXPERTS}
TOP_K=${TOP_K}

MOE_DIM=${MOE_DIM}
EXPERT_HIDDEN_DIM=${EXPERT_HIDDEN_DIM}

BALANCE_LOSS_WEIGHT=${BALANCE_LOSS_WEIGHT}

LEARNING_RATE=${LEARNING_RATE}
MOMENTUM=${MOMENTUM}
WEIGHT_DECAY=${WEIGHT_DECAY}

FISHER_BATCH_SIZE=${FISHER_BATCH_SIZE}
MINIMUM_KFAC_SAMPLES=${MINIMUM_KFAC_SAMPLES}
RELATIVE_DAMPING=${RELATIVE_DAMPING}
MAX_WHITENING_GAIN=${MAX_WHITENING_GAIN}
KFAC_SERVER_DEVICE=${KFAC_SERVER_DEVICE}
EOF


# ============================================================
# 保存实际执行命令
# ============================================================

{
    echo "CUDA_VISIBLE_DEVICES=${GPU_KFAC} \\"
    printf '%q ' "${KFAC_CMD[@]}"
    echo
} > "${META_DIR}/kfac_command.txt"

{
    echo "CUDA_VISIBLE_DEVICES=${GPU_UNIFORM} \\"
    printf '%q ' "${UNIFORM_CMD[@]}"
    echo
} > "${META_DIR}/uniform_command.txt"


# ============================================================
# 保存 GPU 信息
# ============================================================

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "${META_DIR}/nvidia_smi.txt" || true
fi


# ============================================================
# Ctrl+C 时终止两个实验
# ============================================================

PID_KFAC=""
PID_UNIFORM=""

cleanup() {
    echo
    echo "[INFO] Stopping paired experiment..."

    if [[ -n "${PID_KFAC}" ]] && kill -0 "${PID_KFAC}" 2>/dev/null; then
        kill "${PID_KFAC}" 2>/dev/null || true
    fi

    if [[ -n "${PID_UNIFORM}" ]] && kill -0 "${PID_UNIFORM}" 2>/dev/null; then
        kill "${PID_UNIFORM}" 2>/dev/null || true
    fi
}

trap cleanup INT TERM


# ============================================================
# 打印实验信息
# ============================================================

echo "============================================================"
echo "Paired experiment"
echo "============================================================"
echo
echo "Run:"
echo "  ${RUN_NAME}"
echo
echo "Seed:"
echo "  ${SEED}"
echo
echo "KFAC:"
echo "  physical GPU = ${GPU_KFAC}"
echo "  output       = ${KFAC_OUTPUT}"
echo "  log          = ${KFAC_LOG}"
echo
echo "Uniform:"
echo "  physical GPU = ${GPU_UNIFORM}"
echo "  output       = ${UNIFORM_OUTPUT}"
echo "  log          = ${UNIFORM_LOG}"
echo
echo "Common:"
echo "  alpha        = ${ALPHA}"
echo "  experts      = ${NUM_EXPERTS}"
echo "  top-k        = ${TOP_K}"
echo "  batch        = ${CLIENT_BATCH_SIZE}"
echo "  lr           = ${LEARNING_RATE}"
echo "  LB           = ${BALANCE_LOSS_WEIGHT}"
echo "  rounds       = ${NUM_ROUNDS}"
echo
echo "============================================================"


# ============================================================
# 启动 KFAC
#
# CUDA_VISIBLE_DEVICES=物理GPU
# 进程内部始终使用 cuda:0
# ============================================================

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES="${GPU_KFAC}" \
"${KFAC_CMD[@]}" \
> "${KFAC_LOG}" 2>&1 &

PID_KFAC=$!

echo "[START] KFAC"
echo "        PID=${PID_KFAC}"


# ============================================================
# 启动 Uniform
# ============================================================

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES="${GPU_UNIFORM}" \
"${UNIFORM_CMD[@]}" \
> "${UNIFORM_LOG}" 2>&1 &

PID_UNIFORM=$!

echo "[START] Uniform"
echo "        PID=${PID_UNIFORM}"

echo
echo "Logs:"
echo "  tail -f ${KFAC_LOG}"
echo "  tail -f ${UNIFORM_LOG}"
echo


# ============================================================
# 等待实验完成
# ============================================================

set +e

wait "${PID_KFAC}"
KFAC_STATUS=$?

wait "${PID_UNIFORM}"
UNIFORM_STATUS=$?

set -e


# ============================================================
# 汇总退出状态
# ============================================================

echo
echo "============================================================"
echo "Experiment finished"
echo "============================================================"
echo
echo "KFAC status:"
echo "  ${KFAC_STATUS}"
echo
echo "Uniform status:"
echo "  ${UNIFORM_STATUS}"
echo
echo "Results:"
echo "  ${RUN_ROOT}"
echo

if [[ "${KFAC_STATUS}" -ne 0 ]]; then
    echo "[ERROR] KFAC failed. Check:"
    echo "        ${KFAC_LOG}"
fi

if [[ "${UNIFORM_STATUS}" -ne 0 ]]; then
    echo "[ERROR] Uniform failed. Check:"
    echo "        ${UNIFORM_LOG}"
fi

if [[ "${KFAC_STATUS}" -ne 0 || "${UNIFORM_STATUS}" -ne 0 ]]; then
    exit 1
fi

echo "[OK] Both experiments completed successfully."