#!/bin/bash
#SBATCH --job-name=fmri-pretrain
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=640G
#SBATCH --time=10-00:00:00
# --time is a ceiling, not an estimate: the recipe measured 60-123 H100-hours per run (mean 85).
# set your own account/partition/qos

# Usage: sbatch scripts/pretrain.sbatch <run_name> [extra fmrifm.train args...]
#
# A fresh run resolves configs/model.yaml plus one corpus from configs/data/ into
# logs/<run_name>/config.yaml and never resolves them again: every later attempt,
# including Slurm's automatic requeue, trains from that frozen copy. So pass the
# seed as PRETRAIN_SEED rather than as a trailing argument -- a requeued attempt
# with trailing arguments hits the "does not accept extra args" guard below and
# dies in a second.
#
#   PRETRAIN_DATA_CONFIG=configs/data/openneuro_22373h.yaml \
#     sbatch scripts/pretrain.sbatch openneuro-native

set -euo pipefail

# Slurm runs a spooled copy of this script, so find the project from the submit
# directory: the repository root, or the current directory.
TOP="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$TOP"

RUN_NAME="${1:-}"
: "${RUN_NAME:?usage: sbatch $(basename "$0") <run_name> [extra fmrifm.train args...]}"
shift
EXTRA_ARGS=("$@")

RUN_DIR="logs/${RUN_NAME}"
RUN_CONFIG="${RUN_DIR}/config.yaml"
CHECKPOINT="${RUN_DIR}/checkpoints/last.ckpt"

RUN_HAS_CHECKPOINTS=0
if compgen -G "${RUN_DIR}/checkpoints/*" > /dev/null; then
    RUN_HAS_CHECKPOINTS=1
fi

USE_EXISTING_CONFIG=0
if (( RUN_HAS_CHECKPOINTS )) && [[ ! -f "$RUN_CONFIG" ]]; then
    echo "checkpoint artifacts exist without immutable ${RUN_CONFIG}" >&2
    echo "restore the resolved config before resuming, or use a new run name" >&2
    exit 1
elif [[ -f "$RUN_CONFIG" ]]; then
    USE_EXISTING_CONFIG=1
    if (( ${#EXTRA_ARGS[@]} > 0 )); then
        echo "restart uses immutable ${RUN_CONFIG} and does not accept extra args" >&2
        echo "start a fresh run with a new name to change its configuration" >&2
        exit 1
    fi
    if (( RUN_HAS_CHECKPOINTS )) && [[ ! -f "$CHECKPOINT" ]]; then
        echo "${RUN_NAME} has checkpoint artifacts but no last.ckpt" >&2
        exit 1
    fi
fi

mkdir -p "${RUN_DIR}/slurm" "${RUN_DIR}/checkpoints"

export PYTHONUNBUFFERED=1
# 16 loader workers share far fewer physical cores. Left unset, torch sizes its
# intra-op pool from the cgroup and every worker inherits it, so they thrash.
# Nothing in the step needs CPU threads: the model is on the GPU and the loader
# is IO-bound on object storage.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# The recipe has died allocating 94 MiB with 60 MiB free out of 79 GiB --
# fragmentation, not exhaustion.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if (( USE_EXISTING_CONFIG )); then
    RESUME_ARGS=()
    if [[ -f "$CHECKPOINT" ]]; then
        RESUME_ARGS=(--checkpoint.load_path "$CHECKPOINT" --checkpoint.save_path "$CHECKPOINT")
        echo "resuming ${RUN_NAME} from ${CHECKPOINT}"
    else
        echo "starting ${RUN_NAME} from immutable ${RUN_CONFIG} (no checkpoint)"
    fi
else
    CONFIG_ARGS=(--config "${PRETRAIN_CONFIG:-configs/model.yaml}")
    if [[ -n "${PRETRAIN_DATA_CONFIG:-}" ]]; then
        CONFIG_ARGS+=(--config "$PRETRAIN_DATA_CONFIG")
    fi
    if [[ -n "${PRETRAIN_SEED:-}" ]]; then
        CONFIG_ARGS+=(--training.seed "$PRETRAIN_SEED")
    fi
    # The letter keeps hex ids from parsing as scientific notation.
    WANDB_ID="${PRETRAIN_WANDB_ID:-$(python -c 'import secrets; print("r" + secrets.token_hex(4))')}"

    PENDING_CONFIG="$(mktemp "${RUN_CONFIG}.pending.XXXXXX")"
    resolve_status=0
    python -m fmrifm.train "${CONFIG_ARGS[@]}" \
        --checkpoint.save_path "$CHECKPOINT" \
        --wandb.name "$RUN_NAME" --wandb.id "$WANDB_ID" --wandb.save_dir "$RUN_DIR" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        --print_config > "$PENDING_CONFIG" || resolve_status=$?
    if (( resolve_status )); then
        echo "config resolution failed; output preserved at ${PENDING_CONFIG}" >&2
        exit "$resolve_status"
    fi
    # --print_config runs before dataclass validation; validate before promoting.
    if ! python -c \
        'import sys; from fmrifm.config import parse_config; parse_config(["--config", sys.argv[1]])' \
        "$PENDING_CONFIG"; then
        echo "resolved config is not valid; preserved at ${PENDING_CONFIG}" >&2
        exit 1
    fi
    mv "$PENDING_CONFIG" "$RUN_CONFIG"
    RESUME_ARGS=()
fi

srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --output="${RUN_DIR}/slurm/job-%j-rank%t.out" \
    --error="${RUN_DIR}/slurm/job-%j-rank%t.err" \
    python -m fmrifm.train --config "$RUN_CONFIG" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
