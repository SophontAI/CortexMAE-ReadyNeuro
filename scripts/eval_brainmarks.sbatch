#!/bin/bash
#SBATCH --job-name=brainmarks-eval
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=20G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-13
# set your own account/partition/qos

# Seven benchmark targets x two coordinate spaces = 14 array tasks, one frozen
# encoder each. Tasks 0-3 are the two attention-probe targets; 4-13 are the five
# phenotype targets, which read out with logistic regression.
#
# Usage: sbatch scripts/eval_brainmarks.sbatch LOG_DIR [MODEL_NAME] [OUTPUT_ROOT] \
#            [CHECKPOINT_PATH] [CONFIG_PATH]

set -euo pipefail

# Slurm runs a spooled copy of this script, so find the project from the submit
# directory: the repository root, or the current directory.
TOP="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$TOP"

USAGE="sbatch scripts/eval_brainmarks.sbatch LOG_DIR [MODEL_NAME] [OUTPUT_ROOT]"
USAGE+=" [CHECKPOINT_PATH] [CONFIG_PATH]"
LOG_DIR="${1:?Usage: $USAGE}"
LOG_DIR="${LOG_DIR%/}"
MODEL_NAME="${2:-$(basename "$LOG_DIR")}"
OUTPUT_ROOT="${3:-output/$MODEL_NAME}"
CHECKPOINT_PATH="${4:-$LOG_DIR/checkpoints/last.ckpt}"
CONFIG_PATH="${5:-}"

DATASETS=(
    hcpya_task21
    nsd_cococlip
    aabc_age
    aabc_sex
    ppmi_dx
    abide_dx
    adhd200_dx
)
DRIVERS=(
    probe
    probe
    logistic
    logistic
    logistic
    logistic
    logistic
)
SPACES=(native mni)

NUM_SPACES=${#SPACES[@]}
NUM_TASKS=$(( ${#DATASETS[@]} * NUM_SPACES ))
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is unset; submit this script with sbatch}"
if (( TASK_ID < 0 || TASK_ID >= NUM_TASKS )); then
    echo "array task $TASK_ID is outside [0, $((NUM_TASKS - 1))]" >&2
    exit 1
fi

dataset="${DATASETS[$(( TASK_ID / NUM_SPACES ))]}"
driver="${DRIVERS[$(( TASK_ID / NUM_SPACES ))]}"
space="${SPACES[$(( TASK_ID % NUM_SPACES ))]}"
model="fmrifm_${space}"

# #SBATCH cannot interpolate a positional OUTPUT_ROOT, so redirect once the task
# is resolved rather than through --output.
SLURM_LOG_DIR="${OUTPUT_ROOT}/slurm"
mkdir -p "$SLURM_LOG_DIR" "$OUTPUT_ROOT"
printf -v task_label '%02d' "$TASK_ID"
TASK_LOG="${SLURM_LOG_DIR}/task-${task_label}-${dataset}-${space}-job-${SLURM_ARRAY_JOB_ID:-manual}.log"
echo "Brainmarks task log: $TASK_LOG"
exec >"$TASK_LOG" 2>&1

[[ -f "$CHECKPOINT_PATH" ]] || {
    echo "checkpoint does not exist: $CHECKPOINT_PATH" >&2
    exit 1
}
if [[ -n "$CONFIG_PATH" && ! -f "$CONFIG_PATH" ]]; then
    echo "config does not exist: $CONFIG_PATH" >&2
    exit 1
fi

# The evaluation stores are read over S3; set AWS_ACCESS_KEY_ID,
# AWS_SECRET_ACCESS_KEY and AWS_ENDPOINT_URL for your own endpoint first.
EVAL_ROOT="${EVAL_ROOT:-s3://<RESTRICTED_ROOT>}"
export BRAINMARKS_DATA_ROOT="$EVAL_ROOT"
export BRAINMARKS_PRIVATE_DATA_ROOT="$EVAL_ROOT"
export HCPYA_FMRI_EVAL_ROOT="${HCPYA_FMRI_EVAL_ROOT:-${EVAL_ROOT}/hcp-ya-task21-eval}"
export AABC_FMRI_EVAL_ROOT="${AABC_FMRI_EVAL_ROOT:-${EVAL_ROOT}/aabc-eval}"
export PPMI_FMRI_EVAL_ROOT="${PPMI_FMRI_EVAL_ROOT:-${EVAL_ROOT}/ppmi-eval}"
export ABIDE_FMRI_EVAL_ROOT="${ABIDE_FMRI_EVAL_ROOT:-${EVAL_ROOT}/abide-eval}"
export ADHD200_FMRI_EVAL_ROOT="${ADHD200_FMRI_EVAL_ROOT:-${EVAL_ROOT}/adhd200-eval}"
# Every other benchmark holds both spaces in one store; NSD is the exception, so
# its root carries the space rather than being selected inside the store.
export NSD_FMRI_EVAL_ROOT="${NSD_FMRI_EVAL_ROOT:-${EVAL_ROOT}/nsd-cococlip-eval-${space}}"

# Shards are read straight into RAM per task; this bounds the parallel S3 reads.
export FMRI_ARROW_READ_WORKERS="${FMRI_ARROW_READ_WORKERS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_OVERRIDES=(
    "model_kwargs.ckpt_path=$CHECKPOINT_PATH"
    "model_kwargs.compile_model=true"
)
if [[ -n "$CONFIG_PATH" ]]; then
    MODEL_OVERRIDES+=("model_kwargs.cfg_path=$CONFIG_PATH")
fi

echo "brainmarks eval: task=$TASK_ID driver=$driver model=$model dataset=${dataset}_readyneuro"
echo "output_root=$OUTPUT_ROOT checkpoint=$CHECKPOINT_PATH"

# PPMI, ABIDE and ADHD-200 join their labels from the target maps in the pinned
# Brainmarks commit's datasets/ tree, which the loaders fetch on their own. AABC's two per-subject
# maps are participant-level phenotype and are NOT in the public suite -- point
# AABC_TARGETS_ROOT at a directory holding aabc_target_map_{age_open,sex}.json.
DATASET_OVERRIDES=()
if [[ "$dataset" == aabc_* ]]; then
    : "${AABC_TARGETS_ROOT:?aabc_age/aabc_sex need AABC_TARGETS_ROOT=<dir or s3:// prefix holding aabc_target_map_age_open.json and aabc_target_map_sex.json>}"
    DATASET_OVERRIDES+=("dataset_kwargs.targets_root=$AABC_TARGETS_ROOT")
fi

if [[ "$driver" == "probe" ]]; then
    python -m brainmarks.main_probe \
        "$model" patch attn "${dataset}_readyneuro" \
        --overrides \
        "output_root=$OUTPUT_ROOT" \
        "${MODEL_OVERRIDES[@]}" \
        "batch_size=32" \
        "accum_iter=4" \
        ${DATASET_OVERRIDES[@]+"${DATASET_OVERRIDES[@]}"}
else
    python -m brainmarks.main_logistic \
        "$model" patch "${dataset}_readyneuro" \
        --overrides \
        "output_root=$OUTPUT_ROOT" \
        "${MODEL_OVERRIDES[@]}" \
        "batch_size=3" \
        ${DATASET_OVERRIDES[@]+"${DATASET_OVERRIDES[@]}"}
fi
