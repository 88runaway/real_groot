#!/usr/bin/env bash
# 从 train_config.yaml 加载参数并启动 GR00T 微调。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/train_config.yaml"
DRY_RUN=false
OVERRIDE_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash examples/our_robot/train.sh [options]
  --config <yaml>           指定配置文件（默认: train_config.yaml）
  --dataset-path <path>     覆盖 YAML 数据集路径
  --base-model-path <path>  覆盖 YAML 基础模型路径
  --output-dir <path>       覆盖 YAML 输出目录
  --max-steps <int>         覆盖 YAML 训练步数
  --num-gpus <int>          覆盖 YAML GPU 数量
  --dry-run                 仅打印配置和命令
EOF
}

# 先提取配置文件，其余参数稍后覆盖 YAML。
while [ "$#" -gt 0 ]; do
    case "$1" in
        --config) CONFIG_FILE="$2"; shift 2 ;;
        *) OVERRIDE_ARGS+=("$1"); shift ;;
    esac
done

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "[ERROR] 配置文件不存在: ${CONFIG_FILE}" >&2
    exit 1
fi

yaml_get() {
    python3 - "${CONFIG_FILE}" "$1" "$2" <<'PY'
import sys
import yaml

path, dotted_key, default = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    value = yaml.safe_load(f)
try:
    for key in dotted_key.split("."):
        value = value[key]
except (KeyError, TypeError):
    value = default
if isinstance(value, bool):
    print(str(value).lower())
else:
    print(value)
PY
}

# YAML 值。
BASE_MODEL_PATH="$(yaml_get paths.base_model_path "")"
BACKBONE_MODEL_NAME="$(yaml_get paths.backbone_model_name "nvidia/Cosmos-Reason2-2B")"
DATASET_PATH="$(yaml_get paths.dataset_path "")"
OUTPUT_DIR="$(yaml_get paths.output_dir "/tmp/our_robot_finetune")"
EXPERIMENT_NAME="$(yaml_get task.experiment_name "our_robot_right_arm")"
CHUNK_SIZE="$(yaml_get action.chunk_size "16")"

MAX_STEPS="$(yaml_get training.max_steps "10000")"
SAVE_STEPS="$(yaml_get training.save_steps "1000")"
SAVE_TOTAL_LIMIT="$(yaml_get training.save_total_limit "5")"
LEARNING_RATE="$(yaml_get training.learning_rate "1e-4")"
WEIGHT_DECAY="$(yaml_get training.weight_decay "1e-5")"
WARMUP_RATIO="$(yaml_get training.warmup_ratio "0.05")"
GRADIENT_ACCUMULATION_STEPS="$(yaml_get training.gradient_accumulation_steps "1")"
GLOBAL_BATCH_SIZE="$(yaml_get training.global_batch_size "32")"
DATALOADER_NUM_WORKERS="$(yaml_get training.dataloader_num_workers "8")"
SHARD_SIZE="$(yaml_get training.shard_size "1024")"
NUM_SHARDS_PER_EPOCH="$(yaml_get training.num_shards_per_epoch "100000")"
EPISODE_SAMPLING_RATE="$(yaml_get training.episode_sampling_rate "0.1")"

TUNE_LLM="$(yaml_get model.tune_llm "false")"
TUNE_VISUAL="$(yaml_get model.tune_visual "false")"
TUNE_PROJECTOR="$(yaml_get model.tune_projector "true")"
TUNE_DIFFUSION_MODEL="$(yaml_get model.tune_diffusion_model "true")"
STATE_DROPOUT_PROB="$(yaml_get model.state_dropout_prob "0.2")"

BRIGHTNESS="$(yaml_get augmentation.color_jitter.brightness "0.3")"
CONTRAST="$(yaml_get augmentation.color_jitter.contrast "0.4")"
SATURATION="$(yaml_get augmentation.color_jitter.saturation "0.5")"
HUE="$(yaml_get augmentation.color_jitter.hue "0.08")"
USE_PERCENTILES="$(yaml_get augmentation.use_percentiles "true")"

NUM_GPUS="$(yaml_get hardware.num_gpus "1")"
USE_WANDB="$(yaml_get logging.use_wandb "false")"
WANDB_PROJECT="$(yaml_get logging.wandb_project "finetune-gr00t-n1d7")"

# 命令行覆盖 YAML。
set -- "${OVERRIDE_ARGS[@]}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --base-model-path) BASE_MODEL_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "[ERROR] 未知参数: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [ -z "${BASE_MODEL_PATH}" ] || [ -z "${DATASET_PATH}" ]; then
    echo "[ERROR] YAML 中必须设置 paths.base_model_path 和 paths.dataset_path" >&2
    exit 1
fi
if [ ! -d "${BASE_MODEL_PATH}" ]; then
    echo "[ERROR] 基础模型目录不存在: ${BASE_MODEL_PATH}" >&2
    exit 1
fi
if [ ! -d "${DATASET_PATH}" ]; then
    echo "[ERROR] 数据集目录不存在: ${DATASET_PATH}" >&2
    exit 1
fi

# chunk_size 由模态配置决定；启动前与 YAML 同步。
python3 - "${SCRIPT_DIR}/our_robot_config.py" "${CHUNK_SIZE}" <<'PY'
import re
import sys

path, chunk_size = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    content = f.read()
updated, count = re.subn(
    r"delta_indices=list\(range\(0,\s*\d+\)\)",
    f"delta_indices=list(range(0, {int(chunk_size)}))",
    content,
    count=1,
)
if count != 1:
    raise RuntimeError("未找到 action.delta_indices 配置")
with open(path, "w", encoding="utf-8") as f:
    f.write(updated)
PY

MODALITY_CONFIG="${SCRIPT_DIR}/our_robot_config.py"
COLOR_JITTER_PARAMS="brightness ${BRIGHTNESS} contrast ${CONTRAST} saturation ${SATURATION} hue ${HUE}"

bool_flag() {
    if [ "$2" = "true" ]; then
        printf -- "--%s" "$1"
    else
        printf -- "--no-%s" "$1"
    fi
}

MODEL_FLAGS=(
    "$(bool_flag tune-llm "${TUNE_LLM}")"
    "$(bool_flag tune-visual "${TUNE_VISUAL}")"
    "$(bool_flag tune-projector "${TUNE_PROJECTOR}")"
    "$(bool_flag tune-diffusion-model "${TUNE_DIFFUSION_MODEL}")"
)

LAUNCH_CMD=(
    bash examples/finetune.sh
    --base-model-path "${BASE_MODEL_PATH}"
    --backbone-model-name "${BACKBONE_MODEL_NAME}"
    --dataset-path "${DATASET_PATH}"
    --modality-config-path "${MODALITY_CONFIG}"
    --embodiment-tag NEW_EMBODIMENT
    --output-dir "${OUTPUT_DIR}"
    --experiment-name "${EXPERIMENT_NAME}"
    --wandb-project "${WANDB_PROJECT}"
    --state-dropout-prob "${STATE_DROPOUT_PROB}"
    --color-jitter-params "${COLOR_JITTER_PARAMS}"
    --use-percentiles "${USE_PERCENTILES}"
    --
    --learning-rate "${LEARNING_RATE}"
    --weight-decay "${WEIGHT_DECAY}"
    --warmup-ratio "${WARMUP_RATIO}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
    --save-total-limit "${SAVE_TOTAL_LIMIT}"
    "${MODEL_FLAGS[@]}"
)

export NUM_GPUS MAX_STEPS SAVE_STEPS GLOBAL_BATCH_SIZE
export DATALOADER_NUM_WORKERS SHARD_SIZE NUM_SHARDS_PER_EPOCH EPISODE_SAMPLING_RATE
if [ "${USE_WANDB}" = "true" ]; then
    export USE_WANDB=1
else
    export USE_WANDB=0
fi

echo "[INFO] 配置文件: ${CONFIG_FILE}"
echo "[INFO] 模型: ${BASE_MODEL_PATH}"
echo "[INFO] 数据集: ${DATASET_PATH}"
echo "[INFO] 输出: ${OUTPUT_DIR}"
echo "[INFO] steps=${MAX_STEPS}, lr=${LEARNING_RATE}, batch=${GLOBAL_BATCH_SIZE}, chunk=${CHUNK_SIZE}"
echo "[INFO] GPUs=${NUM_GPUS}, W&B=${USE_WANDB}"

cd "${REPO_DIR}"
if [ "${DRY_RUN}" = true ]; then
    printf "[DRY-RUN] "
    printf "%q " "${LAUNCH_CMD[@]}"
    printf "\n"
    exit 0
fi

exec "${LAUNCH_CMD[@]}"
