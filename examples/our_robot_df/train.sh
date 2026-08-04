#!/usr/bin/env bash
# GR00T-DF: Block-wise Diffusion Forcing 微调启动脚本
# 从 train_config.yaml 加载参数并启动带 DF 的 GR00T 微调。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/train_config.yaml"
DRY_RUN=false
OVERRIDE_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash examples/our_robot_df/train.sh [options]
  --config <yaml>           指定配置文件（默认: train_config.yaml）
  --dataset-path <path>     覆盖 YAML 数据集路径
  --base-model-path <path>  覆盖 YAML 基础模型路径
  --output-dir <path>       覆盖 YAML 输出目录
  --max-steps <int>         覆盖 YAML 训练步数
  --num-gpus <int>          覆盖 YAML GPU 数量
  --df-block-size <int>     覆盖 DF block size
  --df-mix-prob <float>     覆盖 DF mix probability
  --dry-run                 仅打印配置和命令
EOF
}

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
elif isinstance(value, list):
    print(",".join(str(v) for v in value))
else:
    print(value)
PY
}

# YAML 值
BASE_MODEL_PATH="$(yaml_get paths.base_model_path "")"
BACKBONE_MODEL_NAME="$(yaml_get paths.backbone_model_name "nvidia/Cosmos-Reason2-2B")"
DATASET_PATH="$(yaml_get paths.dataset_path "")"
OUTPUT_DIR="$(yaml_get paths.output_dir "/tmp/our_robot_df_finetune")"
EXPERIMENT_NAME="$(yaml_get task.experiment_name "our_robot_df")"
CHUNK_SIZE="$(yaml_get action.chunk_size "40")"

MAX_STEPS="$(yaml_get training.max_steps "15000")"
SAVE_STEPS="$(yaml_get training.save_steps "1000")"
SAVE_TOTAL_LIMIT="$(yaml_get training.save_total_limit "10")"
LEARNING_RATE="$(yaml_get training.learning_rate "6e-5")"
WEIGHT_DECAY="$(yaml_get training.weight_decay "1e-5")"
WARMUP_RATIO="$(yaml_get training.warmup_ratio "0.1")"
GRADIENT_ACCUMULATION_STEPS="$(yaml_get training.gradient_accumulation_steps "1")"
GLOBAL_BATCH_SIZE="$(yaml_get training.global_batch_size "256")"
DATALOADER_NUM_WORKERS="$(yaml_get training.dataloader_num_workers "16")"
SHARD_SIZE="$(yaml_get training.shard_size "1024")"
NUM_SHARDS_PER_EPOCH="$(yaml_get training.num_shards_per_epoch "100000")"
EPISODE_SAMPLING_RATE="$(yaml_get training.episode_sampling_rate "0.1")"

TUNE_LLM="$(yaml_get model.tune_llm "true")"
TUNE_VISUAL="$(yaml_get model.tune_visual "false")"
TUNE_PROJECTOR="$(yaml_get model.tune_projector "true")"
TUNE_DIFFUSION_MODEL="$(yaml_get model.tune_diffusion_model "true")"
STATE_DROPOUT_PROB="$(yaml_get model.state_dropout_prob "0.25")"

# Diffusion Forcing 参数
DF_ENABLED="$(yaml_get diffusion_forcing.enabled "true")"
DF_BLOCK_SIZE="$(yaml_get diffusion_forcing.block_size "5")"
DF_MIX_PROB="$(yaml_get diffusion_forcing.mix_prob "1.0")"
DF_BLOCK_TIME_SAMPLING="$(yaml_get diffusion_forcing.block_time_sampling "monotone")"
DF_REWEIGHT_GAMMA="$(yaml_get diffusion_forcing.reweight_gamma "0.5")"
DF_PHASE_ALPHA="$(yaml_get diffusion_forcing.phase_alpha "1.0")"

# Tactile 参数
TAC_ENABLED="$(yaml_get tactile.enabled "false")"
TAC_ENCODER_PATH="$(yaml_get tactile.encoder_path "")"
TAC_SENSOR_NAME="$(yaml_get tactile.sensor_name "GelSightMini")"
TAC_ENCODER_OUTPUT_DIM="$(yaml_get tactile.encoder_output_dim "1536")"
TAC_FREEZE_BACKBONE="$(yaml_get tactile.freeze_backbone "true")"
TAC_NUM_TOKENS="$(yaml_get tactile.num_tokens "2")"
TAC_FUNC_AREA_INDICES="$(yaml_get tactile.func_area_indices "")"

BRIGHTNESS="$(yaml_get augmentation.color_jitter.brightness "0.3")"
CONTRAST="$(yaml_get augmentation.color_jitter.contrast "0.4")"
SATURATION="$(yaml_get augmentation.color_jitter.saturation "0.5")"
HUE="$(yaml_get augmentation.color_jitter.hue "0.08")"
USE_PERCENTILES="$(yaml_get augmentation.use_percentiles "true")"

NUM_GPUS="$(yaml_get hardware.num_gpus "8")"
USE_WANDB="$(yaml_get logging.use_wandb "false")"
WANDB_PROJECT="$(yaml_get logging.wandb_project "finetune-gr00t-n1d7-df")"

# 命令行覆盖
set -- "${OVERRIDE_ARGS[@]}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --base-model-path) BASE_MODEL_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --df-block-size) DF_BLOCK_SIZE="$2"; shift 2 ;;
        --df-mix-prob) DF_MIX_PROB="$2"; shift 2 ;;
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

# 同步 chunk_size 到模态配置
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

# DF flags
DF_FLAGS=()
if [ "${DF_ENABLED}" = "true" ]; then
    DF_FLAGS+=(--df-enabled)
else
    DF_FLAGS+=(--no-df-enabled)
fi
DF_FLAGS+=(
    --df-block-size "${DF_BLOCK_SIZE}"
    --df-mix-prob "${DF_MIX_PROB}"
    --df-block-time-sampling "${DF_BLOCK_TIME_SAMPLING}"
    --df-reweight-gamma "${DF_REWEIGHT_GAMMA}"
    --df-phase-alpha "${DF_PHASE_ALPHA}"
)

# Tactile flags
TAC_FLAGS=()
if [ "${TAC_ENABLED}" = "true" ]; then
    TAC_FLAGS+=(--tactile-enabled)
else
    TAC_FLAGS+=(--no-tactile-enabled)
fi
if [ -n "${TAC_ENCODER_PATH}" ]; then
    TAC_FLAGS+=(--tactile-encoder-path "${TAC_ENCODER_PATH}")
fi
TAC_FLAGS+=(
    --tactile-sensor-name "${TAC_SENSOR_NAME}"
    --tactile-encoder-output-dim "${TAC_ENCODER_OUTPUT_DIM}"
    --tactile-num-tokens "${TAC_NUM_TOKENS}"
)
if [ -n "${TAC_FUNC_AREA_INDICES}" ]; then
    TAC_FLAGS+=(--tactile-func-area-indices "${TAC_FUNC_AREA_INDICES}")
fi
if [ "${TAC_FREEZE_BACKBONE}" = "true" ]; then
    TAC_FLAGS+=(--tactile-freeze-backbone)
else
    TAC_FLAGS+=(--no-tactile-freeze-backbone)
fi

LAUNCH_CMD=(
    torchrun --nproc_per_node="${NUM_GPUS}" --standalone
    "${SCRIPT_DIR}/launch_finetune_df.py"
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
    --learning-rate "${LEARNING_RATE}"
    --weight-decay "${WEIGHT_DECAY}"
    --warmup-ratio "${WARMUP_RATIO}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
    --save-steps "${SAVE_STEPS}"
    --save-total-limit "${SAVE_TOTAL_LIMIT}"
    --max-steps "${MAX_STEPS}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
    --shard-size "${SHARD_SIZE}"
    --num-shards-per-epoch "${NUM_SHARDS_PER_EPOCH}"
    --episode-sampling-rate "${EPISODE_SAMPLING_RATE}"
    "${MODEL_FLAGS[@]}"
    "${DF_FLAGS[@]}"
    "${TAC_FLAGS[@]}"
)

export NUM_GPUS MAX_STEPS SAVE_STEPS GLOBAL_BATCH_SIZE
export DATALOADER_NUM_WORKERS SHARD_SIZE NUM_SHARDS_PER_EPOCH EPISODE_SAMPLING_RATE
if [ "${USE_WANDB}" = "true" ]; then
    export USE_WANDB=1
else
    export USE_WANDB=0
fi

echo "═══════════════════════════════════════════════════════════"
echo "  GR00T-DF: Block-wise Diffusion Forcing Finetuning"
echo "═══════════════════════════════════════════════════════════"
echo "[INFO] 配置文件: ${CONFIG_FILE}"
echo "[INFO] 模型: ${BASE_MODEL_PATH}"
echo "[INFO] 数据集: ${DATASET_PATH}"
echo "[INFO] 输出: ${OUTPUT_DIR}"
echo "[INFO] steps=${MAX_STEPS}, lr=${LEARNING_RATE}, batch=${GLOBAL_BATCH_SIZE}, chunk=${CHUNK_SIZE}"
echo "[INFO] GPUs=${NUM_GPUS}, W&B=${USE_WANDB}"
echo "[DF]   enabled=${DF_ENABLED}, block_size=${DF_BLOCK_SIZE}, mix_prob=${DF_MIX_PROB}"
echo "[DF]   sampling=${DF_BLOCK_TIME_SAMPLING}, gamma=${DF_REWEIGHT_GAMMA}, phase_alpha=${DF_PHASE_ALPHA}"
echo "[TAC]  enabled=${TAC_ENABLED}, encoder=${TAC_ENCODER_PATH:-none}, sensor=${TAC_SENSOR_NAME}, freeze=${TAC_FREEZE_BACKBONE}"
echo "[TAC]  func_areas=${TAC_FUNC_AREA_INDICES:-auto}"
echo "═══════════════════════════════════════════════════════════"

cd "${REPO_DIR}"
if [ "${DRY_RUN}" = true ]; then
    printf "[DRY-RUN] "
    printf "%q " "${LAUNCH_CMD[@]}"
    printf "\n"
    exit 0
fi

exec "${LAUNCH_CMD[@]}"
