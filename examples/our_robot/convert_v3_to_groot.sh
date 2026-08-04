#!/usr/bin/env bash
#
# 将 LeRobot v3.0 数据集转换为 GR00T LeRobot v2 格式
#
# 流程:
#   1. 调用 Isaac-GR00T 自带的 convert_v3_to_v2.py 做格式转换
#   2. 拷贝 modality.json 到转换后的 meta/ 目录
#   3. 补充 annotation 列（如果 parquet 中没有）
#   4. 计算 stats / relative_stats
#
# 用法:
#   bash examples/our_robot/convert_v3_to_groot.sh \
#       --input-dir /path/to/lerobot_v3_dataset \
#       --output-dir /path/to/output \
#       [--task-description "Stand the bottle upright."]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_DIR=""
OUTPUT_DIR=""
TASK_DESCRIPTION="Stand the bottle upright."
DATASET_NAME="lift_can_orig"

usage() {
    cat <<'EOF'
Usage: bash examples/our_robot/convert_v3_to_groot.sh \
  --input-dir <LeRobot v3.0 数据集路径> \
  --output-dir <输出根目录> \
  [--dataset-name <数据集名称, 默认: lift_can_orig>] \
  [--task-description <任务描述, 默认: "Stand the bottle upright.">]
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input-dir)       INPUT_DIR="$2";        shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";       shift 2 ;;
        --dataset-name)    DATASET_NAME="$2";     shift 2 ;;
        --task-description) TASK_DESCRIPTION="$2"; shift 2 ;;
        --help|-h)         usage; exit 0 ;;
        *)                 echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

for required_var in INPUT_DIR OUTPUT_DIR; do
    if [ -z "${!required_var}" ]; then
        echo "[ERROR] Missing required argument: ${required_var}" >&2
        usage >&2
        exit 1
    fi
done

CONVERTED_DIR="${OUTPUT_DIR}/${DATASET_NAME}"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  LeRobot v3.0 → GR00T LeRobot v2 转换"
echo "══════════════════════════════════════════════════════"
echo "  输入目录:  ${INPUT_DIR}"
echo "  输出目录:  ${CONVERTED_DIR}"
echo "  任务描述:  ${TASK_DESCRIPTION}"
echo "══════════════════════════════════════════════════════"
echo ""

# ════════════════════════════════════════════════════════
# Step 1: v3.0 → v2.1 格式转换
# ════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════"
echo "  Step 1/4: LeRobot v3.0 → v2.1 格式转换"
echo "═══════════════════════════════════════════════"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] 输入目录不存在: ${INPUT_DIR}"
    exit 1
fi

cd "${REPO_DIR}"

# 将 v3 数据集拷贝到 output-dir 下再就地转换
mkdir -p "${OUTPUT_DIR}"
if [ -d "${CONVERTED_DIR}" ]; then
    echo "[WARN] 输出目录已存在，跳过拷贝: ${CONVERTED_DIR}"
else
    echo "[INFO] 拷贝数据集到 ${CONVERTED_DIR} ..."
    cp -r "${INPUT_DIR}" "${CONVERTED_DIR}"
fi

echo "[INFO] 运行 convert_v3_to_v2.py ..."
uv run --project scripts/lerobot_conversion \
    python scripts/lerobot_conversion/convert_v3_to_v2.py \
    --repo-id "${DATASET_NAME}" \
    --root "${OUTPUT_DIR}"

echo "[INFO] Step 1 完成"

# ════════════════════════════════════════════════════════
# Step 2: 拷贝 modality.json
# ════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════"
echo "  Step 2/4: 拷贝 modality.json"
echo "═══════════════════════════════════════════════"

cp "${SCRIPT_DIR}/modality.json" "${CONVERTED_DIR}/meta/modality.json"
echo "[INFO] modality.json 已拷贝到 ${CONVERTED_DIR}/meta/"

# ════════════════════════════════════════════════════════
# Step 3: 补充 annotation 列和 tasks.jsonl
# ════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════"
echo "  Step 3/4: 补充语言标注"
echo "═══════════════════════════════════════════════"

python3 - <<PYEOF
"""为转换后的 GR00T LeRobot v2 数据集补充 annotation 列."""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

dataset_dir = Path("${CONVERTED_DIR}")
task_desc = "${TASK_DESCRIPTION}"

# --- 确保 tasks.jsonl 存在且包含任务描述 ---
tasks_jsonl = dataset_dir / "meta" / "tasks.jsonl"
if tasks_jsonl.exists():
    with open(tasks_jsonl) as f:
        lines = f.readlines()
    tasks = [json.loads(line) for line in lines if line.strip()]
    task_descs = [t["task"] for t in tasks]
    if task_desc not in task_descs:
        new_idx = max(t["task_index"] for t in tasks) + 1
        with open(tasks_jsonl, "a") as f:
            f.write(json.dumps({"task_index": new_idx, "task": task_desc}) + "\n")
        task_idx = new_idx
        print(f"[INFO] 追加任务: task_index={new_idx}, task={task_desc!r}")
    else:
        task_idx = tasks[task_descs.index(task_desc)]["task_index"]
        print(f"[INFO] 任务已存在: task_index={task_idx}")
else:
    tasks_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(tasks_jsonl, "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_desc}) + "\n")
    task_idx = 0
    print(f"[INFO] 创建 tasks.jsonl: task_index=0, task={task_desc!r}")

# --- 为每个 episode parquet 添加 annotation 列 ---
data_dir = dataset_dir / "data"
parquet_files = sorted(data_dir.rglob("*.parquet"))
print(f"[INFO] 处理 {len(parquet_files)} 个 parquet 文件 ...")

modified = 0
for pf in parquet_files:
    table = pq.read_table(str(pf))
    col_name = "annotation.human.task_description"
    if col_name not in table.column_names:
        n = table.num_rows
        anno_col = pa.array([task_idx] * n, type=pa.int64())
        table = table.append_column(col_name, anno_col)
        pq.write_table(table, str(pf))
        modified += 1

print(f"[INFO] 添加 annotation 列: {modified} 个文件已更新")

# --- 更新 info.json 添加 annotation feature ---
info_path = dataset_dir / "meta" / "info.json"
if info_path.exists():
    with open(info_path) as f:
        info = json.load(f)
    if "annotation.human.task_description" not in info.get("features", {}):
        info["features"]["annotation.human.task_description"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print("[INFO] info.json 已更新: 添加 annotation feature")
PYEOF

echo "[INFO] Step 3 完成"

# ════════════════════════════════════════════════════════
# Step 4: 计算统计量
# ════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════"
echo "  Step 4/4: 计算 stats & relative_stats"
echo "═══════════════════════════════════════════════"

cd "${REPO_DIR}"
uv run --no-sync python gr00t/data/stats.py \
    --dataset-path "${CONVERTED_DIR}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${SCRIPT_DIR}/our_robot_config.py"

echo "[INFO] Step 4 完成"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  全部转换完成！"
echo "  GR00T 数据集路径: ${CONVERTED_DIR}"
echo "══════════════════════════════════════════════════════"
