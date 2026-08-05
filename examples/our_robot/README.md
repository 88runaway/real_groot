cd /mnt/netdata/Team/Personal/chenyiyang/zjb/Isaac-GR00T
uv sync --python 3.12

# 双臂 RealMan + Sharpa 灵巧手 GR00T 微调

在自采集的 LeRobot v3.0 数据上微调 GR00T N1.7 模型，仅训练右臂 + 右手。

## 机器人规格

| 项目 | 规格 |
|------|------|
| 机器人 | double_realman_follower（双臂 RealMan） |
| 灵巧手 | Sharpa 22-dof × 2 |
| 关节维度 | 58 维（左臂7 + 左手22 + 右臂7 + 右手22） |
| 训练部分 | 仅右臂(7) + 右手(22) = 29 维 |
| 相机 | ego (left_top 480×640) + right_wrist (480×640) |
| 帧率 | 30 FPS |

## 文件说明

| 文件 | 作用 |
|------|------|
| `modality.json` | 数据模态映射：定义 state/action 切片、视频 key 映射、annotation 配置 |
| `our_robot_config.py` | GR00T 训练模态配置：指定输入/输出模态、时序采样、动作表示 |
| `convert_v3_to_groot.sh` | 数据转换脚本：LeRobot v3.0 → GR00T LeRobot v2 全流程 |
| `train.sh` | 训练启动脚本：封装 `examples/finetune.sh` |

## 快速开始

### 1. 数据转换

将 LeRobot v3.0 格式的自采集数据转换为 GR00T 可用的 v2 格式：

```bash
cd /mnt/netdata/Team/Personal/chenyiyang/zjb/Isaac-GR00T

bash examples/our_robot/convert_v3_to_groot.sh \
    --input-dir /mnt/netdata/Team/Personal/chenyiyang/zjb/data/real/chemistry_experiment \
    --output-dir /mnt/netdata/Team/Personal/chenyiyang/zjb/data/groot/chemistry_experiment \
    --dataset-name chemistry_experiment \
    --task-description "Pick up the dropper, draw solution from the Erlenmeyer flask, and dispense it into the beaker."
```

### 2. 启动训练

训练参数统一在 `train_config.yaml` 中管理（路径、lr、steps、chunk_size 等），直接运行即可：

```bash
bash examples/our_robot/train.sh
```

预览将执行的命令（不实际训练）：

```bash
bash examples/our_robot/train.sh --dry-run
```

临时覆盖某个参数（不改 yaml）：

```bash
# 示例：只跑 2000 步做快速验证
bash examples/our_robot/train.sh --max-steps 2000 --dry-run
```

### 3. 开环评估

```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path /mnt/netdata/Team/Personal/chenyiyang/zjb/data/groot/lift_can_orig \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path /mnt/netdata/Team/Personal/chenyiyang/zjb/ckpts/groot_our_robot/checkpoint-10000 \
    --traj-ids 0 \
    --execution-horizon 16 \
    --steps 400 \
    --modality-keys right_arm right_hand
```

## 设计决策

### 为什么只训练右臂？

采集的 `lift_can_orig` 数据仅使用右臂操作，左臂始终静止。
`modality.json` 中 state/action 只映射了 `[29:36]`（right_arm）和 `[36:58]`（right_hand），
GR00T 只会对这些维度计算损失和预测动作。

### 动作表示

- **right_arm**: `RELATIVE`（相对动作）— 预测与当前关节角度的差值，泛化性更好
- **right_hand**: `ABSOLUTE`（绝对动作）— 灵巧手直接预测目标位置，更适合抓取等离散动作

### 关于触觉数据

当前配置未使用触觉数据（视觉触觉图像和 F/T 传感器）。
如需加入触觉，可以：
1. 将触觉视频作为额外的 video modality 添加到 `modality.json` 和 config 中
2. 将 `observation.tactile_f6` 的 30 维（右手 5 指 × 6 轴）拼接到 state 中
