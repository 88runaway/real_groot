# GR00T-DF: Block-wise Diffusion Forcing + FTP 触觉注入

基于 `our_robot` 标准配置，启用 **Block-wise Diffusion Forcing** + **FTP 触觉编码器**。

## 架构概览

```
                    ┌──────────────────────────────────┐
 Language ──────────┤                                  │
                    │  Qwen3-VL Backbone (cross-attn)  │
 Visual (ego+wrist)─┤                                  │
                    └──────────┬───────────────────────┘
                               │ backbone_features
                    ┌──────────▼───────────────────────┐
                    │       DiT Action Head            │
 State ──────┐      │  ┌─────────────────────────┐    │
             ├─────→│  │ [state | tactile | action]│   │
 Tactile ────┘      │  │   1   +   2    +   40    │   │
 (FTP encoder)      │  └─────────────────────────┘    │
                    │  AdaLN: t=0  t=0   t=per_block  │
                    └──────────────────────────────────┘
```

**改动要点**:
- State 使用 timestep=0 (clean observation，不随 block noise 变化)
- Tactile tokens 使用 timestep=0 (clean conditioning signal)
- Action 每个 block 有独立 timestep（Diffusion Forcing）

## 核心改进

### 1. Block-wise Diffusion Forcing

将 action horizon (40 步) 切分为 8 个 block (每 block 5 步)：
- **Monotone 调度**：前 block 更干净 → 近期动作更精确
- **金字塔推理**：Block 0 最先收敛 → 低延迟响应
- **损失重加权**：补偿前 block 训练覆盖不足

### 2. FTP 触觉编码器

- 预训练 ViT-based 触觉 encoder（T3 + Trunk 架构）
- 每个手指的 deformation image → 1 个 token (1536D)
- Freeze backbone，仅训练 output_proj + DiT 注意力

## 文件说明

| 文件 | 作用 |
|------|------|
| `our_robot_config.py` | 模态配置（video/state/action/语言/触觉） |
| `modality.json` | 数据模态映射（state/action 切片、video key 映射） |
| `train_config.yaml` | 统一训练配置（路径、DF 参数、触觉参数、超参） |
| `launch_finetune_df.py` | 自定义启动脚本（注入 DF/Tactile 参数） |
| `train.sh` | 一键训练启动脚本 |
| `preprocess_tactile.py` | 触觉视频预处理：拼接帧 → 单指视频 |

## 完整流程

### Step 0: 环境安装

```bash
cd /mnt/netdata/Team/Personal/chenyiyang/zjb/Isaac-GR00T
uv sync --python 3.12
```

### Step 1: 数据转换 (LeRobot v3.0 → GR00T v2)

```bash
bash examples/our_robot/convert_v3_to_groot.sh \
    --input-dir /mnt/netdata/Team/Personal/chenyiyang/zjb/data/real/chemistry_experiment \
    --output-dir /mnt/netdata/Team/Personal/chenyiyang/zjb/data/groot_tactile \
    --dataset-name chemistry_experiment \
    --task-description "Pick up the dropper, draw solution from the Erlenmeyer flask, and dispense it into the beaker."
```

此脚本自动执行 5 步：
1. **v3 → v2 格式转换**（拷贝 + convert_v3_to_v2.py）
2. **拷贝 modality.json** 到 meta/
3. **补充 annotation 列**（语言标注 task_description）
4. **触觉视频预处理**（拆分拼接 1200×240 → 单指 224×224）
5. **计算 stats**（state/action 的 min/max/mean/std）

### Step 1b: 单独运行触觉预处理（可选）

如果数据集已经转换但还没处理触觉：

```bash
# 预处理所有手指（双手 × 5指），训练时再选择使用哪些
python3 examples/our_robot_df/preprocess_tactile.py \
    --dataset-path /path/to/groot_dataset \
    --num-fingers 5 \
    --finger-indices 0,1,2,3,4 \
    --target-size 224 \
    --source-keys "observation.images.tactile_deform_right,observation.images.tactile_deform_left"
```

**输入**: `tactile_deform_right` + `tactile_deform_left` (各 1200×240, 5 指拼接)

**输出** (共 10 个视频 key):
- `observation.images.tactile_finger_right_0` ~ `tactile_finger_right_4` (右手 5 指)
- `observation.images.tactile_finger_left_0` ~ `tactile_finger_left_4` (左手 5 指)

训练时通过 `modality.json` 选择实际使用的手指（如只用右手拇指+食指）：

```json
"tactile_finger_right_0": {"original_key": "observation.images.tactile_finger_right_0"},
"tactile_finger_right_1": {"original_key": "observation.images.tactile_finger_right_1"}
```

### Step 1c: 单独计算/重新计算 norm（可选）

当 state/action 维度变化时需要重新计算：

```bash
cd /mnt/netdata/Team/Personal/chenyiyang/zjb/Isaac-GR00T

uv run --no-sync python gr00t/data/stats.py \
    --dataset-path /path/to/groot_dataset \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/our_robot_df/our_robot_config.py
```

**注意**：触觉图像不参与 stats 计算（它通过 FTP encoder 单独处理，normalize 到 [0,1]）。
stats 只覆盖 state 和 action 的 min/max（用于归一化到 [-1, 1]）。

### Step 2: 配置 train_config.yaml

修改 `examples/our_robot_df/train_config.yaml` 中的关键路径：

```yaml
paths:
  base_model_path: /path/to/groot_base        # 预训练模型
  backbone_model_name: /path/to/cosmos_base   # VLM backbone
  dataset_path: /path/to/groot_dataset        # Step 1 的输出
  output_dir: /path/to/output_checkpoints     # 训练输出

task:
  description: "Your task description here."
```

### Step 3: 启动训练

```bash
bash examples/our_robot_df/train.sh
```

预览命令：

```bash
bash examples/our_robot_df/train.sh --dry-run
```

覆盖参数：

```bash
# 快速验证
bash examples/our_robot_df/train.sh --max-steps 2000 --num-gpus 1

# 调整 DF block size
bash examples/our_robot_df/train.sh --df-block-size 4
```

## 关于 Norm 计算

| 模态 | 是否需要 stats | 说明 |
|------|---------------|------|
| state (right_arm/hand) | **需要** | min/max 归一化到 [-1, 1] |
| action (right_arm/hand) | **需要** | 同上 + relative action 转换 |
| visual (ego/wrist) | 不需要 | VLM processor 自带 normalize |
| tactile (finger images) | **不需要** | FTP encoder 输入直接 [0,1] |
| language | 不需要 | tokenizer 处理 |

**何时需要重新计算 stats**：
- 切换数据集时
- 修改 state/action 维度映射时 (`modality.json` 的 start/end)
- 切换 arm_mode (right ↔ bimanual) 时

## 配置参数

### Diffusion Forcing

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `block_size` | 5 | 每 block 的 action token 数 |
| `mix_prob` | 1.0 | DF 训练概率 (vs 标准 FM) |
| `block_time_sampling` | monotone | 噪声分配策略 |
| `reweight_gamma` | 0.0 | 损失重加权 (0=不加权) |
| `phase_alpha` | 1.0 | Beta 分布 alpha |

### 触觉 (FTP)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `encoder_path` | - | FTP 预训练权重目录 |
| `sensor_name` | SharpaWave | T3 encoder 对应传感器 |
| `num_tokens` | 2 | 触觉 token 数 (= 活跃手指数) |
| `func_area_indices` | [24, 25] | FTP function area 索引 |
| `num_fingers` | 5 | 拼接图中总手指数 |
| `finger_indices` | [0, 1] | 提取的手指索引 |
| `target_size` | 224 | resize 目标尺寸 |

### 单臂/双臂切换

在 `train_config.yaml` 中设置 `arm_mode`:
- `right`: 仅右臂 (29维，默认)
- `bimanual`: 双臂 (58维)

## 后续计划

- [x] Phase 1: Block-wise Diffusion Forcing
- [x] Phase 2: FTP 触觉编码器 + deformation image 预处理
- [ ] Phase 3: Block-aligned attention（触觉局部修正）
- [ ] Phase 4: 交互式推理（块级流式输出）
