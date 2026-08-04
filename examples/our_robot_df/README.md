# GR00T-DF: Block-wise Diffusion Forcing 微调

基于 `our_robot` 标准配置，启用 **Block-wise Diffusion Forcing** 架构改进。

## 架构改进

相比标准 GR00T flow matching，本版本实现了以下核心改进：

### 1. 块级独立噪声调度 (Block-wise Noise Scheduling)

将 action horizon (40 步) 切分为 8 个 block (每 block 5 步)，每个 block 有独立的噪声水平：

- **Monotone 调度**：前面的 block 噪声更低（更干净），后面的 block 噪声更高
- 训练时通过 `phase ~ Beta(alpha, 1)` 采样，控制"clean frontier"位置
- 支持与标准 flow matching 混合训练 (`mix_prob`)

### 2. 金字塔推理 (Pyramid Inference Schedule)

推理时不再对所有 token 均匀去噪：

- Block 0 最先完成去噪 → 最近的动作最先确定
- Block 7 最后完成 → 远期动作保持灵活性
- 效果：近期动作精准确定性高，远期动作允许后续修正

### 3. 损失重加权 (Loss Reweighting)

Monotone 调度下，前面的 block 训练覆盖不足。通过 `(num_blocks/(k+1))^gamma` 重加权补偿。

## 文件说明

| 文件 | 作用 |
|------|------|
| `our_robot_config.py` | 模态配置（与标准版一致） |
| `train_config.yaml` | 训练配置，包含 DF 专属参数 |
| `launch_finetune_df.py` | 自定义启动脚本，注入 DF 参数到模型 |
| `train.sh` | 一键训练启动脚本 |
| `modality.json` | 数据模态映射 |

## 快速开始

### 启动训练

```bash
bash examples/our_robot_df/train.sh
```

预览命令：

```bash
bash examples/our_robot_df/train.sh --dry-run
```

### 覆盖参数

```bash
# 减小 block size（更细粒度的块）
bash examples/our_robot_df/train.sh --df-block-size 4

# 混合 50% DF + 50% 标准 FM
bash examples/our_robot_df/train.sh --df-mix-prob 0.5
```

## 配置参数说明

### Diffusion Forcing 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | true | 是否启用 DF |
| `block_size` | 5 | 每 block 包含的 action token 数 |
| `mix_prob` | 1.0 | 训练时使用 DF 的概率 (vs 标准 FM) |
| `block_time_sampling` | monotone | 块时间采样策略 |
| `reweight_gamma` | 0.5 | 损失重加权系数 |
| `phase_alpha` | 1.0 | Beta(alpha,1) 的 alpha 参数 |

### 采样策略

- **monotone**：前 block 更干净，与推理时 pyramid 调度对齐。推荐。
- **independent**：各 block 完全独立采样，更随机但与推理不对齐。

### 设计决策

- `action_horizon=40, block_size=5` → 8 blocks，平衡粒度与计算量
- `mix_prob=1.0`：纯 DF 训练，完全对齐 pyramid 推理
- `reweight_gamma=0.5`：适度补偿，过大会让早期 block 主导训练

## 与标准版的区别

| 维度 | our_robot (标准) | our_robot_df (DF) |
|------|-----------------|------------------|
| 噪声调度 | 全局单一时间步 | 块级独立时间步 |
| 推理去噪 | 均匀 4 步 Euler | 金字塔调度 |
| 近期动作 | 与远期同步生成 | 优先生成（更确定） |
| 输出目录 | groot_our_robot | groot_our_robot_df |
| W&B project | finetune-gr00t-n1d7 | finetune-gr00t-n1d7-df |

## 后续计划

- [ ] Phase 2: 集成 FTP 触觉编码器
- [ ] Phase 3: Block-aligned attention（触觉局部修正）
- [ ] Phase 4: 交互式推理（块级流式输出）
