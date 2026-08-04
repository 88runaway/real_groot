"""
双臂 RealMan + Sharpa 灵巧手 模态配置（仅右臂）
Diffusion Forcing 版本 — 模态配置与标准版一致

数据来源: LeRobot v3.0 格式 → 转换为 GR00T LeRobot v2 后使用
机器人:    double_realman_follower
关节布局:  observation.state / action 共 58 维
           [0:7]   左臂 7-dof    (不使用)
           [7:29]  左手 22-dof   (不使用)
           [29:36] 右臂 7-dof    ← right_arm
           [36:58] 右手 22-dof   ← right_hand
相机:      ego (left_top), right_wrist
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

our_robot_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego", "right_wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "right_arm",
            "right_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 40)),
        modality_keys=[
            "right_arm",
            "right_hand",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(our_robot_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
