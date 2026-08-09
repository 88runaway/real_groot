"""
双臂 RealMan + Sharpa 灵巧手 模态配置（双臂模式）
Diffusion Forcing 版本 — 支持 block-aligned 触觉时间对齐

数据来源: LeRobot v3.0 格式 → 转换为 GR00T LeRobot v2 后使用
机器人:    double_realman_follower
关节布局:  observation.state / action 共 58 维
           [0:7]   左臂 7-dof    ← left_arm
           [7:29]  左手 22-dof   ← left_hand
           [29:36] 右臂 7-dof    ← right_arm
           [36:58] 右手 22-dof   ← right_hand
相机:      ego (left_top), left_wrist, right_wrist
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

_ACTION_HORIZON = 40
_BLOCK_SIZE = 5
_NUM_BLOCKS = _ACTION_HORIZON // _BLOCK_SIZE
_TACTILE_DELTA_INDICES = [b * _BLOCK_SIZE for b in range(_NUM_BLOCKS)]

our_robot_bimanual_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego", "left_wrist", "right_wrist"],
    ),
    "tactile_video": ModalityConfig(
        delta_indices=_TACTILE_DELTA_INDICES,
        modality_keys=["tactile_finger_left_0", "tactile_finger_left_1", "tactile_finger_left_2"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "left_hand",
            "right_arm",
            "right_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, _ACTION_HORIZON)),
        modality_keys=[
            "left_arm",
            "left_hand",
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

register_modality_config(our_robot_bimanual_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
