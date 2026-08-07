"""
Launch finetuning with Block-wise Diffusion Forcing for GR00T N1.7.

在标准 launch_finetune 基础上，注入 Diffusion Forcing 配置到 model config 中。
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


@dataclass
class DiffusionForcingConfig:
    """Diffusion Forcing specific parameters."""

    enabled: bool = True
    block_size: int = 5
    mix_prob: float = 1.0
    block_time_sampling: str = "monotone"
    reweight_gamma: float = 0.5
    phase_alpha: float = 1.0


@dataclass
class DFFinetuneConfig(FinetuneConfig):
    """FinetuneConfig extended with Diffusion Forcing parameters."""

    df_enabled: bool = True
    """Enable block-wise diffusion forcing."""

    df_block_size: int = 5
    """Number of action tokens per block."""

    df_mix_prob: float = 1.0
    """Probability of using DF vs standard flow matching during training."""

    df_block_time_sampling: str = "monotone"
    """Block time sampling strategy: 'monotone' or 'independent'."""

    df_reweight_gamma: float = 0.5
    """Loss reweighting gamma for monotone scheduling."""

    df_phase_alpha: float = 1.0
    """Beta distribution alpha for phase sampling in monotone mode."""

    # Tactile injection parameters
    tactile_enabled: bool = False
    """Enable FTP tactile encoder injection."""

    tactile_encoder_path: str = ""
    """Path to FTP model checkpoint directory."""

    tactile_sensor_name: str = "GelSightMini"
    """Sensor name for T3 encoder checkpoint (e.g. GelSightMini, SharpaWave)."""

    tactile_encoder_output_dim: int = 1536
    """Output dim of tactile encoder (must match input_embedding_dim)."""

    tactile_freeze_backbone: bool = True
    """Freeze FTP backbone, only train output_proj."""

    tactile_num_tokens: int = 2
    """Number of tactile tokens (auto-computed from finger masks)."""

    tactile_func_area_indices: str = ""
    """Comma-separated function area indices (e.g. '24,25')."""

    tactile_target_size: int = 224
    """Target image size for each finger (square)."""

    tactile_num_fingers: int = 5
    """Total number of fingers in concatenated tactile image (per hand)."""

    tactile_finger_indices: str = ""
    """Comma-separated finger indices to extract from concatenated image (e.g. '0,1,2')."""

    tactile_block_aligned: bool = False
    """Block-aligned attention: action block k only attends to tactile block k."""

    tactile_attend_self: bool = True
    """Whether tactile tokens can attend to each other in self-attention."""


def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    ft_config = tyro.cli(DFFinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # Standard finetune config mapping
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    config.model.use_percentiles = ft_config.use_percentiles
    if (ft_config.shortest_image_edge is None) != (ft_config.crop_fraction is None):
        raise ValueError("shortest_image_edge and crop_fraction must be set together")
    if ft_config.shortest_image_edge is not None:
        config.model.shortest_image_edge = ft_config.shortest_image_edge
        config.model.crop_fraction = ft_config.crop_fraction
        config.model.image_crop_size = None
        config.model.image_target_size = None
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = ft_config.backbone_model_name
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    # === Diffusion Forcing parameters ===
    config.model.use_diffusion_forcing = ft_config.df_enabled
    config.model.df_block_size = ft_config.df_block_size
    config.model.df_mix_prob = ft_config.df_mix_prob
    config.model.df_block_time_sampling = ft_config.df_block_time_sampling
    config.model.df_reweight_gamma = ft_config.df_reweight_gamma
    config.model.df_phase_alpha = ft_config.df_phase_alpha

    # === Tactile injection parameters ===
    config.model.use_tactile = ft_config.tactile_enabled
    config.model.tactile_encoder_path = ft_config.tactile_encoder_path
    config.model.tactile_sensor_name = ft_config.tactile_sensor_name
    config.model.tactile_encoder_output_dim = ft_config.tactile_encoder_output_dim
    config.model.tactile_freeze_backbone = ft_config.tactile_freeze_backbone
    config.model.num_tactile_tokens = ft_config.tactile_num_tokens
    if ft_config.tactile_func_area_indices:
        config.model.tactile_func_area_indices = [
            int(x.strip()) for x in ft_config.tactile_func_area_indices.split(",")
        ]
    else:
        config.model.tactile_func_area_indices = None
    config.model.tactile_target_size = ft_config.tactile_target_size
    config.model.tactile_num_fingers = ft_config.tactile_num_fingers
    if ft_config.tactile_finger_indices:
        config.model.tactile_finger_indices = [
            int(x.strip()) for x in ft_config.tactile_finger_indices.split(",")
        ]
    else:
        config.model.tactile_finger_indices = None
    config.model.tactile_block_aligned = ft_config.tactile_block_aligned
    config.model.tactile_attend_self = ft_config.tactile_attend_self

    # Training config
    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.ds_weights_alpha = ft_config.ds_weights_alpha

    config.training.save_only_model = ft_config.save_only_model
    config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    print(f"[DF] Diffusion Forcing enabled: {ft_config.df_enabled}")
    print(f"[DF] block_size={ft_config.df_block_size}, mix_prob={ft_config.df_mix_prob}")
    print(f"[DF] sampling={ft_config.df_block_time_sampling}, gamma={ft_config.df_reweight_gamma}")
    print(f"[TAC] Tactile enabled: {ft_config.tactile_enabled}")
    if ft_config.tactile_enabled:
        print(f"[TAC] encoder_path={ft_config.tactile_encoder_path}")
        print(f"[TAC] sensor={ft_config.tactile_sensor_name}")
        print(f"[TAC] output_dim={ft_config.tactile_encoder_output_dim}, freeze={ft_config.tactile_freeze_backbone}")
        print(f"[TAC] func_area_indices={ft_config.tactile_func_area_indices}")
        print(f"[TAC] block_aligned={ft_config.tactile_block_aligned}, attend_self={ft_config.tactile_attend_self}")

    run(config)
