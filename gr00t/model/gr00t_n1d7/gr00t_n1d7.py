# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from gr00t.model.modules.ftp_encoder import FTPTactileEncoder


logger = logging.getLogger(__name__)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        # Tactile encoder
        self.use_tactile = config.use_tactile
        if self.use_tactile:
            self.tactile_encoder = FTPTactileEncoder(
                output_dim=config.tactile_encoder_output_dim
            )
            if config.tactile_encoder_path:
                self.tactile_encoder.load_pretrained(
                    config.tactile_encoder_path,
                    sensor_name=config.tactile_sensor_name,
                    freeze_backbone=config.tactile_freeze_backbone,
                )
            self.num_tactile_tokens = config.num_tactile_tokens
            self.tactile_func_area_indices = config.tactile_func_area_indices or list(
                range(config.num_tactile_tokens)
            )

        # Pin the time-sampling Beta to CPU/fp32 explicitly. The action head can
        # be instantiated under a meta / no_init_weights default-device context
        # (e.g. nested from_pretrained). A Beta built from bare Python floats
        # would then place its concentration tensors on the meta device (or in
        # the active default dtype, e.g. bf16). With validate_args enabled that
        # already fails here in __init__ (Beta's internal .item() check cannot
        # run on meta); even with validation off, sample_time would later raise
        # or return garbage. Explicit device/dtype here makes the sampler depend
        # only on the config, not on the construction-time device/dtype context,
        # so the noise schedule is identical across SDPA/FA2/FA4 and meta vs.
        # real-device loads. config is the canonical source for these values.
        self.beta_dist = Beta(
            torch.tensor(float(config.noise_beta_alpha), dtype=torch.float32, device="cpu"),
            torch.tensor(float(config.noise_beta_beta), dtype=torch.float32, device="cpu"),
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        # Re-apply tactile backbone freeze: the loop above (`p.requires_grad = True`)
        # unfreezes everything, including any freeze applied by load_pretrained earlier.
        if self.use_tactile and self.config.tactile_freeze_backbone:
            self.tactile_encoder._freeze_backbone()
            trainable_tac = sum(
                p.numel() for p in self.tactile_encoder.parameters() if p.requires_grad
            )
            logger.debug(
                f"Tactile encoder backbone re-frozen; trainable tactile params: {trainable_tac}"
            )
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)

        if self.config.use_diffusion_forcing:
            # Read pre-sampled block_c from data pipeline (DF tactile alignment)
            ext_block_c = getattr(action_input, "block_c", None)
            if ext_block_c is None and isinstance(action_input, dict):
                ext_block_c = action_input.get("block_c")
            if ext_block_c is not None:
                ext_block_c = ext_block_c.to(device=device, dtype=torch.long)

            t, t_per_token, loss_weights, block_c = self._sample_df_time(
                actions.shape[0], actions.shape[1], device, actions.dtype,
                external_block_c=ext_block_c,
            )
        else:
            t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
            t_per_token = t[:, None, None]  # (B,1,1) for broadcast
            loss_weights = None
            block_c = None

        noisy_trajectory = (1 - t_per_token) * noise + t_per_token * actions
        velocity = actions - noise

        if self.config.use_diffusion_forcing:
            # Per-token discretized timesteps: [B, action_horizon]
            t_discrete_per_token = (t_per_token.squeeze(-1) * self.num_timestep_buckets).long()
            action_features = self.action_encoder(noisy_trajectory, t_discrete_per_token, embodiment_id)

            # Per-token temb for DiT:
            # state = 0 (clean observation, no noise)
            # tactile = 0 (clean conditioning signal)
            # action = per-block timestep
            state_t = torch.zeros(actions.shape[0], 1, device=device, dtype=torch.long)
            t_for_dit = torch.cat(
                [state_t, t_discrete_per_token], dim=1
            )  # [B, 1+action_horizon]
        else:
            t_discretized = (t * self.num_timestep_buckets).long()
            action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
            t_for_dit = t_discretized

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Encode tactile (data pipeline already selected the correct temporal frame)
        tactile_features = self._encode_tactile(action_input, device)

        # Join state, tactile (optional), and action along sequence dimension.
        if tactile_features is not None:
            sa_embs = torch.cat((state_features, tactile_features, action_features), dim=1)
            # Tactile positions get zero timestep (pure conditioning, no flow-matching)
            if self.config.use_diffusion_forcing:
                nt = tactile_features.shape[1]
                tac_t = torch.zeros(actions.shape[0], nt, device=device, dtype=torch.long)
                t_for_dit = torch.cat(
                    [t_for_dit[:, :1], tac_t, t_for_dit[:, 1:]], dim=1
                )
            elif t_for_dit.dim() == 1:
                pass  # scalar timestep, no adjustment needed

            # Build tactile attention mask (block_aligned / attend_self control)
            sa_attn_mask = self._build_tactile_attention_mask(
                batch_size=actions.shape[0],
                num_tactile_tokens=tactile_features.shape[1],
                action_horizon=actions.shape[1],
                device=device,
                block_c=block_c,
            )
        else:
            sa_embs = torch.cat((state_features, action_features), dim=1)
            sa_attn_mask = None
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_for_dit,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
                self_attention_mask=sa_attn_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_for_dit,
                return_all_hidden_states=True,
                self_attention_mask=sa_attn_mask,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask

        # DF active mask: exclude near-clean tokens whose loss is dominated by
        # irreducible Var[noise]=1.0 (matches Pi0's active_mask = (time > 5e-4)).
        # GR00T convention: t=noise_s is clean; Pi0 convention: t=0 is clean.
        if self.config.use_diffusion_forcing and t_per_token is not None:
            df_active = (t_per_token < self.config.noise_s - 5e-4).to(dtype=action_loss.dtype)
            action_loss = action_loss * df_active

        if loss_weights is not None:
            action_loss = action_loss * loss_weights

        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _sample_df_time(
        self, batch_size, action_horizon, device, dtype,
        external_block_c: torch.Tensor | None = None,
    ):
        """Sample block-wise diffusion forcing timesteps.

        Args:
            external_block_c: [B] pre-sampled block index from data pipeline.
                When provided (monotone mode), phase is derived from it so that
                the noise schedule is consistent with the loaded tactile frame.

        Returns:
            t: [B] representative scalar time (for compatibility)
            t_per_token: [B, action_horizon, 1] per-token time for noise mixing
            loss_weights: [B, action_horizon, 1] per-block loss reweighting or None
            block_c: [B] current block index per sample (for block_aligned mask)
        """
        block_size = self.config.df_block_size
        num_blocks = action_horizon // block_size
        assert action_horizon % block_size == 0, (
            f"action_horizon ({action_horizon}) must be divisible by df_block_size ({block_size})"
        )

        use_df = torch.rand(batch_size, device=device) < self.config.df_mix_prob

        if self.config.df_block_time_sampling == "monotone":
            if external_block_c is not None:
                # Phase derived from pre-sampled block_c:
                # block_c = floor(phase * num_blocks)  =>
                # phase ∈ [block_c/nb, (block_c+1)/nb)
                block_c = external_block_c  # [B]
                phase = (block_c.float() + torch.rand(batch_size, device=device, dtype=dtype)) / num_blocks
            else:
                # No external block_c (legacy path): sample phase freely
                phase_dist = Beta(
                    torch.tensor(float(self.config.df_phase_alpha), device="cpu"),
                    torch.tensor(1.0, device="cpu"),
                )
                phase = phase_dist.sample([batch_size]).to(device, dtype=dtype)
                block_c = (phase * num_blocks).long().clamp(0, num_blocks - 1)

            block_indices = torch.arange(num_blocks, device=device, dtype=dtype)  # [nb]
            # GR00T convention: t=0 noisy, t=1 clean.
            # Earlier blocks (small k) should be cleaner (higher t).
            t_blocks = (
                phase.unsqueeze(1) * num_blocks / (block_indices.unsqueeze(0) + 1.0)
            ).clamp(0, 1) * self.config.noise_s  # [B, nb]

            # Loss reweighting: compensate for early blocks seeing less gradient
            if self.config.df_reweight_gamma > 0:
                weights = (num_blocks / (block_indices + 1.0)) ** self.config.df_reweight_gamma
                weights = weights.unsqueeze(0).expand(batch_size, -1)  # [B, nb]
                weights = weights.repeat_interleave(block_size, dim=1).unsqueeze(-1)  # [B, H, 1]
                loss_weights = weights.to(dtype=dtype)
            else:
                loss_weights = None
        else:
            # Independent: each block samples independently, no frontier block
            t_blocks = self.beta_dist.sample([batch_size, num_blocks]).to(device, dtype=dtype)
            t_blocks = (1 - t_blocks) * self.config.noise_s  # [B, nb]
            loss_weights = None
            block_c = external_block_c if external_block_c is not None else torch.zeros(
                batch_size, device=device, dtype=torch.long
            )

        # Fallback: standard flow matching time (single scalar per sample)
        t_standard = self.sample_time(batch_size, device=device, dtype=dtype)  # [B]

        # Expand block times to per-token: [B, nb] -> [B, action_horizon]
        t_per_token_df = t_blocks.repeat_interleave(block_size, dim=1)  # [B, H]

        # Mix DF vs standard
        t_per_token_standard = t_standard.unsqueeze(1).expand(-1, action_horizon)  # [B, H]
        t_per_token = torch.where(
            use_df.unsqueeze(1), t_per_token_df, t_per_token_standard
        )  # [B, H]

        # Representative scalar time
        t = t_per_token.mean(dim=1)  # [B]

        # Reshape for noise mixing
        t_per_token = t_per_token.unsqueeze(-1)  # [B, H, 1]

        if loss_weights is not None:
            loss_weights = torch.where(
                use_df.unsqueeze(1).unsqueeze(2), loss_weights, torch.ones_like(loss_weights)
            )

        return t, t_per_token, loss_weights, block_c

    def _encode_tactile(
        self,
        action_input: BatchFeature,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Encode tactile deformation images if available.

        Reads pre-split finger images from action_input["tactile_deform"]
        ([B, N_fingers, 3, H, W]) and encodes each finger with the FTP encoder.

        The data pipeline already selects the correct temporal frame
        (block_c-aligned for DF training, observation-time for inference),
        so no temporal selection is needed here.

        Falls back to legacy per-field naming (tactile_0, tactile_1, ...) if
        tactile_deform is not present.

        Returns:
            [B, num_tactile_tokens, input_embedding_dim] or None if tactile disabled.
        """
        if not self.use_tactile:
            return None

        # Primary path: pre-split deformation images from processor
        tactile_deform = getattr(action_input, "tactile_deform", None)
        if tactile_deform is None and isinstance(action_input, dict):
            tactile_deform = action_input.get("tactile_deform")

        if tactile_deform is not None:
            tactile_deform = tactile_deform.to(device=device)

            # tactile_deform: [B, N_fingers, 3, H, W]
            B, N = tactile_deform.shape[:2]
            tokens = []
            for i in range(N):
                area_idx = self.tactile_func_area_indices[i] if i < len(self.tactile_func_area_indices) else i
                finger_img = tactile_deform[:, i]  # [B, 3, H, W]
                tok = self.tactile_encoder(finger_img, func_area_idx=area_idx)  # (B, 1, D)
                tokens.append(tok)
            tactile_tokens = torch.cat(tokens, dim=1)  # (B, num_tactile_tokens, D)
            return tactile_tokens.to(dtype=next(self.model.parameters()).dtype)

        # Fallback: legacy per-field naming
        tokens = []
        for i, area_idx in enumerate(self.tactile_func_area_indices):
            tactile_img = getattr(action_input, f"tactile_{i}", None)
            if tactile_img is None and i == 0:
                tactile_img = getattr(action_input, "tactile_left", None)
            if tactile_img is None and i == 1:
                tactile_img = getattr(action_input, "tactile_right", None)

            if tactile_img is None:
                return None

            tok = self.tactile_encoder(tactile_img, func_area_idx=area_idx)  # (B, 1, D)
            tokens.append(tok)

        tactile_tokens = torch.cat(tokens, dim=1)  # (B, num_tactile_tokens, D)
        return tactile_tokens.to(device=device, dtype=next(self.model.parameters()).dtype)

    def _build_tactile_attention_mask(
        self,
        batch_size: int,
        num_tactile_tokens: int,
        action_horizon: int,
        device: torch.device,
        block_c: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Build self-attention mask for tactile attention control.

        Implements two mechanisms:
        1. tactile_attend_self=False: Block tactile tokens from attending to each other.
           Tactile tokens become passive KV providers — action tokens can attend to them
           but they don't attend to each other.
        2. tactile_block_aligned=True: Only action block c can attend to tactile tokens.
           Other action blocks are blocked from attending tactile (enforces temporal
           local conditioning with diffusion forcing).

        Sequence layout: [state(1) | tactile(nt) | action(action_horizon)]

        Args:
            batch_size: Batch size B.
            num_tactile_tokens: Number of tactile tokens (nt).
            action_horizon: Total action tokens.
            device: Tensor device.
            block_c: Per-sample current block index [B] for block_aligned.
                Required when tactile_block_aligned=True during training.

        Returns:
            Boolean attention mask [B, seq_len, seq_len] where True=attend,
            or None if no masking needed.
        """
        need_no_self = not self.config.tactile_attend_self
        need_block_align = (
            self.config.tactile_block_aligned and self.config.use_diffusion_forcing
        )

        if not need_no_self and not need_block_align:
            return None

        nt = num_tactile_tokens
        seq_len = 1 + nt + action_horizon  # state + tactile + action

        # Start with full attention (all True)
        mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)

        # Indices
        tac_start = 1
        tac_end = 1 + nt
        act_start = 1 + nt

        # 1. tactile_attend_self=False: block tactile→tactile attention
        if need_no_self:
            # Tactile queries (rows tac_start:tac_end) cannot attend to
            # tactile keys (cols tac_start:tac_end)
            mask[:, tac_start:tac_end, tac_start:tac_end] = False

        # 2. block_aligned: only action block c can attend to tactile
        if need_block_align and block_c is not None:
            block_size = self.config.df_block_size
            num_blocks = action_horizon // block_size

            # For each action token, determine which block it belongs to
            # action token at position act_start + j belongs to block j // block_size
            for k in range(num_blocks):
                blk_start = act_start + k * block_size
                blk_end = act_start + (k + 1) * block_size
                # Per-sample: block if k != c[b]
                not_current = (block_c != k)  # [B]
                # Block these action tokens from attending to tactile
                mask[not_current, blk_start:blk_end, tac_start:tac_end] = False

        return mask

    def _blockwise_time_schedule(self, num_steps: int, num_blocks: int, device, dtype):
        """Build a pyramid time schedule for blockwise inference.

        Each block k starts at noise level 0 and becomes fully clean at
        step = ceil((k+1)/num_blocks * num_steps). All blocks are denoised
        simultaneously but at different rates, forming a "pyramid" schedule.

        Returns:
            schedule: [num_steps+1, num_blocks] time values from 0 (noise) to 1 (clean)
        """
        schedule = torch.zeros(num_steps + 1, num_blocks, device=device, dtype=dtype)
        for k in range(num_blocks):
            clean_step = int(((k + 1) / num_blocks) * num_steps)
            for s in range(num_steps + 1):
                if s <= clean_step:
                    schedule[s, k] = s / max(clean_step, 1)
                else:
                    schedule[s, k] = 1.0
        return schedule

    def _blockwise_denoise(
        self,
        actions: torch.Tensor,
        state_features: torch.Tensor,
        vl_embeds: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        vel_strength: torch.Tensor,
        tactile_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Blockwise pyramid denoising for diffusion forcing inference.

        Instead of uniform Euler steps, different blocks progress at different rates.
        Earlier blocks are denoised faster (pyramid schedule).
        """
        batch_size = actions.shape[0]
        device = actions.device
        dtype = actions.dtype
        action_horizon = self.config.action_horizon
        block_size = self.config.df_block_size
        num_blocks = action_horizon // block_size
        num_steps = self.num_inference_timesteps

        schedule = self._blockwise_time_schedule(num_steps, num_blocks, device, dtype)

        for step in range(num_steps):
            # Current time per block: [num_blocks]
            t_curr = schedule[step]      # where each block currently is
            t_next = schedule[step + 1]  # where each block should go

            # Per-block dt
            dt_blocks = t_next - t_curr  # [num_blocks]

            # Expand to per-token: [action_horizon]
            t_per_token = t_curr.repeat_interleave(block_size)  # [H]
            dt_per_token = dt_blocks.repeat_interleave(block_size)  # [H]

            # Discretize for encoder: [B, H]
            t_discrete = (t_per_token * self.num_timestep_buckets).long()
            t_discrete = t_discrete.unsqueeze(0).expand(batch_size, -1)  # [B, H]

            # State = 0 (clean observation)
            state_t = torch.zeros(batch_size, 1, device=device, dtype=torch.long)

            action_features = self.action_encoder(actions, t_discrete, embodiment_id)

            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            if tactile_features is not None:
                sa_embs = torch.cat((state_features, tactile_features, action_features), dim=1)
            else:
                sa_embs = torch.cat((state_features, action_features), dim=1)

            # Build per-token timestep for DiT: [B, 1+(nt)+H]
            # state=0, tactile=0, action=per-block time
            t_for_dit = torch.cat([state_t, t_discrete], dim=1)
            if tactile_features is not None:
                nt = tactile_features.shape[1]
                tac_t = torch.zeros(batch_size, nt, device=device, dtype=torch.long)
                t_for_dit = torch.cat([t_for_dit[:, :1], tac_t, t_for_dit[:, 1:]], dim=1)

            # Inference mask: only tactile_attend_self control (no block_aligned during inference)
            sa_attn_mask = None
            if tactile_features is not None and not self.config.tactile_attend_self:
                sa_attn_mask = self._build_tactile_attention_mask(
                    batch_size=batch_size,
                    num_tactile_tokens=tactile_features.shape[1],
                    action_horizon=action_horizon,
                    device=device,
                    block_c=None,
                )

            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=t_for_dit,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                    self_attention_mask=sa_attn_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=t_for_dit,
                    self_attention_mask=sa_attn_mask,
                )

            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -action_horizon:]

            # Per-token Euler integration with block-specific dt
            dt_expanded = dt_per_token.unsqueeze(0).unsqueeze(-1)  # [1, H, 1]
            actions = actions + dt_expanded * pred_velocity * vel_strength

        return actions

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        vel_strength = torch.ones_like(actions)

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # Run denoising steps.
        # Encode tactile for inference
        tactile_features = self._encode_tactile(action_input, device)

        if self.config.use_diffusion_forcing:
            actions = self._blockwise_denoise(
                actions, state_features, vl_embeds,
                embodiment_id, backbone_output, vel_strength,
                tactile_features=tactile_features,
            )
        else:
            for t in range(self.num_inference_timesteps):
                t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
                t_discretized = int(t_cont * self.num_timestep_buckets)

                # Embed noised action trajectory.
                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
                # Add position embedding.
                if self.config.add_pos_embed:
                    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                # Join state, tactile (optional), and action along sequence dimension.
                if tactile_features is not None:
                    sa_embs = torch.cat((state_features, tactile_features, action_features), dim=1)
                else:
                    sa_embs = torch.cat((state_features, action_features), dim=1)

                # Build inference attention mask (only tactile_attend_self control)
                sa_attn_mask = None
                if tactile_features is not None and not self.config.tactile_attend_self:
                    sa_attn_mask = self._build_tactile_attention_mask(
                        batch_size=batch_size,
                        num_tactile_tokens=tactile_features.shape[1],
                        action_horizon=self.config.action_horizon,
                        device=device,
                        block_c=None,
                    )

                # Run model forward.
                if self.config.use_alternate_vl_dit:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                        image_mask=backbone_output.image_mask,
                        backbone_attention_mask=backbone_output.backbone_attention_mask,
                        self_attention_mask=sa_attn_mask,
                    )
                else:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                        self_attention_mask=sa_attn_mask,
                    )
                pred = self.action_decoder(model_output, embodiment_id)

                pred_velocity = pred[:, -self.action_horizon :]

                # Update actions using euler integration.
                actions = actions + (1.0 / self.num_inference_timesteps) * pred_velocity * vel_strength

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    if config.backbone_model_type == "qwen":
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported backbone model type: {config.backbone_model_type}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
