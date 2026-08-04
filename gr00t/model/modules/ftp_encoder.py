"""FTP Tactile Encoder for GR00T integration.

Loads pretrained weights from ftp1-policy checkpoint structure:
    checkpoint_dir/
        hpt_tokenizer/
            {SensorName}_image_224_224_3.safetensors   (T3 ViT encoder)
            shared_image_chunk_encoder.safetensors      (Trunk + image_proj)
        model.safetensors                              (func_area_idx_embedding + unified_proj)

Architecture per tactile image (B, 3, 224, 224) → (B, 1, output_dim):
    ViT Encoder (3 blocks, 768-dim):  patch_embed → cls + pos → 3 T3 blocks → (B, 197, 768)
    Trunk (9 blocks, 768-dim):        9 blocks + LN → CLS → (B, 768)
    image_proj: Linear(768 → 512)
    + func_area_idx_embedding[area_idx]
    unified_proj: LayerNorm → Linear → GELU → Linear  (all 512-dim)
    output_proj: Linear(512 → output_dim)              (new, random init)
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_EMBED_DIM = 768
_HEADS = 12
_T3_DEPTH = 3
_TRUNK_DEPTH = 9
_MLP_RATIO = 4
_PATCH_SIZE = 16
_IMAGE_SIZE = 224
_NUM_PATCHES = (_IMAGE_SIZE // _PATCH_SIZE) ** 2  # 196
_SEQ_LEN = _NUM_PATCHES + 1  # 197
_INTERMEDIATE = 512
_LN_EPS = 1e-6
_TOTAL_FUNC_AREAS = 48


class _FTPAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _FTPMlp(nn.Module):
    def __init__(self, dim: int, mlp_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _FTPBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=_LN_EPS)
        self.attn = _FTPAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=_LN_EPS)
        self.mlp = _FTPMlp(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FTPTactileEncoder(nn.Module):
    """FTP Tactile Encoder matching ftp1-policy checkpoint structure.

    Encodes tactile images (224x224x3) into tokens for GR00T's DiT.
    Supports multiple function areas via learned embeddings.

    Usage::

        encoder = FTPTactileEncoder(output_dim=1536)
        encoder.load_pretrained("/path/to/ftp_model", sensor_name="GelSightMini")
        left_tok  = encoder(left_img,  func_area_idx=24)  # (B, 1, 1536)
        right_tok = encoder(right_img, func_area_idx=25)  # (B, 1, 1536)
        tactile_tokens = torch.cat([left_tok, right_tok], dim=1)  # (B, 2, 1536)
    """

    def __init__(self, output_dim: int = 1536, num_func_areas: int = _TOTAL_FUNC_AREAS):
        super().__init__()
        D = _EMBED_DIM
        H = _HEADS
        M = _MLP_RATIO
        I = _INTERMEDIATE

        # --- ViT Encoder (T3, 3 blocks) ---
        self.patch_embed = nn.Conv2d(3, D, kernel_size=_PATCH_SIZE, stride=_PATCH_SIZE)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_embed = nn.Parameter(torch.zeros(1, _SEQ_LEN, D))
        self.t3_blocks = nn.ModuleList([_FTPBlock(D, H, M) for _ in range(_T3_DEPTH)])

        # --- Trunk (shared_chunk_encoder, 9 blocks) ---
        self.trunk_blocks = nn.ModuleList([_FTPBlock(D, H, M) for _ in range(_TRUNK_DEPTH)])
        self.trunk_norm = nn.LayerNorm(D, eps=_LN_EPS)

        # --- image_proj: 768 → 512 ---
        self.image_proj = nn.Linear(D, I)

        # --- func_area_idx_embedding: (48, 512) ---
        self.func_area_idx_embedding = nn.Embedding(num_func_areas, I)

        # --- unified_proj: LayerNorm → Linear → GELU → Linear ---
        # Matches checkpoint keys: unified_proj.{0,1,3} (index 2 is GELU)
        self.unified_proj = nn.Sequential(
            nn.LayerNorm(I, eps=_LN_EPS),  # [0]
            nn.Linear(I, I),                # [1]
            nn.GELU(),                      # [2]
            nn.Linear(I, I),                # [3]
        )

        # --- output_proj: 512 → output_dim (NEW, random init) ---
        self.output_proj = nn.Linear(I, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, func_area_idx: int = 0) -> torch.Tensor:
        """Encode one tactile image → 1 global token.

        Args:
            x: (B, 3, 224, 224) float tensor in [0, 1].
            func_area_idx: Function area index (0-47) for position embedding.
        Returns:
            (B, 1, output_dim) token embedding.
        """
        B = x.shape[0]

        # Patch embed → (B, 196, 768)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Prepend CLS + positional embedding
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 197, 768)
        x = x + self.pos_embed

        # T3 ViT blocks
        for block in self.t3_blocks:
            x = block(x)

        # Trunk blocks + norm
        for block in self.trunk_blocks:
            x = block(x)
        x = self.trunk_norm(x)

        # CLS token → (B, 768)
        cls_feat = x[:, 0]

        # image_proj → (B, 512)
        feat = self.image_proj(cls_feat)

        # Function area embedding
        feat = feat + self.func_area_idx_embedding.weight[func_area_idx]

        # unified_proj
        feat = self.unified_proj(feat)

        # output_proj → (B, output_dim)
        feat = self.output_proj(feat)

        return feat.unsqueeze(1)  # (B, 1, output_dim)

    def load_pretrained(
        self,
        checkpoint_dir: str,
        sensor_name: str = "GelSightMini",
        freeze_backbone: bool = True,
    ):
        """Load pretrained weights from ftp1-policy checkpoint directory.

        Expected structure:
            checkpoint_dir/
                hpt_tokenizer/{sensor_name}_image_224_224_3.safetensors
                hpt_tokenizer/shared_image_chunk_encoder.safetensors
                model.safetensors

        Args:
            checkpoint_dir: Path to ftp_model checkpoint directory.
            sensor_name: Sensor name for T3 encoder (e.g. "GelSightMini", "SharpaWave").
            freeze_backbone: If True, freeze all pretrained layers; only output_proj trains.
        """
        path = Path(checkpoint_dir)

        # 1. Load T3 ViT encoder
        t3_file = path / "hpt_tokenizer" / f"{sensor_name}_image_224_224_3.safetensors"
        if t3_file.exists():
            self._load_t3_encoder(t3_file)
        else:
            logger.warning(f"[FTPEncoder] T3 encoder not found: {t3_file}")

        # 2. Load shared trunk + image_proj
        trunk_file = path / "hpt_tokenizer" / "shared_image_chunk_encoder.safetensors"
        if trunk_file.exists():
            self._load_trunk(trunk_file)
        else:
            logger.warning(f"[FTPEncoder] Trunk not found: {trunk_file}")

        # 3. Load func_area_idx_embedding + unified_proj from model.safetensors
        model_file = path / "model.safetensors"
        if model_file.exists():
            self._load_model_weights(model_file)
        else:
            logger.warning(f"[FTPEncoder] model.safetensors not found: {model_file}")

        if freeze_backbone:
            self._freeze_backbone()

        logger.info(f"[FTPEncoder] Loaded pretrained from {checkpoint_dir} (sensor={sensor_name})")

    def _load_t3_encoder(self, path: Path):
        """Load T3 ViT encoder from {sensor}_image_224_224_3.safetensors."""
        from safetensors.torch import load_file

        state = load_file(str(path))

        # patch_embed
        if "vit_encoder.patch_embed.proj.weight" in state:
            self.patch_embed.weight.data = state["vit_encoder.patch_embed.proj.weight"]
            self.patch_embed.bias.data = state["vit_encoder.patch_embed.proj.bias"]
        if "vit_encoder.cls_token" in state:
            self.cls_token.data = state["vit_encoder.cls_token"]
        if "vit_encoder.pos_embed" in state:
            self.pos_embed.data = state["vit_encoder.pos_embed"]

        # T3 blocks
        for i in range(_T3_DEPTH):
            blk = self.t3_blocks[i]
            prefix = f"vit_encoder.blocks.{i}"
            if f"{prefix}.norm1.weight" in state:
                blk.norm1.weight.data = state[f"{prefix}.norm1.weight"]
                blk.norm1.bias.data = state[f"{prefix}.norm1.bias"]
                blk.attn.qkv.weight.data = state[f"{prefix}.attn.qkv.weight"]
                blk.attn.qkv.bias.data = state[f"{prefix}.attn.qkv.bias"]
                blk.attn.proj.weight.data = state[f"{prefix}.attn.proj.weight"]
                blk.attn.proj.bias.data = state[f"{prefix}.attn.proj.bias"]
                blk.norm2.weight.data = state[f"{prefix}.norm2.weight"]
                blk.norm2.bias.data = state[f"{prefix}.norm2.bias"]
                blk.mlp.fc1.weight.data = state[f"{prefix}.mlp.fc1.weight"]
                blk.mlp.fc1.bias.data = state[f"{prefix}.mlp.fc1.bias"]
                blk.mlp.fc2.weight.data = state[f"{prefix}.mlp.fc2.weight"]
                blk.mlp.fc2.bias.data = state[f"{prefix}.mlp.fc2.bias"]

        logger.info(f"[FTPEncoder] Loaded T3 encoder from {path}")

    def _load_trunk(self, path: Path):
        """Load Trunk + image_proj from shared_image_chunk_encoder.safetensors."""
        from safetensors.torch import load_file

        state = load_file(str(path))

        # Trunk blocks
        for i in range(_TRUNK_DEPTH):
            blk = self.trunk_blocks[i]
            prefix = f"shared_chunk_encoder.blocks.{i}"
            if f"{prefix}.norm1.weight" in state:
                blk.norm1.weight.data = state[f"{prefix}.norm1.weight"]
                blk.norm1.bias.data = state[f"{prefix}.norm1.bias"]
                blk.attn.qkv.weight.data = state[f"{prefix}.attn.qkv.weight"]
                blk.attn.qkv.bias.data = state[f"{prefix}.attn.qkv.bias"]
                blk.attn.proj.weight.data = state[f"{prefix}.attn.proj.weight"]
                blk.attn.proj.bias.data = state[f"{prefix}.attn.proj.bias"]
                blk.norm2.weight.data = state[f"{prefix}.norm2.weight"]
                blk.norm2.bias.data = state[f"{prefix}.norm2.bias"]
                blk.mlp.fc1.weight.data = state[f"{prefix}.mlp.fc1.weight"]
                blk.mlp.fc1.bias.data = state[f"{prefix}.mlp.fc1.bias"]
                blk.mlp.fc2.weight.data = state[f"{prefix}.mlp.fc2.weight"]
                blk.mlp.fc2.bias.data = state[f"{prefix}.mlp.fc2.bias"]

        if "shared_chunk_encoder.norm.weight" in state:
            self.trunk_norm.weight.data = state["shared_chunk_encoder.norm.weight"]
            self.trunk_norm.bias.data = state["shared_chunk_encoder.norm.bias"]

        # image_proj
        if "image_proj.weight" in state:
            self.image_proj.weight.data = state["image_proj.weight"]
            self.image_proj.bias.data = state["image_proj.bias"]

        logger.info(f"[FTPEncoder] Loaded Trunk from {path}")

    def _load_model_weights(self, path: Path):
        """Load func_area_idx_embedding + unified_proj from model.safetensors."""
        from safetensors.torch import load_file

        state = load_file(str(path))
        prefix = "hpt_tactile_encoder"

        # func_area_idx_embedding
        key = f"{prefix}.func_area_idx_embedding.weight"
        if key in state:
            self.func_area_idx_embedding.weight.data = state[key]

        # unified_proj: Sequential(LayerNorm[0], Linear[1], GELU[2], Linear[3])
        key_ln_w = f"{prefix}.unified_proj.0.weight"
        if key_ln_w in state:
            self.unified_proj[0].weight.data = state[f"{prefix}.unified_proj.0.weight"]
            self.unified_proj[0].bias.data = state[f"{prefix}.unified_proj.0.bias"]
            self.unified_proj[1].weight.data = state[f"{prefix}.unified_proj.1.weight"]
            self.unified_proj[1].bias.data = state[f"{prefix}.unified_proj.1.bias"]
            self.unified_proj[3].weight.data = state[f"{prefix}.unified_proj.3.weight"]
            self.unified_proj[3].bias.data = state[f"{prefix}.unified_proj.3.bias"]

        logger.info(f"[FTPEncoder] Loaded model weights from {path}")

    def _freeze_backbone(self):
        """Freeze all pretrained layers, only output_proj remains trainable."""
        for name, param in self.named_parameters():
            if "output_proj" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[FTPEncoder] Backbone frozen. Trainable: {trainable:,} / {total:,} params"
        )
