"""Verify fix: FTP encoder weights correct after from_pretrained + re-load."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from pathlib import Path
import importlib
config_path = Path("examples/our_robot_df/our_robot_config.py")
sys.path.append(str(config_path.parent))
importlib.import_module(config_path.stem)

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.modules.ftp_encoder import FTPTactileEncoder
from gr00t.configs.base_config import get_default_config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
from transformers import AutoProcessor, BatchFeature
import tree

embodiment_tag = EmbodimentTag.resolve("NEW_EMBODIMENT")
dataset_path = "/mnt/netdata/Team/Personal/chenyiyang/zjb/data/groot_tactile/lift_can"
mc = MODALITY_CONFIGS[embodiment_tag.value]

config = get_default_config().load_dict({
    "data": {"download_cache": False, "datasets": [{"dataset_paths": [dataset_path], "mix_ratio": 1.0, "embodiment_tag": embodiment_tag.value}]},
})
config.load_config_path = None
config.model.use_percentiles = True
config.model.use_relative_action = True
config.model.model_name = "/mnt/netdata/Team/Personal/chenyiyang/zjb/ckpts/cosmos_base"
config.model.use_tactile = True
config.model.tactile_target_size = 224
config.model.tactile_num_fingers = 5
config.model.tactile_finger_indices = [0, 1, 2]
config.model.use_diffusion_forcing = True
config.model.df_block_size = 5
config.model.df_mix_prob = 1.0
config.model.df_block_time_sampling = "monotone"
config.model.df_reweight_gamma = 0.0
config.model.df_phase_alpha = 1.0
config.model.load_bf16 = False
config.model.reproject_vision = False
config.model.backbone_trainable_params_fp32 = True
config.model.num_tactile_tokens = 3
config.model.tactile_encoder_path = "/mnt/netdata/Team/Personal/chenyiyang/zjb/ckpts/ftp_model"
config.model.tactile_sensor_name = "SharpaWave"
config.model.tactile_encoder_output_dim = 1536
config.model.tactile_freeze_backbone = False
config.model.tactile_func_area_indices = [24, 25, 26]
config.model.tactile_block_aligned = True
config.model.tactile_attend_self = False
config.training.start_from_checkpoint = "/mnt/netdata/Team/Personal/chenyiyang/zjb/ckpts/groot_base"
config.training.num_gpus = 1
config.training.output_dir = "/tmp/debug_loss"
config.data.shard_size = 1024
config.data.episode_sampling_rate = 0.1
config.data.num_shards_per_epoch = 100

save_cfg_dir = Path("/tmp/debug_loss/experiment_cfg")
save_cfg_dir.mkdir(parents=True, exist_ok=True)
pipeline = Gr00tN1d7Pipeline(config, save_cfg_dir)

print("Loading model (with fix)...")
model = pipeline._create_model()

# 1. Check FTP weights
ftp = model.action_head.tactile_encoder
nan_params = sum(torch.isnan(p).sum().item() for p in ftp.parameters())
print(f"\nFTP encoder NaN params: {nan_params}")

# Compare with standalone
ftp_standalone = FTPTactileEncoder(output_dim=1536)
ftp_standalone.load_pretrained("/mnt/netdata/Team/Personal/chenyiyang/zjb/ckpts/ftp_model", sensor_name="SharpaWave", freeze_backbone=False)

for check_name in ["patch_embed.weight", "cls_token", "trunk_blocks.0.attn.qkv.weight", "trunk_norm.weight"]:
    p_loaded = dict(ftp.named_parameters())[check_name]
    p_standalone = dict(ftp_standalone.named_parameters())[check_name]
    match = torch.allclose(p_loaded.float().cpu(), p_standalone.float().cpu(), atol=1e-3)
    print(f"  {check_name}: match={match}")

# 2. Test FTP encoder forward in BF16
model.to(device="cuda", dtype=torch.bfloat16)
model.train()

x = torch.randn(2, 3, 224, 224, device="cuda", dtype=torch.bfloat16) * 0.5 + 0.5
with torch.no_grad():
    out = model.action_head.tactile_encoder(x, func_area_idx=24)
    print(f"\nFTP forward (BF16): NaN={torch.isnan(out).any()}, min={out.min():.4f}, max={out.max():.4f}")

# 3. Full forward pass
processor = AutoProcessor.from_pretrained(
    config.training.start_from_checkpoint,
    modality_configs={embodiment_tag.value: mc},
    use_percentiles=True, model_name=config.model.model_name, model_type="qwen",
    max_action_horizon=40, use_relative_action=True,
    trust_remote_code=True, local_files_only=True,
)
processor.set_tactile_config(num_fingers=5, finger_indices=[0, 1, 2], target_size=224)
dataset_factory = DatasetFactory(config=config)
train_dataset, _ = dataset_factory.build(processor=processor)
collator = processor.collator

it = iter(train_dataset)
samples = [next(it) for _ in range(2)]
batch = collator(samples)

print("\n=== FULL FORWARD PASS ===")
outputs = model(batch["inputs"])
loss = outputs["loss"]
print(f"loss = {loss.item():.6f}")
print(f"loss is NaN? {torch.isnan(loss).item()}")
print(f"action_loss max = {outputs['action_loss'].max().item():.6f}")
print(f"action_mask sum = {outputs['action_mask'].sum().item():.0f}")

if not torch.isnan(loss):
    print(f"\n*** FIX VERIFIED: loss is {loss.item():.6f} (not NaN/0) ***")
else:
    print(f"\n*** FIX NOT WORKING: loss is still NaN ***")

print("\nDone.")
