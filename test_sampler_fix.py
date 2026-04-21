"""Quick test: load old gt_clip_max=0.5 weights, run inference with fixed sampler."""
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

from graver import models as model_registry
from graver.datasets.cascaded_feats import ImageConditionedCascadedBlockFeats
from graver.dataset_toolkits.mesh2block import BLOCK_DIM
from graver.modules.sparse.basic import SparseTensor
from graver.pipelines.samplers import FlowGuidanceIntervalSampler
from graver.datasets.block_feats import BlockFeats
from graver.trainers.flow_matching.mixins.image_conditioned import (
    ImageConditionedMixin as ImageCondHelper,
)

# --- Config ---
CKPT_DIR = "ckpt/cascaded_v5_full"
CONFIG = "ckpt/cascaded_v5_full/config.json"
STEP = 60000
EMA = 0.9999
GT_CLIP_MAX = 0.5
NUM_SAMPLES = 4
OUTPUT_DIR = "eval_sampler_fix"
DEVICE = "cuda:0"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load config ---
with open(CONFIG) as f:
    cfg = json.load(f)

# --- Load model ---
model_cfg = cfg["models"]["denoiser"]
model = getattr(model_registry, model_cfg["name"])(**model_cfg["args"])
ckpt_path = f"{CKPT_DIR}/ckpts/denoiser_ema{EMA}_step{STEP:07d}.pt"
print(f"Loading {ckpt_path}")
sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing:
    print(f"  MISSING: {missing[:5]}")
if unexpected:
    print(f"  UNEXPECTED: {unexpected[:5]}")
if not missing and not unexpected:
    print(f"  All {len(sd)} keys loaded")
model.eval().to(DEVICE)

# --- Load dataset ---
ds = ImageConditionedCascadedBlockFeats(
    roots=["/cfs/yizhao/TRAIN"], max_block_num=12000, max_samples=100,
    require_pred_mask=False,
)

# --- Image encoder (same as eval.py) ---
class ImageEncoder:
    def __init__(self, device):
        self.device = device
        self.helper = ImageCondHelper(image_cond_model='dinov2_vitl14_reg')
        self.helper._init_image_cond_model()
        self.helper.image_cond_model['model'] = (
            self.helper.image_cond_model['model'].to(device)
        )

    @torch.no_grad()
    def encode_tensor(self, img_tensor):
        """img_tensor: [3, H, W] float32 in [0, 1]"""
        img = img_tensor.unsqueeze(0).to(self.device)
        img = self.helper.image_cond_model['transform'](img)
        feats = self.helper.image_cond_model['model'](img, is_training=True)['x_prenorm']
        return F.layer_norm(feats, feats.shape[-1:])

print("Loading DINOv2 encoder...")
encoder = ImageEncoder(DEVICE)

# --- Sampler ---
sampler = FlowGuidanceIntervalSampler()

# --- Run inference ---
for i in range(min(NUM_SAMPLES, len(ds))):
    sample = ds[i]

    coords = sample["coords"]  # [T, 3]
    submask = sample["submask"]  # [T, R^3]
    gt_feats = sample["fine_feats"]  # [T, D^3]
    cond_img = sample["cond"]  # [3, H, W] tensor

    T = coords.shape[0]
    print(f"\nSample {i}: T={T}")

    # Build batch coords [T, 4] with batch_idx=0
    batch_coords = torch.cat([
        torch.zeros(T, 1, dtype=torch.int32),
        coords,
    ], dim=1).to(DEVICE)

    # Build dilated voxel mask (same as CascadedFeatsTrainer)
    R = round(submask.shape[1] ** (1.0 / 3.0))
    D = BLOCK_DIM
    scale = D // R
    sub_3d = submask.reshape(T, 1, R, R, R).float()
    voxel_3d = F.interpolate(sub_3d, scale_factor=float(scale), mode="nearest")
    raw_mask = voxel_3d.reshape(T, D**3)
    vol = raw_mask.reshape(T, 1, D, D, D)
    for _ in range(2):
        vol = F.max_pool3d(vol, kernel_size=3, stride=1, padding=1)
    voxel_mask = vol.reshape(T, D**3).to(DEVICE)

    # Encode condition image
    with torch.no_grad():
        cond = encoder.encode_tensor(cond_img)

    # Noise (noise_scale=1.0 matches old v5_full config)
    bg_fill = GT_CLIP_MAX
    noise_scale = 1.0
    noise_raw = noise_scale * torch.randn(T, model.token_dim, device=DEVICE)
    noise_raw = noise_raw * voxel_mask + (1.0 - voxel_mask) * bg_fill
    noise = SparseTensor(feats=noise_raw, coords=batch_coords)

    # Sample with fixed sampler
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        result = sampler.sample(
            model, noise=noise, cond=cond,
            neg_cond=torch.zeros_like(cond),
            voxel_mask=voxel_mask, bg_fill=bg_fill,
            cfg_strength=3.0, cfg_interval=(0.1, 1.0),
            steps=50, verbose=True,
        )

    # Post-process (same as training: hard-fill + clamp)
    pred = result.samples.feats.float()
    pred = pred * voxel_mask + (1.0 - voxel_mask) * bg_fill
    pred = pred.clamp(0.0, bg_fill)

    # Save mesh + normal
    coords_np = coords.numpy().astype(np.int32)
    pred_np = pred.cpu().numpy().astype(np.float32)
    mesh_path = os.path.join(OUTPUT_DIR, f"sample_{i:03d}.ply")
    normal_path = os.path.join(OUTPUT_DIR, f"sample_{i:03d}_normal.jpg")

    try:
        BlockFeats.tokens_to_mesh(coords_np, pred_np, mesh_path, verbose=True)
        if os.path.exists(mesh_path):
            BlockFeats.render_normal_grid(
                mesh_path, normal_path, resolution=1024, radius=1.75, verbose=True,
            )
            print(f"  Saved {mesh_path} + {normal_path}")
        else:
            print(f"  Mesh generation failed")
    except Exception as e:
        print(f"  Error: {e}")

    # Also save GT mesh for comparison
    gt_mesh = os.path.join(OUTPUT_DIR, f"sample_{i:03d}_gt.ply")
    gt_normal = os.path.join(OUTPUT_DIR, f"sample_{i:03d}_gt_normal.jpg")
    try:
        BlockFeats.tokens_to_mesh(
            coords_np, gt_feats.numpy().astype(np.float32),
            gt_mesh, verbose=True,
        )
        if os.path.exists(gt_mesh):
            BlockFeats.render_normal_grid(
                gt_mesh, gt_normal, resolution=1024, radius=1.75, verbose=True,
            )
    except Exception as e:
        print(f"  GT mesh error: {e}")

    torch.cuda.empty_cache()

print("\nDone! Check", OUTPUT_DIR)
