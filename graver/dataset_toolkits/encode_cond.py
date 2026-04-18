"""
离线编码 DINOv2 条件特征.

对每个实例的每个 renders_cond 视角, 执行:
  1. 加载图片 → crop/resize 518×518 (与在线 ImageConditionedMixin.get_instance 完全一致)
  2. DINOv2 ViT-L/14-reg 前向 → [1369, 1024] patch tokens
  3. 保存为 .pt 文件

目录结构:
  {root}/renders_cond/{sha256}/         ← 原始图片 + transforms.json
  {root}/dinov2_cond/{sha256}/          ← 编码后特征
      000.pt  001.pt  ...               ← 每个视角一个 [1369, 1024] tensor

用法:
  python -m graver.dataset_toolkits.encode_cond \
      --root /path/to/dataset \
      --model dinov2_vitl14_reg \
      --batch_size 16 \
      --num_workers 4
"""

import os
import json
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import pandas as pd


# ---- 与 ImageConditionedMixin.get_instance 完全一致的图片预处理 ----

def load_and_preprocess_image(image_path: str, image_size: int = 518) -> torch.Tensor:
    """
    加载单张 RGBA 图片, 做 bbox crop + resize + alpha premultiply.
    返回 [3, image_size, image_size] float32 tensor, 值域 [0, 1].

    逻辑与 components.py ImageConditionedMixin.get_instance 完全一致.
    """
    image = Image.open(image_path)

    # bbox crop (与在线代码一致)
    alpha = np.array(image.getchannel(3))
    bbox = np.array(alpha).nonzero()
    if bbox[0].size == 0:
        # 全透明图: 直接返回黑图
        return torch.zeros(3, image_size, image_size)
    bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    aug_size_ratio = 1.2
    aug_hsize = hsize * aug_size_ratio
    aug_center = [center[0], center[1]]
    aug_bbox = [
        int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize),
        int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize),
    ]
    image = image.crop(aug_bbox)

    # resize + alpha premultiply
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha = image.getchannel(3)
    image = image.convert('RGB')
    image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    alpha = torch.tensor(np.array(alpha)).float() / 255.0
    image = image * alpha.unsqueeze(0)

    return image


@torch.no_grad()
def encode_dataset(
    root: str,
    model_name: str = 'dinov2_vitl14_reg',
    model_path: str = None,
    image_size: int = 518,
    batch_size: int = 16,
    device: str = 'cuda',
    force: bool = False,
):
    """
    遍历 metadata.csv, 对所有 cond_rendered=True 的实例编码 DINOv2 特征.
    """
    # ---- 加载 metadata ----
    metadata_path = os.path.join(root, 'metadata.csv')
    assert os.path.exists(metadata_path), f"metadata.csv not found at {root}"
    metadata = pd.read_csv(metadata_path)
    metadata['cond_rendered'] = metadata['cond_rendered'].fillna(False).astype(bool)
    instances = metadata[metadata['cond_rendered']]['sha256'].values.tolist()
    print(f"[encode_cond] {len(instances)} instances with cond_rendered=True")

    # ---- 加载 DINOv2 ----
    print(f"[encode_cond] Loading {model_name}...")
    if model_path and os.path.isdir(model_path):
        dinov2 = torch.hub.load(model_path, model_name, source='local', pretrained=True)
    else:
        dinov2 = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
    dinov2 = dinov2.eval().to(device)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # ---- 遍历实例 ----
    output_base = os.path.join(root, 'dinov2_cond')
    skipped = 0
    encoded = 0
    errors = 0

    for sha256 in tqdm(instances, desc="Encoding DINOv2"):
        cond_dir = os.path.join(root, 'renders_cond', sha256)
        out_dir = os.path.join(output_base, sha256)
        transforms_path = os.path.join(cond_dir, 'transforms.json')

        if not os.path.exists(transforms_path):
            errors += 1
            continue

        with open(transforms_path) as f:
            meta = json.load(f)
        frames = meta['frames']
        n_views = len(frames)

        # 检查是否已编码完成
        if not force and os.path.isdir(out_dir):
            existing = [f for f in os.listdir(out_dir) if f.endswith('.pt')]
            if len(existing) >= n_views:
                skipped += 1
                continue

        os.makedirs(out_dir, exist_ok=True)

        # ---- 加载所有视角图片 ----
        images = []
        valid_indices = []
        for i, frame in enumerate(frames):
            image_path = os.path.join(cond_dir, frame['file_path'])
            if not os.path.exists(image_path):
                continue
            try:
                img = load_and_preprocess_image(image_path, image_size)
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                print(f"  [WARN] {sha256} view {i}: {e}")

        if not images:
            errors += 1
            continue

        # ---- 分批编码 ----
        images_tensor = torch.stack(images)  # [V, 3, 518, 518]

        for start in range(0, len(images_tensor), batch_size):
            end = min(start + batch_size, len(images_tensor))
            batch = images_tensor[start:end].to(device)
            batch = normalize(batch)

            features = dinov2(batch, is_training=True)['x_prenorm']
            patchtokens = F.layer_norm(features, features.shape[-1:])
            # patchtokens: [B, 1369, 1024]

            for j in range(end - start):
                view_idx = valid_indices[start + j]
                out_path = os.path.join(out_dir, f'{view_idx:03d}.pt')
                torch.save(patchtokens[j].cpu(), out_path)

        encoded += 1

    print(f"[encode_cond] Done: encoded={encoded}, skipped={skipped}, errors={errors}")

    # ---- 更新 metadata ----
    col = 'dinov2_cond_encoded'
    if col not in metadata.columns:
        metadata[col] = False

    for sha256 in instances:
        out_dir = os.path.join(output_base, sha256)
        if os.path.isdir(out_dir):
            n_pt = len([f for f in os.listdir(out_dir) if f.endswith('.pt')])
            metadata.loc[metadata['sha256'] == sha256, col] = n_pt > 0

    metadata.to_csv(metadata_path, index=False)
    print(f"[encode_cond] Updated {col} column in metadata.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Offline DINOv2 feature encoding")
    parser.add_argument('--root', type=str, required=True,
                        help='Dataset root (contains metadata.csv, renders_cond/)')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='DINOv2 model name')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Local path to dinov2 hub repo (optional)')
    parser.add_argument('--image_size', type=int, default=518)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--force', action='store_true',
                        help='Re-encode even if .pt files exist')
    args = parser.parse_args()

    encode_dataset(
        root=args.root,
        model_name=args.model,
        model_path=args.model_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        force=args.force,
    )
