"""
Offline mask encoding: run frozen mask model on all data samples,
inject predicted submask into the existing block npz file under
`{BLOCK_FOLDER}/{sha256}.npz` as the `pred_mask` array.

CLI style mirrors encode_block.py: --root / --device / --rank / --world_size.

Usage (single GPU):
    python encode_mask.py --root /path/to/dataset --device cuda \\
        --mask_config ... --mask_weight ...

Usage (multi-GPU on one node, 8 GPUs):
    for i in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$i python encode_mask.py --root ... \\
            --rank $i --world_size 8 --device cuda \\
            --mask_config ... --mask_weight ... &
    done; wait

Usage (multi-node): launch with the appropriate global --rank on each node.

Idempotent: skips samples whose npz already has a `pred_mask` field.
"""
import os
os.umask(0)  # keep parity with encode_block.py (CFS 777 perms)

import argparse
import json
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from graver import models
from graver.modules.sparse.basic import SparseTensor
from graver.dataset_toolkits.mesh2block import BLOCK_FOLDER, COL_PREFIX, SUBMASK_DIM
from graver.trainers.flow_matching.mixins.image_conditioned import (
    ImageConditionedMixin as ImageCondHelper,
)


# ----------------------------------------------------------------------
# Model / encoder
# ----------------------------------------------------------------------

def load_mask_model(config_path, weight_path, device='cuda'):
    with open(config_path) as f:
        cfg = json.load(f)
    model_cfg = cfg['models']['denoiser']
    model = getattr(models, model_cfg['name'])(**model_cfg['args'])
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    return model, n


class ImageEncoder:
    def __init__(self, device='cuda'):
        self.device = device
        self.helper = ImageCondHelper(image_cond_model='dinov2_vitl14_reg')
        self.helper._init_image_cond_model()
        self.helper.image_cond_model['model'] = (
            self.helper.image_cond_model['model'].to(device)
        )

    @torch.no_grad()
    def encode(self, image_pil):
        img = np.array(image_pil.convert('RGB')).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img = self.helper.image_cond_model['transform'](img)
        feats = self.helper.image_cond_model['model'](img, is_training=True)['x_prenorm']
        return F.layer_norm(feats, feats.shape[-1:])


@torch.no_grad()
def predict_mask(model, cond, coords, device='cuda', threshold=0.4):
    """Run mask model: coords + cond -> binary submask [T, SUBMASK_DIM]."""
    T = coords.shape[0]
    batch_coords = torch.cat([
        torch.zeros(T, 1, device=device, dtype=torch.int32),
        coords.to(device),
    ], dim=1)

    dummy = SparseTensor(
        feats=torch.zeros(T, model.token_dim, device=device),
        coords=batch_coords,
    )
    t = torch.zeros(1, device=device)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        pred = model(dummy, t, cond)

    return (torch.sigmoid(pred.feats.float()) > threshold).float()


# ----------------------------------------------------------------------
# npz I/O: add `pred_mask` field atomically
# ----------------------------------------------------------------------

def npz_has_pred_mask(npz_path: str) -> bool:
    if not os.path.exists(npz_path):
        return False
    try:
        with np.load(npz_path) as data:
            return 'pred_mask' in data.files
    except Exception:
        return False


def write_pred_mask_into_npz(npz_path: str, pred_mask: np.ndarray):
    """Load existing arrays, add pred_mask, atomic-write back.

    Writes through a file object (not a path string) so that
    `np.savez_compressed` never auto-appends another `.npz` to the name.
    """
    with np.load(npz_path) as data:
        arrays = {k: data[k] for k in data.files}
    arrays['pred_mask'] = pred_mask.astype(np.float16)

    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(npz_path) + '.tmp.',
        dir=os.path.dirname(npz_path),
        suffix='.npz',
    )
    os.close(tmp_fd)
    try:
        with open(tmp_path, 'wb') as fp:
            np.savez_compressed(fp, **arrays)
        # Guard: final path must be exactly `npz_path` (single .npz ext).
        # If any legacy/buggy path produced `{npz_path}.npz`, drop it too.
        stray = npz_path + '.npz'
        if os.path.exists(stray) and stray != tmp_path:
            try:
                os.remove(stray)
            except OSError:
                pass
        os.replace(tmp_path, npz_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Encode pred mask into block npz')
    # CLI aligned with encode_block.py
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Max samples to process (0 = all)')
    parser.add_argument('--max_block_num', type=int, default=15000)
    parser.add_argument('--instances', type=str, default=None,
                        help='Path to txt of sha256s, or comma-separated list')
    # Mask-specific
    parser.add_argument('--mask_config', type=str, required=True)
    parser.add_argument('--mask_weight', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--image_size', type=int, default=518)
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if npz already has pred_mask.')
    opt = parser.parse_args()

    # Device setup (one device per process, same as encode_block worker)
    if opt.device.startswith('cuda') and torch.cuda.is_available():
        if ':' in opt.device:
            torch.cuda.set_device(opt.device)
        device = opt.device
    else:
        device = 'cpu'

    print(f'[encode_mask rank={opt.rank}/{opt.world_size}] device={device}',
          flush=True)

    # Load and filter metadata (identical on every rank -> deterministic shard)
    meta_path = os.path.join(opt.root, 'metadata.csv')
    metadata = pd.read_csv(meta_path)

    if opt.instances:
        ids = (open(opt.instances).read().splitlines()
               if os.path.exists(opt.instances)
               else opt.instances.split(','))
        metadata = metadata[metadata['sha256'].isin(ids)]

    metadata = metadata[metadata[f'{COL_PREFIX}_block_status'] == 'success']
    metadata = metadata[metadata[f'{COL_PREFIX}_num_blocks'] <= opt.max_block_num]
    metadata = metadata[metadata['cond_rendered'].fillna(False).astype(bool)]
    metadata = metadata.reset_index(drop=True)

    # Contiguous shard, identical to encode_block.py
    total = len(metadata)
    metadata = metadata.iloc[total * opt.rank // opt.world_size:
                             total * (opt.rank + 1) // opt.world_size]

    if opt.max_samples > 0:
        metadata = metadata.iloc[:opt.max_samples]

    metadata = metadata.reset_index(drop=True)
    n_shard = len(metadata)
    print(f'[encode_mask rank={opt.rank}] shard size: {n_shard} '
          f'(global total: {total})', flush=True)

    # Split done / todo
    tasks = []
    skipped = 0
    for i in range(n_shard):
        sha = metadata.at[i, 'sha256']
        npz_path = os.path.join(opt.root, BLOCK_FOLDER, f'{sha}.npz')
        if not os.path.exists(npz_path):
            continue
        if (not opt.force) and npz_has_pred_mask(npz_path):
            skipped += 1
            continue
        tasks.append((sha, npz_path))

    print(f'[encode_mask rank={opt.rank}] {skipped} already done, '
          f'{len(tasks)} to process', flush=True)

    if not tasks:
        return

    # Load models on this rank's device
    mask_model, n_params = load_mask_model(opt.mask_config, opt.mask_weight, device)
    print(f'[encode_mask rank={opt.rank}] mask model: {n_params:.1f}M params',
          flush=True)
    encoder = ImageEncoder(device)

    from PIL import Image
    t0 = time.time()
    ok = 0
    fail = 0

    pbar = tqdm(tasks, desc=f'rank{opt.rank}', position=opt.rank,
                leave=(opt.rank == 0))
    for sha256, npz_path in pbar:
        try:
            if (not opt.force) and npz_has_pred_mask(npz_path):
                continue

            with np.load(npz_path) as data:
                coords = torch.from_numpy(data['coords']).int()

            image_root = os.path.join(opt.root, 'renders_cond', sha256)
            transforms_path = os.path.join(image_root, 'transforms.json')
            if not os.path.exists(transforms_path):
                fail += 1
                continue
            with open(transforms_path) as f:
                transforms = json.load(f)
            frame = transforms['frames'][0]
            image_path = os.path.join(image_root, frame['file_path'])
            image = Image.open(image_path)

            alpha = np.array(image.getchannel(3))
            nz = alpha.nonzero()
            if nz[0].size == 0:
                h, w = alpha.shape
                bbox = [0, 0, w - 1, h - 1]
            else:
                bbox = [nz[1].min(), nz[0].min(), nz[1].max(), nz[0].max()]
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2 * 1.2
            crop = [int(cx - hsize), int(cy - hsize),
                    int(cx + hsize), int(cy + hsize)]
            image = image.crop(crop).resize(
                (opt.image_size, opt.image_size), Image.LANCZOS)
            alpha_ch = image.getchannel(3)
            image = image.convert('RGB')
            img_t = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            alpha_t = torch.tensor(np.array(alpha_ch)).float() / 255.0
            cond_pil = Image.fromarray(
                (img_t * alpha_t.unsqueeze(0)).permute(1, 2, 0).mul(255)
                .clamp(0, 255).byte().numpy()
            )

            cond = encoder.encode(cond_pil)

            pred_submask = predict_mask(
                mask_model, cond, coords,
                device=device, threshold=opt.threshold,
            ).cpu().numpy().astype(np.float16)

            write_pred_mask_into_npz(npz_path, pred_submask)
            ok += 1
        except Exception as e:
            fail += 1
            tqdm.write(f'[rank{opt.rank}] {sha256}: {type(e).__name__}: {e}')
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (ok + fail) % 50 == 0:
            elapsed = time.time() - t0
            speed = (ok + fail) / max(elapsed, 1e-6)
            pbar.set_postfix(ok=ok, fail=fail, sps=f'{speed:.2f}')

    elapsed = time.time() - t0
    print(f'[rank{opt.rank}] Done: ok={ok}, fail={fail}, '
          f'elapsed={elapsed:.0f}s '
          f'({(ok+fail)/max(elapsed,1e-6):.2f} samples/s)', flush=True)


if __name__ == '__main__':
    main()
