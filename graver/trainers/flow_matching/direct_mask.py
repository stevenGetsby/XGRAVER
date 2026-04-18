"""
Direct Mask Prediction Trainer v3.
Uses DirectMaskModel (Fourier coords + CLS mod).
BCE loss with pos_weight.
"""
from typing import *
import os
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from ...datasets.block_feats import BlockFeats
from ...modules import sparse as sp
from ...utils.data_utils import cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.image_conditioned import ImageConditionedMixin


class DirectMaskTrainer(FlowMatchingTrainer):

    def __init__(self, *args, recall_weight: float = 3.0,
                 surface_threshold: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.recall_weight = recall_weight
        self.surface_threshold = surface_threshold
        print(f"[DirectMaskTrainer] recall_weight={recall_weight}, "
              f"surface_threshold={surface_threshold}")

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(
            self.dataset, shuffle=True, batch_size=self.batch_size_per_gpu)
        num_gpus = max(torch.cuda.device_count(), 1)
        num_workers = max(1, min(os.cpu_count() // num_gpus, 16))
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size_per_gpu,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler)
        self.data_iterator = cycle(self.dataloader)

    def training_losses(self, x_0=None, cond=None, **kwargs):
        assert x_0 is not None
        B, device = len(x_0.layout), x_0.device
        cond = self.get_cond(cond, **kwargs)

        # Dummy input: model uses coords internally via DirectMaskModel
        # x_0.feats = GT submask, x_0.coords = block coords
        dummy_input = x_0.replace(torch.zeros_like(x_0.feats))
        t = torch.zeros(B, device=device)

        pred = self.training_models['denoiser'](dummy_input, t, cond)

        gt = x_0.feats  # binary: 0 or 1
        pos_weight = torch.tensor(self.recall_weight, device=device)
        bce = F.binary_cross_entropy_with_logits(
            pred.feats, gt, pos_weight=pos_weight, reduction='none')

        loss = torch.stack([bce[sl].mean() for sl in x_0.layout]).mean()

        with torch.no_grad():
            pred_prob = torch.sigmoid(pred.feats)
            pb = (pred_prob > self.surface_threshold).float()
            tp = (pb * gt).sum()
            fp = (pb * (1 - gt)).sum()
            fn = ((1 - pb) * gt).sum()
            iou = (tp / (tp + fp + fn).clamp(min=1)).item()
            prec = (tp / (tp + fp).clamp(min=1)).item()
            rec = (tp / (tp + fn).clamp(min=1)).item()

        terms = edict(bce=loss, loss=loss, train_iou=iou,
                      train_prec=prec, train_rec=rec)
        return terms, {}

    # ---- Normal overview grid ----

    @staticmethod
    def _compose_normal_overviews(snapshot_dir, num_samples):
        from PIL import Image
        details_dir = os.path.join(snapshot_dir, 'details')
        for tag, sfx in [('pred', '_normal.jpg'), ('gt', '_gt_normal.jpg')]:
            paths = [os.path.join(details_dir, f'sample_{i:03d}{sfx}') for i in range(num_samples)]
            imgs = []
            for p in paths:
                try: imgs.append(Image.open(p) if os.path.exists(p) else None)
                except Exception: imgs.append(None)
            if not any(imgs): continue
            ref = next(im for im in imgs if im is not None)
            w, h = ref.size
            imgs = [im if im else Image.new('RGB', (w, h), (0, 0, 0)) for im in imgs]
            ncol = min(len(imgs), 4)
            nrow = (len(imgs) + ncol - 1) // ncol
            grid = Image.new('RGB', (w * ncol, h * nrow))
            for i, im in enumerate(imgs):
                r, c = divmod(i, ncol)
                grid.paste(im, (c * w, r * h))
            out = os.path.join(snapshot_dir, f'overview_{tag}.jpg')
            grid.save(out, quality=90)

    def snapshot(self, suffix=None, num_samples=4, batch_size=1, verbose=False):
        from ...dataset_toolkits.mesh2block import BLOCK_FOLDER, BLOCK_DIM

        suffix = suffix or f'step{self.step:07d}'
        snapshot_dir = os.path.join(self.output_dir, 'samples', suffix)
        details_dir = os.path.join(snapshot_dir, 'details')
        if self.rank == 0:
            os.makedirs(details_dir, exist_ok=True)
            print(f'\n[Snapshot] Step {self.step}: {num_samples} direct mask predictions')

        states = {n: m.training for n, m in self.models.items()}
        for m in self.models.values(): m.eval()

        try:
            ds = self.test_dataset if hasattr(self, 'test_dataset') and self.test_dataset else self.dataset
            per_rank = int(np.ceil(num_samples / self.world_size))
            my_s = self.rank * per_rank
            my_e = min((self.rank + 1) * per_rank, num_samples)
            indices = torch.randperm(len(ds))[:num_samples][my_s:my_e]

            tp, fp, fn, total = 0., 0., 0., 0.
            tmps = []
            SUB = 8

            def _up(mask, bd=BLOCK_DIM, sr=SUB):
                T = mask.shape[0]; s = bd // sr
                return F.interpolate(mask.reshape(T, 1, sr, sr, sr),
                                     scale_factor=float(s), mode='nearest').reshape(T, -1)

            for li, idx in enumerate(indices):
                idx = int(idx); gi = my_s + li; name = f'sample_{gi:03d}'
                data = ds[idx]

                # Encode image
                cond_img = data['cond'].unsqueeze(0).cuda()
                cond = self.encode_image(cond_img)

                coords_int = data['coords']  # [T, 3]
                batch_coords = torch.cat([
                    torch.zeros(coords_int.shape[0], 1, dtype=torch.int32), coords_int], 1).cuda()

                model = self.models['denoiser']
                if hasattr(model, 'module'): model = model.module
                token_dim = model.token_dim
                dummy_feats = torch.zeros(batch_coords.shape[0], token_dim, device='cuda')
                dummy_sp = sp.SparseTensor(feats=dummy_feats, coords=batch_coords)
                t = torch.zeros(1, device='cuda')

                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred = self.models['denoiser'](dummy_sp, t, cond)

                pred_mask = (torch.sigmoid(pred.feats.float()) > self.surface_threshold).float()
                gt_mask = data['submask'].cuda()

                # Metrics
                pred_flat = pred_mask.reshape(-1, SUB**3)
                gt_flat = gt_mask.reshape(-1, SUB**3)
                tp += (pred_flat * gt_flat).sum().item()
                fp += (pred_flat * (1 - gt_flat)).sum().item()
                fn += ((1 - pred_flat) * gt_flat).sum().item()
                total += gt_flat.numel()

                # Load GT fine_feats for normal rendering
                root, inst = ds.instances[idx]
                try:
                    with np.load(os.path.join(root, BLOCK_FOLDER, f'{inst}.npz')) as z:
                        coords_np = z['coords'].astype(np.int32)
                        raw = z['fine_feats']
                        feats = raw.astype(np.float32) if raw.dtype == np.float16 else raw.copy()
                except Exception:
                    torch.cuda.empty_cache()
                    continue

                # Render pred and GT normal maps
                for tag, mask_t in [('gt', gt_mask), ('pred', pred_mask)]:
                    vx = (_up(mask_t).cpu().numpy() > 0.5)
                    masked = feats.copy()
                    masked[~vx] = 1.0  # non-surface → UDF far

                    sfx = '_gt_normal.jpg' if tag == 'gt' else '_normal.jpg'
                    ply = os.path.join(details_dir, f'{name}_{tag}_tmp.ply')
                    nrm = os.path.join(details_dir, f'{name}{sfx}')
                    try:
                        torch.cuda.empty_cache()
                        BlockFeats.tokens_to_mesh(coords_np, masked, ply, verbose=False)
                        if os.path.exists(ply):
                            tmps.append(ply)
                            BlockFeats.render_normal_grid(ply, nrm, resolution=512, radius=1.75, verbose=False)
                    except Exception as e:
                        print(f'  [Rank {self.rank}] {tag} render error ({name}): {e}')
                        torch.cuda.empty_cache()

                torch.cuda.empty_cache()

            # Cleanup temp ply files
            for p in tmps:
                try: os.remove(p)
                except OSError: pass

            # All-reduce metrics
            if self.world_size > 1:
                stats = torch.tensor([tp, fp, fn, total], device='cuda')
                dist.all_reduce(stats)
                tp, fp, fn, total = stats.tolist()
                dist.barrier()

            if self.rank == 0:
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                iou = tp / max(tp + fp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                pos_total = max(total, 1)
                metrics = {'iou': iou, 'precision': prec, 'recall': rec, 'f1': f1,
                           'pos_gt': (tp + fn) / pos_total,
                           'pos_pred': (tp + fp) / pos_total}
                with open(os.path.join(snapshot_dir, 'metrics.txt'), 'w') as f:
                    for k, v in metrics.items():
                        line = f'{k}: {v:.4f}'
                        print(f'  {line}')
                        f.write(line + '\n')
                self._compose_normal_overviews(snapshot_dir, num_samples)
                print('  Done.')
        finally:
            for n, m in self.models.items():
                m.train(states[n])


class ImageConditionedDirectMaskTrainer(ImageConditionedMixin, DirectMaskTrainer):
    pass
