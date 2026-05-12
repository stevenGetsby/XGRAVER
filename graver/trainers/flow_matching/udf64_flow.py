"""
UDF64 Flow Matching Trainer.

Flow matching on 64-dim min-pooled UDF (4³ resolution).
At snapshot time, threshold the ODE output to extract a binary mask,
then compute IoU / recall / precision vs GT submask.
No normal map rendering.
"""
from typing import *
import os
import functools

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from ...modules import sparse as sp
from ...pipelines import samplers
from ...utils.data_utils import cycle, BalancedResumableSampler
from ...dataset_toolkits.mesh2block import MC_THRESHOLD
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import ImageConditionedMixin


class UDF64FlowTrainer(FlowMatchingTrainer):
    """
    Flow matching on 64-dim min-pooled UDF.
    Mask is extracted from ODE output by thresholding.
    """

    def __init__(
        self,
        *args,
        surface_weight: float = 4.0,
        noise_scale: float = 1.0,
        mask_threshold: float = 0.4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.surface_weight = surface_weight
        self.noise_scale = noise_scale
        self.mask_threshold = mask_threshold
        self.voxel = MC_THRESHOLD
        print(f"[UDF64FlowTrainer] surface_weight={surface_weight}, "
              f"noise_scale={noise_scale}, mask_threshold={mask_threshold}")

    # ── Dataloader ──

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(
            self.dataset, shuffle=True, batch_size=self.batch_size_per_gpu)
        num_gpus = max(torch.cuda.device_count(), 1)
        num_workers = max(1, min(os.cpu_count() // num_gpus, 8))
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size_per_gpu,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=functools.partial(
                self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler)
        self.data_iterator = cycle(self.dataloader)

    # ── SparseTensor helpers ──

    @staticmethod
    def _expand_t(t, layout, T):
        counts = torch.tensor(
            [sl.stop - sl.start for sl in layout], device=t.device, dtype=torch.long)
        return t.repeat_interleave(counts).unsqueeze(1)

    def _sp_diffuse(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0.feats) * self.noise_scale
        t_tok = self._expand_t(t, x_0.layout, x_0.feats.shape[0])
        x_t_feats = t_tok * x_0.feats + (1 - t_tok) * noise
        return x_0.replace(x_t_feats)

    # ── Training losses ──

    def training_losses(self, x_0=None, submask_64=None, cond=None, **kwargs):
        assert x_0 is not None
        B, device = len(x_0.layout), x_0.device
        cond = self.get_cond(cond, **kwargs)

        gt = x_0.feats  # [T, 64] UDF in [0, 1]

        # Sample t and diffuse
        t = self.sample_t(B).to(device).float()
        noise = torch.randn_like(gt) * self.noise_scale
        x_t = self._sp_diffuse(x_0, t, noise)

        # Model prediction
        pred = self.training_models['denoiser'](x_t, t, cond)

        # v-loss: ||pred - gt||² / (1-t)²
        t_tok = self._expand_t(t, x_0.layout, gt.shape[0])
        residual = pred.feats - gt
        v_loss_raw = (residual / (1 - t_tok).clamp(min=0.05)) ** 2

        # Surface weighting: UDF <= 0.4 (SURFACE_THRESHOLD) = surface
        surface_mask = (gt <= self.mask_threshold).float()
        weighted = (1.0 + (self.surface_weight - 1.0) * surface_mask) * v_loss_raw
        flow_loss = torch.stack([weighted[sl].mean() for sl in x_0.layout]).mean()
        loss = flow_loss

        # Mask metrics from UDF threshold
        with torch.no_grad():
            pred_mask = (gt < self.mask_threshold).float()  # from GT UDF for train metric
            pred_mask_model = (pred.feats.clamp(0, 1) < self.mask_threshold).float()
            if submask_64 is not None:
                gt_mask = submask_64.to(device)
            else:
                gt_mask = pred_mask  # fallback

            tp = (pred_mask_model * gt_mask).sum()
            fp = (pred_mask_model * (1 - gt_mask)).sum()
            fn = ((1 - pred_mask_model) * gt_mask).sum()
            iou = (tp / (tp + fp + fn).clamp(min=1)).item()
            prec = (tp / (tp + fp).clamp(min=1)).item()
            rec = (tp / (tp + fn).clamp(min=1)).item()

            # UDF L1 on surface band
            surface_l1 = (residual.abs() * surface_mask).sum() / surface_mask.sum().clamp(min=1)

        terms = edict(
            loss=loss, flow_loss=flow_loss,
            surface_l1=surface_l1.item(),
            mask_iou=iou, mask_prec=prec, mask_rec=rec,
        )
        return terms, {}

    # ── Snapshot: ODE → mask metrics ──

    def snapshot(self, suffix=None, num_samples=4, batch_size=1,
                 steps=50, verbose=False, **kwargs):
        suffix = suffix or f'step{self.step:07d}'
        snapshot_dir = os.path.join(self.output_dir, 'samples', suffix)
        if self.rank == 0:
            os.makedirs(snapshot_dir, exist_ok=True)
            print(f'\n[Snapshot] Step {self.step}: {num_samples} ODE samples → mask metrics')

        states = {n: m.training for n, m in self.models.items()}
        for m in self.models.values():
            m.eval()

        try:
            ds = self.test_dataset if hasattr(self, 'test_dataset') and self.test_dataset else self.dataset
            per_rank = int(np.ceil(num_samples / self.world_size))
            my_s = self.rank * per_rank
            my_e = min((self.rank + 1) * per_rank, num_samples)
            indices = torch.randperm(len(ds))[:num_samples][my_s:my_e]

            tp, fp, fn = 0., 0., 0.
            udf_l1_sum, udf_count = 0., 0.

            sampler = self.get_sampler()

            for li, idx in enumerate(indices):
                idx = int(idx)
                data = ds[idx]

                cond_img = data['cond'].unsqueeze(0).cuda()
                cond = self.encode_image(cond_img)

                coords_int = data['coords']
                batch_coords = torch.cat([
                    torch.zeros(coords_int.shape[0], 1, dtype=torch.int32),
                    coords_int,
                ], 1).cuda()

                model = self.models['denoiser']
                if hasattr(model, 'module'):
                    model = model.module
                token_dim = model.token_dim
                T = batch_coords.shape[0]

                # Initial noise
                noise = torch.randn(T, token_dim, device='cuda') * self.noise_scale
                x_init = sp.SparseTensor(feats=noise, coords=batch_coords)

                # ODE solve
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    result = sampler.sample(
                        model=self.models['denoiser'],
                        noise=x_init,
                        cond=cond,
                        steps=steps,
                    )

                pred_sp = result.samples
                pred_udf = pred_sp.feats.float().clamp(0, 1)  # [T, 64]
                pred_mask = (pred_udf < self.mask_threshold).float()

                gt_mask = data['submask_64'].cuda()
                gt_udf = data['udf_64'].cuda()

                # Mask metrics
                tp += (pred_mask * gt_mask).sum().item()
                fp += (pred_mask * (1 - gt_mask)).sum().item()
                fn += ((1 - pred_mask) * gt_mask).sum().item()

                # UDF L1 on GT surface
                surface = (gt_udf < self.mask_threshold)
                if surface.any():
                    udf_l1_sum += (pred_udf[surface] - gt_udf[surface]).abs().sum().item()
                    udf_count += surface.sum().item()

                torch.cuda.empty_cache()

            # All-reduce
            if self.world_size > 1:
                stats = torch.tensor([tp, fp, fn, udf_l1_sum, udf_count], device='cuda')
                dist.all_reduce(stats)
                tp, fp, fn, udf_l1_sum, udf_count = stats.tolist()
                dist.barrier()

            if self.rank == 0:
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                iou = tp / max(tp + fp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                udf_l1 = udf_l1_sum / max(udf_count, 1)

                metrics = {
                    'iou': iou, 'precision': prec, 'recall': rec, 'f1': f1,
                    'udf_surface_l1': udf_l1,
                }
                with open(os.path.join(snapshot_dir, 'metrics.txt'), 'w') as f:
                    for k, v in metrics.items():
                        line = f'{k}: {v:.4f}'
                        print(f'  {line}')
                        f.write(line + '\n')
                print('  Done.')
        finally:
            for n, m in self.models.items():
                m.train(states[n])


class ImageConditionedUDF64FlowTrainer(ImageConditionedMixin, UDF64FlowTrainer):
    pass
