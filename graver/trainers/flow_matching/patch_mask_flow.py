"""
Patch2 -> Patch4 mask flow matching.

Oracle experiment: derive a 2^3 support mask from the GT 4^3 mask,
then train a conditional flow to refine that support into the 4^3 mask.
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
from ...utils.data_utils import cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.image_conditioned import ImageConditionedMixin


class Patch2ToPatch4FlowTrainer(FlowMatchingTrainer):
    """Conditional flow: GT patch2 support -> generated patch4 mask."""

    def __init__(
        self,
        *args,
        surface_weight: float = 4.0,
        support_weight: float = 1.0,
        outside_weight: float = 0.05,
        noise_scale: float = 1.0,
        mask_threshold: float = 0.5,
        use_heun: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.surface_weight = surface_weight
        self.support_weight = support_weight
        self.outside_weight = outside_weight
        self.noise_scale = noise_scale
        self.mask_threshold = mask_threshold
        self.use_heun = use_heun
        print(
            f"[Patch2ToPatch4FlowTrainer] surface_weight={surface_weight}, "
            f"support_weight={support_weight}, outside_weight={outside_weight}, "
            f"noise_scale={noise_scale}, mask_threshold={mask_threshold}, "
            f"use_heun={use_heun}"
        )

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

    @staticmethod
    def _expand_t(t: torch.Tensor, layout, token_count: int) -> torch.Tensor:
        counts = torch.tensor(
            [sl.stop - sl.start for sl in layout], device=t.device, dtype=torch.long)
        return t.repeat_interleave(counts).unsqueeze(1)

    @staticmethod
    def _patch4_to_patch2(mask4: torch.Tensor) -> torch.Tensor:
        vol = mask4.reshape(mask4.shape[0], 1, 4, 4, 4)
        patch2 = F.max_pool3d(vol, kernel_size=2, stride=2)
        return patch2.reshape(mask4.shape[0], 8)

    @staticmethod
    def _patch2_to_support4(patch2: torch.Tensor) -> torch.Tensor:
        vol = patch2.reshape(patch2.shape[0], 1, 2, 2, 2)
        support4 = F.interpolate(vol, scale_factor=2, mode='nearest')
        return support4.reshape(patch2.shape[0], 64)

    def _diffuse(self, x_0: sp.SparseTensor, t: torch.Tensor,
                 noise: torch.Tensor) -> sp.SparseTensor:
        t_tok = self._expand_t(t, x_0.layout, x_0.feats.shape[0])
        x_t_feats = t_tok * x_0.feats + (1.0 - t_tok) * noise
        return x_0.replace(x_t_feats)

    def training_losses(self, x_0=None, cond=None, **kwargs):
        assert x_0 is not None
        B, device = len(x_0.layout), x_0.device
        cond = self.get_cond(cond, **kwargs)

        gt4 = x_0.feats.float()
        patch2 = self._patch4_to_patch2(gt4)
        support4 = self._patch2_to_support4(patch2)

        t = self.sample_t(B).to(device).float()
        noise = torch.randn_like(gt4) * self.noise_scale
        noise = noise * support4
        x_t = self._diffuse(x_0.replace(gt4), t, noise)
        x_t = x_t.replace(x_t.feats * support4)

        pred = self.training_models['denoiser'](x_t, t, cond, submask=patch2)

        t_tok = self._expand_t(t, x_0.layout, gt4.shape[0])
        diff = ((pred.feats.float() - gt4) / (1.0 - t_tok).clamp(min=0.05)).pow(2)

        active_weight = self.outside_weight + (self.support_weight - self.outside_weight) * support4
        pos_weight = 1.0 + (self.surface_weight - 1.0) * gt4
        weighted = diff * active_weight * pos_weight
        flow_loss = torch.stack([weighted[sl].mean() for sl in x_0.layout]).mean()

        with torch.no_grad():
            pred_score = pred.feats.float() * support4
            pred_mask = (pred_score > self.mask_threshold).float()
            gt_mask = gt4
            tp = (pred_mask * gt_mask).sum()
            fp = (pred_mask * (1.0 - gt_mask)).sum()
            fn = ((1.0 - pred_mask) * gt_mask).sum()
            iou = (tp / (tp + fp + fn).clamp(min=1)).item()
            prec = (tp / (tp + fp).clamp(min=1)).item()
            rec = (tp / (tp + fn).clamp(min=1)).item()
            support_pos = support4.mean().item()
            pred_pos = pred_mask.mean().item()
            gt_pos = gt_mask.mean().item()

        terms = edict(
            loss=flow_loss,
            flow_loss=flow_loss,
            train_iou=iou,
            train_prec=prec,
            train_rec=rec,
            support_pos=support_pos,
            pred_pos=pred_pos,
            gt_pos=gt_pos,
        )
        return terms, {}

    def snapshot(self, suffix=None, num_samples=32, batch_size=1,
                 steps=50, verbose=False, **kwargs):
        suffix = suffix or f'step{self.step:07d}'
        snapshot_dir = os.path.join(self.output_dir, 'samples', suffix)
        if self.rank == 0:
            os.makedirs(snapshot_dir, exist_ok=True)
            print(f'\n[Snapshot] Step {self.step}: patch2->patch4 flow metrics')

        states = {n: m.training for n, m in self.models.items()}
        for m in self.models.values():
            m.eval()

        try:
            ds = self.test_dataset if hasattr(self, 'test_dataset') and self.test_dataset else self.dataset
            per_rank = int(np.ceil(num_samples / self.world_size))
            my_s = self.rank * per_rank
            my_e = min((self.rank + 1) * per_rank, num_samples)
            indices = torch.randperm(len(ds))[:num_samples][my_s:my_e]

            tp = fp = fn = 0.0
            support_tp = support_fp = support_fn = 0.0
            sampler = self.get_sampler()

            for idx in indices:
                data = ds[int(idx)]
                cond_img = data['cond'].unsqueeze(0).cuda()
                cond = self.encode_image(cond_img)

                coords = torch.cat([
                    torch.zeros(data['coords'].shape[0], 1, dtype=torch.int32),
                    data['coords'],
                ], dim=1).cuda()
                gt4 = data['submask'].cuda().float()
                patch2 = self._patch4_to_patch2(gt4)
                support4 = self._patch2_to_support4(patch2)

                noise = torch.randn_like(gt4) * self.noise_scale * support4
                x_init = sp.SparseTensor(feats=noise, coords=coords)

                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    result = sampler.sample(
                        model=self.models['denoiser'],
                        noise=x_init,
                        cond=cond,
                        steps=steps,
                        verbose=False,
                        use_heun=self.use_heun,
                        submask=patch2,
                        voxel_mask=support4,
                        bg_fill=0.0,
                    )

                pred_score = result.samples.feats.float() * support4
                pred_mask = (pred_score > self.mask_threshold).float()

                tp += (pred_mask * gt4).sum().item()
                fp += (pred_mask * (1.0 - gt4)).sum().item()
                fn += ((1.0 - pred_mask) * gt4).sum().item()

                support_mask = (support4 > 0.5).float()
                support_tp += (support_mask * gt4).sum().item()
                support_fp += (support_mask * (1.0 - gt4)).sum().item()
                support_fn += ((1.0 - support_mask) * gt4).sum().item()
                torch.cuda.empty_cache()

            if self.world_size > 1:
                stats = torch.tensor(
                    [tp, fp, fn, support_tp, support_fp, support_fn],
                    device='cuda', dtype=torch.float64,
                )
                dist.all_reduce(stats)
                tp, fp, fn, support_tp, support_fp, support_fn = stats.tolist()
                dist.barrier()

            if self.rank == 0:
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                iou = tp / max(tp + fp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                support_prec = support_tp / max(support_tp + support_fp, 1)
                support_rec = support_tp / max(support_tp + support_fn, 1)
                support_iou = support_tp / max(support_tp + support_fp + support_fn, 1)
                metrics = {
                    'iou': iou,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'oracle_support_iou': support_iou,
                    'oracle_support_precision': support_prec,
                    'oracle_support_recall': support_rec,
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


class ImageConditionedPatch2ToPatch4FlowTrainer(ImageConditionedMixin, Patch2ToPatch4FlowTrainer):
    pass