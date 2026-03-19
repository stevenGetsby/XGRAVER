"""
Stage 2 trainer: sparse flow matching on per-block submask [0,1].

Standard continuous flow matching (v-loss) on submask values.
"""
from typing import *
import os
import copy
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
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import ImageConditionedMixin


class SparseMaskFlowTrainer(FlowMatchingTrainer):
    """
    Sparse flow matching trainer for per-block submask prediction.
    Submask ∈ [0,1]^512, standard v-loss with recall weighting.
    """

    def __init__(self, *args, recall_weight: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.recall_weight = recall_weight
        print(f"[SparseMaskFlowTrainer] recall_weight={recall_weight}, "
              f"noise_scale={self.noise_scale}")

    # ------------------------------------------------------------------
    # Dataloader (sparse, balanced)
    # ------------------------------------------------------------------

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(
            self.dataset, shuffle=True, batch_size=self.batch_size_per_gpu,
        )
        num_gpus = max(torch.cuda.device_count(), 1)
        num_workers = max(1, min(os.cpu_count() // num_gpus, 16))
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            collate_fn=functools.partial(
                self.dataset.collate_fn, split_size=self.batch_split,
            ),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    # ------------------------------------------------------------------
    # Sparse diffuse / v helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_t(t: torch.Tensor, layout, T: int) -> torch.Tensor:
        counts = torch.tensor(
            [sl.stop - sl.start for sl in layout],
            device=t.device, dtype=torch.long,
        )
        return t.repeat_interleave(counts).unsqueeze(1)

    def diffuse(self, x_0, t, noise=None):
        if isinstance(x_0, sp.SparseTensor):
            if noise is None:
                noise = x_0.replace(torch.randn_like(x_0.feats))
            t_tok = self._expand_t(t, x_0.layout, x_0.feats.shape[0])
            x_t = t_tok * x_0.feats + (1 - t_tok) * noise.feats
            return x_0.replace(x_t)
        return super().diffuse(x_0, t, noise=noise)

    def compute_v_from_x_prediction(self, x_t, x_pred, t):
        if isinstance(x_t, sp.SparseTensor):
            t_tok = self._expand_t(t, x_t.layout, x_t.feats.shape[0])
            v = (x_pred.feats - x_t.feats) / (1 - t_tok).clamp(min=0.05)
            return x_t.replace(v)
        return super().compute_v_from_x_prediction(x_t, x_pred, t)

    # ------------------------------------------------------------------
    # Training losses — MSE v-loss with recall weighting
    # ------------------------------------------------------------------

    def training_losses(self, x_0=None, cond=None, **kwargs) -> Tuple[Dict, Dict]:
        assert x_0 is not None
        B = len(x_0.layout)
        device = x_0.device

        cond = self.get_cond(cond, **kwargs)

        noise_raw = self.noise_scale * torch.randn_like(x_0.feats)
        noise = x_0.replace(noise_raw)
        t = self.sample_t(B).to(device).float()
        x_t = self.diffuse(x_0, t, noise=noise)

        pred = self.training_models['denoiser'](x_t, t, cond)

        # Standard v-loss
        v_target = self.compute_v_from_x_prediction(x_t, x_0, t)
        v_pred = self.compute_v_from_x_prediction(x_t, pred, t)

        # Recall-biased weighting: surface cells (GT=1) get higher weight
        with torch.no_grad():
            w = 1.0 + (self.recall_weight - 1.0) * x_0.feats

        mse = (w * (v_pred.feats - v_target.feats) ** 2).mean()

        # Inline IoU for monitoring (threshold at 0.5)
        with torch.no_grad():
            pred_bin = (pred.feats > 0.5).float()
            gt_bin = (x_0.feats > 0.5).float()
            tp = (pred_bin * gt_bin).sum()
            fp = (pred_bin * (1 - gt_bin)).sum()
            fn = ((1 - pred_bin) * gt_bin).sum()
            train_iou = (tp / (tp + fp + fn).clamp(min=1)).item()

        terms = edict()
        terms["mse"] = mse
        terms["loss"] = mse
        terms["train_iou"] = train_iou

        # Per-bin logging
        t_np = t.cpu().numpy()
        mse_per_b = np.array([
            F.mse_loss(v_pred.feats[sl], v_target.feats[sl]).item()
            for sl in x_0.layout
        ])
        bins = np.digitize(t_np, np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (bins == i).sum():
                terms[f"bin_{i}"] = {"mse": mse_per_b[bins == i].mean()}

        return terms, {}

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=4, batch_size=1, verbose=False):
        if suffix is None:
            suffix = f'step{self.step:07d}'

        n_per_rank = max(1, int(np.ceil(num_samples / self.world_size)))
        local_counts = self._run_snapshot_counts(n_per_rank, batch_size, verbose)

        # All-reduce
        keys = ['tp', 'fp', 'fn', 'sum_gt', 'sum_pred', 'total']
        tensors = {k: torch.tensor(local_counts[k], device='cuda') for k in keys}
        if self.world_size > 1:
            for v in tensors.values():
                dist.all_reduce(v, op=dist.ReduceOp.SUM)
        counts = {k: v.item() for k, v in tensors.items()}

        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        metrics = {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pos_gt': counts['sum_gt'] / max(counts['total'], 1),
            'pos_pred': counts['sum_pred'] / max(counts['total'], 1),
        }

        if self.is_master:
            out_dir = os.path.join(self.output_dir, 'samples', suffix)
            os.makedirs(out_dir, exist_ok=True)
            print(f'\n[Snapshot] {num_samples} samples:')
            with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
                for k, v in metrics.items():
                    line = f'{k}: {v:.4f}'
                    print(f'  {line}')
                    f.write(line + '\n')
            print('  Done.')

    @torch.no_grad()
    def _run_snapshot_counts(self, num_samples, batch_size, verbose):
        snap_ds = (self.test_dataset if hasattr(self, 'test_dataset') and self.test_dataset
                   else copy.deepcopy(self.dataset))
        loader = DataLoader(
            snap_ds, batch_size=batch_size, shuffle=True, num_workers=0,
            collate_fn=snap_ds.collate_fn if hasattr(snap_ds, 'collate_fn') else None,
        )
        sampler = self.get_sampler()

        tp, fp, fn = 0.0, 0.0, 0.0
        sum_gt, sum_pred, total = 0.0, 0.0, 0

        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(loader))
            data = {k: (v.to('cuda') if isinstance(v, sp.SparseTensor)
                        else v[:batch].cuda() if isinstance(v, torch.Tensor)
                        else v)
                    for k, v in data.items()}

            x_0 = data.pop('x_0')
            gt = x_0.feats

            noise = x_0.replace(self.noise_scale * torch.randn_like(x_0.feats))

            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models['denoiser'], noise=noise,
                **args, steps=50, cfg_strength=3.0, verbose=verbose,
            )
            pred = res.samples.feats if hasattr(res.samples, 'feats') else res.samples

            pred_bin = (pred > 0.5).float()
            gt_bin = (gt > 0.5).float()

            tp += (pred_bin * gt_bin).sum().item()
            fp += (pred_bin * (1 - gt_bin)).sum().item()
            fn += ((1 - pred_bin) * gt_bin).sum().item()
            sum_gt += gt_bin.sum().item()
            sum_pred += pred_bin.sum().item()
            total += gt.numel()

        return {'tp': tp, 'fp': fp, 'fn': fn,
                'sum_gt': sum_gt, 'sum_pred': sum_pred, 'total': total}

    def get_sampler(self, **kwargs):
        return samplers.FlowSampler()


# ======================================================================
# CFG / ImageConditioned variants
# ======================================================================

class SparseMaskFlowCFGTrainer(ClassifierFreeGuidanceMixin, SparseMaskFlowTrainer):
    pass


class ImageConditionedSparseMaskFlowCFGTrainer(
    ImageConditionedMixin, SparseMaskFlowCFGTrainer,
):
    def get_sampler(self, **kwargs):
        return samplers.FlowGuidanceIntervalSampler(**kwargs)
