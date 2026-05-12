"""
MaskRefineTrainer: stage-2.5 mask refiner.

Takes a coarse per-block pred_mask (from stage-2) plus image condition and
predicts a refined mask. Loss is BCE vs. GT submask with extra FN penalty
on voxels that the coarse pred_mask missed (gt=1 but pred_mask<0.5), which
is the dominant failure mode at stage-2.
"""
from typing import *

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from .direct_mask import DirectMaskTrainer
from .mixins.image_conditioned import ImageConditionedMixin


class MaskRefineTrainer(DirectMaskTrainer):
    """Refine a coarse pred_mask into a cleaner mask.

    Extra knobs vs. DirectMaskTrainer:
        fn_on_miss_weight: scalar multiplier on the BCE term for voxels where
            gt=1 but the coarse pred_mask < 0.5 (i.e. stage-2 FN). Default 3.0.
    """

    def __init__(self, *args, fn_on_miss_weight: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn_on_miss_weight = float(fn_on_miss_weight)
        print(f"[MaskRefineTrainer] fn_on_miss_weight={fn_on_miss_weight}")

    def training_losses(self, x_0=None, cond=None, pred_submask=None, **kwargs):
        assert x_0 is not None
        assert pred_submask is not None, \
            "MaskRefineTrainer requires 'pred_submask' in the batch (set require_pred_mask=True on the dataset)."
        B, device = len(x_0.layout), x_0.device
        cond = self.get_cond(cond, **kwargs)

        pred_submask = pred_submask.to(device=device, dtype=x_0.feats.dtype)

        # Model input: pred_mask as per-token features (NOT dummy zeros).
        x_in = x_0.replace(pred_submask)
        t = torch.zeros(B, device=device)

        pred_out = self.training_models['denoiser'](x_in, t, cond, return_aux=True)
        if isinstance(pred_out, tuple):
            pred, aux = pred_out
        else:
            pred, aux = pred_out, {}

        gt = x_0.feats  # binary {0, 1}
        pos_weight = torch.tensor(self.recall_weight, device=device)
        bce = F.binary_cross_entropy_with_logits(
            pred.feats, gt, pos_weight=pos_weight, reduction='none')

        # Extra weight on stage-2 FN voxels (gt=1 & pred_submask<0.5)
        if self.fn_on_miss_weight != 1.0:
            miss_fn = ((pred_submask < 0.5) & (gt > 0.5)).float()
            weight = 1.0 + (self.fn_on_miss_weight - 1.0) * miss_fn
            bce = bce * weight

        fine_loss = torch.stack([bce[sl].mean() for sl in x_0.layout]).mean()
        loss = fine_loss

        coarse_loss = None
        if self.coarse_weight > 0 and aux.get('coarse_logits') is not None:
            model = self.training_models['denoiser']
            if hasattr(model, 'module'):
                model = model.module
            coarse_resolution = getattr(model, 'coarse_resolution', self.coarse_resolution)
            gt_coarse = self._pool_to_coarse(gt, coarse_resolution)
            coarse_bce = F.binary_cross_entropy_with_logits(
                aux['coarse_logits'], gt_coarse,
                pos_weight=pos_weight, reduction='none')
            coarse_loss = torch.stack(
                [coarse_bce[sl].mean() for sl in x_0.layout]).mean()
            loss = loss + self.coarse_weight * coarse_loss

        # Metrics: report both coarse (stage-2) and refined IoU/recall so we
        # can directly see the improvement on tensorboard.
        with torch.no_grad():
            pred_prob = torch.sigmoid(pred.feats.float())
            pb = (pred_prob > self.surface_threshold).float()

            tp = (pb * gt).sum()
            fp = (pb * (1 - gt)).sum()
            fn = ((1 - pb) * gt).sum()
            iou = (tp / (tp + fp + fn).clamp(min=1)).item()
            prec = (tp / (tp + fp).clamp(min=1)).item()
            rec = (tp / (tp + fn).clamp(min=1)).item()

            cb = (pred_submask > self.surface_threshold).float()
            ctp = (cb * gt).sum()
            cfp = (cb * (1 - gt)).sum()
            cfn = ((1 - cb) * gt).sum()
            c_iou = (ctp / (ctp + cfp + cfn).clamp(min=1)).item()
            c_rec = (ctp / (ctp + cfn).clamp(min=1)).item()
            c_prec = (ctp / (ctp + cfp).clamp(min=1)).item()

        terms = edict(
            bce=fine_loss, loss=loss,
            train_iou=iou, train_prec=prec, train_rec=rec,
            coarse_iou=c_iou, coarse_prec=c_prec, coarse_rec=c_rec,
            d_iou=iou - c_iou, d_rec=rec - c_rec,
        )
        if coarse_loss is not None:
            terms['coarse_bce'] = coarse_loss
        return terms, {}


class ImageConditionedMaskRefineTrainer(ImageConditionedMixin, MaskRefineTrainer):
    pass
