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
                 surface_threshold: float = 0.5,
                 coarse_weight: float = 0.0,
                 coarse_resolution: int = 4,
                 dice_weight: float = 0.0,
                 focal_gamma: float = 0.0,
                 negative_weight: float = 1.0,
                 density_aux_weight: float = 0.0,
                 fine_aux_weight: float = 0.0,
                 fine_aux_dice_weight: float = 0.0,
                 snapshot_render_normals: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.recall_weight = recall_weight
        self.surface_threshold = surface_threshold
        self.coarse_weight = coarse_weight
        self.coarse_resolution = coarse_resolution
        self.dice_weight = dice_weight
        self.focal_gamma = focal_gamma
        self.negative_weight = negative_weight
        self.density_aux_weight = density_aux_weight
        self.fine_aux_weight = fine_aux_weight
        self.fine_aux_dice_weight = fine_aux_dice_weight
        self.snapshot_render_normals = snapshot_render_normals
        print(f"[DirectMaskTrainer] recall_weight={recall_weight}, "
              f"surface_threshold={surface_threshold}, "
              f"coarse_weight={coarse_weight}, "
              f"dice_weight={dice_weight}, "
              f"focal_gamma={focal_gamma}, "
              f"negative_weight={negative_weight}, "
              f"density_aux_weight={density_aux_weight}, "
              f"fine_aux_weight={fine_aux_weight}, "
              f"fine_aux_dice_weight={fine_aux_dice_weight}")

    @staticmethod
    def _pool_to_coarse(mask: torch.Tensor, coarse_resolution: int) -> torch.Tensor:
        fine_resolution = round(mask.shape[1] ** (1 / 3))
        if fine_resolution ** 3 != mask.shape[1]:
            raise ValueError(f"Invalid mask dim: {mask.shape[1]}")
        if fine_resolution % coarse_resolution != 0:
            raise ValueError(
                f"fine_resolution={fine_resolution} must be divisible by coarse_resolution={coarse_resolution}"
            )

        stride = fine_resolution // coarse_resolution
        vol = mask.reshape(mask.shape[0], 1, fine_resolution, fine_resolution, fine_resolution)
        coarse = F.max_pool3d(vol, kernel_size=stride, stride=stride)
        return coarse.reshape(mask.shape[0], -1)

    @staticmethod
    def _pool_to_density(mask: torch.Tensor, density_resolution: int) -> torch.Tensor:
        fine_resolution = round(mask.shape[1] ** (1 / 3))
        if fine_resolution ** 3 != mask.shape[1]:
            raise ValueError(f"Invalid mask dim: {mask.shape[1]}")
        if fine_resolution == density_resolution:
            return mask
        if fine_resolution % density_resolution != 0:
            raise ValueError(
                f"fine_resolution={fine_resolution} must be divisible by density_resolution={density_resolution}"
            )

        stride = fine_resolution // density_resolution
        vol = mask.reshape(mask.shape[0], 1, fine_resolution, fine_resolution, fine_resolution)
        density = F.avg_pool3d(vol, kernel_size=stride, stride=stride)
        return density.reshape(mask.shape[0], -1)

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(
            self.dataset, shuffle=True, batch_size=self.batch_size_per_gpu)
        num_gpus = max(torch.cuda.device_count(), 1)
        # Cap low to avoid cgroup/RSS pressure on shared hosts (96-worker fleets
        # can SIGKILL). 8 per rank is a reasonable ceiling.
        num_workers = max(1, min(os.cpu_count() // num_gpus, 8))
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size_per_gpu,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None,
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

        model_kwargs = {}
        if 'patch_xy' in kwargs:
            model_kwargs['patch_xy'] = kwargs['patch_xy']
        if 'patch_valid' in kwargs:
            model_kwargs['patch_valid'] = kwargs['patch_valid']

        pred_out = self.training_models['denoiser'](
            dummy_input, t, cond, return_aux=True, **model_kwargs,
        )
        if isinstance(pred_out, tuple):
            pred, aux = pred_out
        else:
            pred, aux = pred_out, {}

        gt = x_0.feats  # binary: 0 or 1
        pos_weight = torch.tensor(self.recall_weight, device=device)
        bce = F.binary_cross_entropy_with_logits(
            pred.feats, gt, pos_weight=pos_weight, reduction='none')
        if self.negative_weight != 1.0:
            class_weight = torch.where(
                gt > 0.5,
                torch.ones_like(gt),
                torch.full_like(gt, self.negative_weight),
            )
            bce = bce * class_weight
        if self.focal_gamma > 0:
            pred_prob_for_loss = torch.sigmoid(pred.feats)
            pt = torch.where(gt > 0.5, pred_prob_for_loss, 1.0 - pred_prob_for_loss)
            bce = bce * (1.0 - pt).clamp(min=1e-4).pow(self.focal_gamma)

        fine_loss = torch.stack([bce[sl].mean() for sl in x_0.layout]).mean()
        loss = fine_loss

        # Dice loss: 1 - 2*TP / (2*TP + FP + FN), per-sample then average
        dice_loss = None
        if self.dice_weight > 0:
            pred_prob = torch.sigmoid(pred.feats)
            dice_per_sample = []
            for sl in x_0.layout:
                p = pred_prob[sl]
                g = gt[sl]
                inter = (p * g).sum()
                union = p.sum() + g.sum()
                dice_per_sample.append(1.0 - (2.0 * inter + 1.0) / (union + 1.0))
            dice_loss = torch.stack(dice_per_sample).mean()
            loss = loss + self.dice_weight * dice_loss

        coarse_loss = None
        if self.coarse_weight > 0 and aux.get('coarse_logits') is not None:
            model = self.training_models['denoiser']
            if hasattr(model, 'module'):
                model = model.module
            coarse_resolution = getattr(model, 'coarse_resolution', self.coarse_resolution)
            gt_coarse = self._pool_to_coarse(gt, coarse_resolution)
            coarse_bce = F.binary_cross_entropy_with_logits(
                aux['coarse_logits'], gt_coarse, pos_weight=pos_weight, reduction='none')
            coarse_loss = torch.stack([coarse_bce[sl].mean() for sl in x_0.layout]).mean()
            loss = loss + self.coarse_weight * coarse_loss

        fine_aux_loss = None
        fine_aux_dice = None
        density_aux_loss = None
        density_aux_mae = None
        density_aux_iou = None
        if self.fine_aux_weight > 0 and aux.get('fine_aux_logits') is not None:
            full_submask = kwargs.get('full_submask', None)
            if full_submask is None:
                raise ValueError(
                    "fine_aux_weight > 0 requires dataset arg return_full_submask=True"
                )
            full_gt = full_submask.to(device=device, dtype=gt.dtype)
            fine_aux_bce = F.binary_cross_entropy_with_logits(
                aux['fine_aux_logits'], full_gt, pos_weight=pos_weight, reduction='none')
            fine_aux_loss = torch.stack(
                [fine_aux_bce[sl].mean() for sl in x_0.layout]).mean()
            loss = loss + self.fine_aux_weight * fine_aux_loss

            if self.fine_aux_dice_weight > 0:
                fine_prob = torch.sigmoid(aux['fine_aux_logits'])
                fine_dice_per_sample = []
                for sl in x_0.layout:
                    p = fine_prob[sl]
                    g = full_gt[sl]
                    inter = (p * g).sum()
                    union = p.sum() + g.sum()
                    fine_dice_per_sample.append(
                        1.0 - (2.0 * inter + 1.0) / (union + 1.0)
                    )
                fine_aux_dice = torch.stack(fine_dice_per_sample).mean()
                loss = loss + self.fine_aux_dice_weight * fine_aux_dice

        if self.density_aux_weight > 0 and aux.get('density_logits') is not None:
            full_submask = kwargs.get('full_submask', None)
            if full_submask is None:
                raise ValueError(
                    "density_aux_weight > 0 requires dataset arg return_full_submask=True"
                )
            model = self.training_models['denoiser']
            if hasattr(model, 'module'):
                model = model.module
            density_resolution = round(model.token_dim ** (1 / 3))
            full_gt = full_submask.to(device=device, dtype=gt.dtype)
            density_gt = self._pool_to_density(full_gt, density_resolution)
            density_bce = F.binary_cross_entropy_with_logits(
                aux['density_logits'], density_gt, reduction='none')
            density_aux_loss = torch.stack(
                [density_bce[sl].mean() for sl in x_0.layout]).mean()
            loss = loss + self.density_aux_weight * density_aux_loss

            with torch.no_grad():
                density_prob = torch.sigmoid(aux['density_logits'].float())
                density_aux_mae = (density_prob - density_gt.float()).abs().mean().item()
                density_pb = (density_prob > self.surface_threshold).float()
                dtp = (density_pb * gt).sum()
                dfp = (density_pb * (1 - gt)).sum()
                dfn = ((1 - density_pb) * gt).sum()
                density_aux_iou = (dtp / (dtp + dfp + dfn).clamp(min=1)).item()

        with torch.no_grad():
            pred_prob = torch.sigmoid(pred.feats)
            pb = (pred_prob > self.surface_threshold).float()
            tp = (pb * gt).sum()
            fp = (pb * (1 - gt)).sum()
            fn = ((1 - pb) * gt).sum()
            iou = (tp / (tp + fp + fn).clamp(min=1)).item()
            prec = (tp / (tp + fp).clamp(min=1)).item()
            rec = (tp / (tp + fn).clamp(min=1)).item()

            coarse_iou = None
            if coarse_loss is not None:
                coarse_gt = self._pool_to_coarse(gt, coarse_resolution)
                coarse_pb = (torch.sigmoid(aux['coarse_logits']) > self.surface_threshold).float()
                ctp = (coarse_pb * coarse_gt).sum()
                cfp = (coarse_pb * (1 - coarse_gt)).sum()
                cfn = ((1 - coarse_pb) * coarse_gt).sum()
                coarse_iou = (ctp / (ctp + cfp + cfn).clamp(min=1)).item()
                coarse_prec = (ctp / (ctp + cfp).clamp(min=1)).item()
                coarse_rec = (ctp / (ctp + cfn).clamp(min=1)).item()

            fine_aux_iou = None
            if fine_aux_loss is not None:
                fine_pb = (torch.sigmoid(aux['fine_aux_logits']) > self.surface_threshold).float()
                ftp = (fine_pb * full_gt).sum()
                ffp = (fine_pb * (1 - full_gt)).sum()
                ffn = ((1 - fine_pb) * full_gt).sum()
                fine_aux_iou = (ftp / (ftp + ffp + ffn).clamp(min=1)).item()

        terms = edict(bce=fine_loss, loss=loss, train_iou=iou,
                      train_prec=prec, train_rec=rec)
        if dice_loss is not None:
            terms['dice'] = dice_loss
        if coarse_loss is not None:
            terms['coarse_bce'] = coarse_loss
            terms['train_coarse_iou'] = coarse_iou
            terms['train_coarse_prec'] = coarse_prec
            terms['train_coarse_rec'] = coarse_rec
        if fine_aux_loss is not None:
            terms['fine_aux_bce'] = fine_aux_loss
            terms['train_fine_aux_iou'] = fine_aux_iou
        if fine_aux_dice is not None:
            terms['fine_aux_dice'] = fine_aux_dice
        if density_aux_loss is not None:
            terms['density_aux_bce'] = density_aux_loss
            terms['density_aux_mae'] = density_aux_mae
            terms['train_density_aux_iou'] = density_aux_iou
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

            model = self.models['denoiser']
            if hasattr(model, 'module'): model = model.module
            token_dim = model.token_dim
            SUB = round(token_dim ** (1/3))  # 8 for 512, 4 for 64

            def _up(mask, bd=BLOCK_DIM, sr=SUB):
                T = mask.shape[0]; s = bd // sr
                return F.interpolate(mask.reshape(T, 1, sr, sr, sr),
                                     scale_factor=float(s), mode='nearest').reshape(T, -1)

            def _pool_fine_aux_logits(logits):
                fine_resolution = round(logits.shape[1] ** (1 / 3))
                if fine_resolution == SUB:
                    return logits
                if fine_resolution % SUB != 0:
                    raise ValueError(
                        f"fine_aux_resolution={fine_resolution} must be divisible by main resolution={SUB}"
                    )
                stride = fine_resolution // SUB
                volume = logits.reshape(logits.shape[0], 1, fine_resolution, fine_resolution, fine_resolution)
                pooled = F.max_pool3d(volume.float(), kernel_size=stride, stride=stride)
                return pooled.reshape(logits.shape[0], -1)

            aux_tp, aux_fp, aux_fn = 0., 0., 0.
            merge_tp, merge_fp, merge_fn = 0., 0., 0.
            aux_seen = False

            for li, idx in enumerate(indices):
                idx = int(idx); gi = my_s + li; name = f'sample_{gi:03d}'
                data = ds[idx]

                # Encode image
                cond_img = data['cond'].unsqueeze(0).cuda()
                cond = self.encode_image(cond_img)

                coords_int = data['coords']  # [T, 3]
                batch_coords = torch.cat([
                    torch.zeros(coords_int.shape[0], 1, dtype=torch.int32), coords_int], 1).cuda()

                dummy_feats = torch.zeros(batch_coords.shape[0], token_dim, device='cuda')
                dummy_sp = sp.SparseTensor(feats=dummy_feats, coords=batch_coords)
                t = torch.zeros(1, device='cuda')
                model_kwargs = {}
                if 'patch_xy' in data:
                    model_kwargs['patch_xy'] = data['patch_xy'].cuda()
                if 'patch_valid' in data:
                    model_kwargs['patch_valid'] = data['patch_valid'].cuda()

                return_aux = hasattr(model, 'fine_aux_predict')
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred_out = self.models['denoiser'](
                        dummy_sp, t, cond, return_aux=return_aux, **model_kwargs,
                    )
                if isinstance(pred_out, tuple):
                    pred, aux = pred_out
                else:
                    pred, aux = pred_out, {}

                pred_mask = (torch.sigmoid(pred.feats.float()) > self.surface_threshold).float()
                gt_mask = data['submask'].cuda()

                # Metrics
                pred_flat = pred_mask.reshape(-1)
                gt_flat = gt_mask.reshape(-1)
                tp += (pred_flat * gt_flat).sum().item()
                fp += (pred_flat * (1 - gt_flat)).sum().item()
                fn += ((1 - pred_flat) * gt_flat).sum().item()
                total += gt_flat.numel()

                if aux.get('fine_aux_logits') is not None:
                    aux_seen = True
                    aux_logits = _pool_fine_aux_logits(aux['fine_aux_logits'])
                    aux_mask = (torch.sigmoid(aux_logits.float()) > self.surface_threshold).float()
                    merge_logits = torch.maximum(pred.feats.float(), aux_logits.float())
                    merge_mask = (torch.sigmoid(merge_logits) > self.surface_threshold).float()

                    aux_flat = aux_mask.reshape(-1)
                    merge_flat = merge_mask.reshape(-1)
                    aux_tp += (aux_flat * gt_flat).sum().item()
                    aux_fp += (aux_flat * (1 - gt_flat)).sum().item()
                    aux_fn += ((1 - aux_flat) * gt_flat).sum().item()
                    merge_tp += (merge_flat * gt_flat).sum().item()
                    merge_fp += (merge_flat * (1 - gt_flat)).sum().item()
                    merge_fn += ((1 - merge_flat) * gt_flat).sum().item()

                if not self.snapshot_render_normals:
                    torch.cuda.empty_cache()
                    continue

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
                stats = torch.tensor([
                    tp, fp, fn, total,
                    aux_tp, aux_fp, aux_fn,
                    merge_tp, merge_fp, merge_fn,
                    1.0 if aux_seen else 0.0,
                ], device='cuda')
                dist.all_reduce(stats)
                (tp, fp, fn, total,
                 aux_tp, aux_fp, aux_fn,
                 merge_tp, merge_fp, merge_fn,
                 aux_seen_count) = stats.tolist()
                aux_seen = aux_seen_count > 0
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
                if aux_seen:
                    aux_prec = aux_tp / max(aux_tp + aux_fp, 1)
                    aux_rec = aux_tp / max(aux_tp + aux_fn, 1)
                    aux_iou = aux_tp / max(aux_tp + aux_fp + aux_fn, 1)
                    merge_prec = merge_tp / max(merge_tp + merge_fp, 1)
                    merge_rec = merge_tp / max(merge_tp + merge_fn, 1)
                    merge_iou = merge_tp / max(merge_tp + merge_fp + merge_fn, 1)
                    metrics.update({
                        'aux_pool_iou': aux_iou,
                        'aux_pool_precision': aux_prec,
                        'aux_pool_recall': aux_rec,
                        'aux_pool_pos_pred': (aux_tp + aux_fp) / pos_total,
                        'merge_iou': merge_iou,
                        'merge_precision': merge_prec,
                        'merge_recall': merge_rec,
                        'merge_pos_pred': (merge_tp + merge_fp) / pos_total,
                    })
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
