from typing import *
import copy
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import utils
import numpy as np
from easydict import EasyDict as edict
from ..basic import BasicTrainer
from ...pipelines import samplers
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import ImageConditionedMixin


class FlowMatchingTrainer(BasicTrainer):
    """
    Dense flow matching trainer for Stage 1 (64³ occupancy).
    """
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {'mean': -0.8, 'std': 0.8},
        },
        cond_noise_std: float = 0.0,
        noise_scale: float = 1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.cond_noise_std = cond_noise_std
        self.noise_scale = noise_scale

    def diffuse(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        return t * x_0 + (1 - t) * noise

    def compute_v_from_x_prediction(self, x_t, x_pred, t):
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        return (x_pred - x_t) / (1 - t).clamp(min=0.05)

    def get_cond(self, cond, **kwargs):
        return cond

    def get_inference_cond(self, **kwargs):
        return kwargs

    def get_sampler(self, **kwargs):
        return samplers.FlowSampler()

    def sample_t(self, batch_size):
        if self.t_schedule['name'] == 'uniform':
            return torch.rand(batch_size)
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            return torch.sigmoid(torch.randn(batch_size) * std + mean)
        raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")

    def training_losses(self, x_0, cond=None, **kwargs):
        noise = torch.randn_like(x_0) * self.noise_scale
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)

        if self.cond_noise_std > 0 and isinstance(cond, torch.Tensor):
            cond = cond + self.cond_noise_std * torch.randn_like(cond)

        pred = self.training_models['denoiser'](x_t, t, cond, **kwargs)
        v_target = self.compute_v_from_x_prediction(x_t, x_0, t)
        v_pred = self.compute_v_from_x_prediction(x_t, pred, t)

        terms = edict()
        terms["mse"] = F.mse_loss(v_pred, v_target)
        terms["loss"] = terms["mse"]

        # Per-bin logging
        t_np = t.cpu().numpy()
        mse_per = np.array([F.mse_loss(v_pred[i], v_target[i]).item() for i in range(x_0.shape[0])])
        bins = np.digitize(t_np, np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (bins == i).sum():
                terms[f"bin_{i}"] = {"mse": mse_per[bins == i].mean()}

        return terms, {}

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_snapshot(self, num_samples, batch_size, verbose=False):
        snap_ds = self.test_dataset if hasattr(self, 'test_dataset') and self.test_dataset else copy.deepcopy(self.dataset)
        loader = DataLoader(snap_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                            collate_fn=snap_ds.collate_fn if hasattr(snap_ds, 'collate_fn') else None)
        sampler = self.get_sampler()
        sample_gt, sample_pred = [], []

        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(loader))
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            noise = torch.randn_like(data['x_0']) * self.noise_scale
            sample_gt.append(data['x_0'])
            del data['x_0']
            args = self.get_inference_cond(**data)

            if getattr(self, 'fp16_mode', None) == 'amp':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    res = sampler.sample(self.models['denoiser'], noise=noise, **args,
                                         steps=50, cfg_strength=3.0, verbose=verbose)
            else:
                res = sampler.sample(self.models['denoiser'], noise=noise, **args,
                                     steps=50, cfg_strength=3.0, verbose=verbose)
            sample_pred.append(res.samples)

        return torch.cat(sample_gt, 0), torch.cat(sample_pred, 0)

    @staticmethod
    def _binary_metric_counts(pred, gt):
        pred_bin = (pred > 0.5).float()
        gt_bin = (gt > 0.5).float()
        return {
            'tp': (pred_bin * gt_bin).sum().item(),
            'fp': (pred_bin * (1 - gt_bin)).sum().item(),
            'fn': ((1 - pred_bin) * gt_bin).sum().item(),
            'sum_gt': gt_bin.sum().item(),
            'sum_pred': pred_bin.sum().item(),
            'total': float(gt_bin.numel()),
        }

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=4, batch_size=4, verbose=False):
        from ...datasets.block_feats import BlockFeats
        from ...dataset_toolkits.mesh2block import BLOCK_FOLDER, BLOCK_DIM, BLOCK_GRID

        if self.is_master:
            print(f'\nSampling {num_samples} images...', end='')
        if suffix is None:
            suffix = f'step{self.step:07d}'

        n_per_rank = int(np.ceil(num_samples / self.world_size))
        gt, pred = self.run_snapshot(n_per_rank, batch_size, verbose)

        # Metrics (all-reduce)
        counts = self._binary_metric_counts(pred, gt)
        tensors = {k: torch.tensor(v, device='cuda') for k, v in counts.items()}
        if self.world_size > 1:
            for v in tensors.values():
                dist.all_reduce(v, op=dist.ReduceOp.SUM)
        counts = {k: v.item() for k, v in tensors.items()}

        # Render voxel visualization
        gt_vis = self.visualize_sample(gt).contiguous().cuda()
        pred_vis = self.visualize_sample(pred).contiguous().cuda()

        vis_images = {}
        if self.world_size > 1:
            for tag, val in [('sample_gt', gt_vis), ('sample', pred_vis)]:
                if self.is_master:
                    all_vis = [torch.empty_like(val) for _ in range(self.world_size)]
                else:
                    all_vis = []
                dist.gather(val, all_vis, dst=0)
                if self.is_master:
                    vis_images[tag] = torch.cat(all_vis, 0)[:num_samples].cpu()
        else:
            vis_images['sample_gt'] = gt_vis[:num_samples].cpu()
            vis_images['sample'] = pred_vis[:num_samples].cpu()

        if self.is_master:
            out_dir = os.path.join(self.output_dir, 'samples', suffix)
            details_dir = os.path.join(out_dir, 'details')
            os.makedirs(details_dir, exist_ok=True)

            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            iou = tp / max(tp + fp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            metrics = {
                'iou': iou, 'precision': precision, 'recall': recall, 'f1': f1,
                'pos_gt': counts['sum_gt'] / max(counts['total'], 1),
                'pos_pred': counts['sum_pred'] / max(counts['total'], 1),
            }
            with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
                for k, v in metrics.items():
                    line = f'{k}: {v:.4f}'
                    print(f'  {line}')
                    f.write(line + '\n')

            for tag, imgs in vis_images.items():
                utils.save_image(imgs, os.path.join(out_dir, f'{tag}_{suffix}.jpg'),
                                 nrow=int(np.sqrt(num_samples)), normalize=True,
                                 value_range=(0, 1))

            # --- Normal map rendering via GT feats ---
            try:
                ds = self.test_dataset if hasattr(self, 'test_dataset') and self.test_dataset else self.dataset
                n_normal = min(num_samples, pred.shape[0], 4)
                tmps = []
                for si in range(n_normal):
                    # Extract predicted active coords from 64³ grid
                    pred_occ = (pred[si, 0] > 0.5).cpu().numpy()   # [64,64,64]
                    gt_occ = (gt[si, 0] > 0.5).cpu().numpy()
                    pred_coords = np.argwhere(pred_occ).astype(np.int32)  # [M, 3]
                    gt_coords = np.argwhere(gt_occ).astype(np.int32)

                    # Load GT feats for this sample
                    idx = si % len(ds)
                    root, inst = ds.instances[idx]
                    npz_path = os.path.join(root, BLOCK_FOLDER, f'{inst}.npz')
                    if not os.path.exists(npz_path):
                        continue
                    with np.load(npz_path) as z:
                        all_coords = z['coords'].astype(np.int32)  # [N, 3]
                        raw = z['fine_feats']
                        all_feats = raw.astype(np.float32) if raw.dtype == np.float16 else raw.copy()

                    # Build coord→feats lookup
                    coord_to_idx = {}
                    for ci in range(all_coords.shape[0]):
                        key = tuple(all_coords[ci])
                        coord_to_idx[key] = ci

                    for tag, coords_arr in [('gt', gt_coords), ('pred', pred_coords)]:
                        matched_coords, matched_feats = [], []
                        for c in coords_arr:
                            key = tuple(c)
                            if key in coord_to_idx:
                                matched_coords.append(c)
                                matched_feats.append(all_feats[coord_to_idx[key]])
                        if len(matched_coords) < 10:
                            continue
                        mc = np.array(matched_coords)
                        mf = np.array(matched_feats)

                        sfx = '_gt_normal.jpg' if tag == 'gt' else '_normal.jpg'
                        ply = os.path.join(details_dir, f'sample_{si:03d}_{tag}_tmp.ply')
                        nrm = os.path.join(details_dir, f'sample_{si:03d}{sfx}')
                        try:
                            BlockFeats.tokens_to_mesh(mc, mf, ply, verbose=False)
                            if os.path.exists(ply):
                                tmps.append(ply)
                                BlockFeats.render_normal_grid(ply, nrm, resolution=512,
                                                             radius=1.75, verbose=False)
                        except Exception as e:
                            print(f'  [Normal] {tag} render error (sample {si}): {e}')

                # Compose overview grids
                from PIL import Image
                for tag, sfx in [('pred', '_normal.jpg'), ('gt', '_gt_normal.jpg')]:
                    paths = [os.path.join(details_dir, f'sample_{i:03d}{sfx}') for i in range(n_normal)]
                    imgs = []
                    for p in paths:
                        try:
                            imgs.append(Image.open(p) if os.path.exists(p) else None)
                        except Exception:
                            imgs.append(None)
                    if not any(imgs):
                        continue
                    ref = next(im for im in imgs if im is not None)
                    w, h = ref.size
                    imgs = [im if im else Image.new('RGB', (w, h), (0, 0, 0)) for im in imgs]
                    ncol = min(len(imgs), 4)
                    nrow = (len(imgs) + ncol - 1) // ncol
                    grid = Image.new('RGB', (w * ncol, h * nrow))
                    for i, im in enumerate(imgs):
                        r, c = divmod(i, ncol)
                        grid.paste(im, (c * w, r * h))
                    opath = os.path.join(out_dir, f'overview_{tag}.jpg')
                    grid.save(opath, quality=90)

                for p in tmps:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            except Exception as e:
                print(f'  [Normal rendering skipped] {e}')

            print(' Done.')


class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    pass


class ImageConditionedFlowMatchingCFGTrainer(ImageConditionedMixin, FlowMatchingCFGTrainer):
    pass



