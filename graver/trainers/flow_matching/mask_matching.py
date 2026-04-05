"""
Stage 2 trainer: sparse flow matching on per-block surface representation.

Supports two modes:
  - Binary mask (legacy): target is {0,1}^512, threshold > 0.5
  - Continuous UDF (default): target is min_pool UDF [0,1]^512,
    surface = small value, threshold < SURFACE_THRESHOLD (0.4).
    Use invert_weight=True to emphasize surface regions in loss.
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

from ...datasets.block_feats import BlockFeats
from ...modules import sparse as sp
from ...pipelines import samplers
from ...utils.data_utils import cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import ImageConditionedMixin


class SparseMaskFlowTrainer(FlowMatchingTrainer):

    def __init__(self, *args, recall_weight: float = 3.0,
                 loss_type: str = "v_loss",
                 surface_threshold: float = 0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.recall_weight = recall_weight
        self.loss_type = loss_type
        self.surface_threshold = surface_threshold
        print(f"[SparseMaskFlowTrainer] recall_weight={recall_weight}, "
              f"loss_type={loss_type}, noise_scale={self.noise_scale}, "
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

    @staticmethod
    def _expand_t(t, layout, T):
        counts = torch.tensor([sl.stop - sl.start for sl in layout],
                              device=t.device, dtype=torch.long)
        return t.repeat_interleave(counts).unsqueeze(1)

    def diffuse(self, x_0, t, noise=None):
        if isinstance(x_0, sp.SparseTensor):
            if noise is None:
                noise = x_0.replace(torch.randn_like(x_0.feats))
            t_tok = self._expand_t(t, x_0.layout, x_0.feats.shape[0])
            return x_0.replace(t_tok * x_0.feats + (1 - t_tok) * noise.feats)
        return super().diffuse(x_0, t, noise=noise)

    def compute_v_from_x_prediction(self, x_t, x_pred, t):
        if isinstance(x_t, sp.SparseTensor):
            t_tok = self._expand_t(t, x_t.layout, x_t.feats.shape[0])
            v = (x_pred.feats - x_t.feats) / (1 - t_tok).clamp(min=0.05)
            return x_t.replace(v)
        return super().compute_v_from_x_prediction(x_t, x_pred, t)

    # ---- Training: pure v-loss ----

    def training_losses(self, x_0=None, cond=None, **kwargs):
        assert x_0 is not None
        B, device = len(x_0.layout), x_0.device
        cond = self.get_cond(cond, **kwargs)

        noise_raw = self.noise_scale * torch.randn_like(x_0.feats)
        noise = x_0.replace(noise_raw)
        t = self.sample_t(B).to(device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        pred = self.training_models['denoiser'](x_t, t, cond)

        with torch.no_grad():
            # UDF target: small value = surface, weight surface regions more
            w = 1.0 + (self.recall_weight - 1.0) * (1.0 - x_0.feats)

        if self.loss_type == "x_loss":
            elem = w * (pred.feats - x_0.feats) ** 2
        else:
            vt = self.compute_v_from_x_prediction(x_t, x_0, t)
            vp = self.compute_v_from_x_prediction(x_t, pred, t)
            elem = w * (vp.feats - vt.feats) ** 2

        loss = torch.stack([elem[sl].mean() for sl in x_0.layout]).mean()

        with torch.no_grad():
            # UDF: surface = pred < threshold
            pb = (pred.feats < self.surface_threshold).float()
            gb = (x_0.feats < self.surface_threshold).float()
            tp = (pb * gb).sum(); fp = (pb * (1 - gb)).sum(); fn = ((1 - pb) * gb).sum()
            iou = (tp / (tp + fp + fn).clamp(min=1)).item()

        terms = edict(mse=loss, loss=loss, train_iou=iou)
        t_np = t.cpu().numpy()
        mse_per_b = np.array([F.mse_loss(pred.feats[sl], x_0.feats[sl]).item() for sl in x_0.layout])
        bins = np.digitize(t_np, np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (bins == i).sum():
                terms[f"bin_{i}"] = {"mse": mse_per_b[bins == i].mean()}
        return terms, {}

    # ---- Snapshot: pred mask -> fill GT feats -> mesh -> normal ----

    @staticmethod
    def _compose_normal_overviews(snapshot_dir, num_samples, verbose=True):
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
            if verbose: print(f'  [Overview] {tag} -> {out}')

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=4, batch_size=1, verbose=False):
        from ...dataset_toolkits.mesh2block import BLOCK_FOLDER, BLOCK_DIM

        if suffix is None: suffix = f'step{self.step:07d}'
        snapshot_dir = os.path.join(self.output_dir, 'samples', suffix)
        details_dir = os.path.join(snapshot_dir, 'details')
        if self.is_master:
            os.makedirs(details_dir, exist_ok=True)
            print(f'\n[Snapshot] Step {self.step}: {num_samples} mask samples')

        states = {n: m.training for n, m in self.models.items()}
        for m in self.models.values(): m.eval()

        try:
            ds = self.test_dataset if hasattr(self, 'test_dataset') and self.test_dataset else self.dataset
            sam = self.get_sampler()
            per_rank = int(np.ceil(num_samples / self.world_size))
            my_s, my_e = self.rank * per_rank, min((self.rank + 1) * per_rank, num_samples)
            indices = torch.randperm(len(ds))[:num_samples][my_s:my_e]

            tp, fp, fn, sg, sp_, tot = 0., 0., 0., 0., 0., 0
            tmps = []
            SUB = 8

            def _up(mask, bd=BLOCK_DIM, sr=SUB):
                T = mask.shape[0]; s = bd // sr
                return F.interpolate(mask.reshape(T, 1, sr, sr, sr),
                                     scale_factor=float(s), mode='nearest').reshape(T, -1)

            for li, idx in enumerate(indices):
                idx = int(idx); gi = my_s + li; name = f'sample_{gi:03d}'

                data = ds[idx]; col = ds.collate_fn([data])
                for k, v in list(col.items()):
                    if isinstance(v, sp.SparseTensor): col[k] = v.to('cuda')
                    elif isinstance(v, torch.Tensor): col[k] = v.cuda()
                x_0 = col.pop('x_0'); gt = x_0.feats

                root, inst = ds.instances[idx]
                with np.load(os.path.join(root, BLOCK_FOLDER, f'{inst}.npz')) as z:
                    coords = z['coords'].astype(np.int32)
                    raw = z['fine_feats']
                    feats = raw.astype(np.float32) if raw.dtype == np.float16 else raw.copy()

                noise = x_0.replace(self.noise_scale * torch.randn_like(gt))
                args = self.get_inference_cond(**col)
                res = sam.sample(self.models['denoiser'], noise=noise,
                                 **args, steps=50, cfg_strength=3.0, verbose=False)
                pred = res.samples.feats if hasattr(res.samples, 'feats') else res.samples
                # UDF: surface = small value
                pm = (pred < self.surface_threshold).float()
                gb = (gt < self.surface_threshold).float()

                tp += (pm * gb).sum().item(); fp += (pm * (1 - gb)).sum().item()
                fn += ((1 - pm) * gb).sum().item()
                sg += gb.sum().item(); sp_ += pm.sum().item(); tot += gt.numel()

                for tag, mask in [('gt', gb), ('pred', pm)]:
                    vx = (_up(mask).cpu().numpy() > 0.5)
                    masked = feats.copy(); masked[~vx] = 1.0
                    sfx = '_gt_normal.jpg' if tag == 'gt' else '_normal.jpg'
                    ply = os.path.join(details_dir, f'{name}_{tag}_tmp.ply')
                    nrm = os.path.join(details_dir, f'{name}{sfx}')
                    try:
                        torch.cuda.empty_cache()
                        BlockFeats.tokens_to_mesh(coords, masked, ply, verbose=False)
                        if os.path.exists(ply):
                            tmps.append(ply)
                            BlockFeats.render_normal_grid(ply, nrm, resolution=512, radius=1.75, verbose=False)
                    except Exception as e:
                        print(f'  [Rank {self.rank}] {tag} render error ({name}): {e}')
                        torch.cuda.empty_cache()

            for p in tmps:
                try: os.remove(p)
                except OSError: pass

            cd = {'tp': tp, 'fp': fp, 'fn': fn, 'sum_gt': sg, 'sum_pred': sp_, 'total': float(tot)}
            ts = {k: torch.tensor(v, device='cuda') for k, v in cd.items()}
            if self.world_size > 1:
                for v in ts.values(): dist.all_reduce(v, op=dist.ReduceOp.SUM)
                dist.barrier()
            c = {k: v.item() for k, v in ts.items()}

            if self.is_master:
                tp, fp, fn = c['tp'], c['fp'], c['fn']
                prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
                iou = tp / max(tp + fp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                metrics = {'iou': iou, 'precision': prec, 'recall': rec, 'f1': f1,
                           'pos_gt': c['sum_gt'] / max(c['total'], 1),
                           'pos_pred': c['sum_pred'] / max(c['total'], 1)}
                with open(os.path.join(snapshot_dir, 'metrics.txt'), 'w') as f:
                    for k, v in metrics.items():
                        line = f'{k}: {v:.4f}'; print(f'  {line}'); f.write(line + '\n')
                self._compose_normal_overviews(snapshot_dir, num_samples)
                print('  Done.')
        finally:
            for n, m in self.models.items(): m.train(states[n])

    def get_sampler(self, **kwargs):
        return samplers.FlowSampler()


class SparseMaskFlowCFGTrainer(ClassifierFreeGuidanceMixin, SparseMaskFlowTrainer):
    pass


class ImageConditionedSparseMaskFlowCFGTrainer(
    ImageConditionedMixin, SparseMaskFlowCFGTrainer,
):
    def get_sampler(self, **kwargs):
        return samplers.FlowGuidanceIntervalSampler(**kwargs)
