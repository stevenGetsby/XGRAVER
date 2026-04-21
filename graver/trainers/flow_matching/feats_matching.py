"""
Stage 3 feats trainer (cascaded).

Uses a dilated pred_mask (from Stage-2 output) as a voxel-level mask:
  voxel_mask = dilate(upsample(pred_submask), k=3, iters=2)

Noise / GT / x_t are hard-filled to `bg_fill` on mask=0 positions so the
model only has to model UDF inside the mask. Loss uses the 3-band mc/near/far
recipe + complexity reweighting, evaluated on mask=1 positions only.
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
from ...dataset_toolkits.mesh2block import MC_THRESHOLD, BLOCK_DIM, SUBMASK_RES
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import ImageConditionedMixin


class SparseFlowMultiTokenTrainer(FlowMatchingTrainer):
    """
    Stage 3 UDF refinement trainer with cascaded pred_mask.
    """

    def __init__(
        self,
        *args,
        lambda_flow: float = 1.0,
        lambda_normal: float = 0.1,
        surface_weight: float = 8.0,
        loss_type: str = "v_loss",
        complexity_boost: float = 2.0,
        cond_noise_std: float = 0.0,
        noise_scale: float = 2.0,
        cascade_dilate_iters: int = 2,
        cascade_dilate_kernel: int = 3,
        cascade_dilate_mode: str = "cube",
        gt_clip_max: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_flow = lambda_flow
        self.lambda_normal = lambda_normal
        self.surface_weight = surface_weight
        self.complexity_boost = complexity_boost
        self.cond_noise_std = cond_noise_std
        self.loss_type = loss_type
        self.noise_scale = noise_scale
        self.voxel = MC_THRESHOLD

        self._cascade_dilate_iters = cascade_dilate_iters
        self._cascade_dilate_kernel = cascade_dilate_kernel
        self._cascade_dilate_mode = str(cascade_dilate_mode).lower()
        assert self._cascade_dilate_mode in ("cube", "cross"), \
            f"cascade_dilate_mode must be cube|cross, got {cascade_dilate_mode}"
        self._gt_clip_max = float(gt_clip_max) if gt_clip_max is not None else None

        print(f"[SparseFlowMultiTokenTrainer] loss_type={self.loss_type}, "
              f"λ_flow={lambda_flow}, λ_normal={lambda_normal}, "
              f"surface_weight={surface_weight}, complexity_boost={complexity_boost}, "
              f"noise_scale={noise_scale}")
        print(f"  cascade: dilate mode={self._cascade_dilate_mode}, "
              f"k={cascade_dilate_kernel}, iters={cascade_dilate_iters}, "
              f"gt_clip_max={self._gt_clip_max}")

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        num_gpus = max(torch.cuda.device_count(), 1)
        cores_per_gpu = os.cpu_count() // num_gpus
        num_workers = max(1, min(cores_per_gpu, 16))

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
        print(f"[DataLoader] num_workers={num_workers} "
              f"(cores_per_gpu={cores_per_gpu}), "
              f"batch_size={self.batch_size_per_gpu}")

    # ------------------------------------------------------------------
    # SparseTensor diffuse / v-prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_t_to_tokens(t: torch.Tensor, layout, T: int) -> torch.Tensor:
        counts = torch.tensor(
            [sl.stop - sl.start for sl in layout],
            device=t.device, dtype=torch.long,
        )
        return t.repeat_interleave(counts).unsqueeze(1)

    def diffuse(self, x_0, t, noise=None):
        if isinstance(x_0, sp.SparseTensor):
            if noise is None:
                noise = x_0.replace(torch.randn_like(x_0.feats))
            t_per_token = self._expand_t_to_tokens(t, x_0.layout, x_0.feats.shape[0])
            x_t_feats = t_per_token * x_0.feats + (1 - t_per_token) * noise.feats
            return x_0.replace(x_t_feats)
        return super().diffuse(x_0, t, noise=noise)

    def compute_v_from_x_prediction(self, x_t, x_pred, t):
        if isinstance(x_t, sp.SparseTensor):
            t_per_token = self._expand_t_to_tokens(t, x_t.layout, x_t.feats.shape[0])
            denom = (1 - t_per_token).clamp(min=0.05)
            v_feats = (x_pred.feats - x_t.feats) / denom
            return x_t.replace(v_feats)
        return super().compute_v_from_x_prediction(x_t, x_pred, t)

    # ------------------------------------------------------------------
    # Voxel mask: submask [T, R^3] -> dilated [T, D^3]
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _upsample_submask(submask: torch.Tensor, block_dim: int = BLOCK_DIM) -> torch.Tensor:
        R = SUBMASK_RES
        T = submask.shape[0]
        sub_3d = submask.reshape(T, 1, R, R, R)
        scale = block_dim // R
        voxel_3d = F.interpolate(sub_3d, scale_factor=scale, mode='nearest')
        return voxel_3d.reshape(T, -1)

    @staticmethod
    @torch.no_grad()
    def _dilate_voxel_mask(voxel_mask: torch.Tensor,
                           kernel: int, iters: int) -> torch.Tensor:
        D = BLOCK_DIM
        T = voxel_mask.shape[0]
        vol = voxel_mask.reshape(T, 1, D, D, D).float()
        pad = kernel // 2
        for _ in range(iters):
            vol = F.max_pool3d(vol, kernel_size=kernel, stride=1, padding=pad)
        return vol.reshape(T, -1)

    _CROSS_KERNEL: Optional[torch.Tensor] = None

    @classmethod
    def _get_cross_kernel(cls, device, dtype):
        k = cls._CROSS_KERNEL
        if k is None or k.device != device or k.dtype != dtype:
            t = torch.zeros(1, 1, 3, 3, 3, device=device, dtype=dtype)
            t[0, 0, 1, 1, 1] = 1
            t[0, 0, 0, 1, 1] = 1; t[0, 0, 2, 1, 1] = 1
            t[0, 0, 1, 0, 1] = 1; t[0, 0, 1, 2, 1] = 1
            t[0, 0, 1, 1, 0] = 1; t[0, 0, 1, 1, 2] = 1
            cls._CROSS_KERNEL = t
        return cls._CROSS_KERNEL

    @classmethod
    @torch.no_grad()
    def _dilate_voxel_mask_cross(cls, voxel_mask: torch.Tensor, iters: int) -> torch.Tensor:
        """6-邻接膨胀：等价于沿 3 个轴各做 k=3 max_pool3d 后取最大。
        比 conv3d 实现快 5-10x（max_pool3d 有专用 kernel）。"""
        D = BLOCK_DIM
        T = voxel_mask.shape[0]
        vol = voxel_mask.reshape(T, 1, D, D, D).float()
        for _ in range(iters):
            vx = F.max_pool3d(vol, (3, 1, 1), 1, (1, 0, 0))
            vy = F.max_pool3d(vol, (1, 3, 1), 1, (0, 1, 0))
            vz = F.max_pool3d(vol, (1, 1, 3), 1, (0, 0, 1))
            vol = torch.maximum(torch.maximum(vx, vy), vz)
        return vol.reshape(T, -1)

    def _build_voxel_mask(self, pred_submask: torch.Tensor) -> torch.Tensor:
        raw = self._upsample_submask(pred_submask)
        if self._cascade_dilate_iters <= 0:
            return raw
        if self._cascade_dilate_mode == "cross":
            return self._dilate_voxel_mask_cross(raw, self._cascade_dilate_iters)
        return self._dilate_voxel_mask(
            raw, self._cascade_dilate_kernel, self._cascade_dilate_iters,
        )

    # ------------------------------------------------------------------
    # Normal loss (UDF 梯度方向一致性)
    # ------------------------------------------------------------------

    @staticmethod
    def _grad_3d(vol: torch.Tensor) -> torch.Tensor:
        vp = F.pad(vol, (1, 1, 1, 1, 1, 1), mode='replicate')
        gx = vp[:, :, 2:, 1:-1, 1:-1] - vp[:, :, :-2, 1:-1, 1:-1]
        gy = vp[:, :, 1:-1, 2:, 1:-1] - vp[:, :, 1:-1, :-2, 1:-1]
        gz = vp[:, :, 1:-1, 1:-1, 2:] - vp[:, :, 1:-1, 1:-1, :-2]
        return torch.cat([gx, gy, gz], dim=1) * 0.5

    def _normal_loss(self, pred_feats: torch.Tensor, gt_feats: torch.Tensor) -> torch.Tensor:
        D = BLOCK_DIM
        v = self.voxel

        with torch.no_grad():
            surface_mask = (gt_feats < v * 2).any(dim=1)
        T_surface = surface_mask.sum().item()
        if T_surface == 0:
            return torch.tensor(0.0, device=pred_feats.device)

        pred_surface = pred_feats[surface_mask]
        gt_surface = gt_feats[surface_mask]

        pred_vol = pred_surface.reshape(T_surface, 1, D, D, D)
        pred_grad = self._grad_3d(pred_vol)

        with torch.no_grad():
            gt_vol = gt_surface.reshape(T_surface, 1, D, D, D)
            gt_grad = self._grad_3d(gt_vol)
            gt_norm = gt_grad.norm(dim=1, keepdim=True).clamp(min=1e-4)
            gt_dir = gt_grad / gt_norm

            w = (gt_surface < v * 2).float()
            w_3d = w.reshape(T_surface, 1, D, D, D)

        pred_norm = pred_grad.norm(dim=1, keepdim=True).clamp(min=1e-4)
        pred_dir = pred_grad / pred_norm

        cos_sim = (pred_dir * gt_dir).sum(dim=1, keepdim=True)
        dir_loss = (1.0 - cos_sim) * w_3d
        eikonal_loss = (pred_norm - 1.0).abs() * w_3d

        with torch.no_grad():
            gt_laplacian = self._grad_3d(gt_norm)
            curvature = gt_laplacian.norm(dim=1, keepdim=True)
            edge_w = 1.0 + 2.0 * (curvature / curvature.max().clamp(min=1e-6))

        loss = (dir_loss + 0.5 * eikonal_loss) * edge_w
        return loss.mean()

    # ------------------------------------------------------------------
    # Complexity reweighting
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _neighborhood_complexity(
        self,
        gt_feats: torch.Tensor,
        coords: torch.Tensor,
        layout: List[slice],
        voxel: float,
        boost: float,
    ) -> torch.Tensor:
        D = BLOCK_DIM
        T = gt_feats.shape[0]
        device = gt_feats.device

        surface_mask = (gt_feats < voxel * 2).any(dim=1)
        block_curvature = torch.zeros(T, device=device)
        T_s = surface_mask.sum().item()

        if T_s > 0:
            gt_vol = gt_feats[surface_mask].reshape(T_s, 1, D, D, D)
            grad = self._grad_3d(gt_vol)
            grad_norm = grad.norm(dim=1).reshape(T_s, -1)
            w = (gt_feats[surface_mask] < voxel * 2).float()
            w_sum = w.sum(dim=1).clamp(min=1)
            mean_gn = (grad_norm * w).sum(dim=1) / w_sum
            var_gn = ((grad_norm - mean_gn.unsqueeze(1)).pow(2) * w).sum(dim=1) / w_sum
            block_curvature[surface_mask] = var_gn.sqrt()

        complexity = torch.zeros(T, device=device)
        offsets = torch.tensor(
            [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],
            device=device, dtype=torch.long,
        )

        for b, sl in enumerate(layout):
            bc = coords[sl][:, 1:].long()
            curv = block_curvature[sl]
            is_surf = surface_mask[sl]
            N_b = bc.shape[0]
            if N_b <= 1:
                complexity[sl] = curv
                continue

            P1, P2 = 100003, 1009
            keys = bc[:, 0] * P1 + bc[:, 1] * P2 + bc[:, 2]

            nb_coords = bc.unsqueeze(1) + offsets.unsqueeze(0)
            nb_keys = (nb_coords[:, :, 0] * P1
                       + nb_coords[:, :, 1] * P2
                       + nb_coords[:, :, 2])

            sorted_keys, sort_idx = keys.sort()
            flat_nb = nb_keys.reshape(-1)
            pos = torch.searchsorted(sorted_keys, flat_nb)
            pos = pos.clamp(max=N_b - 1)
            matched = sorted_keys[pos] == flat_nb
            nb_idx_flat = sort_idx[pos]
            nb_idx_flat[~matched] = 0

            nb_idx = nb_idx_flat.reshape(N_b, 6)
            valid = matched.reshape(N_b, 6)

            nb_curv = curv[nb_idx]
            nb_is_surf = is_surf[nb_idx]
            valid_surf = valid & nb_is_surf

            curv_diff = (curv.unsqueeze(1) - nb_curv).abs() * valid_surf.float()
            n_surf_nb = valid_surf.float().sum(dim=1)

            has_nb = n_surf_nb > 0
            mean_diff = torch.where(
                has_nb, curv_diff.sum(dim=1) / n_surf_nb,
                torch.zeros_like(n_surf_nb),
            )

            local_c = (curv + mean_diff) * (1.0 + n_surf_nb / 6.0)
            complexity[sl] = local_c

        cmax, cmin = complexity.max(), complexity.min()
        if cmax > cmin:
            complexity = (complexity - cmin) / (cmax - cmin)
        else:
            complexity.zero_()

        return 1.0 + boost * complexity

    # ------------------------------------------------------------------
    # Training losses (cascade: pred_mask dilate + hard-fill)
    # ------------------------------------------------------------------

    def training_losses(
        self,
        x_f: sp.SparseTensor = None,
        x_c: sp.SparseTensor = None,
        cond=None,
        submask=None,
        pred_submask=None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        x = x_f if x_f is not None else x_c
        assert x is not None

        B = len(x.layout)
        T = x.feats.shape[0]
        device = x.device
        terms = edict()

        cond = self.get_cond(cond, **kwargs)
        if self.cond_noise_std > 0:
            cond = cond + self.cond_noise_std * torch.randn_like(cond)

        v = self.voxel

        with torch.no_grad():
            # Build dilated voxel mask from pred_submask (fallback: GT submask, fallback: all-ones)
            if pred_submask is not None:
                pred_sub = pred_submask.to(device)
            elif submask is not None:
                pred_sub = submask.to(device)
            else:
                pred_sub = torch.ones(T, SUBMASK_RES ** 3, device=device)
            voxel_mask = self._build_voxel_mask(pred_sub)

            # GT clip (optional) + hard-fill on mask=0
            bg_fill = self._gt_clip_max if self._gt_clip_max is not None else 1.0
            if self._gt_clip_max is not None:
                gt = x.feats.clamp(max=self._gt_clip_max)
            else:
                gt = x.feats
            x_feats_masked = gt * voxel_mask + (1.0 - voxel_mask) * bg_fill
            x_masked = x.replace(x_feats_masked)

            noise_raw = self.noise_scale * torch.randn_like(x.feats) * voxel_mask
            noise = x.replace(noise_raw)
            t = self.sample_t(B).to(device).float()
            x_t = self.diffuse(x_masked, t, noise=noise)
            x_t_feats = x_t.feats * voxel_mask + (1.0 - voxel_mask) * bg_fill
            x_t = x_t.replace(x_t_feats)

        pred = self.training_models["denoiser"](x_t, t, cond)

        x_residual = pred.feats - x_masked.feats
        t_per_token = self._expand_t_to_tokens(t, x.layout, T)

        if self.loss_type == "v_loss":
            v_denom = (1 - t_per_token).clamp(min=0.05)
            diff = (x_residual / v_denom) ** 2
        else:
            diff = x_residual ** 2

        diff = diff * voxel_mask

        with torch.no_grad():
            mc_mask   = (x_masked.feats < v * 2).float()
            far_mask  = (x_masked.feats > v * 3).float()
            near_mask = (1.0 - mc_mask) * (1.0 - far_mask)

        block_loss_mc   = (diff * mc_mask  ).sum(dim=1) / mc_mask  .sum(dim=1).clamp(min=1)
        block_loss_near = (diff * near_mask).sum(dim=1) / near_mask.sum(dim=1).clamp(min=1)
        block_loss_far  = (diff * far_mask ).sum(dim=1) / far_mask .sum(dim=1).clamp(min=1)

        block_loss = (
            self.surface_weight * block_loss_mc
            + 2.0 * block_loss_near
            + 1.0 * block_loss_far
        )

        with torch.no_grad():
            complexity_w = self._neighborhood_complexity(
                x_masked.feats, x.coords, x.layout,
                self.voxel, self.complexity_boost,
            )
        block_loss = block_loss * complexity_w

        per_sample_loss = torch.stack([block_loss[sl].mean() for sl in x.layout])
        terms["flow_loss"] = per_sample_loss.mean()
        terms["complexity_avg"] = complexity_w.mean()

        per_sample_mc   = torch.stack([block_loss_mc[sl].mean()   for sl in x.layout])
        per_sample_near = torch.stack([block_loss_near[sl].mean() for sl in x.layout])
        per_sample_far  = torch.stack([block_loss_far[sl].mean()  for sl in x.layout])
        terms["loss_mc"]   = per_sample_mc.mean()
        terms["loss_near"] = per_sample_near.mean()
        terms["loss_far"]  = per_sample_far.mean()

        terms["loss"] = self.lambda_flow * terms["flow_loss"]

        if self.lambda_normal > 0:
            terms["normal_loss"] = self._normal_loss(pred.feats, x_masked.feats)
            terms["loss"] = terms["loss"] + self.lambda_normal * terms["normal_loss"]

        with torch.no_grad():
            batch_loss = torch.stack([block_loss[sl].mean() for sl in x.layout])
            for lo, hi in [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]:
                mt = (t >= lo) & (t < hi)
                if mt.any():
                    terms[f"loss_t{lo:.1f}_{hi:.1f}"] = batch_loss[mt].mean()

            terms["mask_ratio"] = float(voxel_mask.mean())
            if submask is not None and pred_submask is not None:
                gt_voxel = self._upsample_submask(submask.to(device))
                gt_b = gt_voxel > 0.5
                dil_b = voxel_mask > 0.5
                gt_total = gt_b.sum().clamp(min=1)
                terms["gt_recall"] = float((gt_b & dil_b).sum() / gt_total)
                terms["deep_fn_ratio"] = float((gt_b & ~dil_b).sum() / gt_total)

        return terms, {}

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compose_normal_overviews(
        normalmap_dir: str,
        num_samples: int,
        verbose: bool = True,
    ):
        from PIL import Image

        details_dir = os.path.join(normalmap_dir, "details")

        pred_paths, gt_paths = [], []
        for i in range(num_samples):
            p = os.path.join(details_dir, f"sample_{i:03d}_normal.jpg")
            g = os.path.join(details_dir, f"sample_{i:03d}_gt_normal.jpg")
            pred_paths.append(p if os.path.exists(p) else None)
            gt_paths.append(g if os.path.exists(g) else None)

        def compose_grid(paths, tag):
            for g_idx in range(0, len(paths), 16):
                group = paths[g_idx : g_idx + 16]
                imgs = []
                for p in group:
                    if p is not None:
                        try:
                            imgs.append(Image.open(p))
                        except Exception:
                            imgs.append(None)
                    else:
                        imgs.append(None)

                if not any(imgs):
                    continue

                ref = next(im for im in imgs if im is not None)
                w, h = ref.size

                while len(imgs) < 16:
                    imgs.append(None)
                imgs = [
                    im if im is not None else Image.new("RGB", (w, h), (0, 0, 0))
                    for im in imgs
                ]

                grid = Image.new("RGB", (w * 4, h * 4))
                for row in range(4):
                    for col in range(4):
                        grid.paste(imgs[row * 4 + col], (col * w, row * h))

                out = os.path.join(normalmap_dir, f"overview_{tag}_{g_idx:03d}.jpg")
                grid.save(out, quality=90)
                if verbose:
                    end_idx = min(g_idx + 15, len(paths) - 1)
                    print(f"  [Overview] {tag} samples {g_idx}-{end_idx} -> {out}")

        compose_grid(pred_paths, "pred")
        compose_grid(gt_paths, "gt")

    # ------------------------------------------------------------------
    # Snapshot (cascade: dilated pred mask + hard-fill, same as training)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def snapshot(
        self,
        num_samples: int = 4,
        batch_size: int = 1,
        steps: int = 100,
        cfg_strength: float = 1.5,
        cfg_interval: Tuple[float, float] = (0.1, 1.0),
        **kwargs,
    ):
        verbose = True
        snapshot_dir = os.path.join(
            self.output_dir, "samples", f"step{self.step:07d}",
        )
        details_dir = os.path.join(snapshot_dir, "details")
        if self.is_master:
            os.makedirs(snapshot_dir, exist_ok=True)
            os.makedirs(details_dir, exist_ok=True)
            if verbose:
                print(f"\n[Snapshot] Step {self.step}: "
                      f"{num_samples} samples, {self.world_size} GPUs "
                      f"({self._cascade_dilate_mode}_{self._cascade_dilate_iters}x + hard-fill)")

        if self.world_size > 1:
            dist.barrier()

        model_states = {n: m.training for n, m in self.models.items()}
        for m in self.models.values():
            m.eval()

        try:
            if hasattr(self, 'test_dataset') and self.test_dataset is not None:
                snap_dataset = self.test_dataset
            else:
                snap_dataset = copy.deepcopy(self.dataset)
            dataloader = DataLoader(
                snap_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                collate_fn=getattr(snap_dataset, "collate_fn", None),
            )

            samples_per_rank = int(np.ceil(num_samples / self.world_size))
            my_start = self.rank * samples_per_rank
            my_end = min(my_start + samples_per_rank, num_samples)
            my_count = max(0, my_end - my_start)

            sampler = self.get_sampler()
            model = getattr(self.models["denoiser"], "module", self.models["denoiser"])

            amp_dtype = torch.float16
            use_amp = (hasattr(self, 'accelerator')
                       and self.accelerator.mixed_precision != 'no')

            data_iter = iter(dataloader)
            saved = 0
            tmp_mesh_paths = []

            while saved < my_count:
                try:
                    data = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    data = next(data_iter)

                for k, v in list(data.items()):
                    if hasattr(v, "cuda"):
                        data[k] = v.cuda()

                x_c = data.pop("x_c", None)
                x_f = data.pop("x_f", None)
                data.pop("n_f", None)
                snap_submask = data.pop("submask", None)
                snap_pred_submask = data.pop("pred_submask", None)
                x_gt = x_f if x_f is not None else x_c

                if snap_pred_submask is not None:
                    voxel_mask = self._build_voxel_mask(snap_pred_submask.cuda())
                elif snap_submask is not None:
                    voxel_mask = self._build_voxel_mask(snap_submask.cuda())
                else:
                    voxel_mask = torch.ones_like(x_gt.feats)

                noise_raw = self.noise_scale * torch.randn_like(x_gt.feats)
                noise_raw = noise_raw * voxel_mask
                # Hard-fill mask=0 to bg_fill so the first model forward matches training.
                bg_fill_init = self._gt_clip_max if self._gt_clip_max is not None else 1.0
                noise_raw = noise_raw + (1.0 - voxel_mask) * bg_fill_init
                noise = x_gt.replace(noise_raw)

                args = self.get_inference_cond(**data)
                args['voxel_mask'] = voxel_mask
                if self._gt_clip_max is not None:
                    args['bg_fill'] = self._gt_clip_max

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    res = sampler.sample(
                        model, noise=noise, **args,
                        steps=steps, cfg_strength=cfg_strength,
                        cfg_interval=cfg_interval,
                        use_heun=True, verbose=False,
                    )
                    pred = res.samples

                bg_fill = self._gt_clip_max if self._gt_clip_max is not None else 1.0
                clamp_hi = self._gt_clip_max if self._gt_clip_max is not None else 1.0
                pred_feats = pred.feats * voxel_mask + (1.0 - voxel_mask) * bg_fill
                pred_feats = pred_feats.clamp(0.0, clamp_hi)
                pred = pred.replace(pred_feats)

                if snap_submask is not None:
                    snap_gt_voxel = self._upsample_submask(snap_submask.cuda())
                else:
                    snap_gt_voxel = None

                actual_batch = len(x_gt.layout)
                for b in range(min(actual_batch, my_count - saved)):
                    sl = x_gt.layout[b]
                    global_idx = my_start + saved
                    name = f"sample_{global_idx:03d}"

                    gt_coords = x_gt.coords[sl].cpu().numpy()
                    pred_fine = pred.feats[sl].cpu().float().numpy()
                    gt_fine_np = x_f.feats[sl].cpu().numpy() if x_f is not None else None

                    try:
                        diag = {}
                        pf = pred.feats[sl].cpu().float()
                        vm = voxel_mask[sl].cpu().float()

                        diag['num_tokens'] = int(pf.shape[0])
                        diag['mask_ratio'] = float(vm.mean())

                        surf_b = vm > 0.5
                        far_b = ~surf_b
                        for region, rmask in [('surf', surf_b), ('far', far_b)]:
                            vals = pf[rmask]
                            if vals.numel() > 0:
                                diag[f'{region}_mean'] = float(vals.mean())
                                diag[f'{region}_std'] = float(vals.std())
                                diag[f'{region}_below_mc'] = float(
                                    (vals < MC_THRESHOLD).float().mean())
                                diag[f'{region}_num'] = int(rmask.sum())

                        if snap_gt_voxel is not None:
                            gm = snap_gt_voxel[sl].cpu().float()
                            gt_b = gm > 0.5
                            diag['gt_total'] = int(gt_b.sum())
                            gt_covered = int((surf_b & gt_b).sum())
                            diag['gt_recall'] = gt_covered / max(diag['gt_total'], 1)
                            deep_fn = (~surf_b & gt_b)
                            diag['deep_fn_num'] = int(deep_fn.sum())
                            diag['deep_fn_ratio'] = diag['deep_fn_num'] / max(diag['gt_total'], 1)

                        import json
                        with open(os.path.join(details_dir, f"{name}_diag.json"), 'w') as f:
                            json.dump(diag, f, indent=2)
                    except Exception as e:
                        print(f"  [Rank {self.rank}] Diag error ({name}): {e}")

                    tmp_ply = os.path.join(details_dir, f"{name}_tmp.ply")
                    normal_path = os.path.join(details_dir, f"{name}_normal.jpg")
                    pred_clipped = np.clip(pred_fine, 0.0, 1.0)
                    try:
                        torch.cuda.empty_cache()
                        self.dataset.tokens_to_mesh(gt_coords, pred_clipped, tmp_ply, verbose=False)
                    except Exception as e:
                        print(f"  [Rank {self.rank}] Mesh error ({name}): {e}")
                        torch.cuda.empty_cache()

                    if os.path.exists(tmp_ply):
                        tmp_mesh_paths.append(tmp_ply)
                        try:
                            self.dataset.render_normal_grid(
                                tmp_ply, normal_path, resolution=1024,
                                radius=1.75, verbose=False)
                        except Exception as e:
                            print(f"  [Rank {self.rank}] Render error ({name}): {e}")
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                    if gt_fine_np is not None:
                        gt_tmp_ply = os.path.join(details_dir, f"{name}_gt_tmp.ply")
                        gt_normal_path = os.path.join(details_dir, f"{name}_gt_normal.jpg")
                        gt_clipped = np.clip(gt_fine_np, 0.0, 1.0)
                        try:
                            torch.cuda.empty_cache()
                            self.dataset.tokens_to_mesh(gt_coords, gt_clipped, gt_tmp_ply, verbose=False)
                        except Exception as e:
                            print(f"  [Rank {self.rank}] GT mesh error ({name}): {e}")
                            torch.cuda.empty_cache()

                        if os.path.exists(gt_tmp_ply):
                            tmp_mesh_paths.append(gt_tmp_ply)
                            try:
                                self.dataset.render_normal_grid(
                                    gt_tmp_ply, gt_normal_path, resolution=1024,
                                    radius=1.75, verbose=False)
                            except Exception as e:
                                print(f"  [Rank {self.rank}] GT render error ({name}): {e}")
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()

                    if verbose:
                        print(f"  [Rank {self.rank}] Sampling {saved+1}/{my_count}")
                    saved += 1

            for p in tmp_mesh_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

            if self.world_size > 1:
                dist.barrier()

            if self.is_master:
                self._compose_normal_overviews(snapshot_dir, num_samples, verbose)
                try:
                    import json
                    import glob
                    diag_files = sorted(glob.glob(os.path.join(details_dir, "*_diag.json")))
                    if diag_files:
                        all_diags = []
                        for df in diag_files:
                            with open(df) as f:
                                all_diags.append(json.load(f))
                        summary = {"num_samples": len(all_diags)}
                        keys_to_avg = [
                            'mask_ratio', 'gt_recall', 'deep_fn_ratio',
                            'surf_mean', 'surf_below_mc',
                            'far_mean', 'far_below_mc',
                        ]
                        for k in keys_to_avg:
                            vals = [d[k] for d in all_diags if k in d]
                            if vals:
                                summary[k] = round(sum(vals) / len(vals), 4)
                        for k in ['gt_total', 'deep_fn_num', 'surf_num', 'far_num']:
                            vals = [d[k] for d in all_diags if k in d]
                            if vals:
                                summary[k] = sum(vals)
                        with open(os.path.join(snapshot_dir, "diag_summary.json"), 'w') as f:
                            json.dump(summary, f, indent=2)
                        print(f"  [Snapshot] Diagnostics: {summary}")
                except Exception as e:
                    print(f"  [Snapshot] Diag summary error: {e}")
                if verbose:
                    print(f"  [Snapshot] All {num_samples} samples done")

        finally:
            for n, m in self.models.items():
                m.train(model_states[n])


# ======================================================================
# CFG / ImageConditioned variants
# ======================================================================

class SparseFlowMultiTokenCFGTrainer(
    ClassifierFreeGuidanceMixin, SparseFlowMultiTokenTrainer,
):
    pass


class ImageConditionedSparseFlowMultiTokenCFGTrainer(
    ImageConditionedMixin, SparseFlowMultiTokenCFGTrainer,
):
    def get_sampler(self, **kwargs):
        return samplers.FlowGuidanceIntervalSampler(**kwargs)
