"""
Stage 3 Cascaded Trainer v5: exact feats_matching recipe with dil_2x pred mask.

Design:
  - voxel_mask = dilate(pred_submask, k=3, iters=2)
  - Noise / GT / x_t hard-filled to bg_fill on mask=0
  - Loss: exact feats_matching recipe (mc/near/far 3-band + surface_weight
    + complexity_boost) applied only on mask=1 positions.
  - gt_clip_max optional (clips far-field GT, bg_fill = gt_clip_max or 1.0).
"""
from typing import *
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from ...modules import sparse as sp
from ...pipelines import samplers
from ...dataset_toolkits.mesh2block import MC_THRESHOLD, BLOCK_DIM, SUBMASK_RES
from .feats_matching import SparseFlowMultiTokenTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import ImageConditionedMixin


class CascadedFeatsTrainer(SparseFlowMultiTokenTrainer):
    """
    Cascaded feats trainer v5.
    Uses dilated pred_mask (k=3, iters=2) as voxel_mask -> exact feats_matching recipe.
    """

    def __init__(
        self,
        *args,
        cascade_dilate_iters: int = 2,
        cascade_dilate_kernel: int = 3,
        gt_clip_max: Optional[float] = None,
        **kwargs,
    ):
        # Pop legacy / v14 params that parent doesn't know
        for k in ('cascade_mask_config', 'cascade_mask_weight',
                  'cascade_mask_threshold', 'cascade_lambda_far',
                  'cascade_weak_noise_std', 'cascade_far_weight',
                  'cascade_trunc_threshold', 'w_high', 'w_low'):
            kwargs.pop(k, None)

        self._cascade_dilate_iters = cascade_dilate_iters
        self._cascade_dilate_kernel = cascade_dilate_kernel
        self._gt_clip_max = float(gt_clip_max) if gt_clip_max is not None else None

        super().__init__(*args, **kwargs)

        print(f"[CascadedFeatsTrainer v5] dilate k={cascade_dilate_kernel}, "
              f"iters={cascade_dilate_iters} | gt_clip_max={self._gt_clip_max} | "
              f"exact feats_matching recipe (mc/near/far {self.surface_weight}/3/1 "
              f"+ complexity_boost={self.complexity_boost})")

    # ------------------------------------------------------------------
    # Dilation helpers
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _dilate_voxel_mask(voxel_mask: torch.Tensor,
                           kernel: int, iters: int) -> torch.Tensor:
        """Dilate [T, D^3] binary mask by `iters` iterations of max_pool3d."""
        D = BLOCK_DIM
        T = voxel_mask.shape[0]
        vol = voxel_mask.reshape(T, 1, D, D, D).float()
        pad = kernel // 2
        for _ in range(iters):
            vol = F.max_pool3d(vol, kernel_size=kernel, stride=1, padding=pad)
        return vol.reshape(T, -1)

    def _build_voxel_mask(self, pred_submask: torch.Tensor) -> torch.Tensor:
        """pred_submask [T, R^3] -> dilated voxel mask [T, D^3]."""
        raw = self._upsample_submask(pred_submask)  # [T, 4096]
        return self._dilate_voxel_mask(
            raw, self._cascade_dilate_kernel, self._cascade_dilate_iters,
        )

    # ------------------------------------------------------------------
    # Training losses (exact feats_matching recipe on mask=1)
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

        v = self.voxel  # = MC_THRESHOLD = 0.2

        with torch.no_grad():
            # ---- Build dilated voxel mask from pred_submask ----
            if pred_submask is not None:
                pred_sub = pred_submask.to(device)
            elif submask is not None:
                pred_sub = submask.to(device)
            else:
                pred_sub = torch.ones(T, SUBMASK_RES ** 3, device=device)
            voxel_mask = self._build_voxel_mask(pred_sub)          # [T, D^3]

            # ---- GT clip (optional) + hard-fill on mask=0 ----
            bg_fill = self._gt_clip_max if self._gt_clip_max is not None else 1.0
            if self._gt_clip_max is not None:
                gt = x.feats.clamp(max=self._gt_clip_max)
            else:
                gt = x.feats
            x_feats_masked = gt * voxel_mask + (1.0 - voxel_mask) * bg_fill
            x_masked = x.replace(x_feats_masked)

            # ---- Diffuse ----
            noise_raw = self.noise_scale * torch.randn_like(x.feats) * voxel_mask
            noise = x.replace(noise_raw)
            t = self.sample_t(B).to(device).float()
            x_t = self.diffuse(x_masked, t, noise=noise)
            x_t_feats = x_t.feats * voxel_mask + (1.0 - voxel_mask) * bg_fill
            x_t = x_t.replace(x_t_feats)

        # ---- Forward ----
        pred = self.training_models["denoiser"](x_t, t, cond)

        # ---- Residual ----
        x_residual = pred.feats - x_masked.feats
        t_per_token = self._expand_t_to_tokens(t, x.layout, T)

        if self.loss_type == "v_loss":
            v_denom = (1 - t_per_token).clamp(min=0.05)
            diff = (x_residual / v_denom) ** 2
        else:
            diff = x_residual ** 2

        # Only mask=1 positions
        diff = diff * voxel_mask

        # ---- Exact feats_matching 3-band weighting ----
        with torch.no_grad():
            mc_mask   = (x_masked.feats < v * 2).float()
            far_mask  = (x_masked.feats > v * 3).float()
            near_mask = (1.0 - mc_mask) * (1.0 - far_mask)

        block_loss_mc   = (diff * mc_mask  ).sum(dim=1) / mc_mask  .sum(dim=1).clamp(min=1)
        block_loss_near = (diff * near_mask).sum(dim=1) / near_mask.sum(dim=1).clamp(min=1)
        block_loss_far  = (diff * far_mask ).sum(dim=1) / far_mask .sum(dim=1).clamp(min=1)

        block_loss = (
            self.surface_weight * block_loss_mc
            + 3.0 * block_loss_near
            + 1.0 * block_loss_far
        )

        # Complexity reweighting
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

        # ---- Monitoring (detached) ----
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
    # Snapshot
    # ------------------------------------------------------------------

    @torch.no_grad()
    def snapshot(self, **kwargs):
        """Snapshot v5: dil_2x pred mask + hard-fill (same as training)."""
        import copy
        from torch.utils.data import DataLoader
        import torch.distributed as dist

        num_samples = kwargs.get('num_samples', 4)
        batch_size = kwargs.get('batch_size', 1)
        steps = kwargs.get('steps', 100)
        cfg_strength = kwargs.get('cfg_strength', 1.5)
        cfg_interval = kwargs.get('cfg_interval', (0.1, 1.0))
        verbose = True

        snapshot_dir = os.path.join(self.output_dir, "samples",
                                    f"step{self.step:07d}")
        details_dir = os.path.join(snapshot_dir, "details")
        if self.is_master:
            os.makedirs(snapshot_dir, exist_ok=True)
            os.makedirs(details_dir, exist_ok=True)
            if verbose:
                print(f"\n[Snapshot v5] Step {self.step}: "
                      f"{num_samples} samples, {self.world_size} GPUs "
                      f"(dil_{self._cascade_dilate_iters}x + hard-fill)")

        if self.world_size > 1:
            dist.barrier()

        model_states = {n: m.training for n, m in self.models.items()}
        for m in self.models.values():
            m.eval()

        try:
            snap_dataset = self.test_dataset if hasattr(self, 'test_dataset') and self.test_dataset else self.dataset
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

                # Build voxel mask (same as training)
                if snap_pred_submask is not None:
                    voxel_mask = self._build_voxel_mask(snap_pred_submask.cuda())
                elif snap_submask is not None:
                    voxel_mask = self._build_voxel_mask(snap_submask.cuda())
                else:
                    voxel_mask = torch.ones_like(x_gt.feats)

                # Noise (same as training)
                noise_raw = self.noise_scale * torch.randn_like(x_gt.feats)
                noise_raw = noise_raw * voxel_mask
                noise = x_gt.replace(noise_raw)

                args = self.get_inference_cond(**data)
                args['voxel_mask'] = voxel_mask  # sampler hard-fills each step
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

                # Final hard-fill and clamp
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
                    import json, glob
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

class CascadedFeatsCFGTrainer(ClassifierFreeGuidanceMixin, CascadedFeatsTrainer):
    pass


class ImageConditionedCascadedFeatsCFGTrainer(
    ImageConditionedMixin, CascadedFeatsCFGTrainer,
):
    def get_sampler(self, **kwargs):
        return samplers.FlowGuidanceIntervalSampler(**kwargs)
