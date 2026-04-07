"""
Two-stage trainer: merges Stage 2 (mask) + Stage 3 (feats) into one model.

Key difference from SparseFlowMultiTokenTrainer:
  - No submask conditioning (submask_resolution=0)
  - Adds coarse auxiliary loss: avg_pool GT to coarse_resolution³, 
    use model's coarse_head prediction, CoarseFine loss with λ_coarse
  - No selective masking (mask is gone)
"""
from typing import *
import os
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from ...modules import sparse as sp
from ...pipelines import samplers
from ...dataset_toolkits.mesh2block import MC_THRESHOLD, BLOCK_DIM
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import ImageConditionedMixin


class TwoStageSparseFlowTrainer(FlowMatchingTrainer):
    """
    Two-stage feats trainer with coarse auxiliary loss.
    Model outputs (fine_pred, coarse_pred) when coarse_resolution > 0.
    """

    def __init__(
        self,
        *args,
        lambda_flow: float = 1.0,
        lambda_coarse: float = 0.5,
        lambda_normal: float = 0.1,
        surface_weight: float = 8.0,
        loss_type: str = "v_loss",
        complexity_boost: float = 2.0,
        cond_noise_std: float = 0.0,
        noise_scale: float = 2.0,
        coarse_resolution: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_flow = lambda_flow
        self.lambda_coarse = lambda_coarse
        self.lambda_normal = lambda_normal
        self.surface_weight = surface_weight
        self.complexity_boost = complexity_boost
        self.cond_noise_std = cond_noise_std
        self.loss_type = loss_type
        self.noise_scale = noise_scale
        self.voxel = MC_THRESHOLD
        self.coarse_resolution = coarse_resolution

        print(f"[TwoStageTrainer] loss_type={self.loss_type}, "
              f"λ_flow={lambda_flow}, λ_coarse={lambda_coarse}, "
              f"λ_normal={lambda_normal}, "
              f"surface_weight={surface_weight}, "
              f"coarse_res={coarse_resolution}, "
              f"noise_scale={noise_scale}")

    # ------------------------------------------------------------------
    # Dataloader (same as SparseFlowMultiTokenTrainer)
    # ------------------------------------------------------------------

    def prepare_dataloader(self, **kwargs):
        from ...utils.data_utils import cycle, BalancedResumableSampler
        import functools
        from torch.utils.data import DataLoader

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

    # ------------------------------------------------------------------
    # Coarse target: avg_pool 16³ → coarse_resolution³
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_coarse_target(self, fine_feats: torch.Tensor) -> torch.Tensor:
        """Avg pool [T, BLOCK_DIM³] → [T, coarse_resolution³]."""
        D = BLOCK_DIM
        R = self.coarse_resolution
        T = fine_feats.shape[0]
        vol = fine_feats.reshape(T, 1, D, D, D)
        coarse_vol = F.avg_pool3d(vol, kernel_size=D // R, stride=D // R)
        return coarse_vol.reshape(T, -1)

    # ------------------------------------------------------------------
    # Normal loss (same as SparseFlowMultiTokenTrainer)
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
        T = pred_feats.shape[0]
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
    # Training losses
    # ------------------------------------------------------------------

    def training_losses(
        self,
        x_f: sp.SparseTensor = None,
        x_c: sp.SparseTensor = None,
        cond=None,
        submask=None,
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

        with torch.no_grad():
            noise_raw = self.noise_scale * torch.randn_like(x.feats)
            noise = x.replace(noise_raw)
            t = self.sample_t(B).to(device).float()
            x_t = self.diffuse(x, t, noise=noise)

        # Model forward — returns (fine_pred_sparse, coarse_pred_tensor) 
        model_out = self.training_models["denoiser"](x_t, t, cond)

        if isinstance(model_out, tuple):
            pred, coarse_pred = model_out
        else:
            pred = model_out
            coarse_pred = None

        # ---- Fine flow loss ----
        x_residual = pred.feats - x.feats
        v = self.voxel
        t_per_token = self._expand_t_to_tokens(t, x.layout, T)

        if self.loss_type == "v_loss":
            v_denom = (1 - t_per_token).clamp(min=0.05)
            v_residual = x_residual / v_denom
            diff = v_residual ** 2
        else:
            diff = x_residual ** 2

        with torch.no_grad():
            mc_mask = x.feats < v * 2
            far_mask = x.feats > v * 3
            near_mask = ~mc_mask & ~far_mask

        mc_f = mc_mask.float()
        near_f = near_mask.float()
        far_f = far_mask.float()

        block_loss_mc = (diff * mc_f).sum(dim=1) / mc_f.sum(dim=1).clamp(min=1)
        block_loss_near = (diff * near_f).sum(dim=1) / near_f.sum(dim=1).clamp(min=1)
        block_loss_far = (diff * far_f).sum(dim=1) / far_f.sum(dim=1).clamp(min=1)

        block_loss = (
            self.surface_weight * block_loss_mc
            + 3.0 * block_loss_near
            + 1.0 * block_loss_far
        )

        per_sample_loss = torch.stack([
            block_loss[sl].mean() for sl in x.layout
        ])
        terms["flow_loss"] = per_sample_loss.mean()
        terms["loss"] = self.lambda_flow * terms["flow_loss"]

        # ---- Coarse auxiliary loss ----
        if coarse_pred is not None and self.lambda_coarse > 0:
            with torch.no_grad():
                coarse_target = self._compute_coarse_target(x.feats)  # [T, R³]

            # Same flow matching formulation on coarse space
            coarse_gt_residual = coarse_pred - coarse_target

            if self.loss_type == "v_loss":
                coarse_v = coarse_gt_residual / (1 - t_per_token).clamp(min=0.05)
                coarse_diff = coarse_v ** 2
            else:
                coarse_diff = coarse_gt_residual ** 2

            coarse_block_loss = coarse_diff.mean(dim=1)  # [T]
            per_sample_coarse = torch.stack([
                coarse_block_loss[sl].mean() for sl in x.layout
            ])
            terms["coarse_loss"] = per_sample_coarse.mean()
            terms["loss"] = terms["loss"] + self.lambda_coarse * terms["coarse_loss"]

        # ---- Normal loss ----
        if self.lambda_normal > 0:
            terms["normal_loss"] = self._normal_loss(pred.feats, x.feats)
            terms["loss"] = terms["loss"] + self.lambda_normal * terms["normal_loss"]

        # Per-region monitoring
        with torch.no_grad():
            per_sample_mc = torch.stack([block_loss_mc[sl].mean() for sl in x.layout])
            per_sample_near = torch.stack([block_loss_near[sl].mean() for sl in x.layout])
            per_sample_far = torch.stack([block_loss_far[sl].mean() for sl in x.layout])
            terms["loss_mc"] = per_sample_mc.mean()
            terms["loss_near"] = per_sample_near.mean()
            terms["loss_far"] = per_sample_far.mean()

        return terms, {}

    # ------------------------------------------------------------------
    # Snapshot: inference → mesh → normal-map rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _compose_normal_overviews(normalmap_dir, num_samples, verbose=True):
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
                    print(f"  [Overview] {tag} samples {g_idx}-{min(g_idx+15, len(paths)-1)} -> {out}")

        compose_grid(pred_paths, "pred")
        compose_grid(gt_paths, "gt")

    @torch.no_grad()
    def snapshot(self, num_samples=4, batch_size=1, steps=100,
                 cfg_strength=1.5, cfg_interval=(0.1, 1.0), **kwargs):
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
                      f"{num_samples} samples, {self.world_size} GPUs")

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
                snap_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0,
                collate_fn=getattr(snap_dataset, "collate_fn", None),
            )

            samples_per_rank = int(np.ceil(num_samples / self.world_size))
            my_start = self.rank * samples_per_rank
            my_end = min(my_start + samples_per_rank, num_samples)
            my_count = max(0, my_end - my_start)

            sampler = self.get_sampler()
            model = getattr(
                self.models["denoiser"], "module", self.models["denoiser"],
            )

            amp_dtype = torch.float16
            use_amp = (
                hasattr(self, 'accelerator')
                and self.accelerator.mixed_precision != 'no'
            )

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
                data.pop("submask", None)
                x_gt = x_f if x_f is not None else x_c

                # Two-stage: no submask, just pure noise
                noise_raw = self.noise_scale * torch.randn_like(x_gt.feats)
                noise = x_gt.replace(noise_raw)

                args = self.get_inference_cond(**data)

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    res = sampler.sample(
                        model, noise=noise, **args,
                        steps=steps, cfg_strength=cfg_strength,
                        cfg_interval=cfg_interval,
                        use_heun=True, verbose=False,
                    )
                    pred = res.samples

                # If model returns tuple, extract fine pred
                if isinstance(pred, tuple):
                    pred = pred[0]

                actual_batch = len(x_gt.layout)
                for b in range(min(actual_batch, my_count - saved)):
                    sl = x_gt.layout[b]
                    global_idx = my_start + saved
                    name = f"sample_{global_idx:03d}"

                    gt_coords = x_gt.coords[sl].cpu().numpy()
                    pred_fine = pred.feats[sl].cpu().float().numpy()
                    gt_fine_np = x_gt.feats[sl].cpu().numpy() if x_f is not None else None

                    # -- Pred mesh → normal render --
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
                                radius=1.75, verbose=False,
                            )
                        except Exception as e:
                            print(f"  [Rank {self.rank}] Render error ({name}): {e}")
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                    # -- GT mesh → normal render --
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
                                    radius=1.75, verbose=False,
                                )
                            except Exception as e:
                                print(f"  [Rank {self.rank}] GT render error ({name}): {e}")
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()

                    if verbose:
                        print(f"  [Rank {self.rank}] Sampling {saved+1}/{my_count}")
                    saved += 1

            # Clean up temp meshes
            for p in tmp_mesh_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

            if self.world_size > 1:
                dist.barrier()

            if self.is_master:
                self._compose_normal_overviews(snapshot_dir, num_samples, verbose)

            if self.is_master and verbose:
                print(f"  [Snapshot] All {num_samples} samples done")

        finally:
            for n, m in self.models.items():
                m.train(model_states[n])


# ======================================================================
# CFG / ImageConditioned variants
# ======================================================================

class TwoStageSparseFlowCFGTrainer(
    ClassifierFreeGuidanceMixin, TwoStageSparseFlowTrainer,
):
    pass


class ImageConditionedTwoStageSparseFlowCFGTrainer(
    ImageConditionedMixin, TwoStageSparseFlowCFGTrainer,
):
    def get_sampler(self, **kwargs):
        return samplers.FlowGuidanceIntervalSampler(**kwargs)
