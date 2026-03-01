"""
Stage 2 trainer: sparse flow matching for per-block binary sub-mask.

Simple x-prediction + v-loss on SUBMASK_DIM binary targets.
No surface weighting, no normal loss — just clean flow matching.
"""
from typing import *
import os
import math
import copy
import functools

import numpy as np
import torch
import torch.nn.functional as F
import utils3d
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from ...modules import sparse as sp
from ...pipelines import samplers
from ...utils.data_utils import cycle, BalancedResumableSampler
from ...representations.octree import DfsOctree as Octree
from ...renderers import OctreeRenderer
from ...dataset_toolkits.mesh2block import BLOCK_GRID, SUBMASK_RES
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import ImageConditionedMixin


class SparseMaskFlowTrainer(FlowMatchingTrainer):
    """Flow matching trainer for per-block binary sub-mask prediction."""

    def __init__(self, *args, noise_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_scale = noise_scale

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
    # Training losses
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

        # v-loss (x-prediction parameterization)
        v_target = self.compute_v_from_x_prediction(x_t, x_0, t)
        v_pred = self.compute_v_from_x_prediction(x_t, pred, t)

        terms = edict()
        terms["mse"] = F.mse_loss(v_pred.feats, v_target.feats)
        terms["loss"] = terms["mse"]

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
    # Snapshot: submask → 全局体素 → OctreeRenderer 4 视角渲染
    # ------------------------------------------------------------------

    @staticmethod
    def _submask_to_voxel_coords(block_coords: torch.Tensor,
                                  submask: torch.Tensor,
                                  threshold: float = 0.5) -> torch.Tensor:
        """
        将 per-block submask 展开为全局体素坐标.
        
        block_coords: [N, 4] (batch_idx, bx, by, bz) int
        submask: [N, SUBMASK_DIM] float
        返回: [M, 3] 全局体素坐标 (在 BLOCK_GRID*SUBMASK_RES 空间中)
        """
        R = SUBMASK_RES
        # 构造每个 sub-cell 的局部偏移 [R³, 3]
        ri = torch.arange(R, device=submask.device)
        rx, ry, rz = torch.meshgrid(ri, ri, ri, indexing='ij')
        local = torch.stack([rx, ry, rz], dim=-1).reshape(-1, 3)  # [R³, 3]

        # submask 二值化
        active = submask > threshold  # [N, R³]

        # 每个 block 的全局基坐标 (去掉 batch_idx 列)
        base = block_coords[:, 1:4].long() * R  # [N, 3]

        # 展开: global = base + local_offset
        global_coords = base[:, None, :] + local[None, :, :]  # [N, R³, 3]
        return global_coords[active]  # [M, 3]

    @staticmethod
    def _render_voxels(coords: torch.Tensor, resolution: int) -> torch.Tensor:
        """
        用 OctreeRenderer 渲染体素到 4 视角 1024×1024 拼图.
        coords: [M, 3] 全局体素坐标
        返回: [3, 1024, 1024] RGB image tensor
        """
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'

        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaw_off = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaw_off for y in yaws]
        pitches = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts, ints = [], []
        for yaw, p in zip(yaws, pitches):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(p),
                np.cos(yaw) * np.cos(p),
                np.sin(p),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30.0)).cuda()
            ext = utils3d.torch.extrinsics_look_at(
                orig, torch.zeros(3).float().cuda(),
                torch.tensor([0, 0, 1]).float().cuda())
            intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(ext)
            ints.append(intr)

        representation = Octree(
            depth=10,
            aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
            device='cuda',
            primitive='voxel',
            sh_degree=0,
            primitive_config={'solid': True},
        )
        # 归一化坐标到 [-0.5, 0.5]
        positions = coords.float() / resolution
        representation.position = positions
        representation.depth = torch.full(
            (positions.shape[0], 1),
            int(math.log2(resolution)),
            dtype=torch.uint8, device='cuda',
        )

        image = torch.zeros(3, 1024, 1024, device='cuda')
        for j, (ext, intr) in enumerate(zip(exts, ints)):
            res = renderer.render(representation, ext, intr,
                                  colors_overwrite=positions)
            r, c = j // 2, j % 2
            image[:, r * 512:(r + 1) * 512, c * 512:(c + 1) * 512] = res['color']
        return image

    @torch.no_grad()
    def run_snapshot(self, num_samples: int, batch_size: int, verbose=False):
        """
        采样 submask → 展开为全局体素 → 渲染 4 视角图像.
        返回与 base.snapshot 兼容的格式 (固定 shape image tensors).
        """
        snap_ds = (self.test_dataset if hasattr(self, 'test_dataset') and self.test_dataset
                   else copy.deepcopy(self.dataset))
        loader = DataLoader(
            snap_ds, batch_size=1, shuffle=True, num_workers=0,
            collate_fn=snap_ds.collate_fn if hasattr(snap_ds, 'collate_fn') else None,
        )
        sampler = self.get_sampler()
        vis_resolution = BLOCK_GRID * SUBMASK_RES  # 64*8 = 512

        images_gt, images_pred = [], []
        for i in range(num_samples):
            data = next(iter(loader))
            data = {k: (v.cuda() if isinstance(v, torch.Tensor) else
                        v.to('cuda') if isinstance(v, sp.SparseTensor) else v)
                    for k, v in data.items()}

            x_0 = data.pop('x_0')
            noise = x_0.replace(self.noise_scale * torch.randn_like(x_0.feats))

            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models['denoiser'], noise=noise,
                **args, steps=50, cfg_strength=3.0, verbose=verbose,
            )
            pred_feats = (res.samples.feats if hasattr(res.samples, 'feats')
                          else res.samples)

            # GT 渲染
            gt_coords = self._submask_to_voxel_coords(x_0.coords, x_0.feats)
            if gt_coords.shape[0] > 0:
                images_gt.append(self._render_voxels(gt_coords, vis_resolution))
            else:
                images_gt.append(torch.zeros(3, 1024, 1024, device='cuda'))

            # Pred 渲染
            pred_coords = self._submask_to_voxel_coords(x_0.coords, pred_feats)
            if pred_coords.shape[0] > 0:
                images_pred.append(self._render_voxels(pred_coords, vis_resolution))
            else:
                images_pred.append(torch.zeros(3, 1024, 1024, device='cuda'))

        return {
            'submask_gt': {
                'value': torch.stack(images_gt),
                'type': 'image',
            },
            'submask_pred': {
                'value': torch.stack(images_pred),
                'type': 'image',
            },
        }

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
