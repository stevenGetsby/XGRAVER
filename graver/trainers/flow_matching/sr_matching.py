from typing import *
import os
import copy
import functools
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from ...modules import sparse as sp
from ...pipelines import samplers
from ...utils.data_utils import cycle, BalancedResumableSampler
from ...dataset_toolkits.mesh2block import MC_THRESHOLD, BLOCK_DIM
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import ImageConditionedMixin


class SparseFlowMultiTokenTrainer(FlowMatchingTrainer):

    def __init__(
        self,
        *args,
        lambda_flow: float = 1.0,
        lambda_normal: float = 0.1,
        surface_weight: float = 10.0,
        loss_type: str = "v_loss",
        complexity_boost: float = 4.0,
        cond_noise_std: float = 0.0,
        noise_scale: float = 1.0,
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

        print(f"[MultiTokenTrainer] loss_type={self.loss_type}, "
              f"λ_flow={lambda_flow}, λ_normal={lambda_normal}, "
              f"surface_weight={surface_weight}, "
              f"complexity_boost={complexity_boost}, "
              f"noise_scale={noise_scale}")

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
        num_workers = min(4, max(1, os.cpu_count() // num_gpus))

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
            collate_fn=functools.partial(
                self.dataset.collate_fn, split_size=self.batch_split,
            ),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)
        print(f"[DataLoader] num_workers={num_workers}, "
              f"batch_size={self.batch_size_per_gpu}")

    # ------------------------------------------------------------------
    # SparseTensor diffuse / v-prediction
    # ------------------------------------------------------------------

    def diffuse(self, x_0, t, noise=None):
        if isinstance(x_0, sp.SparseTensor):
            if noise is None:
                noise = x_0.replace(torch.randn_like(x_0.feats))
            t_expand = t.view(-1, 1)
            T = x_0.feats.shape[0]
            t_per_token = torch.zeros(T, 1, device=x_0.device)
            for i, sl in enumerate(x_0.layout):
                t_per_token[sl] = t_expand[i]
            x_t_feats = t_per_token * x_0.feats + (1 - t_per_token) * noise.feats
            return x_0.replace(x_t_feats)
        return super().diffuse(x_0, t, noise=noise)

    def compute_v_from_x_prediction(self, x_t, x_pred, t):
        if isinstance(x_t, sp.SparseTensor):
            T = x_t.feats.shape[0]
            t_per_token = torch.zeros(T, 1, device=x_t.device)
            for i, sl in enumerate(x_t.layout):
                t_per_token[sl] = t[i]
            denom = (1 - t_per_token).clamp(min=0.05)
            v_feats = (x_pred.feats - x_t.feats) / denom
            return x_t.replace(v_feats)
        return super().compute_v_from_x_prediction(x_t, x_pred, t)

    # ------------------------------------------------------------------
    # Normal loss (UDF 梯度方向一致性)
    # ------------------------------------------------------------------

    @staticmethod
    def _grad_3d(vol: torch.Tensor) -> torch.Tensor:
        """Central differences on [N, 1, D, D, D] → [N, 3, D, D, D]"""
        vp = F.pad(vol, (1, 1, 1, 1, 1, 1), mode='replicate')
        gx = vp[:, :, 2:, 1:-1, 1:-1] - vp[:, :, :-2, 1:-1, 1:-1]
        gy = vp[:, :, 1:-1, 2:, 1:-1] - vp[:, :, 1:-1, :-2, 1:-1]
        gz = vp[:, :, 1:-1, 1:-1, 2:] - vp[:, :, 1:-1, 1:-1, :-2]
        return torch.cat([gx, gy, gz], dim=1) * 0.5

    def _normal_loss(self, pred_feats: torch.Tensor, gt_feats: torch.Tensor) -> torch.Tensor:
        D = BLOCK_DIM
        T = pred_feats.shape[0]
        v = self.voxel

        # === 筛选含表面的 block: any(UDF < 2v) per block ===
        with torch.no_grad():
            surface_mask = (gt_feats < v * 2).any(dim=1)     # [T] bool
        T_surface = surface_mask.sum().item()
        if T_surface == 0:
            return torch.tensor(0.0, device=pred_feats.device)

        # 只取含表面的 block
        pred_surface = pred_feats[surface_mask]               # [T_s, D³]
        gt_surface = gt_feats[surface_mask]                   # [T_s, D³]

        pred_vol = pred_surface.reshape(T_surface, 1, D, D, D)
        pred_grad = self._grad_3d(pred_vol)                  # [T_s, 3, D, D, D]

        with torch.no_grad():
            gt_vol = gt_surface.reshape(T_surface, 1, D, D, D)
            gt_grad = self._grad_3d(gt_vol)
            gt_norm = gt_grad.norm(dim=1, keepdim=True).clamp(min=1e-4)
            gt_dir = gt_grad / gt_norm                       # [T_s, 3, D, D, D]

            # MC 插值区 (< 2 voxels): 法线只在此处影响 mesh 质量
            w = (gt_surface < v * 2).float()                  # 二值 mask
            w_3d = w.reshape(T_surface, 1, D, D, D)

        pred_norm = pred_grad.norm(dim=1, keepdim=True).clamp(min=1e-4)
        pred_dir = pred_grad / pred_norm

        # 1. 方向一致性 (Cosine Similarity)
        cos_sim = (pred_dir * gt_dir).sum(dim=1, keepdim=True)  # [T_s, 1, D, D, D]
        dir_loss = (1.0 - cos_sim) * w_3d

        # 2. Eikonal Loss (UDF 梯度模长必须为 1)
        eikonal_loss = (pred_norm - 1.0).abs() * w_3d

        # 3. Edge-aware weighting: 高曲率区域(法线变化大)加权更高
        with torch.no_grad():
            gt_laplacian = self._grad_3d(gt_norm)  # 梯度模长的梯度 → 曲率代理
            curvature = gt_laplacian.norm(dim=1, keepdim=True)
            edge_w = 1.0 + 2.0 * (curvature / curvature.max().clamp(min=1e-6))  # [1, 3]

        loss = (dir_loss + 0.5 * eikonal_loss) * edge_w

        return loss.mean()

    @torch.no_grad()
    def _neighborhood_complexity(
        self,
        gt_feats: torch.Tensor,
        coords: torch.Tensor,
        layout: List[slice],
        voxel: float,
        boost: float,
    ) -> torch.Tensor:
        """
        邻域曲率复杂度: 连通 block 间曲率变化越剧烈 → 权重越大.
        复杂几何 (耳朵、手指、褶皱) = 多个邻近 block 曲率急变的区域.
        """
        D = BLOCK_DIM
        T = gt_feats.shape[0]
        device = gt_feats.device

        # 1. Per-block curvature: 表面体素内梯度模长的标准差
        surface_mask = (gt_feats < voxel * 2).any(dim=1)  # [T]
        block_curvature = torch.zeros(T, device=device)
        T_s = surface_mask.sum().item()

        if T_s > 0:
            gt_vol = gt_feats[surface_mask].reshape(T_s, 1, D, D, D)
            grad = self._grad_3d(gt_vol)
            grad_norm = grad.norm(dim=1).reshape(T_s, -1)  # [T_s, D³]
            w = (gt_feats[surface_mask] < voxel * 2).float()  # [T_s, D³]
            w_sum = w.sum(dim=1).clamp(min=1)
            mean_gn = (grad_norm * w).sum(dim=1) / w_sum
            var_gn = ((grad_norm - mean_gn.unsqueeze(1)).pow(2) * w).sum(dim=1) / w_sum
            block_curvature[surface_mask] = var_gn.sqrt()

        # 2. 邻域曲率梯度: 6-连通邻居间曲率差异
        complexity = torch.zeros(T, device=device)
        offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

        for b, sl in enumerate(layout):
            bc = coords[sl][:, 1:].long()  # [N_b, 3]
            curv = block_curvature[sl]
            is_surf = surface_mask[sl]
            N_b = bc.shape[0]
            if N_b <= 1:
                complexity[sl] = curv
                continue

            # Hash → index (向量化邻居查找)
            keys_t = bc[:, 0] * 100003 + bc[:, 1] * 1009 + bc[:, 2]
            key_to_idx = {k: i for i, k in enumerate(keys_t.tolist())}

            curv_diff_sum = torch.zeros(N_b, device=device)
            n_surf_nb = torch.zeros(N_b, device=device)

            for dx, dy, dz in offsets:
                nb_keys = ((bc[:, 0] + dx) * 100003
                           + (bc[:, 1] + dy) * 1009
                           + (bc[:, 2] + dz))
                nb_idx = torch.tensor(
                    [key_to_idx.get(k, -1) for k in nb_keys.tolist()],
                    device=device, dtype=torch.long,
                )
                valid = nb_idx >= 0
                safe_idx = nb_idx.clamp(min=0)
                valid_surf = valid & is_surf[safe_idx]

                nb_curv = curv[safe_idx]
                curv_diff_sum += (curv - nb_curv).abs() * valid_surf.float()
                n_surf_nb += valid_surf.float()

            has_nb = n_surf_nb > 0
            mean_diff = torch.where(
                has_nb, curv_diff_sum / n_surf_nb,
                torch.zeros_like(n_surf_nb),
            )

            # 复杂度 = 自身曲率 + 邻域曲率变化, 再按邻居密度放大
            local_c = (curv + mean_diff) * (1.0 + n_surf_nb / 6.0)
            complexity[sl] = local_c

        # Normalize → [1, 1 + boost]
        cmax, cmin = complexity.max(), complexity.min()
        if cmax > cmin:
            complexity = (complexity - cmin) / (cmax - cmin)
        else:
            complexity.zero_()

        return 1.0 + boost * complexity

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------

    def training_losses(
        self,
        x_f: sp.SparseTensor = None,
        x_c: sp.SparseTensor = None,
        cond=None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        x = x_f if x_f is not None else x_c
        assert x is not None

        B = len(x.layout)
        T = x.feats.shape[0]
        device = x.device
        terms = edict()

        _profile = os.environ.get('XGRAVER_PROFILE', '0') == '1'
        if _profile:
            torch.cuda.synchronize()
            _t0 = time.time()

        cond = self.get_cond(cond, **kwargs)

        # 条件噪声增强: 训练时给 DINOv2 特征加微弱噪声, 防止过拟合
        if self.cond_noise_std > 0:
            cond = cond + self.cond_noise_std * torch.randn_like(cond)

        if _profile:
            torch.cuda.synchronize()
            _t_cond = time.time() - _t0

        with torch.no_grad():
            noise = x.replace(
                self.noise_scale * torch.randn_like(x.feats)
            )
            t = self.sample_t(B).to(device).float()
            x_t = self.diffuse(x, t, noise=noise)

        if _profile:
            torch.cuda.synchronize()
            _t_prep = time.time() - _t0 - _t_cond

        pred = self.training_models["denoiser"](x_t, t, cond)

        if _profile:
            torch.cuda.synchronize()
            _t_fwd = time.time() - _t0 - _t_cond - _t_prep

        # 模型直接在原始 UDF [0,1] 空间预测 x_0
        x_residual = pred.feats - x.feats
        v = self.voxel

        t_per_token = torch.zeros(T, 1, device=device)
        for i, sl in enumerate(x.layout):
            t_per_token[sl] = t[i]

        if self.loss_type == "v_loss":
            v_denom = (1 - t_per_token).clamp(min=0.05)
            v_residual = x_residual / v_denom
            diff = v_residual ** 2 + 0.2 * v_residual.abs()
        else:
            diff = x_residual ** 2 + 0.2 * x_residual.abs()

        with torch.no_grad():
            mc_mask   = x.feats < v * 2
            far_mask  = x.feats > v * 3
            near_mask = ~mc_mask & ~far_mask

        mc_f   = mc_mask.float()
        near_f = near_mask.float()
        far_f  = far_mask.float()

        block_loss_mc = (diff * mc_f).sum(dim=1) / mc_f.sum(dim=1).clamp(min=1)

        block_loss_near = (diff * near_f).sum(dim=1) / near_f.sum(dim=1).clamp(min=1)
        block_loss_far  = (diff * far_f).sum(dim=1) / far_f.sum(dim=1).clamp(min=1)

        block_loss = (
            self.surface_weight * block_loss_mc
            + 3.0 * block_loss_near
            + 1.0 * block_loss_far
        )

        # Complexity reweighting: 连通 block 曲率变化越陡峭 → 权重越大
        with torch.no_grad():
            complexity_w = self._neighborhood_complexity(
                x.feats, x.coords, x.layout,
                self.voxel, self.complexity_boost,
            )

        block_loss = block_loss * complexity_w
        terms["flow_loss"] = block_loss.mean()
        terms["complexity_avg"] = complexity_w.mean()

        terms["loss_mc"]   = block_loss_mc.mean()
        terms["loss_near"] = block_loss_near.mean()
        terms["loss_far"]  = block_loss_far.mean()

        terms["loss"] = self.lambda_flow * terms["flow_loss"]

        # Normal Loss
        if self.lambda_normal > 0:
            terms["normal_loss"] = self._normal_loss(pred.feats, x.feats)
            terms["loss"] = terms["loss"] + self.lambda_normal * terms["normal_loss"]

        if _profile:
            torch.cuda.synchronize()
            _t_loss = time.time() - _t0 - _t_cond - _t_prep - _t_fwd
            _t_total = time.time() - _t0
            print(f"  [Profile] tokens={T} | "
                  f"cond={_t_cond*1000:.1f}ms "
                  f"prep={_t_prep*1000:.1f}ms "
                  f"fwd={_t_fwd*1000:.1f}ms "
                  f"loss={_t_loss*1000:.1f}ms "
                  f"total={_t_total*1000:.1f}ms")

        return terms, {}

    # ------------------------------------------------------------------
    # 合成大图: 每 16 张法线图 → 4×4 overview
    # ------------------------------------------------------------------

    @staticmethod
    def _compose_normal_overviews(
        normalmap_dir: str,
        num_samples: int,
        verbose: bool = True,
    ):
        from PIL import Image

        details_dir = os.path.join(normalmap_dir, "details")

        # 收集 pred / gt 法线图 (从 details/ 子目录读取)
        pred_paths, gt_paths = [], []
        for i in range(num_samples):
            p = os.path.join(details_dir, f"sample_{i:03d}_normal.jpg")
            g = os.path.join(details_dir, f"sample_{i:03d}_gt_normal.jpg")
            pred_paths.append(p if os.path.exists(p) else None)
            gt_paths.append(g if os.path.exists(g) else None)

        def compose_grid(paths, tag):
            """每 16 张拼 4×4"""
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

                # 取分辨率 (以第一张有效图为准)
                ref = next(im for im in imgs if im is not None)
                w, h = ref.size

                # 补齐: 不足 16 张用黑图占位
                while len(imgs) < 16:
                    imgs.append(None)
                imgs = [
                    im if im is not None else Image.new("RGB", (w, h), (0, 0, 0))
                    for im in imgs
                ]

                # 4列 × 4行
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
    # Snapshot: 推理 + 转 mesh + 渲染法线图 (所有卡参与, 全并行)
    # 覆写 base.snapshot(), 不走 dense tensor 可视化路径
    # ------------------------------------------------------------------

    @torch.no_grad()
    def snapshot(
        self,
        num_samples: int = 4,
        batch_size: int = 1,
        steps: int = 50,
        cfg_strength: float = 1.5,
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
                      f"{num_samples} samples, {self.world_size} GPUs")

        if self.world_size > 1:
            dist.barrier()

        model_states = {n: m.training for n, m in self.models.items()}
        for m in self.models.values():
            m.eval()

        try:
            # 优先使用 test_dataset (固定测试集), 否则 fallback 到训练集
            if hasattr(self, 'test_dataset') and self.test_dataset is not None:
                snap_dataset = self.test_dataset
                if verbose:
                    print(f"  Using test_dataset ({len(snap_dataset)} samples)")
            else:
                snap_dataset = copy.deepcopy(self.dataset)
                if verbose:
                    print(f"  Using train dataset (deepcopy, {len(snap_dataset)} samples)")
            dataloader = DataLoader(
                snap_dataset,
                batch_size=batch_size,
                shuffle=True,
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
            tmp_mesh_paths = []  # 临时 ply 路径, 渲染后删除

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
                x_gt = x_f if x_f is not None else x_c

                noise = x_gt.replace(
                    self.noise_scale * torch.randn_like(x_gt.feats)
                )
                args = self.get_inference_cond(**data)

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    res = sampler.sample(
                        model, noise=noise, **args,
                        steps=steps, cfg_strength=cfg_strength,
                        verbose=False,
                    )
                    pred = res.samples

                actual_batch = len(x_gt.layout)
                for b in range(min(actual_batch, my_count - saved)):
                    sl = x_gt.layout[b]
                    global_idx = my_start + saved
                    name = f"sample_{global_idx:03d}"

                    gt_coords = x_gt.coords[sl].cpu().numpy()
                    pred_fine = pred.feats[sl].cpu().float().numpy()
                    gt_fine_np = None
                    if x_f is not None:
                        gt_fine_np = x_f.feats[sl].cpu().numpy()

                    # -- Pred: 转临时 mesh → 渲染法线图 → 删 mesh --
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
                                tmp_ply, normal_path,
                                resolution=1024, radius=1.75,
                                verbose=False,
                            )
                        except Exception as e:
                            print(f"  [Rank {self.rank}] Render error ({name}): {e}")
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                    # -- GT: 转临时 mesh → 渲染法线图 → 删 mesh --
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
                                    gt_tmp_ply, gt_normal_path,
                                    resolution=1024, radius=1.75,
                                    verbose=False,
                                )
                            except Exception as e:
                                print(f"  [Rank {self.rank}] GT render error ({name}): {e}")
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()

                    if verbose:
                        print(f"  [Rank {self.rank}] Sampling {saved+1}/{my_count}")
                    saved += 1

            # 清理临时 ply
            for p in tmp_mesh_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

            # 同步
            if self.world_size > 1:
                dist.barrier()

            # 合成大图
            if self.is_master:
                self._compose_normal_overviews(snapshot_dir, num_samples, verbose)

            if self.is_master and verbose:
                print(f"  [Snapshot] All {num_samples} samples done")

        finally:
            for n, m in self.models.items():
                m.train(model_states[n])


# ======================================================================
# CFG / ImageConditioned 变体
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
