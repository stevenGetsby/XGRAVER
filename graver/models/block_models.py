from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F
from ..modules import sparse as sp

class BlockEncoder3D(nn.Module):
    def __init__(self, in_res=5, hidden_dim=1024):  # ← in_res=5 for coarse
        super().__init__()
        self.in_res = in_res
        self.in_channels = 1  # ← 添加这个属性
        self.down_res = in_res // 2
        
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),  # ← 输入 1 通道
            nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Flatten()
        )
        
        flatten_dim = 64 * (self.down_res ** 3)
        self.proj = nn.Linear(flatten_dim, hidden_dim)

    def forward(self, x_feats):
        # x_feats: [N, in_res^3] 单通道
        N = x_feats.shape[0]
        # 改为: view(-1, 1, in_res, in_res, in_res)
        x = x_feats.view(N, 1, self.in_res, self.in_res, self.in_res)
        feat = self.net(x)
        return self.proj(feat)
    
    
def pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    B, C, H, W, D = x.shape
    C_ = C // scale_factor**3
    x = x.reshape(B, C_, scale_factor, scale_factor, scale_factor, H, W, D)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(B, C_, H*scale_factor, W*scale_factor, D*scale_factor)


class PixelShuffle3D(nn.Module):
    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        return pixel_shuffle_3d(x, self.scale_factor)




class BlockDecoder3D(nn.Module):
    def __init__(self, hidden_dim=1024, out_res=17):
        super().__init__()
        self.out_res = out_res
        
        # 固定从 4³ 开始，用插值到目标分辨率
        self.start_res = 4
        self.start_dim = 64
        
        self.proj = nn.Linear(hidden_dim, self.start_dim * (self.start_res ** 3))
        # = 64 * 64 = 4096 维，合理！
        
        # 卷积细化（在 4³ 分辨率上）
        self.refine = nn.Sequential(
            nn.Conv3d(self.start_dim, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv3d(32, 1, 3, padding=1),
        )

    def forward_chunk(self, x_chunk):
        N = x_chunk.shape[0]
        
        # [N, hidden_dim] -> [N, start_dim, 4, 4, 4]
        x = self.proj(x_chunk).view(N, self.start_dim, self.start_res, self.start_res, self.start_res)
        
        # 卷积细化
        if self.training:
            x = checkpoint(self.refine, x, use_reentrant=False)
        else:
            x = self.refine(x)
        
        # [N, 1, 4, 4, 4] -> [N, 1, out_res, out_res, out_res]
        x = F.interpolate(x, size=(self.out_res, self.out_res, self.out_res), 
                          mode='trilinear', align_corners=True)
        
        return x.view(N, -1)

    def forward(self, x, chunk_size: int = 30000, chunk: bool = False):
        if not chunk:
            return self.forward_chunk(x)
        
        N = x.shape[0]
        if N <= chunk_size:
            return self.forward_chunk(x)
        
        outputs = []
        for i in range(0, N, chunk_size):
            x_chunk = x[i:min(i + chunk_size, N)]
            outputs.append(self.forward_chunk(x_chunk))
        
        return torch.cat(outputs, dim=0)


# ======================================================================
# Surface-Aware Block Encoder (adaptive depth, chunked, checkpointed)
# ======================================================================

class SurfaceAwareBlockEncoder3D(nn.Module):
    """
    自适应深度 3D 卷积编码器。

    根据输入分辨率自动添加 stride-2 下采样层，将空间尺寸缩减到 ≤5。
    支持 chunked forward + gradient checkpoint，使得高分辨率块
    (13³, 17³, 21³+) 在大量 token (T~30000) 下可训练。

    下采样规则 (Conv3d k=3, s=2, p=1):
      R=13 → 7 → 4   (2 层, 128×4³=8192 → 1024)
      R=9  → 5        (1 层, 64×5³=8000 → 1024)
      R=7  → 4        (1 层, 64×4³=4096 → 1024)
    """

    def __init__(self, in_res: int = 13, hidden_dim: int = 1024, chunk_size: int = 4096):
        super().__init__()
        self.in_res = in_res
        self.in_channels = 1
        self.chunk_size = chunk_size

        layers = [
            nn.Conv3d(1, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
        ]
        ch = 32
        r = in_res

        # 自适应下采样: 缩减到空间 ≤5
        while r > 5:
            ch_out = min(ch * 2, 128)
            layers.extend([
                nn.Conv3d(ch, ch_out, 3, stride=2, padding=1),
                nn.GroupNorm(min(16, ch_out), ch_out),
                nn.SiLU(),
            ])
            r = (r - 1) // 2 + 1   # Conv3d(k=3,s=2,p=1) output formula
            ch = ch_out

        layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(ch * r ** 3, hidden_dim)

    def _forward(self, x_feats: torch.Tensor) -> torch.Tensor:
        N = x_feats.shape[0]
        x = x_feats.view(N, 1, self.in_res, self.in_res, self.in_res)
        return self.proj(self.net(x))

    def forward(self, x_feats: torch.Tensor) -> torch.Tensor:
        N = x_feats.shape[0]
        cs = self.chunk_size

        if N <= cs:
            if self.training:
                return checkpoint(self._forward, x_feats, use_reentrant=False)
            return self._forward(x_feats)

        outputs = []
        for i in range(0, N, cs):
            chunk = x_feats[i:min(i + cs, N)]
            if self.training:
                outputs.append(checkpoint(self._forward, chunk, use_reentrant=False))
            else:
                outputs.append(self._forward(chunk))
        return torch.cat(outputs, dim=0)


# ======================================================================
# Surface-Gated Block Decoder
# ======================================================================

class SurfaceGatedBlockDecoder3D(nn.Module):
    """
    表面门控块解码器: output = base + gate × detail

    架构:
      token [1024]
        → Linear → [C, 4³] 特征体
        ├─ base_head  → Conv(4³) → trilinear → R³  (平滑低频 UDF)
        ├─ up_shared  → ConvTranspose(4³→8³)
        │   ├─ detail_head → Conv(8³) → trilinear → R³  (高频表面残差)
        │   └─ gate_head   → Conv(8³) → trilinear → sigmoid → R³  (表面掩码)
        └─ output = base + gate × detail

    核心思想:
      - base 只能表达 4³ 精度的平滑场 → 捕获低频
      - detail 在 8³ 精度工作 → 能表达更高频的表面细节
      - gate 自动学会在表面打开(≈1), 非表面关闭(≈0)
      - 效果: 模型容量自适应集中在表面区域

    初始化策略:
      - base/detail 输出层 zero-init → 起始输出 ≈ 0, 稳定训练
      - gate bias = -2 → sigmoid(-2) ≈ 0.12, 初始抑制 detail
      - 训练初期先学好 base, gate 逐渐对表面打开
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        out_res: int = 13,
        base_res: int = 4,
        chunk_size: int = 4096,
    ):
        super().__init__()
        self.out_res = out_res
        self.base_res = base_res
        self.chunk_size = chunk_size

        start_dim = 128
        self.proj = nn.Linear(hidden_dim, start_dim * base_res ** 3)

        # === Base 分支: 4³ → 平滑 UDF ===
        self.base_head = nn.Sequential(
            nn.Conv3d(start_dim, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv3d(64, 1, 3, padding=1),
        )

        # === 共享上采样: 4³ → 8³ ===
        self.up_shared = nn.Sequential(
            nn.ConvTranspose3d(start_dim, 64, 4, stride=2, padding=1),  # 4→8
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
        )

        # === Detail 分支: 8³ → 高频表面残差 ===
        self.detail_head = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
            nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv3d(32, 1, 3, padding=1),
        )

        # === Gate 分支: 8³ → sigmoid 表面掩码 ===
        self.gate_head = nn.Sequential(
            nn.Conv3d(64, 16, 3, padding=1),
            nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv3d(16, 1, 3, padding=1),
        )

    def forward_chunk(
        self, x: torch.Tensor, return_components: bool = False,
    ):
        N = x.shape[0]
        R = self.out_res
        BR = self.base_res

        feat = self.proj(x).view(N, -1, BR, BR, BR)

        # Base: 4³ → trilinear → R³ (平滑低频)
        base = self.base_head(feat)
        base = F.interpolate(
            base, size=(R, R, R), mode='trilinear', align_corners=True,
        )

        # 共享上采样: 4³ → 8³
        feat_up = self.up_shared(feat)

        # Detail: 8³ → trilinear → R³ (表面残差)
        detail = self.detail_head(feat_up)
        detail = F.interpolate(
            detail, size=(R, R, R), mode='trilinear', align_corners=True,
        )

        # Gate: 8³ → trilinear → sigmoid → R³ (表面掩码)
        gate = self.gate_head(feat_up)
        gate = torch.sigmoid(F.interpolate(
            gate, size=(R, R, R), mode='trilinear', align_corners=True,
        ))

        output = base + gate * detail     # [N, 1, R, R, R]

        if return_components:
            return (
                output.reshape(N, -1),
                base.reshape(N, -1),
                gate.reshape(N, -1),
            )
        return output.reshape(N, -1)

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
        chunk_size: int = None,
    ):
        cs = chunk_size or self.chunk_size
        N = x.shape[0]

        if N <= cs:
            if self.training:
                return checkpoint(
                    self.forward_chunk, x, return_components,
                    use_reentrant=False,
                )
            return self.forward_chunk(x, return_components)

        # 分块处理
        results = []
        for i in range(0, N, cs):
            chunk = x[i:min(i + cs, N)]
            if self.training:
                results.append(checkpoint(
                    self.forward_chunk, chunk, return_components,
                    use_reentrant=False,
                ))
            else:
                results.append(self.forward_chunk(chunk, return_components))

        if return_components:
            return (
                torch.cat([r[0] for r in results], dim=0),
                torch.cat([r[1] for r in results], dim=0),
                torch.cat([r[2] for r in results], dim=0),
            )
        return torch.cat(results, dim=0)


# ======================================================================
# Multi-Token Patch Encoder:  13³ block → k sub-tokens
# ======================================================================

class PatchBlockEncoder3D(nn.Module):
    """
    多 token 块编码器: 13³ → Conv3d 下采样 → [s³, d] sub-tokens

    数据流:
      [T, 2197] → view [T, 1, 13, 13, 13]
        → Conv3d(1→32, k3, p1)        [T, 32, 13, 13, 13]
        → Conv3d(32→64, k3, s2, p1)   [T, 64,  7,  7,  7]
        → Conv3d(64→C,  k3, s2, p1)   [T,  C,  4,  4,  4]
        → Conv3d(C→C,   k4, s2, p1)   [T,  C,  2,  2,  2]   (s=2, k=8)
        → reshape                      [T,  8,  C]
        → Linear(C → d)               [T,  8,  d]

    其中 k = s³ = 8 sub-tokens, d = model_channels (512)
    """

    def __init__(
        self,
        in_res: int = 13,
        hidden_dim: int = 512,
        sub_grid: int = 2,
        chunk_size: int = 4096,
    ):
        super().__init__()
        self.in_res = in_res
        self.in_channels = 1
        self.sub_grid = sub_grid
        self.num_sub_tokens = sub_grid ** 3   # 8
        self.chunk_size = chunk_size

        # 构建下采样卷积: 13 → 7 → 4 → 2
        layers = [
            nn.Conv3d(1, 32, 3, padding=1),
            nn.GroupNorm(4, 32), nn.SiLU(),
        ]
        ch = 32
        r = in_res   # 13

        # 下采样到 sub_grid (=2)
        target = sub_grid
        while r > target:
            ch_out = min(ch * 2, 256)
            layers.extend([
                nn.Conv3d(ch, ch_out, 3, stride=2, padding=1),
                nn.GroupNorm(min(16, ch_out), ch_out),
                nn.SiLU(),
            ])
            r = (r + 1) // 2   # Conv3d(k=3,s=2,p=1): ceil(r/2)
            ch = ch_out

        self.conv = nn.Sequential(*layers)
        # ch 通道 @ sub_grid³ 空间 → project 到 hidden_dim
        self.proj = nn.Linear(ch, hidden_dim)
        self._conv_out_channels = ch

    def _forward(self, x_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_feats: [N, in_res³]
        Returns:
            [N, k, d]  其中 k = sub_grid³ = 8
        """
        N = x_feats.shape[0]
        s = self.sub_grid
        x = x_feats.view(N, 1, self.in_res, self.in_res, self.in_res)
        feat = self.conv(x)                            # [N, C, s, s, s]
        feat = feat.reshape(N, self._conv_out_channels, s ** 3)  # [N, C, k]
        feat = feat.permute(0, 2, 1)                   # [N, k, C]
        return self.proj(feat)                          # [N, k, d]

    def forward(self, x_feats: torch.Tensor) -> torch.Tensor:
        N = x_feats.shape[0]
        cs = self.chunk_size
        if N <= cs:
            if self.training:
                return checkpoint(self._forward, x_feats, use_reentrant=False)
            return self._forward(x_feats)
        outputs = []
        for i in range(0, N, cs):
            chunk = x_feats[i:min(i + cs, N)]
            if self.training:
                outputs.append(checkpoint(self._forward, chunk, use_reentrant=False))
            else:
                outputs.append(self._forward(chunk))
        return torch.cat(outputs, dim=0)


# ======================================================================
# Multi-Token Patch Decoder:  k sub-tokens → 13³ block
# ======================================================================

class PatchBlockDecoder3D(nn.Module):
    """
    多 token 块解码器: [k, d] sub-tokens → ConvTranspose3d → out_res³

    自动构建上采样路径:
      sub_grid → sub_grid*2 → sub_grid*4 → ... → ≥ out_res → crop

    例:
      out_res=13, sub_grid=2: 2→4→8→16 → crop 13    (3 层)
      out_res=17, sub_grid=2: 2→4→8→16→32 → crop 17  (4 层)

    全程 ConvTranspose3d, 无 trilinear 插值。
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        out_res: int = 13,
        sub_grid: int = 2,
        chunk_size: int = 4096,
    ):
        super().__init__()
        self.out_res = out_res
        self.sub_grid = sub_grid
        self.num_sub_tokens = sub_grid ** 3
        self.chunk_size = chunk_size

        # 计算需要多少层上采样: sub_grid * 2^n >= out_res
        n_up = 0
        r = sub_grid
        while r < out_res:
            r *= 2
            n_up += 1

        # 通道规划: start_ch → 逐层减半, 最低 32
        start_ch = min(256, 32 * (2 ** n_up))
        self.proj = nn.Linear(hidden_dim, start_ch)
        self._start_ch = start_ch

        # 自动构建上采样层
        layers = []
        ch = start_ch
        for i in range(n_up):
            ch_out = max(32, ch // 2)
            layers.extend([
                nn.ConvTranspose3d(ch, ch_out, 4, stride=2, padding=1),
                nn.GroupNorm(min(16, ch_out), ch_out), nn.SiLU(),
                nn.Conv3d(ch_out, ch_out, 3, padding=1),
                nn.GroupNorm(min(16, ch_out), ch_out), nn.SiLU(),
            ])
            ch = ch_out
        self.up = nn.Sequential(*layers)

        # 最终: crop → 1 通道输出
        self.head = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(16, ch), ch), nn.SiLU(),
            nn.Conv3d(ch, 1, 3, padding=1),
        )

        # 上采样后的空间尺寸
        self._up_res = sub_grid * (2 ** n_up)
        print(f"[PatchBlockDecoder3D] {sub_grid}³→{self._up_res}³→crop {out_res}³, "
              f"{n_up} upsample layers, start_ch={start_ch}")

    def forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        s = self.sub_grid
        R = self.out_res

        feat = self.proj(x)                            # [N, k, start_ch]
        feat = feat.permute(0, 2, 1)                   # [N, start_ch, k]
        feat = feat.reshape(N, self._start_ch, s, s, s)

        feat = self.up(feat)                           # [N, ch, up_res, up_res, up_res]

        # 中心裁剪到 out_res (16→15: start=0, 取 [:15])
        up_size = feat.shape[2]
        if up_size != R:
            start = (up_size - R) // 2
            feat = feat[:, :, start:start+R, start:start+R, start:start+R]

        out = self.head(feat)                          # [N, 1, R, R, R]
        return out.reshape(N, -1)                      # [N, R³]

    def forward(self, x: torch.Tensor, chunk_size: int = None) -> torch.Tensor:
        cs = chunk_size or self.chunk_size
        N = x.shape[0]

        if N <= cs:
            if self.training:
                return checkpoint(self.forward_chunk, x, use_reentrant=False)
            return self.forward_chunk(x)

        results = []
        for i in range(0, N, cs):
            chunk = x[i:min(i + cs, N)]
            if self.training:
                results.append(checkpoint(self.forward_chunk, chunk, use_reentrant=False))
            else:
                results.append(self.forward_chunk(chunk))
        return torch.cat(results, dim=0)