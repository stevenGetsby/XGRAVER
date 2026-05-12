"""
DirectMaskModel: Sparse Transformer for direct binary mask prediction.

Input:  Fourier-encoded 3D coords + DINOv2 image features
Output: Binary submask logits per block (512 values each)

Architecture:
    - Fourier position encoding of block coords → Linear → hidden
    - DINOv2 CLS token → Linear → AdaLN modulation
    - N transformer blocks (Pooled + Windowed self-attn, cross-attn to DINOv2, MLP)
    - Optional coarse-to-fine decode head for 3D-aware 8^3 mask prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

from ..modules.norm import LayerNorm32
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules import sparse as sp
from ..modules.sparse.transformer import (
    ModulatedSparseTransformerCrossBlock,
    PooledSparseTransformerCrossBlock,
)


class DirectMaskModel(nn.Module):
    """
    Direct mask prediction model.
    Predicts binary submask from 3D coords + image condition.
    No flow matching, no noise, no timestep.
    """

    def __init__(
        self,
        resolution: int = 8,
        model_channels: int = 768,
        cond_channels: int = 1024,
        num_blocks: int = 8,
        num_heads: Optional[int] = None,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        pe_mode: str = "rope",
        attn_mode: str = "windowed",
        window_size: int = 32,
        full_attn_threshold: int = 8192,
        use_fp16: bool = True,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
        num_pooled_layers: int = 0,
        pool_stride: int = 8,
        pool_stride_coarse: int = 0,
        num_context_registers: int = 0,
        context_start_block: int = 0,
        cross_attn_topk: int = 0,
        local_patch_mode: str = "none",
        num_moe_layers: int = 0,
        num_experts: int = 8,
        moe_top_k: int = 2,
        coord_max: float = 64.0,
        head_type: str = "linear",
        coarse_resolution: int = 4,
        coarse_channels: int = 8,
        coarse_to_fine_mode: str = "independent",
        fine_aux_resolution: int = 0,
        fine_aux_train_only: bool = False,
        density_aux: bool = False,
        dense_query_channels: int = 32,
        dense_query_layers: int = 3,
        dense_query_grid_size: int = 64,
        dense_query_skip_linear: bool = False,
        dense_query_train_only: bool = False,
        use_neighbor_occ: bool = False,
        fourier_dim: int = 0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(f"[DirectMaskModel] ignored kwargs: {list(kwargs.keys())}")

        token_dim = resolution ** 3  # 512 for res=8
        num_heads = num_heads or model_channels // num_head_channels

        self.token_dim = token_dim
        self.resolution = resolution
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_context_registers = num_context_registers
        self.context_start_block = context_start_block
        self.coord_max = coord_max
        self.head_type = head_type
        self.coarse_resolution = coarse_resolution
        self.coarse_channels = coarse_channels
        self.coarse_to_fine_mode = coarse_to_fine_mode
        self.fine_aux_resolution = fine_aux_resolution
        self.fine_aux_train_only = fine_aux_train_only
        self.density_aux = density_aux
        self.dense_query_channels = dense_query_channels
        self.dense_query_layers = dense_query_layers
        self.dense_query_grid_size = dense_query_grid_size
        self.dense_query_skip_linear = dense_query_skip_linear
        self.dense_query_train_only = dense_query_train_only
        self.use_neighbor_occ = use_neighbor_occ
        self.local_patch_mode = local_patch_mode
        self.num_moe_layers = num_moe_layers

        if self.head_type not in {"linear", "coarse_to_fine", "dense_query_conv"}:
            raise ValueError(f"Unsupported head_type: {self.head_type}")
        if self.coarse_to_fine_mode not in {"independent", "additive"}:
            raise ValueError(f"Unsupported coarse_to_fine_mode: {self.coarse_to_fine_mode}")
        if self.local_patch_mode not in {"none", "projected"}:
            raise ValueError(f"Unsupported local_patch_mode: {self.local_patch_mode}")
        if resolution % coarse_resolution != 0:
            raise ValueError(
                f"resolution={resolution} must be divisible by coarse_resolution={coarse_resolution}"
            )
        if self.fine_aux_resolution and self.fine_aux_resolution <= resolution:
            raise ValueError(
                f"fine_aux_resolution={fine_aux_resolution} must be larger than resolution={resolution}"
            )

        # ── Input: Fourier coord encoding → hidden ──
        _fdim = fourier_dim if fourier_dim > 0 else token_dim
        n_freq = _fdim // 6  # 85 freqs per axis for fdim=512
        freqs = 2.0 ** (torch.arange(n_freq, dtype=torch.float32) / n_freq * 6)
        self.register_buffer("fourier_freqs", freqs)
        self.fourier_out_dim = _fdim

        self.coord_embed = nn.Linear(_fdim, model_channels)
        if self.use_neighbor_occ:
            offsets = [
                (dx, dy, dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
                if not (dx == 0 and dy == 0 and dz == 0)
            ]
            self.register_buffer(
                "neighbor_offsets",
                torch.tensor(offsets, dtype=torch.long),
                persistent=False,
            )
            self.neighbor_occ_embed = nn.Sequential(
                nn.Linear(26, 64),
                nn.SiLU(),
                nn.Linear(64, model_channels),
            )

        # ── Condition modulation: DINOv2 CLS token → AdaLN mod ──
        self.cond_mod_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_channels, model_channels),
        )

        if self.local_patch_mode == "projected":
            self.local_patch_proj = nn.Sequential(
                nn.LayerNorm(cond_channels),
                nn.Linear(cond_channels, model_channels),
                nn.SiLU(),
                nn.Linear(model_channels, model_channels),
            )

        # ── Output head ──
        # Note: 之前版本使用 Conv3d 搭配 coarse_channels=8 做 c2f refine,在 N≈batch*max_block_num~6e4
        # 的 "伪 batch" 下 cuDNN Conv3d 在 C=8 的小通道 regime 效率极差(实测 fwd+bwd 167 ms/step,
        # 比 Linear head 慢 ~124×),成为训练瓶颈。改为 Linear-based c2f:从 h_out 直接 Linear 到
        # coarse logits [N, 4³] 与 fine logits [N, 8³],保留 c2f 监督信号,head 代价回到 ~2 ms。
        if self.head_type == "linear":
            self.predict = nn.Linear(model_channels, token_dim)
        elif self.head_type == "coarse_to_fine":
            coarse_dim = coarse_resolution ** 3  # 64
            self.coarse_predict = nn.Linear(model_channels, coarse_dim)
            self.fine_predict = nn.Linear(model_channels, token_dim)
            self._up_factor = resolution // coarse_resolution
            self.i_coarse_resolution = coarse_resolution
            self._fine_resolution = resolution
        elif self.dense_query_skip_linear:
            self.predict = nn.Linear(model_channels, token_dim)
        if self.head_type == "dense_query_conv":
            conv_channels = max(1, int(dense_query_channels))
            self.dense_query_proj = nn.Linear(model_channels, conv_channels)
            conv_layers = []
            in_channels = conv_channels + 4  # sparse token features + valid mask + xyz position
            for layer_idx in range(max(1, int(dense_query_layers))):
                conv_layers.append(nn.Conv3d(
                    in_channels if layer_idx == 0 else conv_channels,
                    conv_channels,
                    kernel_size=3,
                    padding=1,
                ))
                conv_layers.append(nn.SiLU())
            self.dense_query_body = nn.Sequential(*conv_layers)
            self.dense_query_out = nn.Linear(conv_channels, token_dim)
        if self.fine_aux_resolution > 0:
            self.fine_aux_predict = nn.Linear(
                model_channels, self.fine_aux_resolution ** 3,
            )
        if self.density_aux:
            self.density_predict = nn.Linear(model_channels, token_dim)
        self.final_norm = LayerNorm32(
            model_channels, elementwise_affine=False, eps=1e-6
        )

        # ── Context registers (learnable global tokens for cross-attn) ──
        if num_context_registers > 0:
            self.context_registers = nn.Parameter(
                torch.randn(1, num_context_registers, cond_channels) * 0.02
            )
            self.context_register_pos = nn.Parameter(
                torch.randn(1, num_context_registers, cond_channels) * 0.02
            )

        # ── Transformer blocks ──
        blocks = []
        coarse_stride = pool_stride_coarse if pool_stride_coarse > 0 else pool_stride * 2
        for i in range(num_blocks):
            if i < num_pooled_layers:
                half_pooled = num_pooled_layers // 2
                stride_i = coarse_stride if i < half_pooled else pool_stride
                blocks.append(
                    PooledSparseTransformerCrossBlock(
                        model_channels,
                        cond_channels,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        use_checkpoint=use_checkpoint,
                        use_rope=(pe_mode == "rope"),
                        qk_rms_norm=qk_rms_norm,
                        qk_rms_norm_cross=qk_rms_norm_cross,
                        cross_attn_topk=cross_attn_topk,
                        use_moe=(num_blocks - i <= num_moe_layers),
                        num_experts=num_experts,
                        moe_top_k=moe_top_k,
                        pool_stride=stride_i,
                    )
                )
            else:
                blocks.append(
                    ModulatedSparseTransformerCrossBlock(
                        model_channels,
                        cond_channels,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_mode=attn_mode,
                        window_size=window_size,
                        shift_window=(
                            (0, 0, 0)
                            if i % 2 == 0
                            else (window_size // 2,) * 3
                        )
                        if attn_mode == "windowed"
                        else None,
                        full_attn_threshold=full_attn_threshold,
                        use_checkpoint=use_checkpoint,
                        use_rope=(pe_mode == "rope"),
                        qk_rms_norm=qk_rms_norm,
                        qk_rms_norm_cross=qk_rms_norm_cross,
                        cross_attn_topk=cross_attn_topk,
                        use_moe=(num_blocks - i <= num_moe_layers),
                        num_experts=num_experts,
                        moe_top_k=moe_top_k,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

        # ── Init & convert ──
        self._initialize_weights()
        if use_fp16:
            self.blocks.apply(convert_module_to_f16)
        if self.fine_aux_train_only or self.dense_query_train_only:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
        if self.fine_aux_train_only:
            if not hasattr(self, "fine_aux_predict"):
                raise ValueError("fine_aux_train_only=True requires fine_aux_resolution > 0")
            for parameter in self.fine_aux_predict.parameters():
                parameter.requires_grad_(True)
        if self.dense_query_train_only:
            if not hasattr(self, "dense_query_body"):
                raise ValueError("dense_query_train_only=True requires head_type='dense_query_conv'")
            for module in (self.dense_query_proj, self.dense_query_body, self.dense_query_out):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(
            f"[DirectMaskModel] res={resolution}, token_dim={token_dim}, "
            f"hidden={model_channels}x{num_blocks}, heads={num_heads}, "
            f"attn={attn_mode}, w={window_size}, fp16={use_fp16}, "
            f"params={n_params:.1f}M"
        )
        if self.head_type == "coarse_to_fine":
            print(
                f"  head=coarse_to_fine coarse_res={coarse_resolution}, "
                f"coarse_channels={coarse_channels}, mode={coarse_to_fine_mode}"
            )
        if self.head_type == "dense_query_conv":
            print(
                f"  head=dense_query_conv grid={dense_query_grid_size}^3, "
                f"channels={dense_query_channels}, layers={dense_query_layers}, "
                f"skip_linear={dense_query_skip_linear}"
            )
        if self.fine_aux_resolution > 0:
            print(f"  fine_aux_resolution={self.fine_aux_resolution}")
        if self.density_aux:
            print("  density_aux=True (64-dim occupancy-ratio auxiliary head)")
        if self.fine_aux_train_only:
            print("  fine_aux_train_only=True")
        if self.dense_query_train_only:
            print("  dense_query_train_only=True")
        if num_pooled_layers > 0:
            print(
                f"  layers: {num_pooled_layers} pooled "
                f"(stride coarse:{coarse_stride}/fine:{pool_stride}) "
                f"+ {num_blocks - num_pooled_layers} standard"
            )
        if num_context_registers > 0:
            print(f"  registers={num_context_registers}, start_block={context_start_block}")
        if self.local_patch_mode == "projected":
            print("  local_patch=projected block-center bilinear DINO patch")
        if self.use_neighbor_occ:
            print("  neighbor_occ=True (26-neighbor block occupancy embedding)")
        if num_moe_layers > 0:
            n_moe = sum(
                p.numel() for b in self.blocks[-num_moe_layers:] for p in b.mlp.parameters()
            ) / 1e6
            print(f"  MoE: last {num_moe_layers}, {num_experts} experts top-{moe_top_k}, {n_moe:.1f}M")

    # ──────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────

    def _initialize_weights(self):
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_init)
        # Zero-init output head for stable start (让最终 logits 初始为 0)
        if hasattr(self, "predict"):
            nn.init.zeros_(self.predict.weight)
            nn.init.zeros_(self.predict.bias)
        # c2f head: zero-init 最后一层 Linear 即可
        for name in ("coarse_predict", "fine_predict", "fine_aux_predict", "density_predict"):
            head = getattr(self, name, None)
            if isinstance(head, nn.Sequential):
                last = head[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)
            elif isinstance(head, nn.Linear):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        if hasattr(self, "dense_query_out"):
            nn.init.zeros_(self.dense_query_out.weight)
            nn.init.zeros_(self.dense_query_out.bias)
        if hasattr(self, "local_patch_proj"):
            last = self.local_patch_proj[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _dense_position_grid(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid_size = self.dense_query_grid_size
        axis = torch.linspace(-1.0, 1.0, grid_size, device=device, dtype=dtype)
        gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
        grid = torch.stack([gx, gy, gz], dim=0).unsqueeze(0)
        return grid.expand(batch_size, -1, -1, -1, -1)

    def _dense_query_logits(self, h_out: torch.Tensor, x: sp.SparseTensor) -> torch.Tensor:
        grid_size = self.dense_query_grid_size
        batch_size = len(x.layout)
        coords = x.coords.long()
        batch_idx = coords[:, 0]
        xyz = coords[:, 1:]
        if xyz.numel() and (xyz.min() < 0 or xyz.max() >= grid_size):
            raise ValueError(
                f"dense_query_grid_size={grid_size} is incompatible with coords range "
                f"[{xyz.min().item()}, {xyz.max().item()}]"
            )

        h_low = self.dense_query_proj(h_out)
        dense = h_low.new_zeros((batch_size, grid_size, grid_size, grid_size, h_low.shape[-1]))
        dense[batch_idx, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = h_low
        dense = dense.permute(0, 4, 1, 2, 3).contiguous()

        valid = h_low.new_zeros((batch_size, 1, grid_size, grid_size, grid_size))
        valid[batch_idx, 0, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = 1.0
        pos = self._dense_position_grid(batch_size, h_low.device, h_low.dtype)

        dense_input = torch.cat([dense, valid, pos], dim=1)
        hidden_grid = self.dense_query_body(dense_input)
        hidden_tokens = hidden_grid[batch_idx, :, xyz[:, 0], xyz[:, 1], xyz[:, 2]]
        return self.dense_query_out(hidden_tokens)

    def _decode_logits(self, h_out: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.head_type == "linear":
            aux = {}
            if hasattr(self, "fine_aux_predict"):
                aux["fine_aux_logits"] = self.fine_aux_predict(h_out)
            if hasattr(self, "density_predict"):
                aux["density_logits"] = self.density_predict(h_out)
            return self.predict(h_out), aux

        if self.head_type == "dense_query_conv":
            raise RuntimeError("dense_query_conv requires x coords; call _decode_logits_with_coords instead")

        # Linear-based coarse-to-fine. In additive mode, 4^3 logits inherit the
        # 2^3 parent support and the fine head only learns a local residual.
        coarse_logits = self.coarse_predict(h_out)               # [N, cr**3]
        fine_logits = self.fine_predict(h_out)                   # [N, fr**3]
        if self.coarse_to_fine_mode == "additive":
            fine_logits = fine_logits + self._upsample_coarse_logits(coarse_logits)
        aux = {"coarse_logits": coarse_logits}
        if hasattr(self, "fine_aux_predict"):
            aux["fine_aux_logits"] = self.fine_aux_predict(h_out)
        if hasattr(self, "density_predict"):
            aux["density_logits"] = self.density_predict(h_out)
        return fine_logits, aux

    def _upsample_coarse_logits(self, coarse_logits: torch.Tensor) -> torch.Tensor:
        if self._up_factor == 1:
            return coarse_logits
        cr = self.i_coarse_resolution
        coarse = coarse_logits.reshape(-1, 1, cr, cr, cr)
        up = coarse.repeat_interleave(self._up_factor, dim=2)
        up = up.repeat_interleave(self._up_factor, dim=3)
        up = up.repeat_interleave(self._up_factor, dim=4)
        return up.reshape(coarse_logits.shape[0], self.token_dim)

    def coords_to_fourier(self, coords_xyz: torch.Tensor) -> torch.Tensor:
        """[N, 3] int coords → [N, token_dim] Fourier features."""
        coords_norm = coords_xyz.float() / self.coord_max * 2.0 - 1.0
        parts = []
        for d in range(3):
            c = coords_norm[:, d : d + 1]  # [N, 1]
            parts.append(torch.sin(c * self.fourier_freqs))
            parts.append(torch.cos(c * self.fourier_freqs))
        pos = torch.cat(parts, dim=-1)  # [N, n_freq*6]
        if pos.shape[-1] < self.fourier_out_dim:
            pos = F.pad(pos, (0, self.fourier_out_dim - pos.shape[-1]))
        else:
            pos = pos[:, : self.fourier_out_dim]
        return pos

    def coords_to_neighbor_occ(self, coords: torch.Tensor) -> torch.Tensor:
        """[N, 4] batch+xyz coords -> [N, 26] binary neighbor occupancy."""
        coords_long = coords.long()
        batch = coords_long[:, 0]
        xyz = coords_long[:, 1:]
        grid_size = int(round(float(self.coord_max)))

        hashes = (
            batch * (grid_size ** 3)
            + xyz[:, 0] * (grid_size ** 2)
            + xyz[:, 1] * grid_size
            + xyz[:, 2]
        )
        sorted_hashes = torch.sort(hashes).values

        neighbor_xyz = xyz[:, None, :] + self.neighbor_offsets.to(xyz.device)[None, :, :]
        valid = ((neighbor_xyz >= 0) & (neighbor_xyz < grid_size)).all(dim=-1)
        neighbor_hashes = (
            batch[:, None] * (grid_size ** 3)
            + neighbor_xyz[..., 0] * (grid_size ** 2)
            + neighbor_xyz[..., 1] * grid_size
            + neighbor_xyz[..., 2]
        )
        flat_hashes = neighbor_hashes.reshape(-1)
        flat_valid = valid.reshape(-1)

        idx = torch.searchsorted(sorted_hashes, flat_hashes.clamp_min(0))
        found = torch.zeros_like(flat_valid)
        in_range = idx < sorted_hashes.numel()
        if in_range.any():
            found[in_range] = sorted_hashes[idx[in_range]] == flat_hashes[in_range]
        found = found & flat_valid
        return found.reshape(coords.shape[0], 26).to(dtype=torch.float32)

    @staticmethod
    def _patch_tokens_to_grid(cond: torch.Tensor) -> Tuple[torch.Tensor, int]:
        L = cond.shape[1]
        grid_size = int(L ** 0.5)
        while grid_size > 0 and grid_size * grid_size > L:
            grid_size -= 1
        if grid_size <= 0:
            raise ValueError(f"Invalid image context length: {L}")
        patch_count = grid_size * grid_size
        patch_tokens = cond[:, -patch_count:, :]
        patch_grid = patch_tokens.reshape(
            cond.shape[0], grid_size, grid_size, cond.shape[-1]
        ).permute(0, 3, 1, 2).contiguous()
        return patch_grid, grid_size

    def _sample_projected_patches(
        self,
        cond: torch.Tensor,
        patch_xy: torch.Tensor,
        patch_valid: Optional[torch.Tensor],
        layout: List[slice],
    ) -> torch.Tensor:
        patch_grid, _ = self._patch_tokens_to_grid(cond)
        local = torch.zeros(
            patch_xy.shape[0], cond.shape[-1], device=cond.device, dtype=cond.dtype
        )
        for batch_idx, sl in enumerate(layout):
            if sl.start == sl.stop:
                continue
            grid = patch_xy[sl].to(device=cond.device, dtype=cond.dtype).view(1, -1, 1, 2)
            sampled = F.grid_sample(
                patch_grid[batch_idx : batch_idx + 1],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
            if patch_valid is not None:
                valid = patch_valid[sl].to(device=cond.device, dtype=cond.dtype).view(-1, 1)
                sampled = sampled * valid
            local[sl] = sampled
        return local

    # ──────────────────────────────────────────────
    #  Forward
    # ──────────────────────────────────────────────

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,       # unused, kept for trainer compatibility
        cond: torch.Tensor,     # [B, L, cond_channels] DINOv2 features
        return_aux: bool = False,
        patch_xy: Optional[torch.Tensor] = None,
        patch_valid: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> sp.SparseTensor:
        # 1. Fourier coord encoding → embed
        coords_xyz = x.coords[:, 1:]  # [N, 3] drop batch dim
        fourier = self.coords_to_fourier(coords_xyz)
        h_feats = self.coord_embed(fourier).type(self.dtype)

        if self.use_neighbor_occ:
            neighbor_occ = self.coords_to_neighbor_occ(x.coords)
            h_feats = h_feats + self.neighbor_occ_embed(neighbor_occ).type(self.dtype)

        if self.local_patch_mode == "projected" and patch_xy is not None:
            local_patch = self._sample_projected_patches(cond, patch_xy, patch_valid, x.layout)
            h_feats = h_feats + self.local_patch_proj(local_patch).type(self.dtype)

        h = sp.SparseTensor(
            feats=h_feats,
            coords=x.coords,
            shape=torch.Size([len(x.layout), h_feats.shape[-1]]),
            layout=x.layout,
        )

        # 2. CLS token → AdaLN modulation
        cond_cls = cond[:, 0, :]  # [B, cond_channels]
        mod = self.cond_mod_proj(cond_cls).type(self.dtype)

        cond = cond.type(self.dtype)

        # 3. Context registers
        if self.num_context_registers > 0:
            B = cond.shape[0]
            regs = (
                (self.context_registers + self.context_register_pos)
                .expand(B, -1, -1)
                .contiguous()
                .type(self.dtype)
            )
            cond_with_reg = torch.cat([regs, cond], dim=1)
        else:
            cond_with_reg = cond

        # 4. Transformer blocks
        for i, block in enumerate(self.blocks):
            ctx = (
                cond_with_reg
                if (self.num_context_registers > 0 and i >= self.context_start_block)
                else cond
            )
            h = block(h, mod, ctx)

        # 5. Output logits
        h_out = self.final_norm(h.feats)
        h_out = h_out.type(torch.float32)
        if self.head_type == "dense_query_conv":
            logits = self._dense_query_logits(h_out, x)
            if self.dense_query_skip_linear:
                logits = self.predict(h_out) + logits
            aux = {}
            if hasattr(self, "fine_aux_predict"):
                aux["fine_aux_logits"] = self.fine_aux_predict(h_out)
            if hasattr(self, "density_predict"):
                aux["density_logits"] = self.density_predict(h_out)
        else:
            logits, aux = self._decode_logits(h_out)
        pred = x.replace(logits)
        if return_aux:
            return pred, aux
        return pred
