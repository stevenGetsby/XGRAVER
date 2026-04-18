"""
DirectMaskModel: Sparse Transformer for direct binary mask prediction.

Input:  Fourier-encoded 3D coords + DINOv2 image features
Output: Binary submask logits per block (512 values each)

Architecture:
  - Fourier position encoding of block coords → Linear → hidden
  - DINOv2 CLS token → Linear → AdaLN modulation
  - N transformer blocks (Pooled + Windowed self-attn, cross-attn to DINOv2, MLP)
  - Linear head → 512-dim logits per block
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
        coord_max: float = 64.0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(f"[DirectMaskModel] ignored kwargs: {list(kwargs.keys())}")

        token_dim = resolution ** 3  # 512 for res=8
        num_heads = num_heads or model_channels // num_head_channels

        self.token_dim = token_dim
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_context_registers = num_context_registers
        self.context_start_block = context_start_block
        self.coord_max = coord_max

        # ── Input: Fourier coord encoding → hidden ──
        n_freq = token_dim // 6  # 85 freqs per axis for token_dim=512
        freqs = 2.0 ** (torch.arange(n_freq, dtype=torch.float32) / n_freq * 6)
        self.register_buffer("fourier_freqs", freqs)
        self.fourier_out_dim = token_dim

        self.coord_embed = nn.Linear(token_dim, model_channels)

        # ── Condition modulation: DINOv2 CLS token → AdaLN mod ──
        self.cond_mod_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_channels, model_channels),
        )

        # ── Output head ──
        self.predict = nn.Linear(model_channels, token_dim)
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
                    )
                )
        self.blocks = nn.ModuleList(blocks)

        # ── Init & convert ──
        self._initialize_weights()
        if use_fp16:
            self.blocks.apply(convert_module_to_f16)

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(
            f"[DirectMaskModel] res={resolution}, token_dim={token_dim}, "
            f"hidden={model_channels}x{num_blocks}, heads={num_heads}, "
            f"attn={attn_mode}, w={window_size}, fp16={use_fp16}, "
            f"params={n_params:.1f}M"
        )
        if num_pooled_layers > 0:
            print(
                f"  layers: {num_pooled_layers} pooled "
                f"(stride coarse:{coarse_stride}/fine:{pool_stride}) "
                f"+ {num_blocks - num_pooled_layers} standard"
            )
        if num_context_registers > 0:
            print(f"  registers={num_context_registers}, start_block={context_start_block}")

    # ──────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────

    def _initialize_weights(self):
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_init)
        # Zero-init output head for stable start
        nn.init.zeros_(self.predict.weight)
        nn.init.zeros_(self.predict.bias)

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

    # ──────────────────────────────────────────────
    #  Forward
    # ──────────────────────────────────────────────

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,       # unused, kept for trainer compatibility
        cond: torch.Tensor,     # [B, L, cond_channels] DINOv2 features
        **kwargs,
    ) -> sp.SparseTensor:
        # 1. Fourier coord encoding → embed
        coords_xyz = x.coords[:, 1:]  # [N, 3] drop batch dim
        fourier = self.coords_to_fourier(coords_xyz)
        h_feats = self.coord_embed(fourier).type(self.dtype)

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
        logits = self.predict(h_out.type(torch.float32))
        return x.replace(logits)
