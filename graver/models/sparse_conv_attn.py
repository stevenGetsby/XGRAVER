"""
SparseConvAttnModel: Hybrid Sparse Conv + Transformer model.

Each block: SparseConv3d(local) → Self-Attn(global) → Cross-Attn(conditioning) → FFN

This combines the best of both:
- SparseConv3d captures local 3D spatial structure between neighboring blocks
- Self-Attention captures global relationships across all tokens  
- Cross-Attention injects image conditioning
"""
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.norm import LayerNorm32
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock

from .dense_flow import TimestepEmbedder


class SparseConvAttnBlock(nn.Module):
    """
    Hybrid block: SparseConv3d → ModulatedSparseTransformerCrossBlock
    
    The conv operates on sparse 3D coordinates to extract local features,
    then transformer does global self-attn + cross-attn + FFN.
    """
    def __init__(
        self,
        channels: int,
        cond_channels: int,
        num_heads: int,
        mlp_ratio: float = 4,
        conv_kernel: int = 3,
        attn_mode: str = "windowed",
        window_size: int = 32,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = True,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        # Sparse 3D Conv (local spatial feature extraction)
        self.conv_norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.conv = sp.SparseConv3d(channels, channels, conv_kernel)
        
        # Transformer block (global self-attn + cross-attn + FFN)
        self.transformer = ModulatedSparseTransformerCrossBlock(
            channels, cond_channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            use_checkpoint=use_checkpoint,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
            qk_rms_norm_cross=qk_rms_norm_cross,
        )
    
    def forward(self, x: sp.SparseTensor, t_emb: torch.Tensor, cond: torch.Tensor) -> sp.SparseTensor:
        # Conv path: norm → silu → conv → residual
        h = x.replace(self.conv_norm(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv(h)
        x = x + h
        
        # Transformer path: self-attn → cross-attn → FFN
        x = self.transformer(x, t_emb, cond)
        return x


class SparseConvAttnModel(nn.Module):
    """
    Sparse Conv + Attention Hybrid Model for mask flow matching.
    
    Architecture:
      SparseLinear(in→C) → [SparseConvAttnBlock × N] → LayerNorm → Linear(C→out)
    
    Each block: SparseConv3d(local 3D) + Self-Attn(global) + Cross-Attn(image cond) + FFN
    """
    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 512,
        model_channels: int = 768,
        cond_channels: int = 1024,
        num_blocks: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4,
        conv_kernel: int = 3,
        pe_mode: Literal["ape", "rope"] = "rope",
        attn_mode: str = "windowed",
        window_size: int = 32,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.share_mod = share_mod
        self.dtype = torch.float16 if use_fp16 else torch.float32

        if kwargs:
            print(f"[SparseConvAttnModel] ignored: {list(kwargs.keys())}")

        # Input/Output
        self.input_proj = sp.SparseLinear(in_channels, model_channels)
        self.output_proj = nn.Linear(model_channels, out_channels)
        self.final_norm = LayerNorm32(model_channels, elementwise_affine=False, eps=1e-6)

        # Time embedder
        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 9 * model_channels, bias=True),
            )

        # Hybrid blocks
        blocks = []
        for i in range(num_blocks):
            blocks.append(SparseConvAttnBlock(
                model_channels, cond_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                conv_kernel=conv_kernel,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_window=(
                    (0, 0, 0) if i % 2 == 0
                    else (window_size // 2, window_size // 2, window_size // 2)
                ) if attn_mode == "windowed" else None,
                use_checkpoint=use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=qk_rms_norm,
                qk_rms_norm_cross=qk_rms_norm_cross,
            ))
        self.blocks = nn.ModuleList(blocks)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[SparseConvAttnModel] {num_blocks} hybrid blocks, "
              f"ch={model_channels}, heads={num_heads}, conv_k={conv_kernel}, "
              f"attn={attn_mode}, w={window_size}, params={n_params:.1f}M")

    def convert_to_fp16(self):
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.transformer.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.transformer.adaLN_modulation[-1].bias, 0)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor, **kwargs) -> sp.SparseTensor:
        h = self.input_proj(x).type(self.dtype)

        t_emb = self.t_embedder(t * 1000).type(self.dtype)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        for block in self.blocks:
            h = block(h, t_emb, cond)

        h = h.replace(self.final_norm(h.feats))
        out = self.output_proj(h.feats.type(x.dtype))
        return h.replace(out)
