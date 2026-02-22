from typing import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.norm import LayerNorm32
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock, PooledSparseTransformerCrossBlock
from ..dataset_toolkits.mesh2block import BLOCK_DIM
from .dense_flow import TimestepEmbedder


class SparseFlowModel(nn.Module):

    def __init__(
        self,
        bottleneck_dim: int = 128,
        model_channels: int = 256,
        cond_channels: int = 1024,
        num_blocks: int = 8,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "rope",
        attn_mode: str = "windowed",
        window_size: int = 16,
        full_attn_threshold: int = 8192,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        num_moe_layers: int = 0,
        num_experts: int = 8,
        moe_top_k: int = 2,
        cross_attn_topk: int = 0,
        num_context_registers: int = 0,
        context_start_block: int = 0,
        pool_stride: int = 8,
        pool_stride_coarse: int = 0,
        num_pooled_layers: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_context_registers = num_context_registers
        self.context_start_block = context_start_block

        block_dim = BLOCK_DIM ** 3
        out_dim = BLOCK_DIM ** 3

        if kwargs:
            print(f"[SparseFlowModel] ignored: {list(kwargs.keys())}")

        self.embed = nn.Sequential(
            nn.Linear(block_dim, bottleneck_dim, bias=False),
            nn.Linear(bottleneck_dim, model_channels),
        )

        self.predict = nn.Sequential(
            nn.Linear(model_channels, bottleneck_dim),
            nn.Linear(bottleneck_dim, out_dim),
        )

        self.final_norm = LayerNorm32(model_channels, elementwise_affine=False, eps=1e-6)

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 9 * model_channels, bias=True),
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        if num_context_registers > 0:
            self.context_registers = nn.Parameter(
                torch.randn(1, num_context_registers, cond_channels) * 0.02
            )
            self.context_register_pos = nn.Parameter(
                torch.randn(1, num_context_registers, cond_channels) * 0.02
            )

        blocks = []
        coarse_stride = pool_stride_coarse if pool_stride_coarse > 0 else pool_stride * 2
        for i in range(num_blocks):
            if i < num_pooled_layers:
                # Pooled Block: 全部计算在池化空间
                half_pooled = num_pooled_layers // 2
                stride_i = coarse_stride if i < half_pooled else pool_stride
                blocks.append(PooledSparseTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    use_checkpoint=use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                    cross_attn_topk=cross_attn_topk,
                    use_moe=(num_blocks - i <= num_moe_layers),
                    num_experts=num_experts,
                    moe_top_k=moe_top_k,
                    pool_stride=stride_i,
                ))
            else:
                # Standard Block: windowed self-attn + cross-attn + MLP
                blocks.append(ModulatedSparseTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_mode=attn_mode,
                    window_size=window_size,
                    shift_window=(
                        (0, 0, 0) if i % 2 == 0
                        else (window_size // 2, window_size // 2, window_size // 2)
                    ) if attn_mode == "windowed" else None,
                    full_attn_threshold=full_attn_threshold,
                    use_checkpoint=use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                    cross_attn_topk=cross_attn_topk,
                    use_moe=(num_blocks - i <= num_moe_layers),
                    num_experts=num_experts,
                    moe_top_k=moe_top_k,
                ))
        self.blocks = nn.ModuleList(blocks)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[SparseFlowModel] BLOCK_DIM={BLOCK_DIM}, "
              f"{block_dim}>{bottleneck_dim}>{model_channels}"
              f">x{num_blocks}"
              f">{model_channels}>{bottleneck_dim}>{out_dim}, "
              f"attn={attn_mode}, w={window_size}, heads={self.num_heads}, "
              f"fp16={use_fp16}, params={n_params:.1f}M")
        if num_pooled_layers > 0:
            print(f"  layers: {num_pooled_layers} pooled (stride coarse:{coarse_stride}/fine:{pool_stride}) "
                  f"+ {num_blocks - num_pooled_layers} standard")
        if num_context_registers > 0:
            print(f"  registers={num_context_registers}, start_block={context_start_block}")
        if num_moe_layers > 0:
            n_moe = sum(
                p.numel() for b in self.blocks[-num_moe_layers:] for p in b.mlp.parameters()
            ) / 1e6
            print(f"  MoE: last {num_moe_layers}, {num_experts} experts top-{moe_top_k}, {n_moe:.1f}M")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
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
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        return_decoder_feats: bool = False,
    ) -> Union[sp.SparseTensor, Tuple[sp.SparseTensor, sp.SparseTensor]]:
        h_feats = self.embed(x.feats).type(self.dtype)

        h = sp.SparseTensor(
            feats=h_feats,
            coords=x.coords,
            shape=torch.Size([len(x.layout), h_feats.shape[-1]]),
            layout=x.layout,
        )

        t_emb = self.t_embedder(t * 1000).type(self.dtype)
        t_emb_tx = self.adaLN_modulation(t_emb) if self.share_mod else t_emb
        t_emb_tx = t_emb_tx.type(self.dtype)
        cond = cond.type(self.dtype)

        if self.num_context_registers > 0:
            B = t.shape[0]
            registers = (self.context_registers + self.context_register_pos
                         ).expand(B, -1, -1).contiguous().type(self.dtype)
            cond_with_reg = torch.cat([registers, cond], dim=1)
        else:
            cond_with_reg = cond

        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)

        for i, block in enumerate(self.blocks):
            ctx = cond_with_reg if (self.num_context_registers > 0 and i >= self.context_start_block) else cond
            h = block(h, t_emb_tx, ctx)

        h_out = self.final_norm(h.feats)
        udf = self.predict(h_out.type(x.dtype))
        out = x.replace(udf)

        if return_decoder_feats:
            return out, h
        return out
