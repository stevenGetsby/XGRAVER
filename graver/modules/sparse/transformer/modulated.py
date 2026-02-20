from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic import SparseTensor
from ..attention import SparseMultiHeadAttention, SerializeMode
from ..attention.modules import SparseMultiHeadRMSNorm
from ..attention.rope import SparseRotaryPositionEmbedder
from ...norm import LayerNorm32
from .blocks import SparseFeedForwardNet, SparseSwiGLUFFN
from .moe import SparseMoEFFN


class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin", "windowed"] = "shift_window",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        full_attn_threshold: int = 8192,
        cross_attn_topk: int = 0,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
            full_attn_threshold=full_attn_threshold,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            cross_attn_topk=cross_attn_topk,
        )
        if use_moe:
            self.mlp = SparseMoEFFN(
                channels,
                mlp_ratio=mlp_ratio,
                num_experts=num_experts,
                top_k=moe_top_k,
            )
        else:
            self.mlp = SparseSwiGLUFFN(
                channels,
                mlp_ratio=mlp_ratio,
            )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 9 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = mod.chunk(9, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(9, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mca) + shift_mca
        h = self.cross_attn(h, context)
        h = h * gate_mca
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)


class SSASparseTransformerCrossBlock(nn.Module):
    """
    Spatial Sparse Attention (SSA) Block — 高效全局注意力.

    核心创新: Self-attention 中 Q 保持全分辨率, K/V 来自池化的 super-tokens.

    对比:
      - PooledBlock:  Q 也被池化 → 同一 bucket 内所有 token 得到相同更新, 丢高频
      - StandardBlock (windowed): Q/KV 全分辨率但窗口 W=16, 仅局部感受野
      - **SSA**: Q 全分辨率(每 token 唯一输出) + KV 池化(全局感受野)

    架构:
      1. SSA Self-Attention: Q(N) × KV(P) → O(N*P), 全局 & 逐 token
      2. Cross-Attention + MLP: 在池化空间 P 完成 → 高效
      3. 残差传播: self-attn 直接在全分辨率叠加; cross/MLP 残差 unpool 回去

    FLOPs / Layer (N=70K, P≈2K, W=16, D=768):
      PooledBlock:   ~11B  (全池化, 快但有损)
      SSA-Hybrid:    ~308B (全局 self-attn + 池化 cross/MLP)
      StandardBlock: ~773B (窗口 self-attn + 全分辨率 cross/MLP)

      SSA 比 Standard 快 2.5×, 同时拥有全局感受野.
      8 SSA + 4 Standard 仅比 8 Pooled + 4 Standard 多 ~16% 计算量.
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        cross_attn_topk: int = 0,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        pool_stride: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.pool_stride = pool_stride
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.use_rope = use_rope
        self._qk_rms_norm = qk_rms_norm

        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)

        # ---------- SSA Self-Attention ----------
        # Q projection (applied to full-res tokens)
        self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
        # KV projection (applied to pooled super-tokens)
        self.to_kv = nn.Linear(channels, channels * 2, bias=qkv_bias)
        self.to_out = nn.Linear(channels, channels)

        if qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)

        if use_rope:
            self.rope = SparseRotaryPositionEmbedder(
                head_dim=self.head_dim, dim=3,
            )

        # ---------- Cross-Attention (在池化空间) ----------
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            cross_attn_topk=cross_attn_topk,
        )

        # ---------- MLP (在池化空间) ----------
        if use_moe:
            self.mlp = SparseMoEFFN(
                channels, mlp_ratio=mlp_ratio,
                num_experts=num_experts, top_k=moe_top_k,
            )
        else:
            self.mlp = SparseSwiGLUFFN(channels, mlp_ratio=mlp_ratio)

        # ---------- adaLN ----------
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 9 * channels, bias=True),
            )

    # ------------------------------------------------------------------
    # RoPE utilities (Q / K 使用不同坐标空间, 需分别计算)
    # ------------------------------------------------------------------

    def _compute_rope_phases(self, coords_3d: torch.Tensor) -> torch.Tensor:
        """
        为 3D 坐标计算 RoPE phases.

        Args:
            coords_3d: [N, 3] 整数坐标 (不含 batch dim)

        Returns:
            [N, head_dim // 2] complex phases
        """
        self.rope.freqs = self.rope.freqs.to(coords_3d.device)
        phases = self.rope._get_phases(coords_3d.float().reshape(-1))  # [N*3, freq_dim]
        phases = phases.reshape(coords_3d.shape[0], -1)                # [N, 3*freq_dim]
        target_dim = self.head_dim // 2
        if phases.shape[-1] < target_dim:
            pad_n = target_dim - phases.shape[-1]
            phases = torch.cat([
                phases,
                torch.polar(
                    torch.ones(*phases.shape[:-1], pad_n, device=phases.device),
                    torch.zeros(*phases.shape[:-1], pad_n, device=phases.device),
                ),
            ], dim=-1)
        return phases

    # ------------------------------------------------------------------
    # SSA Self-Attention
    # ------------------------------------------------------------------

    def _ssa_self_attn(self, x: SparseTensor, h_normed: SparseTensor) -> SparseTensor:
        """
        SSA Self-Attention: Q 全分辨率 × KV 池化.

        Args:
            x:        原始输入 (用于 replace / 坐标)
            h_normed: 经 adaLN 调制后的输入 (用于 Q/KV 投影)

        Returns:
            SparseTensor: 注意力输出, feats shape [T, D]
        """
        from ..attention.spca import pool_sparse_tensor
        from ..attention.full_attn import sparse_scaled_dot_product_attention

        T = h_normed.feats.shape[0]

        # Q from full-res tokens: [T, D] → [T, H, C]
        q_feats = self.to_q(h_normed.feats)
        q_feats = q_feats.reshape(T, self.num_heads, self.head_dim)

        # Pool for KV: [T, D] → [P, D] → [P, 2*D] → [P, 2, H, C]
        h_pooled, _ = pool_sparse_tensor(h_normed, self.pool_stride)
        P = h_pooled.feats.shape[0]
        kv_feats = self.to_kv(h_pooled.feats)
        kv_feats = kv_feats.reshape(P, 2, self.num_heads, self.head_dim)

        # RoPE: Q 使用全分辨率坐标, K 使用池化坐标 × stride (回到原始空间)
        if self.use_rope:
            q_phases = self._compute_rope_phases(x.coords[:, 1:])             # [T, hd/2]
            q_feats = self.rope._rotary_embedding(q_feats, q_phases)

            k_coords = h_pooled.coords[:, 1:] * self.pool_stride              # [P, 3]
            k_phases = self._compute_rope_phases(k_coords)                     # [P, hd/2]
            k_rotated = self.rope._rotary_embedding(kv_feats[:, 0], k_phases)  # [P, H, C]
            kv_feats = torch.stack([k_rotated, kv_feats[:, 1]], dim=1)         # [P, 2, H, C]

        # QK RMS Norm
        if self._qk_rms_norm:
            q_feats = F.normalize(q_feats.float(), dim=-1)
            q_feats = (q_feats * self.q_rms_norm.gamma * self.q_rms_norm.scale).to(x.dtype)

            k_feats = kv_feats[:, 0]
            k_feats = F.normalize(k_feats.float(), dim=-1)
            k_feats = (k_feats * self.k_rms_norm.gamma * self.k_rms_norm.scale).to(x.dtype)
            kv_feats = torch.stack([k_feats, kv_feats[:, 1]], dim=1)

        # Flash Attention: Q(SparseTensor, full-res layout) × KV(SparseTensor, pooled layout)
        q_sparse = x.replace(q_feats)            # inherits full-res coords / layout
        kv_sparse = h_pooled.replace(kv_feats)    # inherits pooled coords / layout
        out = sparse_scaled_dot_product_attention(q_sparse, kv_sparse)  # → [T, H, C]

        # Output projection: [T, H*C] → [T, D]
        out_feats = out.feats.reshape(T, -1)
        out_feats = self.to_out(out_feats)

        return x.replace(out_feats)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        from ..attention.spca import pool_sparse_tensor, unpool_feats

        if self.share_mod:
            shift_msa, scale_msa, gate_msa, \
            shift_mca, scale_mca, gate_mca, \
            shift_mlp, scale_mlp, gate_mlp = mod.chunk(9, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, \
            shift_mca, scale_mca, gate_mca, \
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(9, dim=1)

        # ====== 1. SSA Self-Attention (全分辨率 Q, 池化 KV) ======
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self._ssa_self_attn(x, h)
        h = h * gate_msa
        x = x + h  # 全分辨率残差 ✓

        # ====== 2-3. Cross-Attention + MLP (池化空间, 高效) ======
        x_pooled, inverse = pool_sparse_tensor(x, self.pool_stride)
        pooled_input_feats = x_pooled.feats.clone()

        # Cross-Attention
        h = x_pooled.replace(self.norm2(x_pooled.feats))
        h = h * (1 + scale_mca) + shift_mca
        h = self.cross_attn(h, context)
        h = h * gate_mca
        x_pooled = x_pooled + h

        # MLP
        h = x_pooled.replace(self.norm3(x_pooled.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x_pooled = x_pooled + h

        # Unpool cross-attn + MLP 残差回全分辨率
        delta = x_pooled.feats - pooled_input_feats  # [P, D]
        delta_unpooled = unpool_feats(delta, inverse)  # [T, D]
        x = x.replace(x.feats + delta_unpooled)

        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)

class PooledSparseTransformerCrossBlock(nn.Module):
    """
    Pooled Sparse Transformer Block: 全部计算在池化后的 super-tokens 上完成.
    
    70K tokens → pool → 2K super-tokens → (self-attn + cross-attn + MLP) → unpool → 70K tokens
    
    MLP/Cross-attn 只处理 ~2K tokens, 节省 ~35× 计算量.
    self-attn 在 2K tokens 上用 full attention (天然全局感受野).
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        cross_attn_topk: int = 0,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        pool_stride: int = 4,
        **kwargs,  # 吸收 window_size 等不需要的参数
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.pool_stride = pool_stride
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        # self-attn: full attention on pooled tokens (P ≈ 2K, 很快)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode="full",
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            cross_attn_topk=cross_attn_topk,
        )
        if use_moe:
            self.mlp = SparseMoEFFN(
                channels,
                mlp_ratio=mlp_ratio,
                num_experts=num_experts,
                top_k=moe_top_k,
            )
        else:
            self.mlp = SparseSwiGLUFFN(
                channels,
                mlp_ratio=mlp_ratio,
            )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 9 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        from ..attention.spca import pool_sparse_tensor, unpool_feats

        # ---- Pool: 70K → ~2K (缓存输入池化结果, 避免二次计算) ----
        x_pooled, inverse = pool_sparse_tensor(x, self.pool_stride)
        pooled_input_feats = x_pooled.feats.clone()  # 保存池化输入

        # ---- adaLN modulation (在 pool 空间) ----
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = mod.chunk(9, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(9, dim=1)

        # ---- Self-Attention (full, on ~2K tokens) ----
        h = x_pooled.replace(self.norm1(x_pooled.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x_pooled = x_pooled + h

        # ---- Cross-Attention (on ~2K tokens) ----
        h = x_pooled.replace(self.norm2(x_pooled.feats))
        h = h * (1 + scale_mca) + shift_mca
        h = self.cross_attn(h, context)
        h = h * gate_mca
        x_pooled = x_pooled + h

        # ---- MLP (on ~2K tokens) ----
        h = x_pooled.replace(self.norm3(x_pooled.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x_pooled = x_pooled + h

        # ---- Unpool: 广播残差回 70K (无二次池化) ----
        delta = x_pooled.feats - pooled_input_feats  # [P, D]
        delta_unpooled = unpool_feats(delta, inverse)  # [T, D]

        return x.replace(x.feats + delta_unpooled)

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)
