"""
MaskRefineModel: stage-2.5 mask refiner.

Input:
    - x.coords : [T, 4] (batch_idx + block coords)
    - x.feats  : [T, token_dim] pred_mask (coarse stage-2 submask, soft or hard)
    - cond     : [B, L, cond_channels] DINOv2 features
Output:
    - refined submask logits [T, token_dim]

Architecture = DirectMaskModel + an extra Linear that embeds the pred_mask
into the token hidden (added to the Fourier coord embedding). The pred_mask
embedder is zero-initialised so the model starts from the pure DirectMask
behaviour and only learns a residual correction.
"""
from typing import *

import torch
import torch.nn as nn

from ..modules import sparse as sp
from .direct_mask import DirectMaskModel


class MaskRefineModel(DirectMaskModel):
    """Mask refiner. Same backbone as DirectMaskModel; takes coarse pred_mask
    as additional per-token input."""

    def __init__(self, *args, pred_mask_gate: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Embed the pred_mask (token_dim -> model_channels), zero-init so that
        # the refiner starts as a plain DirectMask predictor and then learns
        # to use the coarse mask as a residual hint.
        self.pred_mask_embed = nn.Linear(self.token_dim, self.model_channels)
        nn.init.zeros_(self.pred_mask_embed.weight)
        nn.init.zeros_(self.pred_mask_embed.bias)
        self.pred_mask_gate = float(pred_mask_gate)
        print(f"[MaskRefineModel] + pred_mask_embed (zero-init), gate={pred_mask_gate}")

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        return_aux: bool = False,
        **kwargs,
    ):
        # 1. Fourier coord encoding
        coords_xyz = x.coords[:, 1:]
        fourier = self.coords_to_fourier(coords_xyz)
        h_feats = self.coord_embed(fourier)

        # 2. Add pred_mask embedding (residual; zero at init)
        #    x.feats is expected to hold the coarse pred_mask in [0, 1].
        pm = x.feats.to(h_feats.dtype)
        h_feats = h_feats + self.pred_mask_gate * self.pred_mask_embed(pm)

        h_feats = h_feats.type(self.dtype)

        h = sp.SparseTensor(
            feats=h_feats,
            coords=x.coords,
            shape=torch.Size([len(x.layout), h_feats.shape[-1]]),
            layout=x.layout,
        )

        # 3. CLS modulation
        cond_cls = cond[:, 0, :]
        mod = self.cond_mod_proj(cond_cls).type(self.dtype)
        cond = cond.type(self.dtype)

        # 4. Context registers
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

        # 5. Transformer blocks
        for i, block in enumerate(self.blocks):
            ctx = (
                cond_with_reg
                if (self.num_context_registers > 0 and i >= self.context_start_block)
                else cond
            )
            h = block(h, mod, ctx)

        # 6. Output logits
        h_out = self.final_norm(h.feats)
        logits, aux = self._decode_logits(h_out.type(torch.float32))
        pred = x.replace(logits)
        if return_aux:
            return pred, aux
        return pred
