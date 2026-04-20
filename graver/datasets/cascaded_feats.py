"""
Dataset for cascaded feats training.
Extends BlockFeats to load pre-computed pred_mask from cache directory.

Returns both GT submask and pred submask for each sample.
"""
import os
import numpy as np
import torch
from .block_feats import BlockFeats
from .components import ImageConditionedMixin
from ..dataset_toolkits.mesh2block import BLOCK_FOLDER, SUBMASK_DIM


class CascadedBlockFeats(BlockFeats):
    """
    BlockFeats + pred_mask.

    Preferred source: `pred_mask` field inside the block npz file
    (written by `encode_mask.py`).  Falls back to a per-sample
    `{sha256}.npy` cache directory if provided (legacy format).
    If neither is available for a sample, the GT `submask` is used
    as the pred fallback.

    Args:
        pred_mask_dir: (optional, legacy) directory with `{sha256}.npy`
            files holding pred submask.  If set, used to filter
            instances and as a fallback when npz lacks `pred_mask`.
        require_pred_mask: if True, drop instances that have neither
            npz `pred_mask` nor a legacy `.npy` file.
    """

    def __init__(self, *args, pred_mask_dir: str = '',
                 require_pred_mask: bool = True, **kwargs):
        self.pred_mask_dir = pred_mask_dir
        self.require_pred_mask = require_pred_mask
        super().__init__(*args, **kwargs)

        if require_pred_mask:
            before = len(self.instances)
            self.filter_existing_instances(
                lambda root, instance: self._has_pred_mask(root, instance),
                stat_name='pred_mask available',
            )
            after = len(self.instances)
            print(f'  [CascadedBlockFeats] pred_mask: npz field '
                  f'(legacy dir={pred_mask_dir or "none"}), '
                  f'filtered {before} → {after}')
        else:
            print(f'  [CascadedBlockFeats] require_pred_mask=False, '
                  f'GT submask will be used as fallback')

    def _has_pred_mask(self, root, instance):
        # Prefer npz-embedded `pred_mask`.
        npz_path = os.path.join(root, BLOCK_FOLDER, f'{instance}.npz')
        if os.path.exists(npz_path):
            try:
                with np.load(npz_path) as data:
                    if 'pred_mask' in data.files:
                        return True
            except Exception:
                pass
        # Legacy per-sample .npy cache.
        if self.pred_mask_dir:
            if os.path.exists(os.path.join(self.pred_mask_dir, f'{instance}.npy')):
                return True
        return False

    def get_instance(self, root, instance):
        result = super().get_instance(root, instance)

        # 1) Prefer `pred_mask` embedded in the npz.
        npz_path = os.path.join(root, BLOCK_FOLDER, f'{instance}.npz')
        pred = None
        try:
            with np.load(npz_path) as data:
                if 'pred_mask' in data.files:
                    pred = data['pred_mask']
        except Exception:
            pred = None

        # 2) Fallback to legacy .npy cache.
        if pred is None and self.pred_mask_dir:
            legacy_path = os.path.join(self.pred_mask_dir, f'{instance}.npy')
            if os.path.exists(legacy_path):
                pred = np.load(legacy_path)

        # 3) Fallback to GT submask.
        if pred is None:
            result['pred_submask'] = result['submask'].clone()
        else:
            result['pred_submask'] = torch.from_numpy(pred.astype(np.float32))

        return result

    @staticmethod
    def collate_fn(batch, split_size=None):
        """Extend parent collate to also handle pred_submask."""
        from ..modules.sparse.basic import SparseTensor
        from ..utils.data_utils import load_balanced_group_indices

        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices(
                [b['coords'].shape[0] for b in batch], split_size,
            )

        packs = []
        for group in group_idx:
            sub = [batch[i] for i in group]
            coords_list, feats_list, submask_list, pred_submask_list, layout = [], [], [], [], []
            start = 0
            for i, b in enumerate(sub):
                n = b['coords'].shape[0]
                coords_list.append(torch.cat([
                    torch.full((n, 1), i, dtype=torch.int32),
                    b['coords'],
                ], dim=-1))
                feats_list.append(b['fine_feats'])
                submask_list.append(b['submask'])
                if 'pred_submask' in b:
                    pred_submask_list.append(b['pred_submask'])
                layout.append(slice(start, start + n))
                start += n

            all_coords = torch.cat(coords_list, 0)
            all_feats = torch.cat(feats_list, 0)
            all_submask = torch.cat(submask_list, 0)

            pack = {
                'x_f': SparseTensor(
                    coords=all_coords,
                    feats=all_feats,
                    shape=torch.Size([len(group), all_feats.shape[1]]),
                    layout=layout,
                ),
                'submask': all_submask,
            }

            if pred_submask_list:
                pack['pred_submask'] = torch.cat(pred_submask_list, 0)

            # Forward extra keys (image cond etc.)
            exclude = {'coords', 'fine_feats', 'submask', 'pred_submask'}
            for k in sub[0]:
                if k in exclude:
                    continue
                if isinstance(sub[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub])
                elif isinstance(sub[0][k], list):
                    pack[k] = sum([b[k] for b in sub], [])
                else:
                    pack[k] = [b[k] for b in sub]

            packs.append(pack)

        return packs[0] if split_size is None else packs


class ImageConditionedCascadedBlockFeats(ImageConditionedMixin, CascadedBlockFeats):
    pass
