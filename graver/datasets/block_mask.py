"""
Stage 2 dataset: per-block binary surface mask.

Loads binary submask {0,1} from npz.
Each block has 8³=512 sub-voxels, 1 = surface, 0 = empty.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from .components import StandardDatasetBase, ImageConditionedMixin
from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices
from ..dataset_toolkits.mesh2block import SUBMASK_DIM, BLOCK_FOLDER, COL_PREFIX

PRED_MASK_CACHE = 'pred_mask_cache'


class BlockMask(StandardDatasetBase):

    def __init__(self, roots, *, max_block_num=15000, min_block_num=0,
                 min_aesthetic_score=5.0, max_samples=0,
                 require_pred_mask: bool = False,
                 mask_resolution: int = 8,
                 return_full_submask: bool = False):
        """
        require_pred_mask: if True, only keep npz that contains 'pred_mask'
                           and also return it as 'pred_submask' from get_instance.
                           Used by mask-refiner training.
        mask_resolution: output mask spatial resolution per block axis.
                         8 → 512-dim (default), 4 → 64-dim (max-pooled from 512).
        """
        self.max_block_num = max_block_num
        self.min_block_num = min_block_num
        self.min_aesthetic_score = min_aesthetic_score
        self.mask_resolution = mask_resolution
        self.return_full_submask = return_full_submask
        self.max_samples = max_samples
        self.require_pred_mask = require_pred_mask
        self.value_range = (0, 1)
        super().__init__(roots)

        def _has_readable_npz(root, instance):
            npz_path = os.path.join(root, BLOCK_FOLDER, f'{instance}.npz')
            if not (os.path.exists(npz_path) and os.access(npz_path, os.R_OK)):
                return False
            if self.require_pred_mask:
                # pred_mask can live in npz OR in pred_mask_cache/{instance}.npy
                cache_path = os.path.join(root, PRED_MASK_CACHE, f'{instance}.npy')
                if os.path.exists(cache_path):
                    return True
                try:
                    with np.load(npz_path) as data:
                        return 'pred_mask' in data.files
                except Exception:
                    return False
            return True

        self.filter_existing_instances(
            _has_readable_npz,
            stat_name='pred_mask readable' if require_pred_mask else 'npz readable',
        )

        if self.max_samples > 0 and len(self.instances) > self.max_samples:
            self.instances = self.instances[:self.max_samples]
            self.metadata = self.metadata.iloc[:self.max_samples]

        self.loads = [
            max(1, int(self.metadata.at[i, f'{COL_PREFIX}_num_blocks']))
            for i in range(len(self.instances))
        ]

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'{COL_PREFIX}_block_status'] == 'success']
        stats['block success'] = len(metadata)
        metadata = metadata[metadata[f'{COL_PREFIX}_num_blocks'] <= self.max_block_num]
        stats[f'blocks <= {self.max_block_num}'] = len(metadata)
        if self.min_block_num > 0:
            metadata = metadata[metadata[f'{COL_PREFIX}_num_blocks'] >= self.min_block_num]
            stats[f'blocks >= {self.min_block_num}'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        npz_path = os.path.join(root, BLOCK_FOLDER, f'{instance}.npz')
        with np.load(npz_path) as data:
            coords = torch.from_numpy(data['coords']).int()
            if 'submask' in data.files:
                submask = torch.from_numpy(data['submask'].astype(np.float32))
            else:
                submask = torch.ones(coords.shape[0], SUBMASK_DIM)
            full_submask = submask.clone()

            # Pool to coarser resolution if requested
            if self.mask_resolution != 8:
                T = submask.shape[0]
                vol = submask.reshape(T, 1, 8, 8, 8)
                stride = 8 // self.mask_resolution
                submask = F.max_pool3d(vol, kernel_size=stride, stride=stride)
                submask = submask.reshape(T, -1)

            out = {'coords': coords, 'submask': submask}
            if self.return_full_submask:
                out['full_submask'] = full_submask
        if self.require_pred_mask:
            npz_path = os.path.join(root, BLOCK_FOLDER, f'{instance}.npz')
            cache_path = os.path.join(root, PRED_MASK_CACHE, f'{instance}.npy')
            if os.path.exists(cache_path):
                out['pred_submask'] = torch.from_numpy(
                    np.load(cache_path).astype(np.float32)
                )
            else:
                with np.load(npz_path) as data:
                    if 'pred_mask' in data.files:
                        out['pred_submask'] = torch.from_numpy(
                            data['pred_mask'].astype(np.float32)
                        )
                    else:
                        out['pred_submask'] = submask.clone()
        return out

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices(
                [b['coords'].shape[0] for b in batch], split_size,
            )

        packs = []
        for group in group_idx:
            sub = [batch[i] for i in group]
            coords_list, mask_list, pred_list, full_mask_list = [], [], [], []
            patch_xy_list, patch_valid_list, layout = [], [], []
            start = 0
            for i, b in enumerate(sub):
                n = b['coords'].shape[0]
                coords_list.append(torch.cat([
                    torch.full((n, 1), i, dtype=torch.int32), b['coords'],
                ], dim=-1))
                mask_list.append(b['submask'])
                if 'pred_submask' in b:
                    pred_list.append(b['pred_submask'])
                if 'full_submask' in b:
                    full_mask_list.append(b['full_submask'])
                if 'patch_xy' in b:
                    patch_xy_list.append(b['patch_xy'])
                if 'patch_valid' in b:
                    patch_valid_list.append(b['patch_valid'])
                layout.append(slice(start, start + n))
                start += n

            coords = torch.cat(coords_list, 0)
            submask = torch.cat(mask_list, 0)

            pack = {
                'x_0': SparseTensor(
                    coords=coords, feats=submask,
                    shape=torch.Size([len(group), submask.shape[1]]),
                    layout=layout,
                ),
            }
            if pred_list:
                pack['pred_submask'] = torch.cat(pred_list, 0)
            if full_mask_list:
                pack['full_submask'] = torch.cat(full_mask_list, 0)
            if patch_xy_list:
                pack['patch_xy'] = torch.cat(patch_xy_list, 0)
            if patch_valid_list:
                pack['patch_valid'] = torch.cat(patch_valid_list, 0)

            # Forward extra keys (cond, etc.)
            exclude = {
                'coords', 'submask', 'pred_submask', 'full_submask',
                'patch_xy', 'patch_valid',
            }
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


class ImageConditionedBlockMask(ImageConditionedMixin, BlockMask):
    pass
