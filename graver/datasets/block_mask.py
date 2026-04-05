"""
Stage 2 dataset: per-block surface mask.

Two modes controlled by `use_udf`:
  - use_udf=False (legacy): loads binary submask {0,1} from npz.
  - use_udf=True  (default): loads fine_feats, min_pool 16³→8³ to get
    continuous local UDF [0,1]. Surface = small UDF. Threshold at
    SURFACE_THRESHOLD (0.4) recovers the same mask as binary submask.
    This continuous target dramatically improves flow matching convergence.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from .components import StandardDatasetBase, ImageConditionedMixin
from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices
from ..dataset_toolkits.mesh2block import SUBMASK_DIM, BLOCK_FOLDER, COL_PREFIX


class BlockMask(StandardDatasetBase):

    def __init__(self, roots, *, max_block_num=15000, min_block_num=0,
                 min_aesthetic_score=5.0, max_samples=0, use_udf=True):
        self.max_block_num = max_block_num
        self.min_block_num = min_block_num
        self.min_aesthetic_score = min_aesthetic_score
        self.max_samples = max_samples
        self.use_udf = use_udf
        self.value_range = (0, 1)
        super().__init__(roots)
        self.filter_existing_instances(
            lambda root, instance: os.path.exists(os.path.join(root, BLOCK_FOLDER, f'{instance}.npz')),
            stat_name='npz exists',
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
            if self.use_udf and 'fine_feats' in data.files:
                # min_pool fine UDF 16³→8³: continuous target, surface = small value
                raw = data['fine_feats']
                fine = torch.from_numpy(raw.astype(np.float32) if raw.dtype == np.float16 else raw)
                N = fine.shape[0]
                vol = fine.reshape(N, 1, 16, 16, 16)
                submask = -F.max_pool3d(-vol, kernel_size=2, stride=2)  # min_pool
                submask = submask.reshape(N, SUBMASK_DIM).clamp(min=0.0, max=1.0)
            elif 'submask' in data.files:
                submask = torch.from_numpy(data['submask'].astype(np.float32))
            else:
                submask = torch.ones(coords.shape[0], SUBMASK_DIM)
        return {'coords': coords, 'submask': submask}

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
            coords_list, mask_list, layout = [], [], []
            start = 0
            for i, b in enumerate(sub):
                n = b['coords'].shape[0]
                coords_list.append(torch.cat([
                    torch.full((n, 1), i, dtype=torch.int32), b['coords'],
                ], dim=-1))
                mask_list.append(b['submask'])
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

            # Forward extra keys (cond, etc.)
            exclude = {'coords', 'submask'}
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
