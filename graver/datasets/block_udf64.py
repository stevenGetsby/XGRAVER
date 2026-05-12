"""
Stage-2 dataset: per-block 64-dim min-pooled UDF.

Loads fine_feats [T, 4096] from npz, min-pools to [T, 64] (4³).
Also loads GT submask [T, 64] for mask-metric evaluation at snapshot time.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from .components import StandardDatasetBase, ImageConditionedMixin
from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices
from ..dataset_toolkits.mesh2block import SUBMASK_DIM, BLOCK_FOLDER, COL_PREFIX


class BlockUDF64(StandardDatasetBase):

    def __init__(self, roots, *, max_block_num=15000, min_block_num=0,
                 min_aesthetic_score=5.0, max_samples=0):
        self.max_block_num = max_block_num
        self.min_block_num = min_block_num
        self.min_aesthetic_score = min_aesthetic_score
        self.max_samples = max_samples
        self.value_range = (0, 1)
        super().__init__(roots)

        self.filter_existing_instances(
            lambda root, instance: os.path.exists(
                os.path.join(root, BLOCK_FOLDER, f'{instance}.npz')),
            stat_name='npz readable',
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

    @staticmethod
    def _minpool_4096_to_64(feats: torch.Tensor) -> torch.Tensor:
        """Min-pool fine_feats [T, 4096] -> [T, 64] via 3D min-pool k=4 s=4."""
        T = feats.shape[0]
        vol = feats.reshape(T, 1, 16, 16, 16)
        # min-pool = -max_pool(-x)
        pooled = -F.max_pool3d(-vol, kernel_size=4, stride=4)  # [T, 1, 4, 4, 4]
        return pooled.reshape(T, 64)

    def get_instance(self, root, instance):
        npz_path = os.path.join(root, BLOCK_FOLDER, f'{instance}.npz')
        with np.load(npz_path) as data:
            coords = torch.from_numpy(data['coords']).int()
            raw = data['fine_feats']
            if raw.dtype == np.float16:
                fine_feats = torch.from_numpy(raw.astype(np.float32))
            else:
                fine_feats = torch.from_numpy(raw).float()

            # GT submask for metric evaluation
            if 'submask' in data.files:
                submask = torch.from_numpy(data['submask'].astype(np.float32))
            else:
                submask = torch.ones(coords.shape[0], SUBMASK_DIM)

        # Min-pool UDF: [T, 4096] -> [T, 64]
        udf_64 = self._minpool_4096_to_64(fine_feats)

        # Max-pool submask 512->64 for GT mask comparison
        T = submask.shape[0]
        if submask.shape[1] == 512:
            vol = submask.reshape(T, 1, 8, 8, 8)
            submask_64 = F.max_pool3d(vol, kernel_size=2, stride=2).reshape(T, 64)
        else:
            submask_64 = submask

        return {
            'coords': coords,
            'udf_64': udf_64,
            'submask_64': submask_64,
        }

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
            coords_list, udf_list, mask_list, layout = [], [], [], []
            start = 0
            for i, b in enumerate(sub):
                n = b['coords'].shape[0]
                coords_list.append(torch.cat([
                    torch.full((n, 1), i, dtype=torch.int32), b['coords'],
                ], dim=-1))
                udf_list.append(b['udf_64'])
                mask_list.append(b['submask_64'])
                layout.append(slice(start, start + n))
                start += n

            coords = torch.cat(coords_list, 0)
            udf = torch.cat(udf_list, 0)
            submask = torch.cat(mask_list, 0)

            pack = {
                'x_0': SparseTensor(
                    coords=coords, feats=udf,
                    shape=torch.Size([len(group), udf.shape[1]]),
                    layout=layout,
                ),
                'submask_64': submask,
            }

            # Forward extra keys (cond, etc.)
            exclude = {'coords', 'udf_64', 'submask_64'}
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

        return packs[0] if len(packs) == 1 else packs


class ImageConditionedBlockUDF64(ImageConditionedMixin, BlockUDF64):
    pass
