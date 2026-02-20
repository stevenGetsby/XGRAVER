import os
import math
from typing import Union
import numpy as np
import torch
import utils3d
from .components import StandardDatasetBase, ImageConditionedMixin
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer
from ..dataset_toolkits.mesh2block import BLOCK_GRID, BLOCK_INNER


class BlockCoords(StandardDatasetBase):
    """
    Block coordinates dataset for Stage 1 (Dense Flow).
    Loads block coords from npz and constructs BLOCK_GRID³ occupancy grid.
    """

    def __init__(self,
        roots,
        max_block_num: int = 15000,
        min_block_num: int = 0,
        min_aesthetic_score: float = 5.0,
    ):
        self.resolution = BLOCK_GRID
        self.max_block_num = max_block_num
        self.min_block_num = min_block_num
        self.min_aesthetic_score = min_aesthetic_score
        self.value_range = (0, 1)

        self._col_prefix = f'{BLOCK_GRID}_{BLOCK_INNER}'
        self._block_prefix = f'blocks_{self._col_prefix}'

        super().__init__(roots)
        
    def filter_metadata(self, metadata):
        stats = {}

        metadata = metadata[metadata[f'{self._col_prefix}_block_status'] == "success"]
        stats['block successed:'] = len(metadata)

        metadata = metadata[metadata[f'{self._col_prefix}_num_blocks'] <= self.max_block_num]
        stats[f'block num <= {self.max_block_num}:'] = len(metadata)

        if self.min_block_num > 0:
            metadata = metadata[metadata[f'{self._col_prefix}_num_blocks'] >= self.min_block_num]
            stats[f'block num >= {self.min_block_num}:'] = len(metadata)

        return metadata, stats

    def get_instance(self, root, instance):
        npz_path = os.path.join(root, self._block_prefix, f'{instance}.npz')
        with np.load(npz_path) as data:
            coords = data['coords']  # Shape: (N, 3), dtype: uint8/int
        
        coords = torch.from_numpy(coords).long()
        ss = torch.zeros(1, self.resolution, self.resolution, self.resolution, dtype=torch.float32)
        mask = (coords >= 0) & (coords < self.resolution)
        valid_coords = coords[mask.all(dim=1)]
        
        ss[0, valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = 1.0

        return {'x_0': ss}

    @torch.no_grad()
    def visualize_sample(self, ss: Union[torch.Tensor, dict]):
        if isinstance(ss, dict):
            ss = ss.get('x_0', ss.get('ss'))

        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'

        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts, ints = [], []
        for yaw, p in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(p),
                np.cos(yaw) * np.cos(p),
                np.sin(p),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(
                orig, torch.zeros(3).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []
        ss = ss.cuda()
        for i in range(ss.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords = torch.nonzero(ss[i, 0] > 0.5, as_tuple=False)
            representation.position = coords.float() / self.resolution
            representation.depth = torch.full(
                (representation.position.shape[0], 1),
                int(math.log2(self.resolution)),
                dtype=torch.uint8, device='cuda',
            )

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr,
                                      colors_overwrite=representation.position)
                r, c = j // tile[1], j % tile[1]
                image[:, r*512:(r+1)*512, c*512:(c+1)*512] = res['color']
            images.append(image)

        return torch.stack(images)


class ImageConditionedBlockCoords(ImageConditionedMixin, BlockCoords):
    """
    Image-conditioned block coords dataset
    """
    pass