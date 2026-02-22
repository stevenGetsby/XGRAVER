import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import cumesh
import cubvh
from graver.renderers.normal_render import render_random_normals_grid
# ================= 从 mesh2block 导入统一常量 =================
from graver.dataset_toolkits.mesh2block import (
    BLOCK_GRID, BLOCK_INNER, BLOCK_DIM, BLOCK_CORE_VERTS,
    PADDING, SAMPLE_RES, MC_THRESHOLD,
)

MC_THRESHOLD_USE = 1.0 * MC_THRESHOLD
train_dir = "./ckpt/test"


def _gpu_laplacian_smooth(vertices, faces, iterations=1, lam=0.5):
    """
    GPU 上的 Laplacian 平滑 (uniform weights).
    vertices: [V, 3] float CUDA
    faces:    [F, 3] long CUDA
    """
    v = vertices.clone()
    f = faces.long()
    for _ in range(iterations):
        # 建邻接: edge pairs from faces
        e01 = torch.stack([f[:, 0], f[:, 1]], dim=1)
        e12 = torch.stack([f[:, 1], f[:, 2]], dim=1)
        e20 = torch.stack([f[:, 2], f[:, 0]], dim=1)
        edges = torch.cat([e01, e12, e20, e01.flip(1), e12.flip(1), e20.flip(1)], dim=0)  # [6F, 2]

        src, dst = edges[:, 0], edges[:, 1]
        # 邻居坐标求和
        neighbor_sum = torch.zeros_like(v)
        neighbor_cnt = torch.zeros(v.shape[0], 1, device=v.device)
        neighbor_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3), v[src])
        neighbor_cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.shape[0], 1, device=v.device))
        neighbor_cnt = neighbor_cnt.clamp(min=1)
        centroid = neighbor_sum / neighbor_cnt
        v = v + lam * (centroid - v)
    return v, faces


def tokens_to_mesh(indices, tokens, output_path):
    """
    将稀疏 tokens 转换为 Mesh — 全 GPU 流水线.
    indices: [N, 3] block 坐标 (在 BLOCK_GRID³ 网格中)
    tokens: [N, BLOCK_DIM³] UDF 值
    """
    device = torch.device("cuda")
    t_start = time.time()

    n = len(indices)

    # ---- 全部搬上 GPU ----
    coords = torch.as_tensor(indices, dtype=torch.long, device=device)  # [N, 3]
    if coords.shape[1] == 4:
        coords = coords[:, 1:]

    F_full = torch.as_tensor(tokens, dtype=torch.float32, device=device).reshape(
        n, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM,
    )
    p = PADDING
    F_core = F_full[:, p:p + BLOCK_CORE_VERTS,
                       p:p + BLOCK_CORE_VERTS,
                       p:p + BLOCK_CORE_VERTS].contiguous()  # [N, 14, 14, 14]

    # ---- 边界角点平均 (GPU scatter) ----
    max_dim = BLOCK_GRID * BLOCK_INNER + BLOCK_CORE_VERTS
    S1 = max_dim * max_dim
    S2 = max_dim

    ci_arr = torch.arange(BLOCK_CORE_VERTS, device=device)
    ci, cj, ck = torch.meshgrid(ci_arr, ci_arr, ci_arr, indexing="ij")
    local_key = (ci * S1 + cj * S2 + ck).reshape(-1)  # [14³]

    bx = coords[:, 0] * BLOCK_INNER
    by = coords[:, 1] * BLOCK_INNER
    bz = coords[:, 2] * BLOCK_INNER
    base_key = bx * S1 + by * S2 + bz  # [N]

    flat_key = (base_key[:, None] + local_key[None, :]).reshape(-1)  # [N * 14³]
    flat_val = F_core.reshape(-1)  # [N * 14³]

    uniq, inverse = torch.unique(flat_key, return_inverse=True)
    num_uniq = uniq.shape[0]
    val_sum = torch.zeros(num_uniq, dtype=torch.float64, device=device)
    val_cnt = torch.zeros(num_uniq, dtype=torch.float64, device=device)
    val_sum.scatter_add_(0, inverse, flat_val.double())
    val_cnt.scatter_add_(0, inverse, torch.ones_like(flat_val, dtype=torch.float64))
    val_avg = (val_sum / val_cnt).float()

    n_shared = int((val_cnt > 1).sum().item())
    print(f"  Boundary averaging: {flat_val.numel():,} corners -> "
          f"{num_uniq:,} unique ({n_shared:,} shared)")

    F_core = val_avg[inverse].reshape(n, BLOCK_CORE_VERTS, BLOCK_CORE_VERTS, BLOCK_CORE_VERTS)
    del flat_key, flat_val, val_sum, val_cnt, val_avg, inverse, uniq, F_full

    # ---- 提取 8 角点, 筛选有效 voxel (GPU) ----
    vx, vy, vz = torch.meshgrid(
        torch.arange(BLOCK_INNER, device=device),
        torch.arange(BLOCK_INNER, device=device),
        torch.arange(BLOCK_INNER, device=device),
        indexing="ij",
    )
    local_vox = torch.stack([vx, vy, vz], dim=-1).reshape(-1, 3)  # [M, 3]
    lx, ly, lz = local_vox[:, 0], local_vox[:, 1], local_vox[:, 2]

    # [N, M, 8]
    all_logits = torch.stack([
        F_core[:, lx,     ly,     lz    ],
        F_core[:, lx + 1, ly,     lz    ],
        F_core[:, lx + 1, ly + 1, lz    ],
        F_core[:, lx,     ly + 1, lz    ],
        F_core[:, lx,     ly,     lz + 1],
        F_core[:, lx + 1, ly,     lz + 1],
        F_core[:, lx + 1, ly + 1, lz + 1],
        F_core[:, lx,     ly + 1, lz + 1],
    ], dim=-1)

    # [N, M, 3]
    all_global = coords[:, None, :] * BLOCK_INNER + local_vox[None, :, :]

    valid = (torch.isfinite(all_logits).all(dim=-1)
             & (all_logits.min(dim=-1).values < (MC_THRESHOLD_USE + 0.005)))

    valid_coords = all_global[valid].long()
    valid_logits = all_logits[valid].float()
    del all_logits, all_global, F_core

    if valid_coords.shape[0] == 0:
        print(f"⚠️ No valid voxels found for {output_path}")
        return

    print(f"  Voxels for MC: {valid_coords.shape[0]:,}")

    # ---- Marching Cubes (GPU) ----
    t_mc = time.time()
    try:
        v, f = cubvh.sparse_marching_cubes(valid_coords, valid_logits, MC_THRESHOLD_USE)
    except Exception as e:
        print(f"⚠️ MC Error: {e}")
        return
    print(f"  MC: {time.time() - t_mc:.2f}s, {f.shape[0]:,} faces")

    # 归一化顶点到 [-0.5, 0.5] (仍在 GPU)
    v = v.float() / SAMPLE_RES - 0.5

    # ---- CuMesh GPU 清理 + 减面 ----
    t_clean = time.time()
    cm = cumesh.CuMesh()
    cm.init(v.contiguous(), f.int().contiguous())

    cm.remove_duplicate_faces()
    cm.remove_degenerate_faces()
    cm.remove_small_connected_components(min_area=1e-6)
    print(f"  Clean: {time.time() - t_clean:.2f}s, {cm.num_faces:,} faces")

    TARGET_FACES = 8_000_000
    cur_faces = cm.num_faces
    if cur_faces > TARGET_FACES:
        t_dec = time.time()
        cm.simplify(TARGET_FACES, verbose=True)
        print(f"  Decimated: {cur_faces:,} -> {cm.num_faces:,} faces ({time.time() - t_dec:.2f}s)")

    new_v, new_f = cm.read()  # GPU tensors

    # ---- GPU Laplacian smoothing ----
    new_v, new_f = _gpu_laplacian_smooth(new_v, new_f, iterations=1, lam=0.5)

    # ---- 导出 (唯一 CPU 步骤: I/O) ----
    import trimesh
    mesh_out = trimesh.Trimesh(
        vertices=new_v.cpu().numpy(),
        faces=new_f.cpu().numpy(),
        process=False,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mesh_out.export(output_path)
    print(f"  Total: {time.time() - t_start:.2f}s -> {output_path}")


# ================= 主流程 =================
# train_dir = "./ckpt/s_flow_refine/256_64_7_13"
items_dir = os.path.join(train_dir, "snapshots")
output_dir = os.path.join(train_dir, "normalmaps")
os.makedirs(output_dir, exist_ok=True)

print(f"📏 Config: BLOCK_GRID={BLOCK_GRID}, SAMPLE_RES={SAMPLE_RES}")
print(f"📏 Block: {BLOCK_DIM}³ (with pad) -> {BLOCK_CORE_VERTS}³ (center)")
print(f"📏 MC_THRESHOLD={MC_THRESHOLD_USE:.4f}")

item_dir_list = [os.path.join(items_dir, item) for item in os.listdir(items_dir)]
for item_dir in item_dir_list:
    if not os.path.isdir(item_dir):
        continue
    step_dir_name = os.path.basename(item_dir)
    output_step_dir = os.path.join(output_dir, step_dir_name)
    os.makedirs(output_step_dir, exist_ok=True)

    for item in os.listdir(item_dir):
        if not item.endswith(".npz"):
            continue

        item_path = os.path.join(item_dir, item)
        mesh_pred_path = os.path.join(output_step_dir, item.replace(".npz", ".ply"))
        mesh_gt_path   = os.path.join(output_step_dir, item.replace(".npz", "_gt.ply"))
        normal_pred_jpg = os.path.join(output_step_dir, item.replace(".npz", "_normal_grid.jpg"))
        normal_gt_jpg   = os.path.join(output_step_dir, item.replace(".npz", "_gt_normal_grid.jpg"))

        with np.load(item_path) as data:
            coords = data["gt_coords"]
            if coords.shape[1] == 4:
                coords = coords[:, 1:]  # 去掉 batch dim

            pred_tokens = np.clip(data["pred_fine"], 0.0, 1.0)
            gt_tokens   = np.clip(data["gt_fine"], 0.0, 1.0)

        # 重建 Pred
        if not os.path.exists(mesh_pred_path):
            print(f"Reconstructing Pred: {item}")
            try:
                tokens_to_mesh(coords, pred_tokens, mesh_pred_path)
            except Exception as e:
                print(f"Reconstruction Error: {e}")

        if os.path.exists(mesh_pred_path) and not os.path.exists(normal_pred_jpg):
            t_render = time.time()
            try:
                render_random_normals_grid(mesh_pred_path, normal_pred_jpg, resolution=1024, radius=1.75)
                print(f"  Render Pred: {time.time() - t_render:.2f}s")
            except Exception as e:
                print(f"Render Error: {e}")
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # 重建 GT
        if not os.path.exists(mesh_gt_path):
            print(f"Reconstructing GT: {item}")
            tokens_to_mesh(coords, gt_tokens, mesh_gt_path)

        if os.path.exists(mesh_gt_path) and not os.path.exists(normal_gt_jpg):
            t_render = time.time()
            try:
                render_random_normals_grid(mesh_gt_path, normal_gt_jpg, resolution=1024, radius=1.75)
                print(f"  Render GT: {time.time() - t_render:.2f}s")
            except Exception as e:
                print(f"Render Error: {e}")
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

print("✅ Done!")