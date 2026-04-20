"""
Stage-level diagnostic evaluation for GRAVER pipeline.

Generates 6 ablation combinations to isolate which stage (coords/mask/feats)
causes quality degradation, with quantitative metrics:
  - Coords: block-level IoU / Precision / Recall
  - Feats: MSE on GT coords (surface voxels only)
  - Mesh: Chamfer Distance + F-Score vs GT mesh
"""
import argparse
import json
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from easydict import EasyDict as edict
from scipy.spatial import cKDTree

from eval import (
    _load_model_from_weight,
    _ImageEncoder,
    _pick_random_indices,
    load_sample,
    sample_coords,
    sample_mask,
    sample_feats,
    _detect_submask_res,
    _adapt_submask,
    _upsample_submask_to_voxel,
    _save_mesh_and_normal,
)
from graver.dataset_toolkits.mesh2block import BLOCK_DIM


# =====================================================================
# Metrics
# =====================================================================

def coords_iou(pred_coords: torch.Tensor, gt_coords: torch.Tensor):
    """Block-level IoU between predicted and GT coordinates."""
    pred_set = set(map(tuple, pred_coords.numpy().tolist()))
    gt_set = set(map(tuple, gt_coords.numpy().tolist()))
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    iou = tp / max(tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {'iou': iou, 'precision': prec, 'recall': rec,
            'tp': tp, 'fp': fp, 'fn': fn,
            'pred_count': len(pred_set), 'gt_count': len(gt_set)}


def feats_mse(pred_feats: torch.Tensor, gt_feats: torch.Tensor,
              submask: torch.Tensor = None):
    """MSE between predicted and GT feats, optionally only on surface voxels."""
    if submask is not None:
        submask_res = _detect_submask_res(submask)
        voxel_mask = _upsample_submask_to_voxel(submask, submask_res)
        mask = voxel_mask.squeeze() > 0.5
        pred_f = pred_feats[mask] if pred_feats.shape[0] == mask.shape[0] else pred_feats
        gt_f = gt_feats[mask] if gt_feats.shape[0] == mask.shape[0] else gt_feats
    else:
        pred_f = pred_feats
        gt_f = gt_feats
    mse = F.mse_loss(pred_f.float(), gt_f.float()).item()
    return mse


def chamfer_distance(mesh_path_a: str, mesh_path_b: str, n_samples: int = 30000):
    """Chamfer Distance + F-Score between two meshes."""
    try:
        mesh_a = trimesh.load(mesh_path_a, process=False)
        mesh_b = trimesh.load(mesh_path_b, process=False)
    except Exception as e:
        return {'chamfer': float('nan'), 'fscore_001': float('nan'),
                'fscore_005': float('nan'), 'error': str(e)}

    if not hasattr(mesh_a, 'vertices') or len(mesh_a.vertices) < 10:
        return {'chamfer': float('nan'), 'fscore_001': float('nan'),
                'fscore_005': float('nan'), 'error': 'mesh_a too small'}
    if not hasattr(mesh_b, 'vertices') or len(mesh_b.vertices) < 10:
        return {'chamfer': float('nan'), 'fscore_001': float('nan'),
                'fscore_005': float('nan'), 'error': 'mesh_b too small'}

    pts_a = mesh_a.sample(n_samples)
    pts_b = mesh_b.sample(n_samples)

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    dist_a2b, _ = tree_b.query(pts_a)
    dist_b2a, _ = tree_a.query(pts_b)

    cd = (dist_a2b.mean() + dist_b2a.mean()) / 2

    # F-Score at thresholds
    results = {'chamfer': float(cd)}
    for tau in [0.01, 0.05]:
        prec = (dist_a2b < tau).mean()
        rec = (dist_b2a < tau).mean()
        f = 2 * prec * rec / max(prec + rec, 1e-8)
        results[f'fscore_{tau:.3f}'] = float(f)

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='GRAVER stage-level diagnostic')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='./eval_diagnostic')

    parser.add_argument('--coords_weight', type=str, required=True)
    parser.add_argument('--coords_config', type=str, required=True)
    parser.add_argument('--mask_weight', type=str, required=True)
    parser.add_argument('--mask_config', type=str, required=True)
    parser.add_argument('--feats_weight', type=str, required=True)
    parser.add_argument('--feats_config', type=str, required=True)

    parser.add_argument('--mask_threshold', type=float, default=0.5)
    parser.add_argument('--coords_threshold', type=float, default=0.25)
    parser.add_argument('--max_block_num', type=int, default=15000)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--cfg_strength', type=float, default=3.0)
    parser.add_argument('--noise_scale_coords', type=float, default=1.0)
    parser.add_argument('--noise_scale_feats', type=float, default=2.0)
    parser.add_argument('--cfg_interval_min', type=float, default=0.1)
    parser.add_argument('--cfg_interval_max', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 6 ablation combos
    COMBOS = [
        'A_gt',               # GT coords + GT mask + GT feats
        'B_gt_predfeats',     # GT coords + GT mask + pred feats
        'C_mask_only',        # GT coords + pred mask + GT feats
        'D_mask_feats',       # GT coords + pred mask + pred feats
        'E_predcoords_gt',    # pred coords + GT mask(adapted) + GT feats
        'F_full',             # pred coords + pred mask + pred feats
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    for c in COMBOS + ['cond']:
        os.makedirs(os.path.join(args.output_dir, c), exist_ok=True)

    # Load models
    print('Loading coords model ...')
    coords_model = _load_model_from_weight(
        args.coords_config, args.coords_weight, device=args.device)
    print('Loading mask model ...')
    mask_model = _load_model_from_weight(
        args.mask_config, args.mask_weight, device=args.device)
    print('Loading feats model ...')
    feats_model = _load_model_from_weight(
        args.feats_config, args.feats_weight, device=args.device)
    print('Loading image encoder ...')
    encoder = _ImageEncoder(device=args.device)

    # Pick samples
    indices = _pick_random_indices(
        args.data_root, args.num_samples,
        max_block_num=args.max_block_num, max_samples=args.max_samples,
        seed=args.seed)

    cfg_interval = (args.cfg_interval_min, args.cfg_interval_max)

    all_metrics = []

    for seq, idx in enumerate(indices, 1):
        print(f'\n{"=" * 70}')
        print(f'[{seq}/{len(indices)}] dataset_idx={idx}')
        print(f'{"=" * 70}')

        sample = load_sample(
            args.data_root, idx,
            max_block_num=args.max_block_num, max_samples=args.max_samples)

        gt_coords = sample['coords']        # [N, 3] int
        gt_submask = sample['submask']       # [N, R³]
        gt_feats = sample['fine_feats']      # [N, D]
        gt_coords_np = gt_coords.numpy().astype(np.int32)
        n_gt = gt_coords.shape[0]
        print(f'  GT blocks={n_gt}')

        sample['cond_image'].save(
            os.path.join(args.output_dir, 'cond', f'{seq}.jpg'))

        cond = encoder.encode(sample['cond_image'])

        # ---- Stage 1: predict coords ----
        print(f'\n  [Stage 1] Sampling coords ...')
        pred_coords = sample_coords(
            coords_model, cond,
            noise_scale=args.noise_scale_coords,
            cfg_strength=args.cfg_strength,
            cfg_interval=cfg_interval,
            steps=args.steps, device=args.device,
            threshold=args.coords_threshold)
        pred_coords_np = pred_coords.numpy().astype(np.int32)

        c_metrics = coords_iou(pred_coords, gt_coords)
        print(f'    Coords: IoU={c_metrics["iou"]:.4f}  '
              f'Prec={c_metrics["precision"]:.4f}  '
              f'Rec={c_metrics["recall"]:.4f}  '
              f'pred={c_metrics["pred_count"]} gt={c_metrics["gt_count"]}')

        # ---- Stage 2: predict mask on GT coords & pred coords ----
        print(f'  [Stage 2] Mask prediction ...')
        pred_mask_on_gt = sample_mask(
            mask_model, cond, gt_coords,
            device=args.device, threshold=args.mask_threshold)
        pred_mask_res = round(mask_model.token_dim ** (1.0 / 3.0))

        pred_mask_on_pred = sample_mask(
            mask_model, cond, pred_coords,
            device=args.device, threshold=args.mask_threshold)

        # Mask IoU on GT coords
        gt_submask_res = _detect_submask_res(gt_submask)
        if gt_submask_res != pred_mask_res:
            gt_mask_cmp = _adapt_submask(
                gt_submask, gt_submask_res, pred_mask_res).to(args.device)
        else:
            gt_mask_cmp = gt_submask.to(args.device)
        pred_b = (pred_mask_on_gt > 0.5).float()
        gt_b = (gt_mask_cmp > 0.5).float()
        tp = (pred_b * gt_b).sum().item()
        fp = (pred_b * (1 - gt_b)).sum().item()
        fn = ((1 - pred_b) * gt_b).sum().item()
        mask_iou = tp / max(tp + fp + fn, 1)
        mask_prec = tp / max(tp + fp, 1)
        mask_rec = tp / max(tp + fn, 1)
        print(f'    Mask (on GT coords): IoU={mask_iou:.4f}  '
              f'Prec={mask_prec:.4f}  Rec={mask_rec:.4f}')

        # ---- Stage 3: predict feats with various combos ----
        # B: GT coords + GT mask + pred feats
        print(f'  [Stage 3] Sampling feats on GT coords + GT mask ...')
        feats_submask_res = feats_model.submask_resolution
        gt_submask_for_feats = gt_submask
        if feats_submask_res > 0 and gt_submask_res != feats_submask_res:
            gt_submask_for_feats = _adapt_submask(
                gt_submask, gt_submask_res, feats_submask_res)

        pred_feats_B = sample_feats(
            feats_model, cond, gt_coords, gt_submask_for_feats,
            noise_scale=args.noise_scale_feats,
            cfg_strength=args.cfg_strength,
            cfg_interval=cfg_interval,
            steps=args.steps, device=args.device)

        # Feats MSE (on GT coords, GT mask, surface voxels)
        fmse_B = feats_mse(pred_feats_B.cpu(), gt_feats, gt_submask)
        print(f'    Feats MSE (GT coords + GT mask): {fmse_B:.6f}')

        # D: GT coords + pred mask + pred feats
        print(f'  [Stage 3] Sampling feats on GT coords + pred mask ...')
        pred_mask_for_feats = pred_mask_on_gt
        if feats_submask_res > 0 and pred_mask_res != feats_submask_res:
            pred_mask_for_feats = _adapt_submask(
                pred_mask_on_gt, pred_mask_res, feats_submask_res)

        pred_feats_D = sample_feats(
            feats_model, cond, gt_coords, pred_mask_for_feats,
            noise_scale=args.noise_scale_feats,
            cfg_strength=args.cfg_strength,
            cfg_interval=cfg_interval,
            steps=args.steps, device=args.device)

        fmse_D = feats_mse(pred_feats_D.cpu(), gt_feats, gt_submask)
        print(f'    Feats MSE (GT coords + pred mask): {fmse_D:.6f}')

        # F: pred coords + pred mask + pred feats
        if pred_coords.shape[0] >= 5:
            print(f'  [Stage 3] Sampling feats on pred coords + pred mask ...')
            pred_mask_for_feats_F = pred_mask_on_pred
            if feats_submask_res > 0 and pred_mask_res != feats_submask_res:
                pred_mask_for_feats_F = _adapt_submask(
                    pred_mask_on_pred, pred_mask_res, feats_submask_res)

            pred_feats_F = sample_feats(
                feats_model, cond, pred_coords, pred_mask_for_feats_F,
                noise_scale=args.noise_scale_feats,
                cfg_strength=args.cfg_strength,
                cfg_interval=cfg_interval,
                steps=args.steps, device=args.device)
        else:
            pred_feats_F = None

        torch.cuda.empty_cache()

        # ---- Generate meshes for all 6 combos ----
        def _mesh(name, coords_np, feats_np):
            mesh_path = os.path.join(args.output_dir, name, f'{seq}.ply')
            normal_path = os.path.join(args.output_dir, name, f'{seq}_normal.jpg')
            _save_mesh_and_normal(coords_np, feats_np, mesh_path, normal_path)
            return mesh_path

        print(f'\n  Generating meshes ...')
        # A: GT everything
        path_A = _mesh('A_gt', gt_coords_np, gt_feats.numpy().astype(np.float32))

        # B: GT coords + GT mask + pred feats
        path_B = _mesh('B_gt_predfeats', gt_coords_np,
                        pred_feats_B.cpu().numpy().astype(np.float32))

        # C: GT coords + pred mask + GT feats (mask_only)
        voxel_mask_C = _upsample_submask_to_voxel(
            pred_mask_on_gt.cpu(), pred_mask_res)
        feats_C = gt_feats.clone()
        feats_C[voxel_mask_C.squeeze() < 0.5] = 1.0
        path_C = _mesh('C_mask_only', gt_coords_np,
                        feats_C.numpy().astype(np.float32))

        # D: GT coords + pred mask + pred feats
        path_D = _mesh('D_mask_feats', gt_coords_np,
                        pred_feats_D.cpu().numpy().astype(np.float32))

        # E: pred coords + GT mask(adapted) + GT feats
        path_E = None
        if pred_coords.shape[0] >= 5:
            # For E, we need GT feats on pred coords -> only possible for
            # coords that overlap with GT. For non-overlapping, fill with 1.0
            pred_set = set(map(tuple, pred_coords_np.tolist()))
            gt_map = {tuple(c): i for i, c in enumerate(gt_coords_np.tolist())}
            feats_E = np.ones((pred_coords_np.shape[0], gt_feats.shape[1]),
                               dtype=np.float32)
            for i, c in enumerate(pred_coords_np.tolist()):
                ct = tuple(c)
                if ct in gt_map:
                    feats_E[i] = gt_feats[gt_map[ct]].numpy()
            path_E = _mesh('E_predcoords_gt', pred_coords_np, feats_E)

        # F: full pipeline
        path_F = None
        if pred_feats_F is not None:
            path_F = _mesh('F_full', pred_coords_np,
                            pred_feats_F.cpu().numpy().astype(np.float32))

        # ---- Chamfer Distance vs GT ----
        print(f'\n  Computing Chamfer Distance ...')
        cd_results = {}
        for name, path in [('B_gt_predfeats', path_B),
                           ('C_mask_only', path_C),
                           ('D_mask_feats', path_D),
                           ('E_predcoords_gt', path_E),
                           ('F_full', path_F)]:
            if path and os.path.exists(path):
                cd = chamfer_distance(path_A, path)
                cd_results[name] = cd
                print(f'    {name}: CD={cd["chamfer"]:.6f}  '
                      f'F@0.01={cd.get("fscore_0.010", 0):.4f}  '
                      f'F@0.05={cd.get("fscore_0.050", 0):.4f}')

        # ---- Collect per-sample metrics ----
        m = {
            'sample': seq,
            'gt_blocks': n_gt,
            'pred_blocks': c_metrics['pred_count'],
            'coords_iou': c_metrics['iou'],
            'coords_prec': c_metrics['precision'],
            'coords_rec': c_metrics['recall'],
            'mask_iou': mask_iou,
            'mask_prec': mask_prec,
            'mask_rec': mask_rec,
            'feats_mse_gt_mask': fmse_B,
            'feats_mse_pred_mask': fmse_D,
        }
        for name, cd in cd_results.items():
            m[f'cd_{name}'] = cd.get('chamfer', float('nan'))
            m[f'f01_{name}'] = cd.get('fscore_0.010', float('nan'))
            m[f'f05_{name}'] = cd.get('fscore_0.050', float('nan'))

        all_metrics.append(m)

    # ---- Summary ----
    print(f'\n{"=" * 70}')
    print(f'STAGE DIAGNOSTIC SUMMARY ({len(all_metrics)} samples)')
    print(f'{"=" * 70}')

    # Averages
    keys_avg = ['coords_iou', 'coords_prec', 'coords_rec',
                'mask_iou', 'mask_prec', 'mask_rec',
                'feats_mse_gt_mask', 'feats_mse_pred_mask']
    cd_keys = set()
    for m in all_metrics:
        for k in m:
            if k.startswith('cd_') or k.startswith('f01_') or k.startswith('f05_'):
                cd_keys.add(k)
    keys_avg += sorted(cd_keys)

    print(f'\n--- Per-sample ---')
    for m in all_metrics:
        print(f"  Sample {m['sample']}: "
              f"coords_IoU={m['coords_iou']:.4f} "
              f"mask_IoU={m['mask_iou']:.4f} "
              f"feats_MSE(gt_mask)={m['feats_mse_gt_mask']:.6f} "
              f"feats_MSE(pred_mask)={m['feats_mse_pred_mask']:.6f}")

    print(f'\n--- Averages ---')
    for k in keys_avg:
        vals = [m.get(k, float('nan')) for m in all_metrics]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if vals:
            avg = np.mean(vals)
            print(f'  {k}: {avg:.6f}')

    print(f'\n--- Stage Attribution (Chamfer Distance breakdown) ---')
    cd_combos = ['B_gt_predfeats', 'C_mask_only', 'D_mask_feats',
                 'E_predcoords_gt', 'F_full']
    combo_desc = {
        'B_gt_predfeats':  'GT coords + GT mask  + PRED feats  → pure feats error',
        'C_mask_only':     'GT coords + PRED mask + GT feats   → pure mask error',
        'D_mask_feats':    'GT coords + PRED mask + PRED feats → mask+feats error',
        'E_predcoords_gt': 'PRED coords + GT mask + GT feats   → pure coords error',
        'F_full':          'PRED coords + PRED mask + PRED feats → total error',
    }
    for combo in cd_combos:
        key = f'cd_{combo}'
        vals = [m.get(key, float('nan')) for m in all_metrics]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if vals:
            avg = np.mean(vals)
            print(f'  {combo}: CD={avg:.6f}  | {combo_desc.get(combo, "")}')

    print(f'\n--- Diagnosis ---')
    avgs = {}
    for combo in cd_combos:
        key = f'cd_{combo}'
        vals = [m.get(key, float('nan')) for m in all_metrics]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        avgs[combo] = np.mean(vals) if vals else float('nan')

    if not np.isnan(avgs.get('B_gt_predfeats', float('nan'))):
        feats_err = avgs['B_gt_predfeats']
        mask_err = avgs['C_mask_only']
        coords_err = avgs['E_predcoords_gt']
        total = avgs['F_full']
        print(f'  Feats-only error:  CD={feats_err:.6f}')
        print(f'  Mask-only error:   CD={mask_err:.6f}')
        print(f'  Coords-only error: CD={coords_err:.6f}')
        print(f'  Total error:       CD={total:.6f}')
        errors = {'feats': feats_err, 'mask': mask_err, 'coords': coords_err}
        worst = max(errors, key=errors.get)
        print(f'\n  ★ Largest single-stage error: **{worst.upper()}** '
              f'(CD={errors[worst]:.6f})')

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, 'diagnostic_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'\nMetrics saved: {metrics_path}')

    # Generate grids
    print(f'\nGenerating comparison grids ...')
    for combo in COMBOS + ['cond']:
        folder = os.path.join(args.output_dir, combo)
        if combo == 'cond':
            files = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')],
                          key=lambda x: int(x.split('.')[0]))
        else:
            files = sorted([f for f in os.listdir(folder) if f.endswith('_normal.jpg')],
                          key=lambda x: int(x.split('_')[0]))
        if not files:
            continue
        from PIL import Image
        imgs = [Image.open(os.path.join(folder, f)) for f in files]
        # Single row grid
        w, h = imgs[0].size
        grid = Image.new('RGB', (w * len(imgs), h))
        for i, img in enumerate(imgs):
            grid.paste(img, (i * w, 0))
        grid.save(os.path.join(folder, f'grid_all.jpg'))

    print('\nDone!')


if __name__ == '__main__':
    main()
