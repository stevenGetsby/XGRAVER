"""
Test if dilated pred_mask covers gt_mask.
Compute per-sample recall/deep-FN statistics at various dilation kernels.
"""
import os
import sys
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F


# Same constants as project
BLOCK_DIM = 16
SUBMASK_RES = 8


def upsample_submask(submask_flat, block_dim=BLOCK_DIM, submask_res=SUBMASK_RES):
    """[T, R^3] -> [T, D^3] via nearest neighbor."""
    T = submask_flat.shape[0]
    R = submask_res
    scale = block_dim // R
    sub3d = submask_flat.reshape(T, 1, R, R, R)
    voxel3d = F.interpolate(sub3d.float(), scale_factor=scale, mode='nearest')
    return voxel3d.reshape(T, -1)


def dilate_voxel_mask(voxel_mask, kernel=3, iters=1, block_dim=BLOCK_DIM):
    """Dilate [T, D^3] binary mask by `iters` iterations of max_pool3d."""
    T = voxel_mask.shape[0]
    D = block_dim
    vol = voxel_mask.reshape(T, 1, D, D, D).float()
    pad = kernel // 2
    for _ in range(iters):
        vol = F.max_pool3d(vol, kernel_size=kernel, stride=1, padding=pad)
    return vol.reshape(T, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, default="/mnt/data/yizhao/TRAIN")
    ap.add_argument("--pred_mask_dir", type=str,
                    default="/mnt/data/yizhao/TRAIN/pred_mask_cache")
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Collect instances that have both pred_mask cache AND gt npz
    pred_files = sorted(glob.glob(os.path.join(args.pred_mask_dir, "*.npy")))
    print(f"Found {len(pred_files)} pred mask files")

    # Dilation configs: (kernel, iterations, label)
    configs = [
        (1, 0, "raw_pred (k=1)"),
        (3, 1, "dil_1x (k=3, 1 iter)"),
        (3, 2, "dil_2x (k=3, 2 iters)"),
        (3, 3, "dil_3x (k=3, 3 iters)"),
        (5, 1, "dil_k5 (k=5, 1 iter)"),
    ]

    # Accumulators
    stats = {label: {
        "recall_sum": 0.0, "fn_rate_sum": 0.0,
        "mask_ratio_sum": 0.0, "iou_sum": 0.0,
        "per_sample_recall": [],
        "per_sample_fn_rate": [],
        "count": 0,
    } for _, _, label in configs}

    tested = 0
    missing = 0

    for pred_file in pred_files:
        if tested >= args.num_samples:
            break

        sha = os.path.splitext(os.path.basename(pred_file))[0]

        # Find GT npz (blocks_64_15_occ8 folder per inspection)
        candidates = glob.glob(os.path.join(
            args.train_root, "blocks_64_15_occ8", f"{sha}.npz"))
        if not candidates:
            candidates = glob.glob(os.path.join(
                args.train_root, "**", f"{sha}.npz"), recursive=True)
            # Filter out pbr variant
            candidates = [c for c in candidates if "pbr" not in c]
        if not candidates:
            missing += 1
            continue
        gt_path = candidates[0]

        try:
            pred_submask = np.load(pred_file)  # [T, R^3]
            gt_data = np.load(gt_path)
            if "submask" not in gt_data:
                missing += 1
                continue
            gt_submask = gt_data["submask"]  # [T, R^3] or [T, D^3]?
        except Exception as e:
            print(f"  skip {sha}: {e}")
            missing += 1
            continue

        if pred_submask.shape != gt_submask.shape:
            # Could be that GT is stored at voxel resolution already; skip
            print(f"  shape mismatch {sha}: pred={pred_submask.shape}, gt={gt_submask.shape}")
            missing += 1
            continue

        pred_t = torch.from_numpy(pred_submask.astype(np.float32)).to(device)
        gt_t = torch.from_numpy(gt_submask.astype(np.float32)).to(device)

        # Upsample to voxel level
        pred_voxel = upsample_submask(pred_t)  # [T, D^3]
        gt_voxel = upsample_submask(gt_t)
        gt_b = gt_voxel > 0.5
        gt_total = gt_b.sum().item()
        if gt_total == 0:
            continue

        for kernel, iters, label in configs:
            if iters == 0:
                dil = (pred_voxel > 0.5)
            else:
                dil_f = dilate_voxel_mask(pred_voxel, kernel=kernel, iters=iters)
                dil = dil_f > 0.5

            tp = (dil & gt_b).sum().item()
            fp = (dil & ~gt_b).sum().item()
            fn = (~dil & gt_b).sum().item()

            recall = tp / gt_total
            fn_rate = fn / gt_total
            mask_ratio = dil.float().mean().item()
            iou = tp / max(tp + fp + fn, 1)

            s = stats[label]
            s["recall_sum"] += recall
            s["fn_rate_sum"] += fn_rate
            s["mask_ratio_sum"] += mask_ratio
            s["iou_sum"] += iou
            s["per_sample_recall"].append(recall)
            s["per_sample_fn_rate"].append(fn_rate)
            s["count"] += 1

        tested += 1
        if tested % 10 == 0:
            print(f"[{tested}/{args.num_samples}] processed")

    print(f"\nTested {tested} samples, missing {missing}")
    print("=" * 110)
    print(f"{'config':<25} {'mean_recall':>12} {'mean_fn':>10} {'mean_ratio':>12} "
          f"{'mean_iou':>10} {'min_recall':>12} {'p10_recall':>12}")
    print("-" * 110)
    for _, _, label in configs:
        s = stats[label]
        n = s["count"]
        if n == 0:
            continue
        recalls = np.array(s["per_sample_recall"])
        fns = np.array(s["per_sample_fn_rate"])
        print(
            f"{label:<25} "
            f"{s['recall_sum']/n:>12.4f} "
            f"{s['fn_rate_sum']/n:>10.4f} "
            f"{s['mask_ratio_sum']/n:>12.4f} "
            f"{s['iou_sum']/n:>10.4f} "
            f"{recalls.min():>12.4f} "
            f"{np.percentile(recalls, 10):>12.4f}"
        )

    # Distribution of recall for dil_1x
    if stats["dil_1x (k=3, 1 iter)"]["count"] > 0:
        print("\nRecall distribution @ dil_1x:")
        r = np.array(stats["dil_1x (k=3, 1 iter)"]["per_sample_recall"])
        for th in [0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]:
            frac = (r >= th).mean()
            print(f"  recall >= {th:.3f}: {frac*100:.1f}%")


if __name__ == "__main__":
    main()
