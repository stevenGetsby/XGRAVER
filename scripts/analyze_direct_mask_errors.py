import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graver import datasets, models
from graver.modules import sparse as sp
from graver.trainers.flow_matching.mixins.image_conditioned import ImageConditionedMixin


def load_config(path):
    with open(path, "r") as file:
        return json.load(file)


def build_dataset(config):
    dataset_config = config.get("test_dataset") or config["dataset"]
    args = dict(dataset_config.get("args", {}))
    roots = args.pop("roots")
    return getattr(datasets, dataset_config["name"])(roots, **args)


def build_model(config_path, ckpt_path, device):
    config = load_config(config_path)
    model_config = config["models"]["denoiser"]
    args = dict(model_config["args"])
    args["use_fp16"] = False
    model = getattr(models, model_config["name"])(**args)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    model.to(device).eval()
    return model, missing, unexpected


def metric_from_counts(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def quantiles(values):
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float32)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
    }


def bucket_name(gt_count):
    if gt_count == 0:
        return "00"
    if gt_count <= 2:
        return "01-02"
    if gt_count <= 4:
        return "03-04"
    if gt_count <= 8:
        return "05-08"
    if gt_count <= 16:
        return "09-16"
    if gt_count <= 32:
        return "17-32"
    return "33-64"


def update_bucket_stats(stats, gt, pred):
    gt_count = gt.sum(dim=1).long().cpu().numpy()
    pred_count = pred.sum(dim=1).long().cpu().numpy()
    fp_count = ((pred == 1) & (gt == 0)).sum(dim=1).long().cpu().numpy()
    fn_count = ((pred == 0) & (gt == 1)).sum(dim=1).long().cpu().numpy()
    tp_count = ((pred == 1) & (gt == 1)).sum(dim=1).long().cpu().numpy()
    for gtc, pc, fp, fn, tp in zip(gt_count, pred_count, fp_count, fn_count, tp_count):
        bucket = bucket_name(int(gtc))
        item = stats[bucket]
        item["blocks"] += 1
        item["gt"] += int(gtc)
        item["pred"] += int(pc)
        item["tp"] += int(tp)
        item["fp"] += int(fp)
        item["fn"] += int(fn)


def finalize_bucket_stats(stats):
    out = {}
    for bucket in ["00", "01-02", "03-04", "05-08", "09-16", "17-32", "33-64"]:
        item = stats.get(bucket, defaultdict(int))
        tp, fp, fn = item["tp"], item["fp"], item["fn"]
        metric = metric_from_counts(tp, fp, fn)
        out[bucket] = {
            "blocks": item["blocks"],
            "gt": item["gt"],
            "pred": item["pred"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            **metric,
        }
    return out


def finalize_counts(tp, fp, fn, pred_pos, gt_pos, total_voxels):
    return {
        **metric_from_counts(tp, fp, fn),
        "pos_gt": gt_pos / max(total_voxels, 1),
        "pos_pred": pred_pos / max(total_voxels, 1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "gt_pos_count": int(gt_pos),
        "pred_pos_count": int(pred_pos),
    }


def summarize_adaptive_by_pred_count(logits_list, gt_list, thresholds):
    logit_thresholds = {
        threshold: math.log(threshold / (1.0 - threshold))
        for threshold in thresholds
    }
    bucket_order = ["00", "01-02", "03-04", "05-08", "09-16", "17-32", "33-64"]
    stats = {
        bucket: {
            threshold: defaultdict(int)
            for threshold in thresholds
        }
        for bucket in bucket_order
    }
    total_voxels = 0

    for logits, gt in zip(logits_list, gt_list):
        gt_bool = gt > 0.5
        default_pred_count = (logits > 0.0).sum(dim=1).long().cpu().numpy()
        total_voxels += gt.numel()
        for bucket in bucket_order:
            row_idx = [idx for idx, count in enumerate(default_pred_count) if bucket_name(int(count)) == bucket]
            if not row_idx:
                continue
            rows = torch.as_tensor(row_idx, dtype=torch.long)
            logits_bucket = logits[rows]
            gt_bucket = gt_bool[rows]
            for threshold, logit_threshold in logit_thresholds.items():
                pred = logits_bucket > logit_threshold
                item = stats[bucket][threshold]
                item["blocks"] += int(rows.numel())
                item["gt_pos"] += int(gt_bucket.sum().item())
                item["pred_pos"] += int(pred.sum().item())
                item["tp"] += int((pred & gt_bucket).sum().item())
                item["fp"] += int((pred & ~gt_bucket).sum().item())
                item["fn"] += int((~pred & gt_bucket).sum().item())

    best_by_bucket = {}
    combined = defaultdict(int)
    for bucket in bucket_order:
        bucket_rows = []
        for threshold in thresholds:
            item = stats[bucket][threshold]
            metric = finalize_counts(
                item["tp"], item["fp"], item["fn"],
                item["pred_pos"], item["gt_pos"],
                item["blocks"] * 64,
            )
            metric["threshold"] = float(threshold)
            metric["blocks"] = int(item["blocks"])
            bucket_rows.append(metric)
        best = max(bucket_rows, key=lambda row: row["iou"])
        best_by_bucket[bucket] = best
        combined["blocks"] += best["blocks"]
        combined["tp"] += best["tp"]
        combined["fp"] += best["fp"]
        combined["fn"] += best["fn"]
        combined["pred_pos"] += best["pred_pos_count"]
        combined["gt_pos"] += best["gt_pos_count"]

    combined_metric = finalize_counts(
        combined["tp"], combined["fp"], combined["fn"],
        combined["pred_pos"], combined["gt_pos"], total_voxels,
    )
    combined_metric["blocks"] = int(combined["blocks"])
    return {
        "group_key": "pred_count_at_threshold_0.5",
        "combined": combined_metric,
        "best_by_bucket": best_by_bucket,
    }


def summarize_adaptive_by_subvoxel(logits_list, gt_list, thresholds):
    logit_thresholds = {
        threshold: math.log(threshold / (1.0 - threshold))
        for threshold in thresholds
    }
    stats = {
        local_idx: {
            threshold: defaultdict(int)
            for threshold in thresholds
        }
        for local_idx in range(64)
    }
    total_voxels = sum(gt.numel() for gt in gt_list)
    total_blocks = sum(gt.shape[0] for gt in gt_list)

    for logits, gt in zip(logits_list, gt_list):
        gt_bool = gt > 0.5
        for local_idx in range(64):
            logits_col = logits[:, local_idx]
            gt_col = gt_bool[:, local_idx]
            for threshold, logit_threshold in logit_thresholds.items():
                pred = logits_col > logit_threshold
                item = stats[local_idx][threshold]
                item["gt_pos"] += int(gt_col.sum().item())
                item["pred_pos"] += int(pred.sum().item())
                item["tp"] += int((pred & gt_col).sum().item())
                item["fp"] += int((pred & ~gt_col).sum().item())
                item["fn"] += int((~pred & gt_col).sum().item())

    best_by_index = []
    combined = defaultdict(int)
    for local_idx in range(64):
        rows = []
        for threshold in thresholds:
            item = stats[local_idx][threshold]
            metric = finalize_counts(
                item["tp"], item["fp"], item["fn"],
                item["pred_pos"], item["gt_pos"],
                total_blocks,
            )
            metric["threshold"] = float(threshold)
            metric["local_idx"] = local_idx
            metric["xyz"] = [local_idx // 16, (local_idx // 4) % 4, local_idx % 4]
            rows.append(metric)
        best = max(rows, key=lambda row: row["iou"])
        best_by_index.append(best)
        combined["tp"] += best["tp"]
        combined["fp"] += best["fp"]
        combined["fn"] += best["fn"]
        combined["pred_pos"] += best["pred_pos_count"]
        combined["gt_pos"] += best["gt_pos_count"]

    combined_metric = finalize_counts(
        combined["tp"], combined["fp"], combined["fn"],
        combined["pred_pos"], combined["gt_pos"], total_voxels,
    )
    return {
        "group_key": "local_subvoxel_index",
        "combined": combined_metric,
        "threshold_map": [row["threshold"] for row in best_by_index],
        "best_by_index": best_by_index,
    }


def heatmap_to_nested(values):
    return np.asarray(values, dtype=np.int64).reshape(4, 4, 4).tolist()


def make_offsets(radius):
    offsets = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dx, dy, dz))
    return offsets


OFFSETS_R1 = make_offsets(1)
OFFSETS_R2 = [offset for offset in make_offsets(2) if max(map(abs, offset)) == 2]


def mask_global_points(coords, mask):
    nonzero = mask.nonzero(as_tuple=False)
    if nonzero.numel() == 0:
        return []
    block = coords[nonzero[:, 0]].long()
    local = nonzero[:, 1].long()
    gx = block[:, 0] * 4 + local // 16
    gy = block[:, 1] * 4 + (local // 4) % 4
    gz = block[:, 2] * 4 + local % 4
    points = torch.stack([gx, gy, gz], dim=1).cpu().numpy()
    return [tuple(map(int, point)) for point in points]


def count_near(error_points, target_set):
    near1 = near2 = far = 0
    for point in error_points:
        x, y, z = point
        found = False
        for dx, dy, dz in OFFSETS_R1:
            q = (x + dx, y + dy, z + dz)
            if q in target_set:
                near1 += 1
                found = True
                break
        if found:
            continue
        for dx, dy, dz in OFFSETS_R2:
            q = (x + dx, y + dy, z + dz)
            if q in target_set:
                near2 += 1
                found = True
                break
        if not found:
            far += 1
    return near1, near2, far


def print_threshold_table(name, sweep):
    print(f"\n[{name}] threshold sweep")
    print(f"{'thr':>5} {'iou':>8} {'prec':>8} {'rec':>8} {'pos_pred':>9}")
    for row in sweep:
        print(
            f"{row['threshold']:5.2f} {row['iou']:8.4f} "
            f"{row['precision']:8.4f} {row['recall']:8.4f} {row['pos_pred']:9.4f}"
        )


def summarize_model(name, logits_list, gt_list, coords_list, thresholds):
    summary = {
        "threshold_sweep": [],
        "confidence": {},
        "bucket_by_gt_count": {},
        "adaptive_by_pred_count": {},
        "adaptive_by_subvoxel": {},
        "subvoxel_heatmap": {},
        "global_neighbor": {},
        "worst_samples": [],
    }

    total_voxels = sum(gt.numel() for gt in gt_list)
    for threshold in thresholds:
        logit_threshold = math.log(threshold / (1.0 - threshold))
        tp = fp = fn = pred_pos = gt_pos = 0
        for logits, gt in zip(logits_list, gt_list):
            pred = logits > logit_threshold
            gt_bool = gt > 0.5
            tp += int((pred & gt_bool).sum().item())
            fp += int((pred & ~gt_bool).sum().item())
            fn += int((~pred & gt_bool).sum().item())
            pred_pos += int(pred.sum().item())
            gt_pos += int(gt_bool.sum().item())
        row = {
            "threshold": float(threshold),
            **metric_from_counts(tp, fp, fn),
            "pos_gt": gt_pos / max(total_voxels, 1),
            "pos_pred": pred_pos / max(total_voxels, 1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        summary["threshold_sweep"].append(row)

    default_logit_threshold = 0.0
    heat = {key: torch.zeros(64, dtype=torch.long) for key in ["gt", "pred", "tp", "fp", "fn"]}
    buckets = defaultdict(lambda: defaultdict(int))
    confidences = {key: [] for key in ["tp_prob", "fp_prob", "fn_prob", "tn_prob"]}
    neighbor = defaultdict(int)
    worst = []

    for sample_idx, (logits, gt, coords) in enumerate(zip(logits_list, gt_list, coords_list)):
        pred = logits > default_logit_threshold
        gt_bool = gt > 0.5
        tp_mask = pred & gt_bool
        fp_mask = pred & ~gt_bool
        fn_mask = ~pred & gt_bool
        tn_mask = ~pred & ~gt_bool

        probs = torch.sigmoid(logits).float().cpu()
        for key, mask in [
            ("tp_prob", tp_mask),
            ("fp_prob", fp_mask),
            ("fn_prob", fn_mask),
            ("tn_prob", tn_mask),
        ]:
            values = probs[mask.cpu()]
            if values.numel() > 0:
                confidences[key].extend(values.flatten().tolist())

        heat["gt"] += gt_bool.long().cpu().sum(dim=0)
        heat["pred"] += pred.long().cpu().sum(dim=0)
        heat["tp"] += tp_mask.long().cpu().sum(dim=0)
        heat["fp"] += fp_mask.long().cpu().sum(dim=0)
        heat["fn"] += fn_mask.long().cpu().sum(dim=0)
        update_bucket_stats(buckets, gt_bool.cpu(), pred.cpu())

        gt_points = mask_global_points(coords, gt_bool.cpu())
        pred_points = mask_global_points(coords, pred.cpu())
        fp_points = mask_global_points(coords, fp_mask.cpu())
        fn_points = mask_global_points(coords, fn_mask.cpu())
        fp_near1, fp_near2, fp_far = count_near(fp_points, set(gt_points))
        fn_near1, fn_near2, fn_far = count_near(fn_points, set(pred_points))
        neighbor["fp_total"] += len(fp_points)
        neighbor["fp_near_gt_r1"] += fp_near1
        neighbor["fp_near_gt_r2"] += fp_near2
        neighbor["fp_far_gt"] += fp_far
        neighbor["fn_total"] += len(fn_points)
        neighbor["fn_near_pred_r1"] += fn_near1
        neighbor["fn_near_pred_r2"] += fn_near2
        neighbor["fn_far_pred"] += fn_far

        tp = int(tp_mask.sum().item())
        fp = int(fp_mask.sum().item())
        fn = int(fn_mask.sum().item())
        iou = metric_from_counts(tp, fp, fn)["iou"]
        worst.append((iou, sample_idx, tp, fp, fn, int(gt_bool.sum().item()), int(pred.sum().item())))

    summary["confidence"] = {key: quantiles(values) for key, values in confidences.items()}
    summary["bucket_by_gt_count"] = finalize_bucket_stats(buckets)
    summary["adaptive_by_pred_count"] = summarize_adaptive_by_pred_count(logits_list, gt_list, thresholds)
    summary["adaptive_by_subvoxel"] = summarize_adaptive_by_subvoxel(logits_list, gt_list, thresholds)
    summary["subvoxel_heatmap"] = {key: heatmap_to_nested(value.numpy()) for key, value in heat.items()}
    summary["global_neighbor"] = dict(neighbor)
    summary["worst_samples"] = [
        {
            "rank": rank,
            "sample_local_idx": sample_idx,
            "iou": float(iou),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_pos": gt_pos,
            "pred_pos": pred_pos,
        }
        for rank, (iou, sample_idx, tp, fp, fn, gt_pos, pred_pos)
        in enumerate(sorted(worst, key=lambda item: item[0])[:10], 1)
    ]
    return summary


def summarize_delta(base_name, cand_name, base_logits, cand_logits, gt_list):
    totals = defaultdict(int)
    for base, cand, gt in zip(base_logits, cand_logits, gt_list):
        gt_bool = gt > 0.5
        base_pred = base > 0
        cand_pred = cand > 0
        totals["fixed_fn"] += int((~base_pred & gt_bool & cand_pred).sum().item())
        totals["new_fp"] += int((~base_pred & ~gt_bool & cand_pred).sum().item())
        totals["fixed_fp"] += int((base_pred & ~gt_bool & ~cand_pred).sum().item())
        totals["broken_tp"] += int((base_pred & gt_bool & ~cand_pred).sum().item())
        totals["same_tp"] += int((base_pred & gt_bool & cand_pred).sum().item())
        totals["same_fp"] += int((base_pred & ~gt_bool & cand_pred).sum().item())
        totals["same_fn"] += int((~base_pred & gt_bool & ~cand_pred).sum().item())
    return {"base": base_name, "candidate": cand_name, **totals}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CONFIG", "CKPT"), required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--image-cond-model", default="dinov2_vitl14_reg")
    parser.add_argument("--output", default="analysis/direct_mask_errors.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = build_dataset(load_config(args.dataset_config))
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[: args.num_samples].tolist()
    thresholds = [float(item) for item in args.thresholds.split(",")]

    encoder = ImageConditionedMixin(image_cond_model=args.image_cond_model)
    model_specs = []
    for name, config_path, ckpt_path in args.model:
        model, missing, unexpected = build_model(config_path, ckpt_path, device)
        print(f"[{name}] missing={missing}")
        print(f"[{name}] unexpected={unexpected}")
        model_specs.append((name, model))

    logits_by_model = {name: [] for name, _ in model_specs}
    gt_list = []
    coords_list = []
    instance_list = []

    for offset, idx in enumerate(indices, 1):
        data = dataset[int(idx)]
        instance_list.append(dataset.instances[int(idx)] if hasattr(dataset, "instances") else int(idx))
        gt = data["submask"].float().cpu()
        gt_list.append(gt)
        coords_list.append(data["coords"].int().cpu())
        cond_img = data["cond"].unsqueeze(0).to(device)
        with torch.no_grad():
            cond = encoder.encode_image(cond_img)
        coords_int = data["coords"].int()
        batch_coords = torch.cat([
            torch.zeros(coords_int.shape[0], 1, dtype=torch.int32), coords_int,
        ], dim=1).to(device)

        for name, model in model_specs:
            token_dim = model.token_dim
            dummy_feats = torch.zeros(batch_coords.shape[0], token_dim, device=device)
            dummy = sp.SparseTensor(feats=dummy_feats, coords=batch_coords)
            t = torch.zeros(1, device=device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(dummy, t, cond)
            logits_by_model[name].append(pred.feats.float().cpu())

        if offset % 16 == 0:
            print(f"processed {offset}/{len(indices)}")

    result = {
        "num_samples": len(indices),
        "seed": args.seed,
        "indices": indices,
        "instances": instance_list,
        "models": {},
        "delta": [],
    }
    for name, _ in model_specs:
        summary = summarize_model(name, logits_by_model[name], gt_list, coords_list, thresholds)
        result["models"][name] = summary
        print_threshold_table(name, summary["threshold_sweep"])
        print(f"\n[{name}] confidence")
        print(json.dumps(summary["confidence"], indent=2))
        print(f"\n[{name}] bucket_by_gt_count")
        print(json.dumps(summary["bucket_by_gt_count"], indent=2))
        print(f"\n[{name}] adaptive_by_pred_count")
        print(json.dumps(summary["adaptive_by_pred_count"], indent=2))
        print(f"\n[{name}] adaptive_by_subvoxel")
        print(json.dumps({
            "combined": summary["adaptive_by_subvoxel"]["combined"],
            "threshold_map": summary["adaptive_by_subvoxel"]["threshold_map"],
        }, indent=2))
        print(f"\n[{name}] global_neighbor")
        print(json.dumps(summary["global_neighbor"], indent=2))

    if len(model_specs) >= 2:
        base_name = model_specs[0][0]
        for cand_name, _ in model_specs[1:]:
            delta = summarize_delta(base_name, cand_name, logits_by_model[base_name], logits_by_model[cand_name], gt_list)
            result["delta"].append(delta)
            print(f"\n[delta {base_name} -> {cand_name}]")
            print(json.dumps(delta, indent=2))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as file:
        json.dump(result, file, indent=2)
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()