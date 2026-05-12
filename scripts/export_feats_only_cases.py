import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import (
    _ImageEncoder,
    _adapt_submask,
    _detect_submask_res,
    _load_model_from_weight,
    sample_feats,
)
from graver.dataset_toolkits.mesh2block import BLOCK_FOLDER, COL_PREFIX
from graver.datasets.block_feats import BlockFeats


def load_metadata(data_root: str, max_block_num: int, max_samples: int) -> pd.DataFrame:
    metadata = pd.read_csv(os.path.join(data_root, "metadata.csv"))
    metadata = metadata[metadata[f"{COL_PREFIX}_block_status"] == "success"]
    metadata = metadata[metadata["cond_rendered"].fillna(False).astype(bool)]
    if max_block_num > 0:
        metadata = metadata[metadata[f"{COL_PREFIX}_num_blocks"] <= max_block_num]
    metadata = metadata.reset_index(drop=True)
    if max_samples > 0 and len(metadata) > max_samples:
        metadata = metadata.iloc[:max_samples].reset_index(drop=True)
    return metadata


def pick_indices(total: int, num_cases: int, seed: int) -> List[int]:
    rng = np.random.RandomState(seed)
    num_cases = min(num_cases, total)
    return sorted(rng.choice(total, size=num_cases, replace=False).tolist())


def make_cond_image(data_root: str, sha256: str, image_size: int) -> Image.Image:
    image_root = os.path.join(data_root, "renders_cond", sha256)
    with open(os.path.join(image_root, "transforms.json")) as file:
        transforms_meta = json.load(file)
    frame = transforms_meta["frames"][0]
    image = Image.open(os.path.join(image_root, frame["file_path"])).convert("RGBA")

    alpha = np.array(image.getchannel(3))
    nz = alpha.nonzero()
    if nz[0].size == 0:
        height, width = alpha.shape
        bbox = [0, 0, width - 1, height - 1]
    else:
        bbox = [nz[1].min(), nz[0].min(), nz[1].max(), nz[0].max()]
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    half_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2 * 1.2
    crop = [
        int(cx - half_size),
        int(cy - half_size),
        int(cx + half_size),
        int(cy + half_size),
    ]
    image = image.crop(crop).resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha_ch = image.getchannel(3)
    image_rgb = image.convert("RGB")
    img_t = torch.tensor(np.array(image_rgb)).permute(2, 0, 1).float() / 255.0
    alpha_t = torch.tensor(np.array(alpha_ch)).float() / 255.0
    return Image.fromarray(
        (img_t * alpha_t.unsqueeze(0)).permute(1, 2, 0).mul(255).clamp(0, 255)
        .byte().numpy()
    )


def copy_cond_views(data_root: str, sha256: str, case_dir: str, max_views: int) -> Dict:
    src_root = os.path.join(data_root, "renders_cond", sha256)
    dst_root = os.path.join(case_dir, "cond_views")
    os.makedirs(dst_root, exist_ok=True)

    transforms_path = os.path.join(src_root, "transforms.json")
    with open(transforms_path) as file:
        transforms_meta = json.load(file)
    shutil.copy2(transforms_path, os.path.join(dst_root, "transforms.json"))

    frames = transforms_meta.get("frames", [])
    if max_views > 0:
        frames = frames[:max_views]
    copied = []
    for frame in frames:
        rel_path = frame.get("file_path", "")
        if not rel_path:
            continue
        src = os.path.join(src_root, rel_path)
        if not os.path.exists(src):
            continue
        dst_name = os.path.basename(rel_path)
        shutil.copy2(src, os.path.join(dst_root, dst_name))
        copied.append(dst_name)
    return {"cond_root": src_root, "views": copied}


def load_npz(data_root: str, sha256: str):
    npz_path = os.path.join(data_root, BLOCK_FOLDER, f"{sha256}.npz")
    with np.load(npz_path) as data:
        coords = torch.from_numpy(data["coords"]).int()
        raw = data["fine_feats"]
        fine_feats = torch.from_numpy(
            raw.astype(np.float32) if raw.dtype == np.float16 else raw
        ).float()
        if "submask" in data.files:
            submask = torch.from_numpy(data["submask"].astype(np.float32))
        else:
            submask = torch.ones(coords.shape[0], 64)
    return npz_path, coords, fine_feats, submask


def append_manifest(path: str, row: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = not os.path.exists(path)
    columns = [
        "seq",
        "dataset_idx",
        "sha256",
        "status",
        "n_blocks",
        "case_dir",
        "gt_ply",
        "pred_ply",
        "cond_views",
        "error",
    ]
    with open(path, "a") as file:
        if header:
            file.write("\t".join(columns) + "\n")
        values = [str(row.get(column, "")).replace("\n", " ") for column in columns]
        file.write("\t".join(values) + "\n")


def export_case(args, seq: int, dataset_idx: int, row, model, encoder):
    sha256 = row["sha256"]
    case_dir = os.path.join(args.output_dir, "cases", f"{seq:04d}_idx{dataset_idx}_{sha256[:12]}")
    os.makedirs(case_dir, exist_ok=True)
    gt_path = os.path.join(case_dir, "gt.ply")
    pred_path = os.path.join(case_dir, "pred.ply")
    metadata_path = os.path.join(case_dir, "metadata.json")

    cond_info = copy_cond_views(args.data_root, sha256, case_dir, args.max_cond_views)
    cond_image = make_cond_image(args.data_root, sha256, args.image_size)
    cond_image.save(os.path.join(case_dir, "cond_input.jpg"), quality=95)

    npz_path, coords, fine_feats, submask = load_npz(args.data_root, sha256)
    coords_np = coords.numpy().astype(np.int32)

    if args.overwrite or not os.path.exists(gt_path):
        BlockFeats.tokens_to_mesh(
            coords_np,
            fine_feats.numpy().astype(np.float32),
            gt_path,
            verbose=True,
        )

    with torch.no_grad():
        cond = encoder.encode(cond_image)
        feats_submask_res = model.submask_resolution
        submask_res = _detect_submask_res(submask)
        if feats_submask_res > 0 and submask_res != feats_submask_res:
            submask_for_feats = _adapt_submask(submask, submask_res, feats_submask_res)
        else:
            submask_for_feats = submask
        pred_feats = sample_feats(
            model,
            cond,
            coords,
            submask_for_feats,
            noise_scale=args.noise_scale_feats,
            cfg_strength=args.cfg_strength,
            cfg_interval=(args.cfg_interval_min, args.cfg_interval_max),
            steps=args.steps,
            device=args.device,
        )

    if args.overwrite or not os.path.exists(pred_path):
        BlockFeats.tokens_to_mesh(
            coords_np,
            pred_feats.cpu().numpy().astype(np.float32),
            pred_path,
            verbose=True,
        )

    meta = {
        "seq": seq,
        "dataset_idx": int(dataset_idx),
        "sha256": sha256,
        "n_blocks": int(coords.shape[0]),
        "npz_path": npz_path,
        "gt_ply": gt_path,
        "pred_ply": pred_path,
        "cond_views": cond_info,
        "feats_weight": args.feats_weight,
        "feats_config": args.feats_config,
    }
    with open(metadata_path, "w") as file:
        json.dump(meta, file, indent=2)

    return {
        "seq": seq,
        "dataset_idx": dataset_idx,
        "sha256": sha256,
        "status": "ok",
        "n_blocks": int(coords.shape[0]),
        "case_dir": case_dir,
        "gt_ply": gt_path,
        "pred_ply": pred_path,
        "cond_views": len(cond_info["views"]),
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser(description="Export feats-only GRAVER cases")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feats_weight", required=True)
    parser.add_argument("--feats_config", required=True)
    parser.add_argument("--num_cases", type=int, default=200)
    parser.add_argument("--max_block_num", type=int, default=15000)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg_strength", type=float, default=3.0)
    parser.add_argument("--noise_scale_feats", type=float, default=2.0)
    parser.add_argument("--cfg_interval_min", type=float, default=0.1)
    parser.add_argument("--cfg_interval_max", type=float, default=1.0)
    parser.add_argument("--max_cond_views", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)
    os.makedirs(os.path.join(args.output_dir, "cases"), exist_ok=True)

    metadata = load_metadata(args.data_root, args.max_block_num, args.max_samples)
    selected = pick_indices(len(metadata), args.num_cases, args.seed)
    shard = [(seq, idx) for seq, idx in enumerate(selected, 1) if (seq - 1) % args.world_size == args.rank]
    print(
        f"[rank {args.rank}/{args.world_size}] total={len(metadata)} "
        f"selected={len(selected)} shard={len(shard)}",
        flush=True,
    )

    model = _load_model_from_weight(args.feats_config, args.feats_weight, device=args.device)
    encoder = _ImageEncoder(device=args.device)

    manifest_path = os.path.join(args.output_dir, f"manifest_rank{args.rank}.tsv")
    for local_i, (seq, dataset_idx) in enumerate(shard, 1):
        row = metadata.iloc[dataset_idx]
        sha256 = row["sha256"]
        print(
            f"[rank {args.rank}] {local_i}/{len(shard)} "
            f"seq={seq:04d} idx={dataset_idx} sha={sha256}",
            flush=True,
        )
        try:
            manifest_row = export_case(args, seq, dataset_idx, row, model, encoder)
        except Exception as exc:
            traceback.print_exc()
            manifest_row = {
                "seq": seq,
                "dataset_idx": dataset_idx,
                "sha256": sha256,
                "status": "error",
                "n_blocks": row.get(f"{COL_PREFIX}_num_blocks", ""),
                "case_dir": os.path.join(args.output_dir, "cases", f"{seq:04d}_idx{dataset_idx}_{sha256[:12]}"),
                "gt_ply": "",
                "pred_ply": "",
                "cond_views": "",
                "error": repr(exc),
            }
        append_manifest(manifest_path, manifest_row)
        torch.cuda.empty_cache()

    print(f"[rank {args.rank}] done -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()