import os
import argparse
import json
import re
from subprocess import call
from multiprocessing import Pool

import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from tqdm import tqdm

from utils import sphere_hammersley_sequence

BLENDER_PATH = "/home/ubuntu/yizhao/XGRAVER/blender-3.0.1-linux-x64/blender"

_VIEW_PNG_RE = re.compile(r"^\d{3}\.png$", re.IGNORECASE)


def _count_view_pngs(folder: str) -> int:
    if not folder or (not os.path.isdir(folder)):
        return 0
    try:
        return sum(1 for f in os.listdir(folder) if _VIEW_PNG_RE.match(f) is not None)
    except FileNotFoundError:
        return 0


def _is_render_done(output_folder: str, num_views: int) -> bool:
    """
    严格判定：必须 transforms.json 存在 + 视角图(000.png...)数量 >= num_views 才算完成

    兼容两种输出结构：
    - 新版 blender_script/render.py：图片直接在 output_folder 下
    - 老版本：图片在 output_folder/renders 下
    """
    transforms_file = os.path.join(output_folder, "transforms.json")
    if not os.path.exists(transforms_file):
        return False

    # 优先检查 legacy 的 renders/ 子目录；否则检查 output_folder 根目录
    legacy_renders = os.path.join(output_folder, "renders")
    if os.path.isdir(legacy_renders):
        n = _count_view_pngs(legacy_renders)
    else:
        n = _count_view_pngs(output_folder)

    return n >= int(num_views)


def _render_cond(
    file_path: str,
    sha256: str,
    output_dir: str,
    num_views: int = 4,
    engine: str = "BLENDER_EEVEE",
    fast_mode: bool = True,
    ultra_fast_mode: bool = False,
):
    output_folder = os.path.join(output_dir, "renders_cond", sha256)
    os.makedirs(output_folder, exist_ok=True)

    # 严格提前判定：必须 >= num_views 才算完成
    if _is_render_done(output_folder, num_views):
        return {"sha256": sha256, "cond_rendered": True}

    # 生成视角参数（用 sha256 做确定性 seed）
    try:
        seed = int(str(sha256)[:8], 16)
    except (ValueError, TypeError):
        seed = sum(ord(c) for c in str(sha256)) & 0xFFFFFFFF

    rng = np.random.RandomState(seed)
    offset = (float(rng.rand()), float(rng.rand()))

    yaws, pitchs = [], []
    for i in range(int(num_views)):
        y, p = sphere_hammersley_sequence(i, int(num_views), offset)
        yaws.append(y)
        pitchs.append(p)

    fov_min, fov_max = 10, 70
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = rng.uniform(k_min, k_max, (int(num_views),))
    radius = [1 / np.sqrt(k) for k in ks]
    fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]
    views = [{"yaw": y, "pitch": p, "radius": r, "fov": f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

    # 无显示环境兼容：EEVEE 需要显示；无 DISPLAY 则优先 xvfb-run，否则回退 CYCLES
    import shutil as _shutil

    desired_engine = engine
    use_xvfb = False
    if desired_engine == "BLENDER_EEVEE" and not os.environ.get("DISPLAY"):
        if _shutil.which("xvfb-run") is not None:
            use_xvfb = True
        else:
            desired_engine = "CYCLES"
            print(
                f"警告: EEVEE需要显示环境且未找到xvfb-run，自动切换到CYCLES引擎 (sha256={sha256})"
            )

    args = [
        BLENDER_PATH,
        "-b",
        "-P",
        os.path.join(os.path.dirname(__file__), "blender_script", "render.py"),
        "--",
        "--views",
        json.dumps(views),
        "--object",
        os.path.expanduser(file_path),
        "--output_folder",
        output_folder,
        "--resolution",
        "512",
        "--engine",
        desired_engine,
        "--save_mesh",
    ]

    if use_xvfb:
        args = ["xvfb-run", "-a"] + args

    if ultra_fast_mode:
        args.append("--ultra_fast_mode")
    elif fast_mode:
        args.append("--fast_mode")

    stdout_log = os.path.join(output_folder, "blender_stdout.log")
    stderr_log = os.path.join(output_folder, "blender_stderr.log")
    with open(stdout_log, "w") as _out, open(stderr_log, "w") as _err:
        result = call(args, stdout=_out, stderr=_err)

    if result != 0:
        print(f"Blender失败 code={result} sha256={sha256}")

    # 严格检查结果：必须 >= num_views 才算完成
    if _is_render_done(output_folder, num_views):
        return {"sha256": sha256, "cond_rendered": True}

    return {"sha256": sha256, "cond_rendered": False}


def _apply_render_cond(args):
    file_path, sha256, output_dir, num_views, engine, fast_mode, ultra_fast_mode = args
    return _render_cond(
        file_path=file_path,
        sha256=sha256,
        output_dir=output_dir,
        num_views=num_views,
        engine=engine,
        fast_mode=fast_mode,
        ultra_fast_mode=ultra_fast_mode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Directory to save the metadata")
    parser.add_argument("--filter_low_aesthetic_score",type=float,default=None, help="Filter objects with aesthetic score lower than this value")
    parser.add_argument("--instances", type=str, default=None, help="Instances to process")
    parser.add_argument("--num_views", type=int, default=8, help="Number of views to render")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument(
        "--engine",
        type=str,
        default="BLENDER_EEVEE",
        choices=["BLENDER_EEVEE", "CYCLES"],
        help="Blender render engine",
    )
    parser.add_argument("--fast_mode", action="store_true", help="Enable fast rendering mode")
    parser.add_argument("--ultra_fast_mode", action="store_true", help="Enable ultra-fast rendering mode")
    opt = edict(vars(parser.parse_args()))

    metadata_path = os.path.join(opt.root, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise ValueError("metadata.csv not found")
    metadata = pd.read_csv(metadata_path)

    # 兼容字段：优先 original_glb_path；否则退回 local_path
    source_col = "original_glb_path" if "original_glb_path" in metadata.columns else "local_path"
    if source_col not in metadata.columns:
        raise ValueError("Required column original_glb_path or local_path not found in metadata.csv")

    if opt.instances is None:
        metadata = metadata[metadata[source_col].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata["aesthetic_score"] >= opt.filter_low_aesthetic_score]
        # 注意：不再用 metadata 里的 cond_rendered 过滤，避免“磁盘缺图但表里是 True”导致永远不补
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, "r") as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(",")
        metadata = metadata[metadata["sha256"].isin(instances)]

    # 分片处理
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    # 跳过已完成对象：严格检查 transforms + png 数量 >= num_views
    records = []
    to_process = []
    for _, row in metadata.iterrows():
        sha256 = row["sha256"]
        output_folder = os.path.join(opt.root, "renders_cond", sha256)
        if _is_render_done(output_folder, opt.num_views):
            records.append({"sha256": sha256, "cond_rendered": True})
        else:
            to_process.append(row)

    metadata = pd.DataFrame(to_process)
    print(f"Processing {len(metadata)} objects...")

    args_list = []
    for _, row in metadata.iterrows():
        file_path = row[source_col]
        if not os.path.isabs(file_path):
            file_path = os.path.join(opt.root, file_path)

        args_list.append(
            (
                file_path,
                row["sha256"],
                opt.root,
                opt.num_views,
                opt.engine,
                opt.fast_mode,
                opt.ultra_fast_mode,
            )
        )

    rendered = []
    if args_list:
        with Pool(processes=opt.max_workers) as pool:
            with tqdm(
                total=len(args_list),
                desc=f"渲染进度 [{opt.rank+1}/{opt.world_size}]",
                unit="个对象",
                ncols=80,
            ) as pbar:
                for result in pool.imap_unordered(_apply_render_cond, args_list):
                    if result:
                        rendered.append(result)
                    pbar.update(1)

    rendered = [r for r in rendered if r is not None]

    rendered_df = pd.DataFrame.from_records(rendered)
    records_df = pd.DataFrame.from_records(records)
    cond_rendered = pd.concat([rendered_df, records_df], ignore_index=True)

    # 确保空结果也有固定表头，避免 KeyError
    if cond_rendered.empty:
        cond_rendered = pd.DataFrame(columns=["sha256", "cond_rendered"])
    else:
        if "sha256" not in cond_rendered.columns:
            cond_rendered["sha256"] = pd.NA
        if "cond_rendered" not in cond_rendered.columns:
            cond_rendered["cond_rendered"] = pd.NA

    out_csv_path = os.path.join(opt.root, f"cond_rendered_{opt.rank}.csv")
    cond_rendered.to_csv(out_csv_path, index=False)

    successful = int((cond_rendered["cond_rendered"] == True).sum()) if "cond_rendered" in cond_rendered.columns else 0
    failed = int((cond_rendered["cond_rendered"] == False).sum()) if "cond_rendered" in cond_rendered.columns else 0
    print(f"处理完成: 成功 {successful} 个, 失败 {failed} 个. 结果已保存至 {out_csv_path}")
