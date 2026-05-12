#!/usr/bin/env python3
"""Create a lightweight copy of an exported case directory with decimated PLY meshes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def mirror_non_ply_files(src_root: Path, dst_root: Path) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    for src in sorted(src_root.rglob("*")):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if rel.name in {"gt.ply", "pred.ply"} and "cases" in rel.parts:
            jobs.append((str(src), str(dst)))
            continue
        link_or_copy(src, dst)
    return jobs


def decimate_one_open3d(src: Path, dst: Path, ratio: float) -> dict[str, object]:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(src))
    vertices_before = len(mesh.vertices)
    faces_before = len(mesh.triangles)
    if faces_before <= 0:
        raise RuntimeError(f"mesh has no faces: {src}")

    target_faces = max(4, int(round(faces_before * ratio)))
    if target_faces < faces_before:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp.ply")
    if tmp.exists():
        tmp.unlink()
    ok = o3d.io.write_triangle_mesh(str(tmp), mesh, write_ascii=False, compressed=False, print_progress=False)
    if not ok:
        raise RuntimeError(f"failed to write mesh: {dst}")
    tmp.replace(dst)

    return {
        "src": str(src),
        "dst": str(dst),
        "status": "ok",
        "backend": "open3d",
        "vertices_before": vertices_before,
        "faces_before": faces_before,
        "vertices_after": len(mesh.vertices),
        "faces_after": len(mesh.triangles),
        "src_bytes": src.stat().st_size,
        "dst_bytes": dst.stat().st_size,
    }


def decimate_one_cumesh(src: Path, dst: Path, ratio: float, device: str = "cuda:0") -> dict[str, object]:
    import cumesh
    import torch
    import trimesh

    mesh_in = trimesh.load_mesh(str(src), process=False)
    vertices_before = len(mesh_in.vertices)
    faces_before = len(mesh_in.faces)
    if faces_before <= 0:
        raise RuntimeError(f"mesh has no faces: {src}")

    target_faces = max(4, int(round(faces_before * ratio)))
    vertices = torch.as_tensor(mesh_in.vertices, dtype=torch.float32, device=device).contiguous()
    faces = torch.as_tensor(mesh_in.faces, dtype=torch.int32, device=device).contiguous()
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    if target_faces < faces_before:
        mesh.simplify(target_faces, verbose=False)
    new_vertices, new_faces = mesh.read()
    new_mesh = trimesh.Trimesh(
        vertices=new_vertices.detach().cpu().numpy(),
        faces=new_faces.detach().cpu().numpy(),
        process=False,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp.ply")
    if tmp.exists():
        tmp.unlink()
    new_mesh.export(str(tmp))
    tmp.replace(dst)
    return {
        "src": str(src),
        "dst": str(dst),
        "status": "ok",
        "backend": "cumesh",
        "device": device,
        "vertices_before": vertices_before,
        "faces_before": faces_before,
        "vertices_after": int(new_vertices.shape[0]),
        "faces_after": int(new_faces.shape[0]),
        "src_bytes": src.stat().st_size,
        "dst_bytes": dst.stat().st_size,
    }


def decimate_one(
    src_path: str,
    dst_path: str,
    ratio: float,
    overwrite: bool = False,
    backend: str = "open3d",
    device: str = "cuda:0",
) -> dict[str, object]:
    src = Path(src_path)
    dst = Path(dst_path)
    if dst.exists() and dst.stat().st_size > 0 and not overwrite:
        return {
            "src": str(src),
            "dst": str(dst),
            "status": "skipped",
            "backend": backend,
            "dst_bytes": dst.stat().st_size,
        }

    start = time.time()
    if backend == "open3d":
        result = decimate_one_open3d(src, dst, ratio)
    elif backend == "cumesh":
        result = decimate_one_cumesh(src, dst, ratio, device)
    else:
        raise ValueError(f"unsupported backend: {backend}")
    result["seconds"] = round(time.time() - start, 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=["open3d", "cumesh"], default="open3d")
    parser.add_argument("--cuda-devices", default="0", help="Comma-separated CUDA device ids for CuMesh workers")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_root = args.src.resolve()
    dst_root = args.dst.resolve()
    if not src_root.exists():
        raise FileNotFoundError(src_root)
    if not (0.0 < args.ratio <= 1.0):
        raise ValueError("--ratio must be in (0, 1]")

    dst_root.mkdir(parents=True, exist_ok=True)
    jobs = mirror_non_ply_files(src_root, dst_root)
    cuda_devices = [item.strip() for item in args.cuda_devices.split(",") if item.strip()]
    if not cuda_devices:
        cuda_devices = ["0"]
    summary_path = dst_root / "decimate_summary.jsonl"
    error_path = dst_root / "decimate_errors.jsonl"
    completed = 0
    failed = 0

    with summary_path.open("a", encoding="utf-8") as summary_file, error_path.open("a", encoding="utf-8") as error_file:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = []
            for index, (src, dst) in enumerate(jobs):
                if args.backend == "open3d":
                    futures.append(executor.submit(decimate_one, src, dst, args.ratio, args.overwrite))
                else:
                    device = f"cuda:{cuda_devices[index % len(cuda_devices)]}"
                    futures.append(executor.submit(decimate_one, src, dst, args.ratio, args.overwrite, args.backend, device))
            for future in as_completed(futures):
                try:
                    result = future.result()
                    completed += 1
                    summary_file.write(json.dumps(result, sort_keys=True) + "\n")
                    summary_file.flush()
                    print(
                        f"[{completed}/{len(jobs)}] {result['status']} "
                        f"{Path(str(result['dst'])).relative_to(dst_root)}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    payload = {"status": "error", "error": repr(exc)}
                    error_file.write(json.dumps(payload, sort_keys=True) + "\n")
                    error_file.flush()
                    print(f"[error] {exc!r}", flush=True)

    manifest = {
        "src": str(src_root),
        "dst": str(dst_root),
        "ratio": args.ratio,
        "backend": args.backend,
        "cuda_devices": cuda_devices if args.backend == "cumesh" else [],
        "num_mesh_jobs": len(jobs),
        "completed": completed,
        "failed": failed,
    }
    (dst_root / "decimate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit(failed)
    (dst_root / "DONE_DECIMATE25").write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    main()