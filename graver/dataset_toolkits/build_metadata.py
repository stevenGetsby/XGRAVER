import os
os.umask(0)  # 确保所有新建文件/目录权限为 777，解决多用户 CFS 权限问题
import hashlib
import csv
import argparse
from pathlib import Path
from typing import Iterable, List, Dict
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

DEFAULT_FIELDS = ["sha256", "local_path", "rendered"]

# 支持的 3D 文件格式（与 render.py IMPORT_FUNCTIONS 对齐）
SUPPORTED_EXTENSIONS = {".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply", ".usd", ".usda", ".dae", ".abc"}


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA256 hash of a file (read in 1MB chunks for speed)."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(1 << 20), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _process_one(args_tuple):
    """Worker: (file_path_str, root_path_str) -> dict or None"""
    file_path_str, root_path_str = args_tuple
    file_path = Path(file_path_str)
    root_path = Path(root_path_str)
    try:
        sha256 = calculate_sha256(file_path)
        local_path = str(file_path.relative_to(root_path))
        return {"sha256": sha256, "local_path": local_path}
    except Exception as exc:
        print(f"Error processing {file_path}: {exc}")
        return None


def has_render_outputs(root_path: Path, sha256: str) -> bool:
    if not sha256:
        return False
    return (root_path / "render" / sha256).is_dir()


def ensure_field_order(existing: Iterable[str] | None) -> List[str]:
    ordered = list(existing or [])
    for field in DEFAULT_FIELDS:
        if field not in ordered:
            ordered.append(field)
    return ordered


def normalize_row(row: Dict[str, str], fieldnames: Iterable[str]) -> Dict[str, str]:
    for field in fieldnames:
        row.setdefault(field, "")
    return row


def merge_fragments(root_path: Path):
    """Merge other CSV files into metadata.csv and delete them."""
    output_path = root_path / "metadata.csv"
    all_csvs = list(root_path.glob("*.csv"))
    fragment_files = [f for f in all_csvs if f.name != "metadata.csv"]
    
    if not fragment_files:
        return

    print(f"Found {len(fragment_files)} fragment CSV files. Merging...")
    
    dfs = []
    if output_path.exists():
        try:
            df = pd.read_csv(output_path)
            if 'sha256' in df.columns:
                df['sha256'] = df['sha256'].astype(str)
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read existing metadata.csv: {e}")

    for f in fragment_files:
        try:
            df = pd.read_csv(f)
            if 'sha256' in df.columns:
                df['sha256'] = df['sha256'].astype(str)
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if not dfs:
        return

    try:
        full_df = pd.concat(dfs, ignore_index=True)
        # Group by sha256 and take the last non-null value for each column
        merged_df = full_df.groupby('sha256', as_index=False).last()
        
        merged_df.to_csv(output_path, index=False)
        print(f"Merged metadata saved to {output_path}.")
        
        for f in fragment_files:
            try:
                f.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {f}: {e}")
        print("Deleted fragment files.")
    except Exception as e:
        print(f"Error merging fragments: {e}")


def build_metadata(root_dir, raw_dir_name="raw", output_filename="metadata.csv", overwrite=False):
    root_path = Path(root_dir)
    
    if not overwrite:
        merge_fragments(root_path)

    raw_path = root_path / raw_dir_name
    output_path = root_path / output_filename

    if not raw_path.exists():
        print(f"Error: Directory {raw_path} does not exist.")
        return

    existing_rows: List[Dict[str, str]] = []
    existing_files = set()
    fieldnames: List[str] | None = None

    if output_path.exists() and not overwrite:
        print(f"Loading existing metadata from {output_path}...")
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames[:] if reader.fieldnames else []
                for row in reader:
                    row = dict(row)
                    local_path = row.get("local_path")
                    if local_path:
                        existing_files.add(local_path)
                    existing_rows.append(row)
            print(f"Loaded {len(existing_rows)} existing entries.")
        except Exception as exc:
            print(f"Error reading existing metadata: {exc}. Will rebuild from scratch.")
            existing_rows = []
            existing_files.clear()
            fieldnames = None
    elif overwrite and output_path.exists():
        print("Overwrite enabled, rebuilding metadata from scratch.")

    print(f"Scanning for 3D files in {raw_path}...")
    all_files = [f for f in raw_path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    # Filter out already-processed files
    to_process = []
    for file_path in all_files:
        local_path = str(file_path.relative_to(root_path))
        if local_path not in existing_files:
            to_process.append((str(file_path), str(root_path)))

    print(f"Found {len(all_files)} 3D files, {len(to_process)} new to process.")

    new_rows: List[Dict[str, str]] = []
    if to_process:
        num_workers = min(os.cpu_count() or 4, 32, len(to_process))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_one, t): t for t in to_process}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Hashing"):
                result = future.result()
                if result is not None:
                    new_rows.append(result)

    total_new = len(new_rows)
    total_existing = len(existing_rows)

    if fieldnames is None:
        fieldnames = []
    fieldnames = ensure_field_order(fieldnames)

    for row in existing_rows:
        normalize_row(row, fieldnames)
    for row in new_rows:
        normalize_row(row, fieldnames)

    all_rows = existing_rows + new_rows
    if not all_rows:
        print("No entries found. Nothing to write.")
        return

    # Batch check render directories (set lookup instead of per-row stat)
    render_dir = root_path / "render"
    if render_dir.exists():
        rendered_set = set(os.listdir(render_dir))
    else:
        rendered_set = set()
    for row in all_rows:
        sha256 = row.get("sha256", "")
        row["rendered"] = sha256 in rendered_set

    print(f"Writing {len(all_rows)} entries to {output_path} (new: {total_new}, existing: {total_existing}).")
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build metadata.csv for 3D files (glb/stl/ply/obj/fbx/usd/dae/abc).")
    parser.add_argument("--root", type=str, default="/mnt/data/yizhao/JiTData", help="Root directory")
    parser.add_argument("--overwrite", action="store_true", help="Force rebuild metadata from scratch")
    args = parser.parse_args()
    
    build_metadata(args.root, overwrite=args.overwrite)
