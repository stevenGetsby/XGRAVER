"""
Encode meshes to block representation (coords + UDF + submask).

Usage:
    python encode_block.py --root /path/to/dataset --device cuda
    python encode_block.py --root /path/to/dataset --device cuda --max_samples 1000
    python encode_block.py --root /path/to/dataset --device cuda --rank 0 --world_size 4
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from mesh2block import generate_adaptive_udf, COL_PREFIX, BLOCK_FOLDER

worker_device = None


def init_worker_cpu():
    global worker_device
    worker_device = "cpu"


def init_worker_gpu(gpu_queue):
    global worker_device
    gpu_id = gpu_queue.get(timeout=5)
    torch.cuda.set_device(gpu_id)
    worker_device = f"cuda:{gpu_id}"


def process_one(sha256: str, root: str) -> dict:
    """Process one mesh -> block npz. Returns metadata dict."""
    input_path = os.path.join(root, 'renders_cond', sha256, 'mesh.ply')
    output_path = os.path.join(root, BLOCK_FOLDER, f'{sha256}.npz')

    result = {
        'sha256': sha256,
        f'{COL_PREFIX}_num_blocks': 0,
        f'{COL_PREFIX}_block_status': 'failed',
    }

    if not os.path.exists(input_path):
        return result

    try:
        generate_adaptive_udf(input_path, output_path,
                              verbose=False, device=worker_device)
        with np.load(output_path) as data:
            result[f'{COL_PREFIX}_num_blocks'] = len(data['coords'])
            result[f'{COL_PREFIX}_block_status'] = 'success'
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        result[f'{COL_PREFIX}_block_status'] = f'failed: {str(e)[:50]}'

    return result


def is_complete(sha256: str, root: str) -> bool:
    """Check if npz already exists with all required keys."""
    path = os.path.join(root, BLOCK_FOLDER, f'{sha256}.npz')
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as data:
            return {'coords', 'fine_feats', 'submask'}.issubset(data.files)
    except Exception:
        return False


def update_metadata(meta_path: str, records: list):
    """Merge processing results into metadata.csv."""
    if not records:
        return
    cols = ['sha256', f'{COL_PREFIX}_num_blocks', f'{COL_PREFIX}_block_status']
    update = pd.DataFrame.from_records(records)[cols].set_index('sha256')

    orig = pd.read_csv(meta_path)
    for c in cols:
        if c != 'sha256' and c not in orig.columns:
            orig[c] = None
    orig.set_index('sha256', inplace=True)
    orig.update(update)

    new_shas = update.index.difference(orig.index)
    if len(new_shas):
        orig = pd.concat([orig, update.loc[new_shas]])

    orig.reset_index().to_csv(meta_path, index=False)
    print(f"metadata.csv updated ({len(orig)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Encode meshes to block NPZ")
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_samples', type=int, default=0,
                        help="Max meshes to process (0 = all)")
    parser.add_argument('--max_workers', type=int, default=64)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None)
    parser.add_argument('--instances', type=str, default=None)
    parser.add_argument('--force', action='store_true',
                        help="Re-encode even if npz exists (for data format changes)")
    opt = parser.parse_args()

    meta_path = os.path.join(opt.root, 'metadata.csv')
    metadata = pd.read_csv(meta_path)

    # Filter
    if opt.instances:
        ids = (open(opt.instances).read().splitlines()
               if os.path.exists(opt.instances)
               else opt.instances.split(','))
        metadata = metadata[metadata['sha256'].isin(ids)]
    elif opt.filter_low_aesthetic_score:
        metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]

    # Shard
    total = len(metadata)
    metadata = metadata.iloc[total * opt.rank // opt.world_size:
                             total * (opt.rank + 1) // opt.world_size]

    # Limit
    if opt.max_samples > 0:
        metadata = metadata.iloc[:opt.max_samples]

    os.makedirs(os.path.join(opt.root, BLOCK_FOLDER), exist_ok=True)

    # Split done / todo
    records, tasks = [], []
    for _, row in metadata.iterrows():
        sha = row['sha256']
        if not opt.force and is_complete(sha, opt.root):
            records.append({
                'sha256': sha,
                f'{COL_PREFIX}_num_blocks': 0,
                f'{COL_PREFIX}_block_status': 'success',
            })
            continue
        tasks.append((sha, opt.root))

    print(f"[encode_block] {len(records)} done, {len(tasks)} to process "
          f"(rank {opt.rank}/{opt.world_size}, device={opt.device})")

    if not tasks:
        update_metadata(meta_path, records)
        return

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    use_cuda = opt.device.startswith('cuda')
    if use_cuda:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("WARNING: no GPU found, falling back to CPU")
            use_cuda = False

    if use_cuda:
        mgr = multiprocessing.Manager()
        gpu_q = mgr.Queue()
        for i in range(num_gpus):
            gpu_q.put(i)
        pool_kw = dict(max_workers=num_gpus, initializer=init_worker_gpu, initargs=(gpu_q,))
    else:
        pool_kw = dict(max_workers=opt.max_workers, initializer=init_worker_cpu)

    with ProcessPoolExecutor(**pool_kw) as pool:
        futs = [pool.submit(process_one, s, r) for s, r in tasks]
        for f in tqdm(futs, desc=f"Encoding ({'GPU' if use_cuda else 'CPU'})"):
            try:
                records.append(f.result())
            except Exception as e:
                print(f"Error: {e}")

    # Summary
    df = pd.DataFrame.from_records(records)
    ok = df[f'{COL_PREFIX}_block_status'] == 'success'
    print(f"\nSuccess: {ok.sum()}/{len(df)}, "
          f"total blocks: {df.loc[ok, f'{COL_PREFIX}_num_blocks'].sum():,}, "
          f"avg: {df.loc[ok, f'{COL_PREFIX}_num_blocks'].mean():.1f}")

    update_metadata(meta_path, records)


if __name__ == '__main__':
    main()
