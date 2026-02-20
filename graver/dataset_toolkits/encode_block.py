import os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from mesh2block import generate_adaptive_udf, BLOCK_GRID, BLOCK_INNER

COL_PREFIX = f'{BLOCK_GRID}_{BLOCK_INNER}'

worker_device = None


def init_worker_cpu():
    """CPU worker 初始化"""
    global worker_device
    worker_device = "cpu"


def init_worker_gpu(gpu_id):
    """GPU worker 初始化: 绑定到指定 GPU"""
    global worker_device
    torch.cuda.set_device(gpu_id)
    worker_device = f"cuda:{gpu_id}"


def _init_worker_gpu_from_queue(gpu_queue):
    """从队列中取 GPU ID 并初始化"""
    try:
        gpu_id = gpu_queue.get(timeout=5)
    except Exception:
        gpu_id = 0
    init_worker_gpu(gpu_id)


def process_one(sha256: str, root: str) -> dict:
    """处理单个 mesh，生成 block 数据，同时累积 UDF 归一化统计量."""
    
    input_path = os.path.join(root, 'renders_cond', sha256, 'mesh.ply')
    output_npz_path = os.path.join(root, f'blocks_{COL_PREFIX}', f'{sha256}.npz')
    
    result = {
        'sha256': sha256,
        f'{COL_PREFIX}_num_blocks': 0,
        f'{COL_PREFIX}_block_status': 'failed',
    }
    
    # 检查是否已存在且完整
    if os.path.exists(output_npz_path):
        try:
            with np.load(output_npz_path) as data:
                # 验证必要的 key 存在
                required_keys = {'coords', 'fine_feats'}
                if required_keys.issubset(data.files):
                    num_blocks = len(data['coords'])
                    result[f'{COL_PREFIX}_num_blocks'] = num_blocks
                    result[f'{COL_PREFIX}_block_status'] = 'success'
                    return result
        except Exception:
            # 文件损坏，重新生成
            pass
    
    # 输入文件不存在
    if not os.path.exists(input_path):
        return result
    
    try:
        # 调用 mesh2block 生成数据 (使用 worker 分配的 device)
        generate_adaptive_udf(input_path, output_npz_path, verbose=False,
                              device=worker_device)
        
        # 验证生成的文件 + 累积统计量
        with np.load(output_npz_path) as data:
            num_blocks = len(data['coords'])
            result[f'{COL_PREFIX}_num_blocks'] = num_blocks
            result[f'{COL_PREFIX}_block_status'] = 'success'
            
    except Exception as e:
        # 生成失败，删除可能的损坏文件
        if os.path.exists(output_npz_path):
            os.remove(output_npz_path)
        result[f'{COL_PREFIX}_block_status'] = f'failed: {str(e)[:50]}'
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encode meshes to block representation")
    parser.add_argument('--root', type=str, required=True, help="Dataset root directory")
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None)
    parser.add_argument('--instances', type=str, default=None, help="Specific instances to process")
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=64)
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device: 'cpu' or 'cuda'. When 'cuda', workers are distributed across all GPUs.")
    opt = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.join(opt.root, f'blocks_{COL_PREFIX}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 读取 Metadata
    meta_path = os.path.join(opt.root, 'metadata.csv')
    if not os.path.exists(meta_path):
        raise ValueError(f'metadata.csv not found at {meta_path}')
    metadata = pd.read_csv(meta_path)
    
    # 2. 过滤
    if opt.instances:
        if os.path.exists(opt.instances):
            instances = open(opt.instances).read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]
    elif opt.filter_low_aesthetic_score:
        metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
    
    # 3. 分片 (Sharding)
    total_len = len(metadata)
    start = total_len * opt.rank // opt.world_size
    end = total_len * (opt.rank + 1) // opt.world_size
    metadata = metadata.iloc[start:end]
    
    print(f'Processing {len(metadata)} objects (Rank {opt.rank}/{opt.world_size}) (device={opt.device})...')
    print(f'Output: {output_dir}')
    
    # 4. 并行处理
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    use_cuda = opt.device.startswith('cuda')
    records = []
    tasks = [(row['sha256'], opt.root) for _, row in metadata.iterrows()]
    
    if use_cuda:
        # ============================================================
        # GPU 模式: 每张卡一个进程，CUDA_VISIBLE_DEVICES 隔离
        # ============================================================
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("WARNING: --device=cuda but no GPU detected. Falling back to CPU.")
            use_cuda = False
        else:
            effective_workers = min(opt.max_workers, num_gpus)
            print(f"Detected {num_gpus} GPUs. Launching {effective_workers} workers (1 per GPU).")
            
            # 为每个 worker 分配 GPU ID
            manager = multiprocessing.Manager()
            gpu_queue = manager.Queue()
            for i in range(effective_workers):
                gpu_queue.put(i % num_gpus)
            
            with ProcessPoolExecutor(
                max_workers=effective_workers,
                initializer=_init_worker_gpu_from_queue,
                initargs=(gpu_queue,)
            ) as executor:
                futures = [executor.submit(process_one, sha256, root)
                           for sha256, root in tasks]
                
                for future in tqdm(futures, desc="Encoding Blocks (GPU)"):
                    try:
                        res = future.result()
                        records.append(res)
                    except Exception as e:
                        print(f"Error: {e}")
    
    if not use_cuda:
        # ============================================================
        # CPU 模式: 多进程并行
        # ============================================================
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass
        
        print(f"Using CPU with {opt.max_workers} workers.")
        
        with ProcessPoolExecutor(
            max_workers=opt.max_workers,
            initializer=init_worker_cpu,
        ) as executor:
            futures = [executor.submit(process_one, sha256, root)
                       for sha256, root in tasks]
            
            for future in tqdm(futures, desc="Encoding Blocks (CPU)"):
                try:
                    res = future.result()
                    records.append(res)
                except Exception as e:
                    print(f"Error: {e}")
    
    # 5. 更新 metadata.csv + 保存归一化常数
    if records:
        new_df = pd.DataFrame.from_records(records)
        
        # 统计
        success = new_df[f'{COL_PREFIX}_block_status'] == 'success'
        print(f"\n{'='*50}")
        print(f"Success: {success.sum()} / {len(new_df)}")
        print(f"Total blocks: {new_df.loc[success, f'{COL_PREFIX}_num_blocks'].sum():,}")
        print(f"Avg blocks per mesh: {new_df.loc[success, f'{COL_PREFIX}_num_blocks'].mean():.1f}")
        
        # ── 自动 merge 到 metadata.csv ──
        block_cols = ['sha256', f'{COL_PREFIX}_num_blocks', f'{COL_PREFIX}_block_status']
        update_df = new_df[block_cols].copy()
        
        orig_meta = pd.read_csv(meta_path)
        for col in block_cols:
            if col != 'sha256' and col not in orig_meta.columns:
                orig_meta[col] = None
        
        orig_meta.set_index('sha256', inplace=True)
        update_df.set_index('sha256', inplace=True)
        orig_meta.update(update_df)
        
        # 确保新增实例也写入 (不在原 metadata 中的)
        new_shas = update_df.index.difference(orig_meta.index)
        if len(new_shas) > 0:
            orig_meta = pd.concat([orig_meta, update_df.loc[new_shas]])
        
        orig_meta.reset_index(inplace=True)
        orig_meta.to_csv(meta_path, index=False)
        print(f"✅ metadata.csv updated ({len(orig_meta)} rows)")
    else:
        print("No records processed.")