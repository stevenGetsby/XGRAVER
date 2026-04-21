import glob, numpy as np, torch, torch.nn.functional as F, random, sys, time
torch.set_num_threads(8)

D, R = 16, 8
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
t0 = time.time()
print(f'[{time.time()-t0:6.1f}s] scanning...', flush=True)
paths = sorted(glob.glob('/cfs/yizhao/TRAIN/blocks_64_15_occ8/*.npz'))
paths = [p for p in paths if not p.endswith('.npz.npz')]
print(f'[{time.time()-t0:6.1f}s] found {len(paths)} files, using {N}', flush=True)
random.seed(0); random.shuffle(paths); paths = paths[:N]

def upsample_bin(sub_bin):
    T = sub_bin.shape[0]
    return F.interpolate(sub_bin.reshape(T, 1, R, R, R).float(),
                         scale_factor=D // R, mode='nearest').reshape(T, -1)

def cross_kernel():
    k = torch.zeros(1, 1, 3, 3, 3)
    k[0, 0, 1, 1, 1] = 1
    k[0, 0, 0, 1, 1] = 1; k[0, 0, 2, 1, 1] = 1
    k[0, 0, 1, 0, 1] = 1; k[0, 0, 1, 2, 1] = 1
    k[0, 0, 1, 1, 0] = 1; k[0, 0, 1, 1, 2] = 1
    return k

def dilate_cross(vm, iters):
    T = vm.shape[0]
    v = vm.reshape(T, 1, D, D, D).float()
    k = cross_kernel()
    for _ in range(iters):
        v = (F.conv3d(v, k, padding=1) > 0).float()
    return v.reshape(T, -1) > 0.5

def dilate_cube(vm, iters, k=3):
    T = vm.shape[0]
    v = vm.reshape(T, 1, D, D, D).float()
    for _ in range(iters):
        v = F.max_pool3d(v, k, 1, k // 2)
    return v.reshape(T, -1) > 0.5

def dilate_axis(vm, axis, iters=1):
    T = vm.shape[0]
    v = vm.reshape(T, 1, D, D, D).float()
    kshape = [1, 1, 1, 1, 1]
    kshape[2 + axis] = 3
    k = torch.ones(*kshape)
    p = [0, 0, 0]; p[axis] = 1
    for _ in range(iters):
        v = (F.conv3d(v, k, padding=tuple(p)) > 0).float()
    return v.reshape(T, -1) > 0.5

raw_bin = lambda pm: (pm > 0.5).float()
strategies = {
    'dil0 (raw)':      lambda pm: upsample_bin(raw_bin(pm)) > 0.5,
    'cube-k3 x1':      lambda pm: dilate_cube(upsample_bin(raw_bin(pm)), 1, 3),
    'cube-k3 x2':      lambda pm: dilate_cube(upsample_bin(raw_bin(pm)), 2, 3),
    'cross x1 (6nb)':  lambda pm: dilate_cross(upsample_bin(raw_bin(pm)), 1),
    'cross x2 (6nb)':  lambda pm: dilate_cross(upsample_bin(raw_bin(pm)), 2),
    'cross x3 (6nb)':  lambda pm: dilate_cross(upsample_bin(raw_bin(pm)), 3),
    'axis-z x1':       lambda pm: dilate_axis(upsample_bin(raw_bin(pm)), 0, 1),
    'axis-xy x1':      lambda pm: dilate_axis(dilate_axis(upsample_bin(raw_bin(pm)), 1, 1), 2, 1),
    'cross1+cube1':    lambda pm: dilate_cube(dilate_cross(upsample_bin(raw_bin(pm)), 1).float().reshape(-1, D ** 3), 1, 3),
}

recall = {k: [] for k in strategies}
mr     = {k: [] for k in strategies}
fn     = {k: 0  for k in strategies}
gt_num = 0; used = 0

for idx, p in enumerate(paths):
    print(f'[{time.time()-t0:6.1f}s] {idx+1}/{len(paths)} {p.split("/")[-1]}', flush=True)
    try:
        with np.load(p) as d:
            if 'pred_mask' not in d.files:
                continue
            pm = torch.from_numpy(d['pred_mask'].astype(np.float32))
            ff = torch.from_numpy(d['fine_feats'].astype(np.float32))
    except Exception:
        continue
    gt_b = (ff < 0.4)
    g = gt_b.sum().item()
    if g == 0:
        continue
    used += 1; gt_num += g
    for name, fn_ in strategies.items():
        vm_b = fn_(pm)
        recall[name].append((gt_b & vm_b).sum().item() / g)
        mr[name].append(vm_b.float().mean().item())
        fn[name] += (gt_b & ~vm_b).sum().item()

print(f'samples={used}, gt_voxels={gt_num:,}\n')
print(f'{"strategy":22} {"recall":>8} {"mask_ratio":>11} {"fn_ratio":>10}  eff(r/mr)')
for name in strategies:
    r, m = float(np.mean(recall[name])), float(np.mean(mr[name]))
    eff = r / m if m > 0 else 0
    print(f'{name:22} {r:>8.4f} {m:>11.4f} {fn[name] / gt_num:>10.4f}  {eff:>8.3f}')
