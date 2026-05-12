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

# --- 8^3 domain hole-filling (neighbor voting) ----------------------------------
# pm_bin: [T, R^3] float {0,1} -> same shape, with 0-voxels flipped to 1 if neighbor-1 count >= thr.
def _count_n6(vol):
    # vol: [T, 1, R, R, R] float
    k = torch.zeros(1, 1, 3, 3, 3)
    k[0, 0, 0, 1, 1] = 1; k[0, 0, 2, 1, 1] = 1
    k[0, 0, 1, 0, 1] = 1; k[0, 0, 1, 2, 1] = 1
    k[0, 0, 1, 1, 0] = 1; k[0, 0, 1, 1, 2] = 1
    return F.conv3d(vol, k, padding=1)

def _count_n26(vol):
    k = torch.ones(1, 1, 3, 3, 3)
    k[0, 0, 1, 1, 1] = 0
    return F.conv3d(vol, k, padding=1)

def fill_n6(pm_bin, thr):
    T = pm_bin.shape[0]
    v = pm_bin.reshape(T, 1, R, R, R).float()
    c = _count_n6(v)
    filled = ((v > 0.5) | (c >= thr)).float()
    return filled.reshape(T, -1)

def fill_n26(pm_bin, thr):
    T = pm_bin.shape[0]
    v = pm_bin.reshape(T, 1, R, R, R).float()
    c = _count_n26(v)
    filled = ((v > 0.5) | (c >= thr)).float()
    return filled.reshape(T, -1)

def fill_iter(pm_bin, thr, iters, use26=False):
    v = pm_bin
    for _ in range(iters):
        v = fill_n26(v, thr) if use26 else fill_n6(v, thr)
    return v

# closing in 8^3: dilate 1 (cross) then erode 1 (cross)
def closing_8(pm_bin):
    T = pm_bin.shape[0]
    v = pm_bin.reshape(T, 1, R, R, R).float()
    k = torch.zeros(1, 1, 3, 3, 3)
    k[0, 0, 1, 1, 1] = 1
    k[0, 0, 0, 1, 1] = 1; k[0, 0, 2, 1, 1] = 1
    k[0, 0, 1, 0, 1] = 1; k[0, 0, 1, 2, 1] = 1
    k[0, 0, 1, 1, 0] = 1; k[0, 0, 1, 1, 2] = 1
    dil = (F.conv3d(v, k, padding=1) > 0).float()
    # erosion: a voxel stays 1 only if all 7 (center+6) are 1  ->  conv count == 7
    ero = (F.conv3d(dil, k, padding=1) >= 7).float()
    return ero.reshape(T, -1)

raw_bin = lambda pm: (pm > 0.5).float()
strategies = {
    'dil0 (raw)':           lambda pm: upsample_bin(raw_bin(pm)) > 0.5,
    'cross x1 (baseline)':  lambda pm: dilate_cross(upsample_bin(raw_bin(pm)), 1),
    'cross x2':             lambda pm: dilate_cross(upsample_bin(raw_bin(pm)), 2),
    'cube-k3 x1':           lambda pm: dilate_cube(upsample_bin(raw_bin(pm)), 1, 3),
    # N6 hole-fill @ 8^3 then cross x1
    'fill6>=5 + cross1':    lambda pm: dilate_cross(upsample_bin(fill_n6(raw_bin(pm), 5)), 1),
    'fill6>=4 + cross1':    lambda pm: dilate_cross(upsample_bin(fill_n6(raw_bin(pm), 4)), 1),
    'fill6>=3 + cross1':    lambda pm: dilate_cross(upsample_bin(fill_n6(raw_bin(pm), 3)), 1),
    'fill6>=4 x2 + cross1': lambda pm: dilate_cross(upsample_bin(fill_iter(raw_bin(pm), 4, 2)), 1),
    # N26 hole-fill
    'fill26>=17 + cross1':  lambda pm: dilate_cross(upsample_bin(fill_n26(raw_bin(pm), 17)), 1),
    'fill26>=14 + cross1':  lambda pm: dilate_cross(upsample_bin(fill_n26(raw_bin(pm), 14)), 1),
    # closing @ 8^3
    'close8 + cross1':      lambda pm: dilate_cross(upsample_bin(closing_8(raw_bin(pm))), 1),
    # dil0 variants (no 16^3 dilate): 检查 8^3 域能否单凭补洞达到 cross x1 级别
    'fill6>=4 + dil0':      lambda pm: upsample_bin(fill_n6(raw_bin(pm), 4)) > 0.5,
    'fill6>=3 + dil0':      lambda pm: upsample_bin(fill_n6(raw_bin(pm), 3)) > 0.5,
}

recall = {k: [] for k in strategies}
mr     = {k: [] for k in strategies}
fn     = {k: 0  for k in strategies}
gt_num = 0; used = 0

# 8^3 FN/TN neighbor-count diagnostic: pred_mask(8^3) vs submask(8^3)
# For every pred=0 voxel, bucket by #neighbors=1 (0..6) and by category FN / TN.
nb_hist_fn  = torch.zeros(7, dtype=torch.long)   # pred=0 & gt8=1
nb_hist_tn  = torch.zeros(7, dtype=torch.long)   # pred=0 & gt8=0
nb_hist_fn26 = torch.zeros(27, dtype=torch.long)
nb_hist_tn26 = torch.zeros(27, dtype=torch.long)

for idx, p in enumerate(paths):
    print(f'[{time.time()-t0:6.1f}s] {idx+1}/{len(paths)} {p.split("/")[-1]}', flush=True)
    try:
        with np.load(p) as d:
            if 'pred_mask' not in d.files:
                continue
            pm = torch.from_numpy(d['pred_mask'].astype(np.float32))
            ff = torch.from_numpy(d['fine_feats'].astype(np.float32))
            sm = torch.from_numpy(d['submask'].astype(np.float32)) if 'submask' in d.files else None
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

    # --- neighbor diagnostic @ 8^3 ---
    if sm is not None:
        T = pm.shape[0]
        pm8 = (pm > 0.5).float().reshape(T, 1, R, R, R)
        gt8 = (sm > 0.5).reshape(T, 1, R, R, R)
        c6  = _count_n6(pm8).reshape(T, 1, R, R, R).long()
        c26 = _count_n26(pm8).reshape(T, 1, R, R, R).long()
        pred0 = (pm8 < 0.5)
        fn_mask = pred0 & gt8
        tn_mask = pred0 & ~gt8
        for k in range(7):
            nb_hist_fn[k] += ((c6 == k) & fn_mask).sum()
            nb_hist_tn[k] += ((c6 == k) & tn_mask).sum()
        for k in range(27):
            nb_hist_fn26[k] += ((c26 == k) & fn_mask).sum()
            nb_hist_tn26[k] += ((c26 == k) & tn_mask).sum()

print(f'samples={used}, gt_voxels={gt_num:,}\n')
print(f'{"strategy":24} {"recall":>8} {"mask_ratio":>11} {"fn_ratio":>10}  eff(r/mr)')
for name in strategies:
    r, m = float(np.mean(recall[name])), float(np.mean(mr[name]))
    eff = r / m if m > 0 else 0
    print(f'{name:24} {r:>8.4f} {m:>11.4f} {fn[name] / gt_num:>10.4f}  {eff:>8.3f}')

# --- neighbor diagnostic ---
fn_tot = nb_hist_fn.sum().item(); tn_tot = nb_hist_tn.sum().item()
if fn_tot > 0 and tn_tot > 0:
    print(f'\n[N6 neighbor hist] over pred=0 voxels (8^3)  FN_total={fn_tot:,}  TN_total={tn_tot:,}')
    print(f'{"n6":>3} {"P(n6|FN)":>10} {"P(n6|TN)":>10} {"LR=FN/TN":>10} {"precision@flip":>14}')
    for k in range(7):
        pfn = nb_hist_fn[k].item() / fn_tot
        ptn = nb_hist_tn[k].item() / tn_tot
        lr  = (pfn / ptn) if ptn > 0 else float('inf')
        flipped = nb_hist_fn[k].item() + nb_hist_tn[k].item()
        prec = nb_hist_fn[k].item() / flipped if flipped > 0 else 0
        print(f'{k:>3} {pfn:>10.4f} {ptn:>10.4f} {lr:>10.2f} {prec:>14.4f}')
    # cumulative: if we flip all pred=0 with n6 >= thr
    print(f'\n[N6 flip-threshold cumulative]')
    print(f'{"thr":>4} {"recover_fn":>11} {"false_flip":>11} {"prec":>8} {"rec_gain":>10}')
    for thr in range(1, 7):
        rec = nb_hist_fn[thr:].sum().item()
        fls = nb_hist_tn[thr:].sum().item()
        tot = rec + fls
        print(f'{thr:>4} {rec:>11,} {fls:>11,} {(rec/tot if tot>0 else 0):>8.4f} {rec/fn_tot:>10.4f}')

    print(f'\n[N26 flip-threshold cumulative]')
    print(f'{"thr":>4} {"recover_fn":>11} {"false_flip":>11} {"prec":>8} {"rec_gain":>10}')
    for thr in [8, 10, 13, 14, 17, 20]:
        rec = nb_hist_fn26[thr:].sum().item()
        fls = nb_hist_tn26[thr:].sum().item()
        tot = rec + fls
        print(f'{thr:>4} {rec:>11,} {fls:>11,} {(rec/tot if tot>0 else 0):>8.4f} {rec/fn_tot:>10.4f}')
