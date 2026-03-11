# GRAVER 三阶段 Pipeline 设计文档

> 维护日期: 2026-03-06  
> 用途: 记录数据构造、模型设计、训练策略的演进与优化决策

---

## 全局数据常量

源文件: `graver/dataset_toolkits/mesh2block.py`

| 常量 | 值 | 说明 |
|---|---|---|
| BLOCK_GRID | 64 | 占位网格每轴 block 数 (64³) |
| BLOCK_INNER | 15 | 每 block 核心采样间隔 |
| BLOCK_DIM | 16 | 每 block UDF 采样顶点数 (BLOCK_INNER + 1) |
| SAMPLE_RES | 960 | 全局采样分辨率 (64 × 15) |
| VOXEL_SIZE | 1/960 | 体素物理尺寸 |
| TRUNCATION | 5/960 | UDF 截断距离 (5 voxels) |
| MC_THRESHOLD | 0.2 | Marching Cubes 阈值 (归一化 UDF 空间) |
| SURFACE_THRESHOLD | 0.4 | 表面判定阈值 (2 × MC, 精确覆盖 MC 插值带) |
| SUBMASK_RES | 4 | sub-mask 每轴分辨率 |
| SUBMASK_STRIDE | 4 | 每个 sub-cell 覆盖的体素数 (16/4) |
| SUBMASK_DIM | 64 | sub-mask 展平维度 (4³) |

---

## 数据预处理

### 入口
```
encode_block.py --root /data --device cuda [--max_samples N] [--force]
```

### 流程 (`generate_adaptive_udf`)
```
mesh (ply) 
  → 归一化到 [-0.5, 0.5]³ (面数 > 100W 时 GPU 简化)
  → cubvh BVH
  → 64³ 中心点距离查询 → active blocks
  → N × 16³ 精细 UDF 查询 → 截断归一化 [0, 1]
  → 过滤: min(UDF) < 0.4
  → 4³ min-pooling → binary threshold
  → 保存 NPZ
```

### 输出 NPZ 结构
| 键 | 形状 | 类型 | 说明 |
|---|---|---|---|
| coords | (M, 3) | int32 | block 坐标 |
| fine_feats | (M, 4096) | float16 | 16³ UDF |
| submask | (M, 64) | float32 | 4³ 二值 mask (紧凑, 不膨胀) |

### 当前安全余量设计
`SURFACE_THRESHOLD=0.4` 精确覆盖 MC 插值带。Stage 3 直接使用 occ4 hard mask，不再做运行时软膨胀。

---

## Stage 1: Block Coords — 占位网格生成

### 任务
从图像预测 64³ 二值占位网格: 哪些 block 位置含有表面。

### 数据
| 项目 | 值 |
|---|---|
| 数据集类 | `ImageConditionedBlockCoords` |
| 文件 | `graver/datasets/block_coords.py` |
| 格式 | Dense `[1, 64, 64, 64]`, 值 ∈ {0, 1} |
| collate | 标准 batch stack |

### 模型: DenseFlowModel
| 参数 | 值 |
|---|---|
| 文件 | `graver/models/dense_flow.py` |
| 输入 | `[B, 1, 64, 64, 64]` |
| patch_size | 8 → 8³ = 512 tokens |
| bottleneck | 512 → 128 → 768 |
| Transformer | 12 blocks, 12 heads, full attention |
| PE | absolute position embedding |
| 参数量级 | ~200M |

### 训练
| 参数 | 值 |
|---|---|
| Trainer | `ImageConditionedFlowMatchingCFGTrainer` |
| 文件 | `graver/trainers/flow_matching/flow_matching.py` |
| 方法 | Flow Matching, v-loss, x-prediction |
| t 采样 | logitNormal(mean=-0.8, std=0.8) |
| noise_scale | 1.0 |
| lr | 1e-4, AdamW |
| batch_size | 4 / GPU |
| max_steps | 100K |
| grad_clip | AdaptiveGradClipper(max_norm=1.0, p95) |
| EMA | 0.999 |
| 图像条件 | DINOv2 ViT-L/14-reg → [B, 1369, 1024] |

### 推理
ODE 采样 (FlowSampler, Euler/Heun), CFG strength=3.0

### 配置
`configs/flow_matching/block_coords.json`

---

## Stage 2: Block Mask — 子区域 mask 预测

### 任务
给定 Stage 1 输出的 block coords, 预测每个 block 内 4³ = 64 维二值 sub-mask。

### 数据
| 项目 | 值 |
|---|---|
| 数据集类 | `ImageConditionedBlockMask` |
| 文件 | `graver/datasets/block_mask.py` |
| 格式 | SparseTensor, feats = [T, 64], 值 ∈ {0, 1} |
| collate | 负载均衡分组 → SparseTensor |

### 模型: SparseFlowModel (Stage 2 配置)
| 参数 | 值 |
|---|---|
| 文件 | `graver/models/sparse_flow.py` |
| resolution | 4 (= SUBMASK_RES) |
| token_dim | 64 (4³) |
| bottleneck | 无 (直接 Linear 64→768) |
| submask条件 | 无 (submask_resolution=0) |
| Transformer | 8 blocks, 12 heads, windowed (w=32) |
| PE | RoPE |
| 窗口偏移 | Swin 式交替 (0 / w//2) |

### 训练: Bit Diffusion
| 参数 | 值 |
|---|---|
| Trainer | `ImageConditionedSparseMaskFlowCFGTrainer` |
| 文件 | `graver/trainers/flow_matching/mask_matching.py` |
| 核心策略 | **Bit Diffusion**: {0,1} → {-1,+1}, 在对称连续空间做 Flow Matching |
| Loss | recall-biased v-loss |
| 阈值 | pred > 0.0 → surface |
| 监控 | IoU, precision, recall, F1, train_recall, train_precision |
| noise_scale | 1.0 |
| lr | 2e-4, AdamW |
| batch_size | 4 / GPU |
| max_steps | 100K |
| grad_clip | AdaptiveGradClipper(max_norm=2.0, p99) |

### Bit Diffusion 原理
```
训练:
  x_0 = submask ∈ {0, 1}
  x_0_logits = x_0 * 2 - 1       →  {-1, +1}
  x_t = t * x_0_logits + (1-t) * N(0, σ²)
  
推理:
  ODE 50 steps → pred
  pred_bin = (pred > 0.0)         →  {0, 1}
```

**为何不直接用 MSE 回归 {0,1}**: Flow Matching 的 ODE 以 N(0,1) 为起点, 目标若在 [0,1] 则均值偏离 0, 导致 ODE 轨迹系统性偏移。映射到 {-1,+1} 后分布关于原点完美对称。

### 推理
FlowGuidanceIntervalSampler, 50 steps, cfg=3.0

### 配置
`configs/flow_matching/block_mask.json`

---

## Stage 3: Block Feats — 精细 UDF 生成

### 任务
给定 block coords + sub-mask, 预测每个 block 内 16³ = 4096 维连续 UDF ∈ [0, 1]。

### 数据
| 项目 | 值 |
|---|---|
| 数据集类 | `ImageConditionedBlockFeats` |
| 文件 | `graver/datasets/block_feats.py` |
| 格式 | x_f: SparseTensor [T, 4096], submask: [T, 64] |
| collate | 负载均衡分组, submask 单独传递 |

### 模型: SparseFlowModel (Stage 3 配置)
| 参数 | 值 |
|---|---|
| resolution | 16 (= BLOCK_DIM) |
| token_dim | 4096 (16³) |
| bottleneck | 4096 → 128 → 768 (压缩 32×) |
| submask条件 | `Linear(64→128→768)`, 加到 token embedding |
| Transformer | 12 blocks (前 4 层 Pooled, 后 8 层 windowed) |
| num_heads | 12 |
| PE | RoPE |
| window_size | 32 |
| context_registers | 32 个可学习 register token |
| 粗糙 UDF prior | 从邻域缺失推断, 作为额外条件 |

### 训练: 多权重 Flow Matching
| 参数 | 值 |
|---|---|
| Trainer | `ImageConditionedSparseFlowMultiTokenCFGTrainer` |
| 文件 | `graver/trainers/flow_matching/feats_matching.py` |
| Loss | v-loss + normal_loss |
| noise_scale | 2.0 |
| lr | 1e-4, AdamW |
| batch_size | 2 / GPU |
| max_steps | 400K |
| grad_clip | AdaptiveGradClipper(max_norm=4.0, p95) |

### Loss 设计

```
总 Loss = λ_flow × flow_loss + λ_normal × normal_loss
        = 1.0 × flow_loss + 0.1 × normal_loss
```

**flow_loss (三区域加权 v-loss)**:
| 区域 | 条件 | 权重 | 含义 |
|---|---|---|---|
| MC 区 | UDF < 2v (0.4) | 8.0 | 直接影响 mesh 表面质量 |
| 近场 | 2v < UDF < 3v | 3.0 | 表面邻近, 影响梯度平滑 |
| 远场 | UDF > 3v | 1.0 | 远离表面, 要求低 |

**complexity_w (邻域复杂度加权)**:
- 计算 per-block 曲率方差 → 6-连通邻居间曲率差异 → 复杂区域 (耳朵、手指) 权重 ×(1+2.0)
- 范围: [1.0, 3.0]

**normal_loss (表面法线一致性)**:
- 仅在含表面 block 上计算
- 余弦方向 loss: `1 - cos(∇pred, ∇gt)` 
- Eikonal loss: `|∇UDF| - 1`
- Edge-aware: 高曲率区域加权更高

**Selective Masking (hard mask)**:
- submask 直接 nearest 上采样到 16³
- `masked_val = pred * mask + (1-mask) * 1.0`
  - mask=1 区: 完全模型预测
  - mask=0 区: 纯远场 1.0
- loss 仅在 mask=1 区域计算

### 推理
FlowGuidanceIntervalSampler, 100 steps (Heun), cfg=1.5, interval=(0.1, 1.0)

### Mesh 重建 (Snapshot)
```
pred UDF [T, 4096]
  → 边界角点平均 (scatter_add 去重)
  → UDF Sharpening (power=1.5, 加陡零交叉)
  → Sparse Marching Cubes (cubvh, threshold=0.2)
  → CuMesh: 去重面 + 退化面 + 小连通分量
  → GPU Bilateral Smoothing (3 iter, 保边去噪)
  → 导出 PLY
  → nvdiffrast 渲染 4 视角法线图
```

### 配置
`configs/flow_matching/block_feats.json`

---

## 图像条件编码 (共享)

三个 Stage 共用同一套图像条件:

| 项目 | 值 |
|---|---|
| 模型 | DINOv2 ViT-L/14-reg |
| 输入 | 518 × 518 RGB |
| 输出 | [B, 1369, 1024] patch tokens |
| 冻结 | 是 (不训练) |
| CFG | drop_rate 训练时随机, neg_cond=zeros |

---

## Pipeline 总览

```
Image ──DINOv2──→ cond [B, 1369, 1024]
                      │
    ┌─────────────────┼──────────────────┐
    ▼                 ▼                  ▼
 Stage 1           Stage 2            Stage 3
 Dense Flow        Sparse Bit-Diff    Sparse Flow
 DenseFlowModel    SparseFlowModel    SparseFlowModel
 64³ occupancy     64-d mask          4096-d UDF
 {0,1}             {0,1}→{-1,+1}     [0,1]
 100K steps        100K steps         400K steps
    │                 │                  │
    ▼                 ▼                  ▼
 block coords      per-block 4³       per-block 16³
 (N, 3)            mask  (N, 64)      UDF (N, 4096)
    │                 │                  │
    └────────┬────────┘                  │
             │    (coords + mask)        │
             └───────────────────────────┘
                         │
                         ▼
            Sparse MC → CuMesh → Bilateral → Mesh
```

---

## 优化记录

### 2026-03-05: Stage 2 Bit Diffusion 改造
**问题**: 直接在 {0,1} 上做 Flow Matching, ODE 轨迹偏移导致训练-推理 IoU 严重不一致。  
**方案**: Bit Diffusion — 将 {0,1} 映射为 {-1,+1}, 在对称连续空间训练, 推理时以 0.0 为阈值。  
**状态**: 待验证


### 2026-03-06: submask 支持域改为阈值增厚
**问题**: 把膨胀写进数据或运行时都容易造成定义不一致，Stage 2 学的目标和 Stage 3 使用的目标不完全相同。  
**方案**: ~~提高 `SURFACE_THRESHOLD: 0.4 → 0.6`~~ → 已回退，见下一条。  
**状态**: ~~已集成~~ 已回退

### 2026-03-06: 切回 occ4, 去掉运行时软膨胀
**问题**: occ8 Stage 2 长时间 IoU 偏低, 且预测 mask 偏胖; 继续依赖运行时软膨胀会让 Stage 3 条件更松, 难以建立稳定基线。  
**方案**:
- `SUBMASK_RES` 改为 `4`
- Stage 2 / Stage 3 全面切到 occ4 数据与配置
- Stage 3 改回 hard mask: nearest 上采样后直接门控, 不再加膨胀 fringe
**状态**: 已集成

### 2026-03-06: Stage 2 recall 偏向
**问题**: 三阶段链路中，Stage 2 的 false negative 会永久切掉 Stage 3 的生成区域，代价远高于 false positive。  
**方案**: 在 Stage 2 的 v-loss 中对 GT=1 的位置加权，当前权重为 `surface:empty = 3:1`，并显式监控 `train_recall` / `train_precision`。  
**状态**: 已集成

---

## V2 架构草图（可落地版本）

目标：在不推翻当前三阶段框架的前提下，把 `submask` 从“硬门控真值”升级为“高 recall 的支持域条件”，并让 Stage 3 对 Stage 2 误差更鲁棒。

### V2-1 总体结构

```
Image ──DINOv2──→ cond
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
      Stage 1   Stage 2    Stage 3
      coords    support    udf
      64³       4³ mask    16³ field
```

- **Stage 1**: 不变，仍负责粗定位 `coords`
- **Stage 2**: 输出的不是“精确表面 occupancy”，而是 **support mask**
- **Stage 3**: 以 `support mask` 作为强条件，但逐步减少 hard gate 强度

### V2-2 三阶段职责重定义

#### Stage 1：粗定位
- 输入：图像
- 输出：活跃 `block coords`
- 指标重点：召回率优先，宁可多给 block，不要漏 block

#### Stage 2：支持域预测（Support Region Prediction）
- 输入：图像 + `coords`
- 输出：`4³ = 64` 维 binary support mask
- 目标语义：
  - `1` = “这里允许 Stage 3 生成表面附近的 UDF”
  - `0` = “这里大概率是远场”
- 不再把它理解为精确表面标签

#### Stage 3：连续场细化
- 输入：图像 + `coords` + support mask
- 输出：`16³ = 4096` 维 UDF
- 核心要求：
  - 对 support mask 内区域精细生成
  - 对 support mask 外区域保持远场稳定
  - 对 Stage 2 的局部噪声具备一定容错能力

### V2-3 建议的关键改动顺序

#### 版本 A（最小改动，可直接落地）
1. 保持当前三阶段结构
2. Stage 2 继续使用 Bit Diffusion + recall-biased loss
3. Stage 3 继续使用 selective masking，但加入 **submask dropout / bit flip augmentation**

推荐实现：
- 训练 Stage 3 时，对输入 `submask` 做小概率扰动：
  - `1 → 0` 概率很小（如 1%）
  - `0 → 1` 概率略大（如 2%~5%）
- 目的：让 Stage 3 见到“Stage 2 预测误差”，提升鲁棒性

这是 **V2 的第一优先级**。

#### 版本 B（中等改动，建议后续做）
把 Stage 3 从 **硬门控** 改成 **软门控**：

当前：
```
mask=0 → 直接强制 UDF=1.0
```

建议改为：
```
mask 作为 loss weight / noise weight / attention bias
但不绝对裁掉生成自由度
```

具体可落地为：
- `voxel_mask` 继续控制噪声强弱
- 但 `x_t` 和最终 `pred` 不再强制写死为 1.0
- loss 对 `mask=0` 区域给低权重，而不是 0 权重

这样 Stage 3 在 Stage 2 漏掉一些 support cell 时，仍有机会自行修复。

#### 版本 C（大改动，真正上限方案）
把 Stage 2 从 binary support mask 升级为 **continuous local latent**：

- 当前：`Stage 2 -> 64-d binary`
- V2 上限版：`Stage 2 -> K-d continuous latent`（如 32 / 64 / 128 维）

这个 latent 表示：
- 局部表面朝向
- 厚度
- 曲率
- 局部拓扑复杂度

Stage 3 再根据 latent + image + coords 解码 UDF。

这才是真正的“终极上限版”，但工程量明显更大。

### V2-4 推荐的训练目标

#### Stage 2 指标门槛
建议把 Stage 2 是否“可用”定义为：

| 指标 | 目标 |
|---|---|
| snapshot recall | ≥ 0.90 |
| snapshot precision | ≥ 0.55 |
| snapshot IoU | ≥ 0.50 |
| pos_ratio_pred / pos_ratio_gt | 0.9 ~ 1.3 |

解释：
- recall 不够，Stage 3 一定会漏面
- precision 低一点能接受，因为 false positive 代价小

#### Stage 3 指标门槛
建议重点看：
- 法线图是否仍有大面积网状结构
- 窄结构是否断裂
- 平面区域是否出现空洞/断带

### V2-5 推荐落地路线图

#### Step 1（当前即可做）
- 当前版本继续训练 Stage 2 / Stage 3
- 观察 Stage 2 的 recall 是否足够高

#### Step 2（建议下一步实现）
- 在 Stage 3 训练中加入 `submask` 扰动增强
- 验证对 Stage 2 误差的鲁棒性是否提升

#### Step 3（若仍然漏细节）
- 把 Stage 3 的 hard gate 改为 soft gate

#### Step 4（若追求最终上限）
- 设计 `Stage 2.5: local latent predictor`
- 用 continuous latent 替代 binary submask

### V2 一句话定义

> **V2 = 粗定位 + 高 recall 支持域 + 对 Stage 2 误差鲁棒的连续场细化。**

它不是追求更“薄”的 mask，而是追求更“稳”的支持域和更强的 Stage 3 自恢复能力。

