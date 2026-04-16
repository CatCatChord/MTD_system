# Tree-Ring Watermark 复现笔记

## 一、论文简介

**论文标题**: *Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust* (NeurIPS 2023)

**核心思想**: 将水印嵌入在扩散模型生成图像的**初始噪声潜变量（initial latents）**中。具体做法是让该噪声经过二维傅里叶变换（FFT）后，在频域中心携带一个精心构造的“密钥”图案（如环形 `ring`）。

- **生成阶段**: 不修改扩散模型本身，仅替换初始噪声为携带水印的噪声，随后通过标准扩散管道生成图像。
- **检测阶段**: 对生成图像进行 **DDIM Inversion（反演）**，还原出原始噪声潜变量，再对其做 FFT，检查频域中心是否携带该密钥图案。
- **优势**: 不可见（水印在噪声层，不影响视觉质量）；对旋转、裁剪、JPEG 压缩等攻击具有频域鲁棒性。

**关键超参数**:
- `w_channel`: 水印嵌入的通道索引（`-1` 表示所有通道）。
- `w_pattern`: 水印图案类型（`ring`、`rand`、`zeros` 等）。
- `w_radius`: 频域中心水印半径。

---

## 二、复现所需关键函数

按流程分类，以下是必须理解或调试的核心函数：

### 1. 水印准备
| 函数 | 文件 | 作用 |
|------|------|------|
| `get_watermarking_pattern(pipe, args, device)` | `optim_utils.py` | 生成目标水印图案 `gt_patch`。例如 `ring` 模式会生成一个以频域中心为圆心、半径 `w_radius` 的环形复数图案。 |
| `get_watermarking_mask(init_latents_w, args, device)` | `optim_utils.py` | 生成布尔掩码 `watermarking_mask`，标记潜变量中哪些位置需要被水印覆盖。 |
| `circle_mask(size, r)` | `optim_utils.py` | 辅助函数，生成二维圆形掩码。 |

### 2. 水印注入与生成
| 函数 | 文件 | 作用 |
|------|------|------|
| `inject_watermark(init_latents_w, mask, gt_patch, args)` | `optim_utils.py` | **核心**: 将初始噪声潜变量做 FFT，在掩码区域内用 `gt_patch` 覆盖，再做 IFFT 还原回空间域。 |
| `pipe.get_random_latents()` | `inverse_stable_diffusion.py` | 获取标准正态分布的初始潜变量噪声。 |
| `pipe(...)` | `run_tree_ring_watermark.py` | 调用扩散管道生成图像（带水印 / 不带水印）。 |

### 3. 检测与评估
| 函数 | 文件 | 作用 |
|------|------|------|
| `pipe.forward_diffusion(...)` | `inverse_stable_diffusion.py` | **核心**: DDIM 反演，从图像还原回噪声潜变量。 |
| `eval_watermark(reversed_no_w, reversed_w, mask, gt_patch, args)` | `optim_utils.py` | 对反演后的潜变量做 FFT，计算与 `gt_patch` 的 L1 距离，得到检测指标。 |
| `image_distortion(img1, img2, seed, args)` | `optim_utils.py` | 模拟各种攻击（旋转、JPEG、裁剪、高斯模糊、亮度调整等）。 |

---

## 三、报错分析：`RuntimeError: cuFFT error: CUFFT_INTERNAL_ERROR`

### 报错信息
```text
File ".../optim_utils.py", line 216, in inject_watermark
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
RuntimeError: cuFFT error: CUFFT_INTERNAL_ERROR
```

### 根本原因
原版代码加载 Stable Diffusion 模型时使用了 **`torch_dtype=torch.float16`**（并指定 `revision='fp16'`）：

```python
pipe = InversableStableDiffusionPipeline.from_pretrained(
    args.model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    revision='fp16',
)
```

这会导致 `pipe.get_random_latents()` 返回的 `init_latents_w` 是一个 **GPU 上的 `float16` (half) tensor**。当该 tensor 被传入 `torch.fft.fft2()` 时，PyTorch 会尝试调用 **cuFFT (CUDA Fast Fourier Transform)** 库处理 half precision 输入。

然而，在许多常见的 PyTorch / CUDA 驱动版本组合下（尤其是 RTX 4090 等较新显卡 + 某些 conda 环境），**cuFFT 对 `float16` 的复数 FFT 支持不完整或存在底层 Bug**，因此直接抛出 `CUFFT_INTERNAL_ERROR`。

### 解决方案
在进行任何 FFT 运算之前，先将 tensor 从 GPU 移到 CPU，并显式转换为 `float32`：

```python
init_latents_w_cpu = init_latents_w.cpu().float()
init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w_cpu), dim=(-1, -2))
```

计算完成后再通过 `.to(original_device).to(original_dtype)` 转回原来的设备和数据类型。同时，由于经过 FFT → IFFT → 取 `real` 后，latent 的标准差可能会发生轻微漂移，还需要做一次**标准差归一化**以保持原始噪声的数值分布，避免生成图像质量下降。

---

## 四、修改内容与原函数对比

### 1. `run_tree_ring_watermark.py`（主脚本）

| 修改项 | 原版 | 复现版 | 原因/目的 |
|--------|------|--------|-----------|
| **输出目录** | 无 | 添加了 `os.makedirs(args.run_name, exist_ok=True)` | 确保运行前目录已存在，避免报错。 |
| **模型精度** | `torch_dtype=torch.float16, revision='fp16'` | `torch_dtype=torch.float32` | **解决 `ComplexHalf` 不支持问题**。`float16` 的 latent 无法直接做复数 FFT。 |
| **安全过滤器** | 默认启用 | `safety_checker=None, requires_safety_checker=False` | 本地测试时避免安全过滤干扰。 |
| **CLIP 模型** | 默认根据参数加载 | 强制置为 `None`，并打印警告 | 本地测试时跳过 CLIP 评分，减少依赖。 |
| **图像保存** | 不保存中间图 | 增加了 `orig_image_no_w.save(...)` 和 `orig_image_w.save(...)` | 方便本地可视化验证。 |
| **空列表打印** | 直接 `mean(clip_scores)` | 增加了 `if clip_scores:` 判断 | 避免 `reference_model=None` 时 `mean([])` 报错。 |

### 2. `optim_utils.py`（核心水印逻辑）

#### (a) `get_dataset()`
- **原版**: 从 HuggingFace `load_dataset(args.dataset)` 或本地 COCO JSON 加载。
- **复现版**: **硬编码了 100 条测试 prompts**，绕过数据集解析/网络访问问题。

#### (b) `get_watermarking_pattern()` — `ring` / `rand` 模式
- **原版**: 直接对 GPU tensor 做 `torch.fft.fft2(gt_init)`。
- **复现版**:
  ```python
  gt_cpu = gt_init.cpu().float()   # 强制 float32 + CPU
  gt_fft = torch.fft.fft2(gt_cpu)
  gt_patch = torch.fft.fftshift(gt_fft, dim=(-1, -2)).to(device)
  ```
  **目的**: 确保 FFT 在 float32 上执行，避免 GPU 上的 `ComplexHalf` 错误。

#### (c) `inject_watermark()` — 最大改动
- **原版**（简洁版）:
  ```python
  init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
  init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
  init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real
  return init_latents_w
  ```
- **复现版**（健壮版）:
  1. 保存 `original_std`、`original_device`、`original_dtype`。
  2. 将 `init_latents_w` 移到 **CPU float32**。
  3. 保持 `gt_patch` 的**复数类型**（移除了 `.float()`，否则复数信息会丢失）。
  4. CPU 上做 FFT → 掩码赋值 → IFFT。
  5. 取 `real` 后，用 `original_std / current_std` **恢复数值尺度**。
  6. 最后 `.to(original_device).to(original_dtype)` 返回。

#### (d) `eval_watermark()` & `get_p_value()`
- **原版**: 直接在 GPU tensor 上做 FFT。
- **复现版**: 统一改为先 `.cpu().float()` 做 FFT，计算完再移回 device。

---

## 五、遇到的困难及解决方案总结

| 困难 | 具体表现 | 解决方案 |
|------|----------|----------|
| **1. `ComplexHalf` 不支持 / `CUFFT_INTERNAL_ERROR`** | 原版用 `torch.float16` 加载 SD，`torch.fft.fft2` 在 GPU 上报错 `CUFFT_INTERNAL_ERROR`。 | ① 主脚本中将模型加载改为 `torch.float32`；<br>② 在所有 FFT 操作前显式调用 `.cpu().float()`。 |
| **2. 数据集加载失败/解析问题** | 原版依赖 HuggingFace 在线加载或本地 COCO JSON，网络不通或文件缺失则无法运行。 | 将 `get_dataset()` 改为**硬编码 100 条测试 prompts**，零依赖启动。 |
| **3. IFFT 后数值尺度漂移** | 将 latent 移到 CPU 做 FFT→IFFT 后取 `real`，标准差可能与原始 latent 不一致，导致生成图像质量下降或出现纯色。 | 在 `inject_watermark()` 中引入**标准差归一化** (`original_std / current_std`)。 |
| **4. 复数类型丢失** | 修复 float16 问题时若对 `gt_patch` 调用 `.float()`，会导致复数 FFT 结果仅保留实部，破坏水印注入逻辑。 | 在 `inject_watermark()` 中明确保留复数类型：`gt_patch_cpu = gt_patch.cpu()`（去掉 `.float()`）。 |
| **5. 本地无 CLIP 模型/无需评分** | 原版默认尝试加载 `open_clip` 模型计算 CLIP Score，增加运行时间和显存占用。 | 在主脚本中将 `args.reference_model` 强制置为 `None`，并跳过相关代码。 |

---

## 六、验证方法

复现后可以使用如下方式快速验证正确性（参考 `test.py`）：

1. **尺寸检查**: 生成图像应为 `512x512`（或模型配置尺寸）。
2. **纯色检查**: 图像像素标准差应在 `40–80` 之间。若 `< 5` 说明生成的是纯色/异常图，存在 bug。
3. **不可见性检查**: 水印图与无水印图的像素差异应在 `1–5` 之间，肉眼不可见。
4. **检测指标**: 运行结束后观察 `AUC`、`ACC`、`TPR@1%FPR` 等指标。正常情况下 watermark 样本的 `w_metric` 应显著低于非 watermark 样本的 `no_w_metric`。

---

## 七、关键代码片段（修改后的 `inject_watermark`）

```python
def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    # 保存原始数值尺度
    original_std = init_latents_w.std().item()
    original_device = init_latents_w.device
    original_dtype = init_latents_w.dtype
    
    # 移到 CPU，但保持 gt_patch 的原始类型（复数！）
    init_latents_w_cpu = init_latents_w.cpu().float()
    gt_patch_cpu = gt_patch.cpu()  # 保持复数类型
    watermarking_mask_cpu = watermarking_mask.cpu()
    
    # FFT（结果是复数）
    init_latents_w_fft = torch.fft.fftshift(
        torch.fft.fft2(init_latents_w_cpu), dim=(-1, -2)
    )
    
    if args.w_injection == 'complex':
        init_latents_w_fft[watermarking_mask_cpu] = \
            gt_patch_cpu[watermarking_mask_cpu].clone()
    elif args.w_injection == 'seed':
        init_latents_w_cpu[watermarking_mask_cpu] = \
            gt_patch_cpu[watermarking_mask_cpu].clone()
        current_std = init_latents_w_cpu.std().item()
        if current_std > 0:
            init_latents_w_cpu = init_latents_w_cpu * (original_std / current_std)
        return init_latents_w_cpu.to(original_device).to(original_dtype)
    else:
        raise NotImplementedError(f'w_injection: {args.w_injection}')

    # IFFT
    init_latents_w_ifft = torch.fft.ifft2(
        torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))
    )
    init_latents_w_out = init_latents_w_ifft.real  # 取实部
    
    # 恢复数值尺度
    current_std = init_latents_w_out.std().item()
    if current_std > 0:
        init_latents_w_out = init_latents_w_out * (original_std / current_std)
    
    # 回到原设备和类型
    init_latents_w_out = init_latents_w_out.to(original_device).to(original_dtype)
    
    return init_latents_w_out
```
