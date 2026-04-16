---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->

# Tree-Ring Watermark 复现总结

## Fingerprints for Diffusion Images that are Invisible and Robust

**NeurIPS 2023**

---

## 目录

1. 论文研读
2. 原理分析
3. 运行时关键函数分析
4. 复现修改
5. 复现结果

---

<!-- _class: lead -->

# 01 论文研读

---

## 基本信息

- **标题**：*Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust*
- **会议**：NeurIPS 2023
- **任务**：为扩散模型生成的图像添加**不可见且鲁棒**的数字水印

### 核心创新点

| 特点 | 说明 |
|------|------|
| **不可见性** | 水印不嵌入像素层，而是隐藏在**初始噪声潜变量（Latent）**中 |
| **无需修改模型** | 直接替换扩散过程的初始噪声，UNet/VAE/文本编码器均保持原样 |
| **频域鲁棒性** | 在**傅里叶频域中心**注入密钥，对旋转、JPEG、模糊等攻击天然鲁棒 |
| **盲检测** | 检测时无需原 prompt，只需 DDIM Inversion 回噪声即可验证 |

---

## 关键参数

- **`w_pattern`**：水印图案类型
  - `ring`（环形，论文默认）
  - `rand`（随机）
  - `zeros`（零值）
- **`w_channel`**：注入通道索引
  - `0~3` 对应 latent 的 4 个通道
  - `-1` 表示全部通道
- **`w_radius`**：频域中心水印半径
  - 控制密钥覆盖的频域范围大小

---

<!-- _class: lead -->

# 02 原理分析

---

## 生成阶段：水印注入

```text
随机噪声 Latent ──► FFT ──► 频域中心覆盖 Ring 图案 ──► IFFT
                                                        │
                                                        ▼
                                           携带水印的 Latent ──► 标准扩散生成 ──► 图像
```

1. 获取标准高斯噪声 latent $z_0$
2. 对 $z_0$ 做 **2D-FFT** 并 `fftshift` 移到频域中心
3. 在中心半径为 `w_radius` 的环形区域，用预设密钥 `gt_patch` 覆盖频谱
4. **IFFT** 还原回空间域，得到携带水印的 latent
5. 通过 Stable Diffusion 标准管道生成图像

---

## 检测阶段：DDIM 反演

```text
生成图像 ──► VAE Encode ──► 图像 Latent ──► DDIM Inversion (50步反向扩散)
                                                          │
                                                          ▼
                                                    恢复噪声 Latent
                                                          │
                                                          ▼
                                                     FFT + fftshift
                                                          │
                                                          ▼
                                               计算与 gt_patch 的 L1 距离
```

1. 将图像通过 VAE 编码回 latent 空间
2. 使用**空 prompt** 进行 **DDIM Inversion**
3. 对恢复出的 latent 做 FFT，检查频域中心是否匹配 `gt_patch`
4. 用 **L1 距离** 作为检测指标，**ROC/AUC** 评估性能

---

<!-- _class: lead -->

# 03 运行时关键函数分析

---

## 初始化阶段

| 函数 | 文件 | 作用 |
|------|------|------|
| `main(args)` | `run_tree_ring_watermark.py` | 主入口，参数解析、模型加载 |
| `InversableStableDiffusionPipeline.from_pretrained()` | `inverse_stable_diffusion.py` | 加载 SD 完整管道 |
| `get_dataset(args)` | `optim_utils.py` | 加载数据集（prompts） |
| `get_watermarking_pattern()` | `optim_utils.py` | **生成密钥**：FFT 后构造频域 ring 图案 |
| `get_watermarking_mask()` | `optim_utils.py` | **生成掩码**：标记频域注入位置 |

---

## 生成阶段（每轮执行 2 次）

| 函数 | 文件 | 作用 |
|------|------|------|
| `pipe.get_random_latents()` | `inverse_stable_diffusion.py` | 生成标准正态噪声 `[1, 4, 64, 64]` |
| `pipe() / __call__()` | `modified_stable_diffusion.py` | **50 步去噪 + CFG + VAE 解码** |
| `inject_watermark(...)` | `optim_utils.py` | **核心**：FFT → 掩码覆盖 → IFFT → 归一化 |

---

## 反演与评估阶段

| 函数 | 文件 | 作用 |
|------|------|------|
| `transform_img(...)` | `optim_utils.py` | 图像预处理：Resize、Crop、ToTensor、归一化 |
| `pipe.get_image_latents(...)` | `inverse_stable_diffusion.py` | VAE Encode 回 latent |
| `pipe.forward_diffusion(...)` | `inverse_stable_diffusion.py` | **DDIM 反演**：图像 → 噪声 |
| `eval_watermark(...)` | `optim_utils.py` | FFT 后计算与密钥的 L1 距离 |
| `metrics.roc_curve / auc` | `sklearn` | 计算 ROC、AUC、ACC、TPR@1%FPR |

---

<!-- _class: lead -->

# 04 复现修改

---

## 遇到的问题与修复

| 问题 | 现象 | 修复方案 |
|------|------|----------|
| `CUFFT_INTERNAL_ERROR` | `torch.fft.fft2()` 在 GPU `float16` 上崩溃 | 模型加载改为 `torch.float32`；FFT 前 `.cpu().float()` |
| 数据集加载失败 | HuggingFace 在线加载不稳定 | `get_dataset()` 优先读取本地 COCO，失败 fallback |
| IFFT 后数值漂移 | 频域转回空间域后标准差改变 | `inject_watermark()` 中加入**标准差归一化** |
| 复数类型丢失 | `.float()` 会丢失复数信息 | 保留复数：`gt_patch.cpu()` |
| 本地无需 CLIP | open_clip 增加显存开销 | 强制 `reference_model = None` |
| 安全过滤干扰 | 正常图像被误拦截 | `safety_checker=None` |

---

## 核心修改对比：`inject_watermark`

### 原版（报错）

```python
init_latents_w_fft = torch.fft.fftshift(
    torch.fft.fft2(init_latents_w), dim=(-1, -2)
)
init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
init_latents_w = torch.fft.ifft2(
    torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))
).real
return init_latents_w
```

❌ `float16` 导致 `RuntimeError: CUFFT_INTERNAL_ERROR`

---

## 核心修改对比：`inject_watermark`

### 修复版

```python
def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    original_std = init_latents_w.std().item()
    original_device = init_latents_w.device
    original_dtype = init_latents_w.dtype

    # ✅ CPU + float32 执行 FFT
    init_latents_w_cpu = init_latents_w.cpu().float()
    gt_patch_cpu = gt_patch.cpu()  # 保持复数
    watermarking_mask_cpu = watermarking_mask.cpu()

    init_latents_w_fft = torch.fft.fftshift(
        torch.fft.fft2(init_latents_w_cpu), dim=(-1, -2)
    )
    init_latents_w_fft[watermarking_mask_cpu] = \
        gt_patch_cpu[watermarking_mask_cpu].clone()

    init_latents_w_out = torch.fft.ifft2(
        torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))
    ).real

    # ✅ 恢复数值尺度
    current_std = init_latents_w_out.std().item()
    if current_std > 0:
        init_latents_w_out *= (original_std / current_std)

    return init_latents_w_out.to(original_device).to(original_dtype)
```

---

<!-- _class: lead -->

# 05 复现结果

---

## 验证标准

- ✅ 成功生成 `no_w_{i}.png` 和 `w_{i}.png`
- ✅ 两图像素差异在 **1~5** 之间，**肉眼不可见**
- ✅ 图像标准差在 **40~80**，无纯色/异常图
- ✅ 可完整跑通：**生成 → 水印注入 → DDIM 反演 → 评估**

---

## 检测性能指标

| 指标 | 含义 | 理想值 |
|------|------|--------|
| **AUC** | ROC 曲线下面积 | 接近 1.0 |
| **ACC** | 最佳阈值准确率 | 越高越好 |
| **TPR@1%FPR** | FPR ≤ 1% 时的检出率 | **论文核心指标** |

> 循环结束后由 `sklearn.metrics` 自动计算并输出到终端，同时同步到 wandb。

---

## 最终状态

### 报错
- `RuntimeError: cuFFT error: CUFFT_INTERNAL_ERROR`

### 根因
- Stable Diffusion 以 `float16` 加载时，GPU 上的 `torch.fft.fft2` 在 cuFFT 底层不支持 ComplexHalf

### 状态
- ✅ **已修复**，程序可完整跑通全流程

---

<!-- _class: lead -->

# 感谢聆听

## Q & A
