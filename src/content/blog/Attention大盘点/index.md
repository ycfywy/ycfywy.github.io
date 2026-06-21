---
title: 'Attention大盘点'
publishDate: 2026-06-20
updatedDate: 2026-06-20
description: ‘对当前常用的Attention做一个大盘点。'
tags:
  - AI
  - LLM
heroImage: { src: './image.png', color: '#839dd5' }
---



## Attention 全家族大盘点

| 家族分类 | 变体名称 | 核心机制与改动点 | 复杂度 (时间/空间) | 核心解决痛点 | 工业界典型应用场景 |
| --- | --- | --- | --- | --- | --- |
| **通道优化家族** | **MHA (标准多头)** | 每个 Head 拥有完全独立的 $Q, K, V$ 投影矩阵。 | $\mathcal{O}(N^2 \cdot d)$ | 允许模型同时关注不同子空间的信息。 | Transformer 基石、BERT |
| **通道优化家族** | **MQA (多查询)** | 所有 Head 共享**同一个** $K$ 和 $V$ 投影，只有 $Q$ 保持多头。 | $\mathcal{O}(N^2 \cdot d)$ (大幅减小显存常数) | 极大地降低了大模型推理时的 **KV Cache 显存占用**。 | PaLM、ChatGLM |
| **通道优化家族** | **GQA (分组查询)** | 将 Head 分组，**组内共享**一对 $K$ 和 $V$（MHA 与 MQA 的折中）。 | $\mathcal{O}(N^2 \cdot d)$ (显存与效果的黄金平衡) | 在保持大模型表达力的同时，大幅优化推理吞吐量。 | **Llama 3、Mistral、Qwen 等现代大模型标配** |
| **空间几何家族** | **Axial Attention (轴向)** | 将 2D 注意力拆解：先算行（Row）1D 注意力，再算列（Col）1D 注意力。 | $\mathcal{O}(HW(H+W))$ | 破除高分辨率图像或视频全局计算时的 **OOM（显存溢出）** 噩梦。 | 高清图像生成、医学图像分割 |
| **空间几何家族** | **Window / Local (滑窗)** | 将特征图划分为固定大小的局部窗口（如 $7 \times 7$），只在内部计算。 | $\mathcal{O}(HW \cdot w^2)$ | 降低视觉任务的计算复杂度，使其具备局部感知的归纳偏置。 | **Swin Transformer**、Longformer |
| **空间几何家族** | **Criss-Cross (十字)** | 对每个像素，只看它正上、下、左、右“十字路径”上的 Token。 | $\mathcal{O}(HW(H+W))$ | 用极低的计算量通过两层叠加间接获得全图视野。 | CCNet（语义分割） |
| **核变换家族** | **Linear Attention (线性)** | 利用核函数特性改变矩阵乘法结合律：$\phi(Q)(\phi(K)^T V)$。 | $\mathcal{O}(N)$ **(线性复杂度)** | 彻底打破 $\mathcal{O}(N^2)$ 魔咒，使处理**无限长文本**成为可能。 | Performer、Linear Transformer |
| **核变换家族** | **SSD (Mamba 2 内核)** | 将状态空间模型（SSM）与**带有时序衰减因子**的线性注意力在数学上对齐。 | $\mathcal{O}(N)$ | 兼顾了 Transformer 的并行训练优势与 RNN 的常数级推理吞吐。 | **Mamba 2 架构** |
| **硬件加速家族** | **FlashAttention (1/2/3)** | **不改公式**。利用 GPU 的 SRAM 高速缓存，通过**分块（Tiling）**避免显存 I/O 瓶颈。 | 数学上仍为 $\mathcal{O}(N^2)$ (但硬件实际运行快数倍) | 解决显存带宽读写太慢导致的 GPU 算力闲置问题（I/O 墙）。 | **现代大模型训练与推理的底层绝对核心** |
| **硬件加速家族** | **PagedAttention** | **不改公式**。引入操作系统虚拟内存分页思想，将 KV Cache 离散存储。 | 数学复杂度不变 | 彻底解决推理时的显存碎片化，支持极高并发的文本生成。 | **vLLM 推理框架的核心** |
| **模态与拓扑家族** | **Cross-Attention (交叉)** | $Q$ 来自序列 A（如图像/当前层），$K, V$ 来自序列 B（如文本/编码器）。 | $\mathcal{O}(N_A \cdot N_B)$ | 实现不同模态、不同序列之间信息的跨界融合与对齐。 | **Stable Diffusion（文本引导图像生成）**、Transformer Decoder |



