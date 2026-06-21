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



## MHA


$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
$$\text{Attention}(Q', K', V') = \text{softmax}\left(\frac{Q'{K'}^T}{\sqrt{d_k}}\right)V'$$


```py
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        # // 返回的类型会保留原类型 比如 a b都是整数 那么 a//b也是整数
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False) 
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor):
        """
            D dimension必须匹配 d_model

            
            两个tensor做 @ 操作的时候 只会看最后两个维度
        """
        B, L, D = x.shape
        # B, L, 3D -> B, L, 3, num_heads, head_dim
        qkv: torch.Tensor = self.qkv_proj(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        # 把一个 Tensor 沿着指定的Dim 拆散”成一个由多个 Tensor 组成的元组。
        q, k, v = qkv.unbind(dim=2)
        # B, L, num_heads, head_dim -> B, num_heads, L, head_dim
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # B, num_heads, L, L
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = F.softmax(scores, dim=-1)

        # B, num_heads, L, head_dim -> B, L, num_heads, head_dim -> B, L, d_model
        attn_output = ( attn_scores @ v ).transpose(1, 2).reshape(B, L, D)

        return self.out_proj(attn_output)




```



### GQA

相比之下，只是多了一个KV拷贝的操作。

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        assert num_heads % num_kv_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        # 需要复制几份 kv
        self.num_q_per_kv = num_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, d_model, bias= False)

        self.kv_proj = nn.Linear(d_model, 2 * num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor):
        
        B, L, D = x.shape

        q : torch.Tensor = self.q_proj(x)
        kv : torch.Tensor = self.kv_proj(x)

        q = q.reshape(B, L, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)

        kv = kv.reshape(B, L, 2, self.num_kv_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # B, L, num_kv_heads, head_dim -> B, L, num_heads, head_dim
        # 一般会先增加一个维度 在该维度上expand 然后reshape
        if self.num_q_per_kv > 1 : 
            print(k.is_contiguous()) # not contiguous 但是下面调用view不知道为什么没有报错
            k = k.view(B, self.num_kv_heads, 1, L, self.head_dim)
            k = k.expand(B, self.num_kv_heads, self.num_q_per_kv, L, self.head_dim)
            k = k.reshape(B, self.num_heads, L, self.head_dim)

            v = v.view(B, self.num_kv_heads, 1, L, self.head_dim)
            v = v.expand(B, self.num_kv_heads, self.num_q_per_kv, L, self.head_dim)
            v = v.reshape(B, self.num_heads, L, self.head_dim)
            
        scores = (q @ k.transpose(-2, -1)) * self.scale

        attn_scores = F.softmax(scores, dim=-1)
        attn_output = (attn_scores @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(attn_output)
             
```


### Axis Attention
在计算机视觉等领域，H * W个 token，每个token维度是C，都做Attention计算的话，开销很大，时间复杂度是 $O((H * W)^2 * C)$，而使用Axis Attention的方式，时间复杂度降低至 $O(H^2W + W^2H) * C$。可以理解为，做行的Attention时，每一行的token有W个，单行的复杂度为 $O(W^2 * C)$。
```python
class AxisAttentionSingle(nn.Module):
    """ 在指定轴(Row或Col)上计算一维自注意力 """
    def __init__(self, dim, num_heads=8, axis=2): # axis=2 对应 H, axis=3 对应 W (输入为 B,C,H,W)
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.axis = axis
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 核心：通过置换，把非目标轴强行塞入 Batch 维度
        if self.axis == 2:   # 沿 H 轴（行）计算：把 W 合并到 Batch
            x_perm = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)
            tgt_len = H
        else:                # 沿 W 轴（列）计算：把 H 合并到 Batch
            x_perm = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
            tgt_len = W
            
        N_batch = x_perm.shape[0]
        
        # 2. 标准 MHA 矩阵计算
        qkv = self.qkv(x_perm).reshape(N_batch, tgt_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        context = (attn @ v).transpose(1, 2).reshape(N_batch, tgt_len, self.dim)
        out = self.proj(context)
        
        # 3. 逆向重排还原为原图几何形状 [B, C, H, W]
        if self.axis == 2:
            return out.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()
        else:
            return out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

```

