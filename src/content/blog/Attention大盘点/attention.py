import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
            print(k.is_contiguous())
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


# ==================== 测试脚本 ====================
def run_gqa_test():
    # 1. 定义模型参数 (d_model=64, 8个Q头, 2个KV头 -> 每个KV组对应4个Q头)
    D_MODEL = 64
    NUM_HEADS = 32
    NUM_KV_HEADS = 4
    
    print("--- 正在初始化 GQA 模型 ---")
    gqa_layer = GroupedQueryAttention(d_model=D_MODEL, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS)
    
    # 2. 构造输入数据：[Batch_Size=2, Seq_Len=10, D_Model=64]
    # 只要满足 NUM_HEADS // NUM_KV_HEADS > 1，就会触发 if 条件里的维度扩充
    x = torch.randn(2, 10, D_MODEL)
    print(f"输入数据 x 的形状: {list(x.shape)}")
    print("-> 模拟前向传播开始...\n")

    try:
        # 3. 尝试运行前向传播
        output = gqa_layer(x)
        print("✅ [恭喜] 前向传播成功！输出形状为:", list(output.shape))
        
    except RuntimeError as e:
        print("❌ [前向传播崩溃！完美复现内存机制报错]")
        print("-" * 80)
        print(e)
        print("-" * 80)
        print("\n💡 报错原因分析：")
        print("正如期望的那样，数据流运行到 `k.reshape(...)` 时炸了。")
        print("因为经过 `transpose` 和 `expand` 之后，k 的内存布局在 PyTorch 底层已经散掉了。")
        print("此时如果不加上 `.contiguous()`，PyTorch 没办法直接把 `num_kv_heads` 和 `num_q_per_kv` 揉成一条连续的 `num_heads` 轴。")

if __name__ == "__main__":
    run_gqa_test()
        
















