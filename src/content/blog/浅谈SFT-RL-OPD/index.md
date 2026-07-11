---
title: '浅谈SFT-RL-OPD'
publishDate: 2026-07-11
updatedDate: 2026-07-11
description: '最近一段时间，OPD也是非常的火啊。我们紧跟热点，从KL散度视角理解SFT、RL和OPD的联系与区别，并结合代码讲解OPD如何实现'
tags:
  - Post-train
heroImage: { src: './image.png', color: '#fb0919' }
---




## 正向 KL 与反向 KL 散度

### KL 散度定义

KL 散度衡量两个概率分布之间的差异，定义为：

$$D_{KL}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$

KL 散度具有两个关键性质：
- **非负性**：$D_{KL}(P \parallel Q) \geq 0$，当且仅当 $P=Q$ 时取等号
- **不对称性**：$D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$，这正是正向/反向 KL 讨论的核心

### 正向 KL 散度（Forward KL）

正向 KL 散度以真实分布 $P$ 为参考，优化 $Q_\theta$ 去匹配 $P$：

$$D_{KL} = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$

从计算公式上理解，当我们最小化正向KL散度的时候，有一个特性：

- $P(x)$比较大的时候 $Q(x)$不能小，必须跟上。因此，当$P(x)$有多个peek的时候，$Q(x)$只能尽量去拟合。
- $P(x)$比较小的时候，那就无所谓了，因为作为乘数这一项的KL一定是小的。


正向KL散度还有一个特性，就是在LLM训练过程中与最小化交叉熵是等价的。这边考虑单个token：

交叉熵的公式长这样：
$$\mathcal{L} = -\sum_{i=1}^{C} y_i \log \hat{y}_i$$
对于LLM这种ground truth = one hot的情况下， 只有一个$y_{i}$能取值为1，这种情况下对于整个长度为$T$ 的序列，有：
$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} \log P_{\theta}(y_t \mid X, y_{1}, \dots, y_{t-1})$$

同理，对于单个token，最小化其正向KL散度，等同于：

$$D_{KL}(P_{\text{data}} \parallel P_{\theta}) = \sum_{w \in \text{Vocab}} P_{\text{data}}(w) \log \frac{P_{\text{data}}(w)}{P_{\theta}(w)}$$
其中$P_{data}$是一个one-hot， $P_{\theta}$是一个长度为 vocab_size 的概率分布。带入进去，其实就等同于单个token的交叉熵。

### 反向 KL 散度（Reverse KL）

反向 KL 散度以模型分布 $Q$ 为参考，优化其去匹配 $P$：

$$D_{KL}(Q \parallel P) = \sum Q(x) \log \frac{Q(x)}{P(x)}$$

相比之下， $Q(x)$不敢在$P(x)$比较小的地方放置太高的概率，但是可以忽略掉一些$P(x)$比较高的地方。因此，表现出来的就是mode-seeking，不会去完整的拟合真实分布。

### 正向 vs 反向 KL 对比

| 特性 | 正向 KL $D_{KL}(P \parallel Q)$ | 反向 KL $D_{KL}(Q \parallel P)$ |
|------|-------------------------------|------------------------------|
| 行为 | Mean-Seeking（均值寻求） | Mode-Seeking（模式寻求） |
| 覆盖性 | 覆盖 $P$ 的所有模式 | 锁定 $P$ 的某个单一模式 |
| 惩罚重点 | $P$ 有大值但 $Q$ 很小 | $Q$ 有大值但 $P$ 很小 |
| 典型场景 | SFT / MLE 训练 | KL 约束的 RL 训练 |


## SFT: 正向 KL 下的监督微调

Supervised Fine-Tuning（SFT）是从预训练到对齐的第一步。给定高质量对话数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$，其中 $x$ 为 prompt，$y$ 为理想回复。

### SFT 的目标函数

SFT 本质上是最小化数据分布 $P_{\text{data}}$ 与模型分布 $\pi_\theta$ 之间的正向 KL 散度：

$$\mathcal{L}_{\text{SFT}}(\theta) = D_{KL}(P_{\text{data}} \parallel \pi_\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}}\left[\log \pi_\theta(y \mid x)\right] + \text{const}$$

展开为标准交叉熵损失：

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \sum_{t=1}^{T_i} \log \pi_\theta(y_i^t \mid x_i, y_i^{<t})$$

其中 $y_i^t$ 是第 $i$ 条回复的第 $t$ 个 token。

### 为什么 SFT 用正向 KL？

1. **数据来自固定分布**：SFT 的数据集是预先收集的高质量数据，$P_{\text{data}}$ 是固定的，我们优化模型去拟合它
2. **Mean-Seeking 是优点**：我们希望模型学到数据中所有高质量的回复模式，而不是只锁定某一个
3. **等价于 MLE**：正向 KL 的最小化等价于最大似然估计，训练稳定、易于优化

### SFT 的局限

SFT 只教会模型"模仿"数据中的回复，但无法区分回复的优劣——数据集中所有样本被同等对待。此外，SFT 只在数据分布上训练，没有在模型自身生成分布上进行探索。


## RL: 反向 KL 约束下的强化学习

在 RLHF（RL from Human Feedback）阶段，我们使用奖励模型 $r_\phi(x, y)$ 来指导策略优化，同时用反向 KL 散度约束策略不要偏离 SFT 模型太远。

### RL 的目标函数

$$\max_\theta \ \mathbb{E}_{x \sim \mathcal{D},\ y \sim \pi_\theta(\cdot \mid x)}\left[r_\phi(x, y)\right] - \beta \cdot D_{KL}\left(\pi_\theta(\cdot \mid x) \parallel \pi_{\text{ref}}(\cdot \mid x)\right)$$

其中：
- $\pi_\theta$：当前策略（正在优化的模型）
- $\pi_{\text{ref}}$：参考策略（通常是 SFT 后的模型）
- $r_\phi$：奖励模型
- $\beta$：KL 惩罚系数，控制约束强度

### 为什么 RL 用反向 KL？

1. **Mode-Seeking 是优点**：我们希望在奖励高的区域集中概率质量，而不是在所有可能的回复上"摊大饼"。反向 KL 天然倾向于锁定高奖励模式
2. **避免 Reward Hacking**：KL 约束阻止策略走到奖励模型未曾见过（因而可能评分不准）的区域——反向 KL 惩罚 $\pi_\theta$ 在 $\pi_{\text{ref}}$ 低概率区域的探索 （模型几乎不太会生出 $\pi_{ref}$生成概率低的token）
3. **保持生成质量**：$\pi_{\text{ref}}$（SFT 模型）保证了基本的流畅性和合理性，反向 KL 确保策略不会退化


## Knowledge Distillation

传统的distillation其实就是从teacher中采样，然后让学生直接SFT这个数据，最小化正向KL散度。

$$D_{KL}(p_T||p_\theta)
=
\sum_y p_T(y|x)
\log
\frac{p_T(y|x)}
{p_\theta(y|x)}
$$

不过这里$P_T$其实是固定的one-hot分布，相当于Teacher 生成答案，Student 学习这个答案。

## OPD On-Policy Distillation

传统的KD存在的问题是什么：今天你让学生直接去学习教师模型的答案。但是由于学生模型和教师模型的policy根本不一样，学生模型很有可能会输出教师根本不存在的分布啊。比如同样是写一个sort算法，学生模型写的是快速排序，教师模型写的是归并。也就是训练数据和policy分布不一致的问题，这个在RL中就是off-policy，即：采样训练的数据model与当前的model不是同一个。此外，KD的监督粒度也太粗了，只有one-hot有loss，其他vocab的分布没有做纠正。

OPD的实际公式为：

$$\min_\theta \mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ D_{\text{KL}} \big( \pi_{S_\theta}(\cdot\vert{}x) \parallel \pi_T(\cdot\vert{}x) \big) \right]$$

学生模型在自己说得出来的那些话（$\pi_S$ 概率高的地方）里，去检查教师是不是也赞同（$\pi_T$ 是否也高）。如果学生自己说了一句话，而教师觉得极其荒谬（$\pi_T \to 0$），那么这一项会给出一个极大的惩罚。这迫使学生模型专注于生成高质量、确定性强的安全文本。


## SFT RL OPD之间的关系


我们可以用一个通用的强化学习奖励最大化框架，把 SFT、On-Policy Distillation 和标准 RL（如 PPO）串联在一起。如果我们把所有这类对齐问题都抽象为：在 Prompt 分布下，寻找一个策略 $\pi$，使其既能满足某种“评判标准”（奖励或教师导向），又不会偏离初始基座模型太远。

其实OPD就相当于一个没有Reward Model的RL，只有KL的惩罚。教师模型的监督成了Reward Model。用 RL 算法（如 PPO 或 REINFORCE），去最大化学生模型生成的文本在教师模型下的 Log 似然得分。 教师模型在这里充当了一个“完美且连续”的 Token 级 Reward 制造机。

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{S_\theta}(\cdot\vert{}x)} \left[ \log \pi_T(y\vert{}x) - \log \pi_{S_\theta}(y\vert{}x) \right]$$



- SFT 是 Off-policy 的模仿：照着教师写好的历史卷子抄，不知道自己写偏了该怎么办。

- On-Policy Distillation 是 On-policy 的围观指导：学生自己写卷子，教师在旁边看着，一旦学生下笔（生成 Token），教师就实时告诉他它的概率分布该长什么样。

- RL (如 PPO) 是 On-policy 的盲盒探索：学生自己写卷子，交上去之后只有一个环境给的模糊总分（Reward），学生得自己摸索哪一步写得好。

## 代码实战 opd

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 1. 加载模型
# ============================================================
student = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype=torch.bfloat16,
).cuda()
teacher = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
).cuda()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# 冻结教师模型：教师只提供监督信号，不更新参数
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False


# ============================================================
# 2. 准备 prompt 数据
# ============================================================
def build_prompt_input(prompt_text, tokenizer, max_length=512):
    """
    将自然语言 prompt 转为模型输入。
    使用 chat_template 保证格式与训练时一致。
    """
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return inputs


# ============================================================
# 3. OPD 训练循环
# ============================================================
prompts = [
    "解释一下什么是机器学习",
    "用 Python 写一个快速排序算法",
    "量子计算的基本原理是什么？",
    "介绍一下黑洞的形成过程",
]

optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)
student.train()

max_new_tokens = 128
temperature = 1.0  # 软化分布的 temperature，1.0 表示不缩放

for epoch in range(3):
    for idx, prompt in enumerate(prompts):
        # --- Tokenize prompt ---
        inputs = build_prompt_input(prompt, tokenizer)
        input_ids = inputs["input_ids"].cuda()          # (1, prompt_len)
        attention_mask = inputs["attention_mask"].cuda()
        prompt_len = input_ids.shape[1]                  # prompt 的 token 数

        # ============================================================
        # Step A: Student 在自己的策略 π_S 下生成回复（on-policy 采样）
        #
        # 关键：这里是学生「自己写」，不是抄教师的答案。
        # 采样出来的轨迹 y ~ π_S(·|x) 构成了本次训练的「数据」。
        # ============================================================
        with torch.no_grad():
            generated = student.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # generated 已包含 prompt + 新生成的 token
        full_ids = generated                              # (1, prompt_len + gen_len)
        gen_len = full_ids.shape[1] - prompt_len
        if gen_len == 0:
            continue  # 模型直接预测了 EOS，跳过

        # ============================================================
        # Step B: 教师模型在「同样的轨迹」上做前向传播
        #
        # 教师不生成，只在学生写出的序列上做一次 forward，
        # 拿到每个位置的完整 vocab 分布 π_T(·|x, y_<t)。
        # ============================================================
        with torch.no_grad():
            t_out = teacher(input_ids=full_ids)
            t_logits = t_out.logits                       # (1, total_len, vocab_size)

        # ============================================================
        # Step C: 学生模型在「同样的轨迹」上做前向传播
        #
        # 学生也做一次 forward（保留梯度），拿到自己在该轨迹上
        # 每个位置的分布 π_S(·|x, y_<t)。
        # ============================================================
        s_out = student(input_ids=full_ids)
        s_logits = s_out.logits                           # (1, total_len, vocab_size)

        # ============================================================
        # Step D: 计算 OPD Loss —— 反向 KL 散度
        #
        # OPD 目标：min D_KL( π_S || π_T )
        #   = Σ_y π_S(y) · [ log π_S(y) - log π_T(y) ]
        #
        # 逐 token 计算（以 vocab_size=3 为例）：
        #   假设某个位置学生 logits=[2.0, 1.0, 0.1]，教师 logits=[1.0, 2.0, 0.1]
        #   经过 softmax：
        #     π_S = [0.59, 0.22, 0.19]    ← 学生更偏好 token A
        #     π_T = [0.24, 0.67, 0.09]    ← 教师更偏好 token B
        #
        #   kl = 0.59·log(0.59/0.24) + 0.22·log(0.22/0.67) + 0.19·log(0.19/0.09)
        #      = 0.53 - 0.24 + 0.14 = 0.43
        #
        #   本质：以学生概率 π_S(y) 为权重，加权每个词上「学生对教师的对数比」。
        #   学生高度相信而教师不认可的 token → 大惩罚（mode-seeking 效应）。
        #
        # 只在「生成 token」的位置计算 loss（prompt 部分是给定的，不参与训练）。
        #
        # 注意 logits 的对齐：
        #   logits[t] 预测的是第 t+1 个 token。
        #   第 0 个生成 token 位于 full_ids 的位置 prompt_len，
        #   预测它的 logit 位于 logits 的位置 prompt_len - 1。
        #   因此生成部分的 logits 范围是 [prompt_len-1, total_len-2]，
        #   即 logits[:, prompt_len-1 : -1, :]。
        # ============================================================
        s_logits_gen = s_logits[:, prompt_len - 1 : -1, :] / temperature
        t_logits_gen = t_logits[:, prompt_len - 1 : -1, :] / temperature

        # 计算 log-softmax 和 softmax
        s_log_prob = F.log_softmax(s_logits_gen, dim=-1)   # log π_S(·|context), (1, gen_len, V)
        t_log_prob = F.log_softmax(t_logits_gen, dim=-1)   # log π_T(·|context), (1, gen_len, V)
        s_prob     = F.softmax(s_logits_gen, dim=-1)        # π_S(·|context),      (1, gen_len, V)

        # 反向 KL：对每个生成 token 位置，在 vocab 维度求和
        #   kl_per_token[t] = Σ_{y ∈ vocab} π_S(y|ctx_t) · log( π_S(y|ctx_t) / π_T(y|ctx_t) )
        kl_per_token = (s_prob * (s_log_prob - t_log_prob)).sum(dim=-1)  # (1, gen_len)

        # 对所有生成 token 取平均作为最终 loss
        # 相当于 E_{t ~ generated_positions}[ D_KL(π_S(·|ctx_t) || π_T(·|ctx_t)) ]
        loss = kl_per_token.mean()

        # ============================================================
        # Step E: 反向传播 & 更新学生参数
        # ============================================================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 1 == 0:
            print(
                f"Epoch {epoch+1} | Step {idx} | Loss: {loss.item():.4f} | "
                f"Gen len: {gen_len} | Prompt: {prompt[:30]}..."
            )

    print(f"=== Epoch {epoch+1} 完成 ===\n")
```