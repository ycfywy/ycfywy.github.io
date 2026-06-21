---
title: 'RL大串烧'
publishDate: 2026-06-21
updatedDate: 2026-06-21
description: '讨论当前比较火爆的RL算法及其实现——从Policy Gradient到PPO、GRPO、GSPO、RLOO、DPO'
tags:
  - 算法
  - RL
heroImage: { src: './image.png', color: '#839dd5' }
---

## 目录

- [目录](#目录)
- [1. 引言：RL 在 LLM 时代的新角色](#1-引言rl-在-llm-时代的新角色)
- [2. 基础概念：一条 Trajectory 里有什么](#2-基础概念一条-trajectory-里有什么)
  - [2.1 传统 RL 视角](#21-传统-rl-视角)
  - [2.2 LLM RLHF 视角](#22-llm-rlhf-视角)
  - [2.3 两个核心数据结构](#23-两个核心数据结构)
- [3. Policy Gradient：一切的起点](#3-policy-gradient一切的起点)
  - [3.1 策略梯度定理](#31-策略梯度定理)
  - [3.2 最简单的 Advantage：Reward-to-Go + Baseline](#32-最简单的-advantagereward-to-go--baseline)
  - [3.3 代码实现](#33-代码实现)
- [4. PPO：Proximal Policy Optimization](#4-ppoproximal-policy-optimization)
  - [4.1 Importance Sampling \& Ratio](#41-importance-sampling--ratio)
  - [4.2 Clipped Surrogate Objective](#42-clipped-surrogate-objective)
  - [4.3 GAE：Generalized Advantage Estimation](#43-gaegeneralized-advantage-estimation)
  - [4.4 完整 Loss \& Value Function](#44-完整-loss--value-function)
- [5. GRPO：Group Relative Policy Optimization](#5-grpogroup-relative-policy-optimization)
  - [5.1 为什么不再需要 Value Function](#51-为什么不再需要-value-function)
  - [5.2 Group-based Advantage](#52-group-based-advantage)
  - [5.3 Token-level Clipping + KL 惩罚](#53-token-level-clipping--kl-惩罚)
- [6. GSPO：Sequence-level 的改良](#6-gsposequence-level-的改良)
- [7. RLOO：REINFORCE Leave-One-Out](#7-rlooreinforce-leave-one-out)
  - [7.1 Leave-One-Out Baseline](#71-leave-one-out-baseline)
  - [7.2 为什么 RLOO 依然有效](#72-为什么-rloo-依然有效)
- [8. DPO：Direct Preference Optimization](#8-dpodirect-preference-optimization)
  - [8.1 从 RLHF 到 DPO](#81-从-rlhf-到-dpo)
  - [8.2 Bradley-Terry 模型 \& DPO Loss](#82-bradley-terry-模型--dpo-loss)
  - [8.3 DPO 的优缺点](#83-dpo-的优缺点)
- [9. 六种算法全景对比](#9-六种算法全景对比)
  - [9.1 Loss 函数速查](#91-loss-函数速查)
  - [9.2 架构依赖对比](#92-架构依赖对比)
  - [9.3 选择建议](#93-选择建议)
- [10. 代码导航](#10-代码导航)


## 1. 引言：RL 在 LLM 时代的新角色

2017 年，PPO 在 OpenAI 的论文里横空出世，成为 Deep RL 的标配算法。2022 年，InstructGPT / ChatGPT 用 PPO 做 RLHF，让强化学习杀进了大语言模型（LLM）的后训练流程。到了 2024–2025 年，**DeepSeek-R1** 带火了 GRPO——一种不需要 Value Function 的 group-based 方法。与此同时，DPO 作为 RLHF 的 "轻量级替代"，也在偏好对齐领域占据一席之地。

这篇文章从最基础的 Policy Gradient 讲起，逐步推导到 PPO、GRPO、GSPO、RLOO、DPO，**把每个算法的 Loss 函数长什么样、Advantage 怎么算、跟别的算法差在哪** 一次性讲清楚。对应的完整代码实现见 [code.py](./code.py)。


## 2. 基础概念：一条 Trajectory 里有什么

在深入算法之前，先对齐几个核心概念。以下同时覆盖**传统 RL（如 CartPole）** 和 **LLM RLHF** 两种视角。

### 2.1 传统 RL 视角

一条 **轨迹 (Trajectory)** 是 agent 与环境交互产生的序列：

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T)
$$

| 符号 | 含义 | 例子 |
|---|---|---|
| $s_t$ | 第 $t$ 步的状态 | CartPole 的 (位置, 速度, 角度, 角速度) |
| $a_t$ | 第 $t$ 步的动作 | 向左/向右推 |
| $r_t$ | 即时奖励 | 杆子没倒给 +1 |
| $\pi_\theta(a \mid s)$ | 策略：在状态 $s$ 下选动作 $a$ 的概率 | 神经网络输出 softmax 后的分布 |
| $V(s)$ | 状态价值：从 $s$ 出发的期望累积奖励 | Value Network 的输出 |
| $A(s, a)$ | 优势函数：动作 $a$ 比平均水平好多少 | $A = Q - V$ 或 GAE 估计 |

核心目标：**最大化期望累积奖励** $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \gamma^t r_t\right]$。

### 2.2 LLM RLHF 视角

在 LLM 的 RLHF 中，"状态" 和 "动作" 的含义变了：

| 传统 RL | LLM RLHF 对应 |
|---|---|
| State $s_t$ | 当前已生成的 token 序列 $x_{<t}$（prompt + 已生成的 response tokens） |
| Action $a_t$ | 下一个要生成的 token $x_t$（从 vocabulary 中选一个） |
| Policy $\pi_\theta$ | 语言模型本身：$\pi_\theta(x_t \mid x_{<t})$ |
| Episode 结束 | 生成 EOS token 或达到 max_length |
| Reward $r$ | Reward Model 对整条 response 的打分（标量，在序列结束时给） |

关键区别：**LLM 的 reward 是 sequence-level 的**——只有整条回答生成完了才有一个标量分数，不像传统 RL 每步都有 $r_t$。这也是 GRPO 等方法出现的重要原因。

### 2.3 两个核心数据结构

对应代码中的两个 dataclass，是整个训练流程的数据载体：

**Trajectory**（PPO 用，传统 RL 环境）：
- `states: [T, state_dim]` — 每步的状态
- `actions: [T]` — 每步的动作
- `log_probs: [T]` — 旧策略下 $\log \pi_{\text{old}}(a_t \mid s_t)$
- `rewards: [T]` — 每步的即时奖励
- `values: [T]` — Value net 估计的 $V(s_t)$
- `advantages: [T]` — GAE 估计的 $A_t$
- `returns: [T]` — 折扣回报 $R_t$

**GroupTrajectory**（GRPO / GSPO / RLOO 用，LLM 场景）：
- `response_ids: [G, seq_len]` — prompt + response 的 token ids
- `action_masks: [G, seq_len]` — 标记哪些位置是 response token（1）vs prompt（0）
- `log_probs: [G, seq_len]` — 旧策略下每个 token 的 $\log \pi_{\text{old}}$
- `rewards: [G]` — Reward Model 对每条 response 的打分
- `ref_log_probs: [G, seq_len]` — 参考策略的 log prob（用于 KL 惩罚，可选）

```python
@dataclass
class Trajectory:
    """单条轨迹"""
    states: torch.Tensor          # [T, state_dim]
    actions: torch.Tensor         # [T]
    log_probs: torch.Tensor       # [T]  旧策略下的 log π(a|s)
    rewards: torch.Tensor         # [T]
    dones: torch.Tensor           # [T]
    values: Optional[torch.Tensor] = None  # [T]  V(s)（PPO 用）
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None


@dataclass
class GroupTrajectory:
    """GRPO / GSPO / RLOO 的一条 group 数据（同一 prompt 的 G 条回答）"""
    prompt_ids: torch.Tensor          # [G, seq_len]
    response_ids: torch.Tensor        # [G, seq_len]
    action_masks: torch.Tensor        # [G, seq_len]  1=response token, 0=prompt token
    log_probs: torch.Tensor           # [G, seq_len]  token-level log probs
    rewards: torch.Tensor             # [G]  scalar reward per response
    ref_log_probs: Optional[torch.Tensor] = None  # 参考策略 log probs（GRPO KL）
```


## 3. Policy Gradient：一切的起点

### 3.1 策略梯度定理

Policy Gradient 是所有 policy-based RL 算法的基石。核心思想直接而优雅：

> 好动作 → 提高它的概率；坏动作 → 降低它的概率。好坏由 **Advantage** $A_t$ 来衡量。

策略梯度定理给出了梯度的无偏估计：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A_t \right]
$$

对应的 Loss 函数（注意负号，因为要做梯度**下降**）：

$$
\mathcal{L}_{\text{PG}} = -\mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_t \log \pi_\theta(a_t \mid s_t) \cdot A_t \right]
$$

### 3.2 最简单的 Advantage：Reward-to-Go + Baseline

最朴素的做法：$A_t = R_t - b$，其中 $R_t = \sum_{k=t}^T \gamma^{k-t} r_k$ 是 return，$b$ 是 baseline（降低方差用）。

在 LLM 场景中只有序列级 reward，$R_t$ 退化为：response token 位置全用最终 reward $r$，prompt token 位置不计算 loss。Baseline 用同一个 group 内多条 response 的均值。

$$
\mathcal{L}_{\text{REINFORCE}} = -\frac{1}{G} \sum_{i=1}^G \left[ \frac{1}{|\text{resp}_i|} \sum_{t \in \text{resp}_i} \log \pi_\theta(a_t \mid s_t) \cdot (r_i - \bar{r}) \right]
$$

其中 $\bar{r} = \frac{1}{G}\sum_i r_i$ 是 group mean baseline，$|\text{resp}_i|$ 是第 $i$ 条 response 的 token 数（用于归一化）。

这已经是一个可以工作的 RLHF 算法了——但方差很大，训练不稳定。后面的 PPO / GRPO 都是在这个基础上做改进。

### 3.3 代码实现

```python
def reinforce_loss_fn(log_probs: torch.Tensor,         # [G, T]
                      old_log_probs: torch.Tensor,     # [G, T]
                      action_masks: torch.Tensor,      # [G, T]  1=response
                      rewards: torch.Tensor,           # [G]
                      config: RLConfig) -> torch.Tensor:
    """
    Vanilla REINFORCE:
      L = - 1/G Σ_i [ log π_θ(a_i|s_i) * (r_i - baseline) ]
      这里用 group mean 作为 baseline。
    """
    G = log_probs.shape[0]
    mask = action_masks.float()
    resp_lens = mask.sum(dim=-1) + 1e-8

    seq_log_probs = (log_probs * mask).sum(dim=-1) / resp_lens

    # baseline = group mean
    baseline = rewards.mean()
    advantages = rewards - baseline

    policy_loss = -(seq_log_probs * advantages).mean()
    return policy_loss
```


## 4. PPO：Proximal Policy Optimization

PPO 的核心洞察：**Policy Gradient 对步长极其敏感**。策略更新太大 → 新策略和旧策略差太远 → 采集的数据不再有效（off-policy 问题）→ 训练崩溃。

PPO 的解决方案：**Trust Region 思想**——用 clipping 限制每次更新的幅度。

### 4.1 Importance Sampling & Ratio

PPO 使用 **importance sampling** 来复用旧策略采集的数据：

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$

代码中就是一句 `ratio = torch.exp(log_probs - old_log_probs.detach())`。当 $r_t \approx 1$ 时，新旧策略接近；偏离 1 越远，数据越 "过期"。

### 4.2 Clipped Surrogate Objective

PPO 的 clipped objective 是整个算法的灵魂：

$$
\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t\left[ \min\left( r_t(\theta) \cdot A_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_t \right) \right]
$$

这个 `min` 的含义非常精巧：

| 情况 | $A_t > 0$（好动作） | $A_t < 0$（坏动作） |
|---|---|---|
| ratio 变大 | 说明策略在 "放大" 这个动作 | 说明策略在给坏动作更多概率 |
| min 的作用 | **clip 当上限**，阻止过度放大（$r_t$ 超过 $1+\varepsilon$ 后不增加收益） | **clip 当下限**，防止 gradient 因为是"坏动作"而反向放大（$r_t$ 低于 $1-\varepsilon$ 后不增加惩罚） |

本质上：**PPO 鼓励好动作，但每次最多放大 $1+\varepsilon$ 倍；抑制坏动作，但每次最多缩小到 $1-\varepsilon$ 倍。** $\varepsilon$ 通常取 0.1~0.2。

### 4.3 GAE：Generalized Advantage Estimation

PPO 还需要一个好的 $A_t$ 估计器。**GAE (Schulman et al., 2016)** 是标配：

$$
\begin{aligned}
\delta_t &= r_t + \gamma V(s_{t+1}) - V(s_t) \quad \text{(TD error)} \\
A_t^{\text{GAE}(\gamma, \lambda)} &= \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
\end{aligned}
$$

$\lambda$ 控制了 bias-variance trade-off：$\lambda=0$ 退化为 TD(0)（低方差高 bias），$\lambda=1$ 退化为 Monte Carlo（高方差低 bias）。

```python
def compute_gae(rewards: torch.Tensor,
                values: torch.Tensor,
                dones: torch.Tensor,
                gamma: float,
                gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation（PPO 用）。
    返回: advantages [T], returns [T]
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns
```

### 4.4 完整 Loss & Value Function

PPO 的完整 Loss 有三项：

$$
\mathcal{L}_{\text{PPO}} = -\mathcal{L}^{\text{CLIP}} + c_1 \cdot \underbrace{(V_\theta(s_t) - R_t)^2}_{\text{Value Loss}} - c_2 \cdot \underbrace{\mathbb{E}[\mathcal{H}(\pi_\theta(\cdot \mid s_t))]}_{\text{Entropy Bonus}}
$$

- **Value Loss**：让 Critic 学会准确估计 $V(s)$，用于 GAE 计算
- **Entropy Bonus**：鼓励策略保持一定的随机性，防止过早 collapse 到确定性策略

```python
def ppo_loss_fn(log_probs: torch.Tensor,
                old_log_probs: torch.Tensor,
                advantages: torch.Tensor,
                returns: torch.Tensor,
                values: torch.Tensor,
                entropy: torch.Tensor,
                config: RLConfig) -> torch.Tensor:
    """
    PPO Clipped Objective:
      L^CLIP = min( r_t(θ) * A_t,  clip(r_t(θ), 1-ε, 1+ε) * A_t )
      L^VF   = (V_θ(s) - R_t)^2
      L^ENT  = entropy bonus
      L      = -L^CLIP + c1 * L^VF - c2 * L^ENT
    """
    ratio = torch.exp(log_probs - old_log_probs.detach())

    # Clipped surrogate
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - config.clip_epsilon,
                               1 + config.clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus
    entropy_loss = entropy.mean()

    return (policy_loss
            + config.value_coef * value_loss
            - config.entropy_coef * entropy_loss)
```


## 5. GRPO：Group Relative Policy Optimization

GRPO（DeepSeekMath, 2024）被 DeepSeek-R1 带火的核心原因：**彻底干掉了 Value Function**。

### 5.1 为什么不再需要 Value Function

PPO 需要同时训练 4 个模型：Policy、Value、Reward、Reference。GRPO 砍掉了 Value Function——那 advantage 怎么算？

答案：**对同一个 prompt，采样 $G$ 条不同的 response，用组内相对排名来算 advantage。** 这比训一个 Value Network 更简单、更稳定，尤其在 LLM 场景中 reward 本身就是 sequence-level 的情况下。

```
PPO:   prompt → 1条response → Reward Model → 用 Value Net 算 advantage → 更新
GRPO:  prompt → G条response → Reward Model → 组内标准化算 advantage → 更新
```

### 5.2 Group-based Advantage

对同一个 prompt 的 $G$ 条 response $\{y_1, \dots, y_G\}$，Reward Model 给出分数 $\{r_1, \dots, r_G\}$。Advantage 就是 z-score 标准化：

$$
A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}
$$

```python
def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    GRPO / GSPO 的 group-based advantage：
    A_i = (r_i - mean(r_group)) / std(r_group)
    """
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    return (rewards - mean_r) / std_r
```

直觉：好于平均 → 正 advantage（增强这些 token），差于平均 → 负 advantage（抑制）。

### 5.3 Token-level Clipping + KL 惩罚

GRPO 在 loss 层面采用和 PPO 一样的 **clipped surrogate**，但是 **per-token** 的：

$$
\mathcal{L}_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^G \frac{1}{|\text{resp}_i|} \sum_{t \in \text{resp}_i} \min\left( r_{i,t}(\theta) \cdot A_i,\; \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_i \right)
$$

其中 $r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} \mid s_{i,t})}{\pi_{\text{old}}(a_{i,t} \mid s_{i,t})}$ 是**每个 token 的 ratio**。

注意一个细节：$A_i$ 是**序列级**的标量（同一个 response 的所有 token 共享同一个 advantage），但 ratio $r_{i,t}$ 和 clipping 是 **token 级**的。这意味着模型可以精细地决定每个 token 应该被增强还是抑制。

此外，GRPO 通常还加一个 **KL 散度惩罚** 来防止策略偏离参考模型太远：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GRPO}} + \beta \cdot D_{KL}(\pi_{\text{old}} \parallel \pi_{\text{ref}})
$$

$$
D_{KL} \approx \frac{1}{|\text{resp}|} \sum_{t \in \text{resp}} \big( \log \pi_{\text{old}}(a_t \mid s_t) - \log \pi_{\text{ref}}(a_t \mid s_t) \big)
$$

```python
def grpo_loss_fn(log_probs: torch.Tensor,         # [G, T]
                 old_log_probs: torch.Tensor,     # [G, T]
                 action_masks: torch.Tensor,      # [G, T]  1=response
                 rewards: torch.Tensor,           # [G]
                 ref_log_probs: Optional[torch.Tensor],  # [G, T]
                 config: RLConfig) -> torch.Tensor:
    """
    GRPO Objective (token-level):
      A_i = (r_i - mean(r_group)) / std(r_group)

      L_GRPO = - 1/G Σ_i [ 1/|resp_i| Σ_t
                 min( r_{i,t}(θ) * A_i,  clip(r_{i,t}(θ), 1-ε, 1+ε) * A_i )
                 - β * D_KL(π_θ || π_ref) ]
    """
    G = log_probs.shape[0]
    advantages = compute_group_advantages(rewards)  # [G]

    # 逐 token 的 ratio（只对 response token）
    ratio = torch.exp(log_probs - old_log_probs.detach())  # [G, T]

    # 按 response 维度扩展 advantage
    adv = advantages.unsqueeze(-1)  # [G, 1]

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - config.clip_epsilon,
                               1 + config.clip_epsilon) * adv
    clip_loss = torch.min(surr1, surr2)

    # 只在 response token 上计算 loss
    mask = action_masks.float()
    resp_lens = mask.sum(dim=-1, keepdim=True) + 1e-8  # [G, 1]
    per_resp_loss = (clip_loss * mask).sum(dim=-1) / resp_lens.squeeze(-1)
    policy_loss = -per_resp_loss.mean()

    # KL 惩罚（可选）
    kl_loss = torch.tensor(0.0)
    if ref_log_probs is not None:
        kl_per_token = old_log_probs - ref_log_probs  # [G, T]
        kl_loss = (kl_per_token * mask).sum() / mask.sum()

    return policy_loss + config.kl_coef * kl_loss
```


## 6. GSPO：Sequence-level 的改良

GSPO（Group-level Sequence-level Policy Optimization）和 GRPO 几乎一模一样，唯一的区别在 ratio 的粒度：

| | GRPO | GSPO |
|---|---|---|
| Ratio 计算 | **Token-level**：每个 token 单独算 $r_{i,t}$ | **Sequence-level**：整条 response 算一个 $r_i$ |
| Clip 范围 | $[1-\varepsilon, 1+\varepsilon]$ | $[1/\mu,\; \mu]$（对称于 1） |
| 梯度信号 | 每个 token 独立 | 整条 sequence 共享 |

GSPO 的序列级 ratio：

$$
r_i(\theta) = \exp\left( \frac{1}{|\text{resp}_i|}\sum_{t \in \text{resp}_i} \log \pi_\theta(a_t \mid s_t) - \frac{1}{|\text{resp}_i|}\sum_{t \in \text{resp}_i} \log \pi_{\text{old}}(a_t \mid s_t) \right)
$$

注意：这里用的是 **per-token 平均**（而不是求和），避免长 response 天然拥有更大的 ratio。

$$
\mathcal{L}_{\text{GSPO}} = -\frac{1}{G} \sum_{i=1}^G \min\left( r_i(\theta) \cdot A_i,\; \text{clip}(r_i(\theta), \frac{1}{\mu}, \mu) \cdot A_i \right) + \beta \cdot D_{KL}
$$

**为什么用 $1/\mu \sim \mu$ 而不是 $1-\varepsilon \sim 1+\varepsilon$？** 因为 sequence-level ratio 的变化范围比 token-level ratio 大得多（整条序列的 log prob 差异累积），用对称的乘法 clip（$1/\mu$ 到 $\mu$）更合理。$\mu$ 通常取 2~5，提供了比 $\varepsilon=0.2$ 宽得多的 trust region。

```python
def gspo_loss_fn(log_probs: torch.Tensor,
                 old_log_probs: torch.Tensor,
                 action_masks: torch.Tensor,
                 rewards: torch.Tensor,
                 ref_log_probs: Optional[torch.Tensor],
                 config: RLConfig) -> torch.Tensor:
    """
    GSPO Objective (sequence-level):
      与 GRPO 的区别在于 ratio 是序列级别的。

      r_i(θ) = exp( Σ_{t∈resp} log π_θ - Σ_{t∈resp} log π_old )  [per-token avg]

      A_i = (r_i - mean(r_group)) / std(r_group)

      L_GSPO = - 1/G Σ_i [ min( r_i(θ) * A_i,
                                 clip(r_i(θ), 1/μ, μ) * A_i )
                            - β * D_KL(π_θ || π_ref) ]
    """
    G = log_probs.shape[0]
    mask = action_masks.float()
    resp_lens = mask.sum(dim=-1) + 1e-8

    # 序列级别 log prob：对 response token 求 per-token 平均
    seq_log_probs = (log_probs * mask).sum(dim=-1) / resp_lens
    old_seq_log_probs = (old_log_probs * mask).sum(dim=-1) / resp_lens

    # 序列级别 ratio
    seq_ratio = torch.exp(seq_log_probs - old_seq_log_probs.detach())  # [G]

    advantages = compute_group_advantages(rewards)

    # GSPO 用 1/μ ~ μ 的对称 clip
    surr1 = seq_ratio * advantages
    surr2 = torch.clamp(seq_ratio, 1.0 / config.gspo_mu,
                                    config.gspo_mu) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # KL 惩罚
    kl_loss = torch.tensor(0.0)
    if ref_log_probs is not None:
        kl_per_token = old_log_probs - ref_log_probs
        kl_loss = (kl_per_token * mask).sum() / mask.sum()

    return policy_loss + config.kl_coef * kl_loss
```


## 7. RLOO：REINFORCE Leave-One-Out

RLOO 是 GRPO 的 "极简版"——**不要 clipping、不要 KL 惩罚**，只用 leave-one-out 做 baseline，纯粹的 REINFORCE 变体。

### 7.1 Leave-One-Out Baseline

对同一个 prompt 的 $G$ 条 response，第 $i$ 条的 baseline 是**其他 $G-1$ 条的平均 reward**：

$$
b_i = \frac{1}{G-1} \sum_{j \neq i} r_j
$$

$$
A_i = r_i - b_i
$$

和 GRPO 的 group-mean 标准化相比，RLOO 的 baseline 排除了自身，因此是 **无偏** 的（$\mathbb{E}[b_i] = \mathbb{E}[r]$ 且 $b_i$ 与 $r_i$ 条件独立）。

```python
def compute_rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    RLOO baseline：对每条响应 i，
    baseline = mean(rewards of other K-1 samples)。
    A_i = r_i - baseline_i
    """
    K = len(rewards)
    total = rewards.sum()
    advantages = torch.zeros(K)
    for i in range(K):
        baseline = (total - rewards[i]) / (K - 1)
        advantages[i] = rewards[i] - baseline
    return advantages
```

### 7.2 为什么 RLOO 依然有效

RLOO 的 loss 极其简单：

$$
\mathcal{L}_{\text{RLOO}} = -\frac{1}{G} \sum_{i=1}^G \left[ \frac{1}{|\text{resp}_i|} \sum_{t \in \text{resp}_i} \log \pi_\theta(a_t \mid s_t) \cdot A_i \right]
$$

没有 clipping、没有 KL、没有 value network。它之所以能工作：

1. **Leave-one-out baseline 显著降低了方差**（相比用 group mean）
2. **Policy Gradient 本身是 unbiased 的**，只要 advantage 估计够好
3. **足够多的 group samples（$G \ge 4$）** 提供了稳定的相对比较信号

RLOO 的缺点也很明显：没有 trust region，更新步长完全依赖 learning rate 控制，可能在 reward 分布极端时不稳定。

```python
def rloo_loss_fn(log_probs: torch.Tensor,         # [G, T]
                 old_log_probs: torch.Tensor,     # [G, T]
                 action_masks: torch.Tensor,      # [G, T]  1=response
                 rewards: torch.Tensor,           # [G]
                 config: RLConfig) -> torch.Tensor:
    """
    RLOO (REINFORCE Leave-One-Out):
      无需 value function。对每个 group 的 G 条响应：
        baseline_i = mean(r_{j≠i})
        A_i = r_i - baseline_i

      L_RLOO = - 1/G Σ_i [ 1/|resp_i| Σ_t log π_θ(a_t|s_t) * A_i ]
    """
    G = log_probs.shape[0]
    mask = action_masks.float()
    resp_lens = mask.sum(dim=-1) + 1e-8

    advantages = compute_rloo_advantages(rewards)  # [G]

    # 序列级 log prob（per-token 平均）
    seq_log_probs = (log_probs * mask).sum(dim=-1) / resp_lens  # [G]

    policy_loss = -(seq_log_probs * advantages).mean()
    return policy_loss
```


## 8. DPO：Direct Preference Optimization

DPO（Rafailov et al., 2023）走了一条完全不同的路：**根本不需要 Reward Model**。

### 8.1 从 RLHF 到 DPO

标准 RLHF 流程是三步曲：
1. SFT：监督微调
2. 训练 Reward Model（从人类偏好对）
3. 用 PPO/GRPO 优化 policy 来最大化 reward

DPO 直接跳过第 2、3 步，**在偏好数据上直接优化 policy**——把 policy 本身当作一个隐式的 reward 函数。

### 8.2 Bradley-Terry 模型 & DPO Loss

DPO 的数学基础是 Bradley-Terry 偏好模型：给定两个回答 $y_w$（chosen / 更好的）和 $y_l$（rejected / 更差的），人类偏好 $y_w > y_l$ 的概率为：

$$
P(y_w \succ y_l) = \sigma\big( r(y_w) - r(y_l) \big)
$$

DPO 的关键洞察：最优 policy $\pi^*$ 和 reward 函数有以下关系（来自 RLHF 目标函数的闭式解）：

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

代入 Bradley-Terry 后 $Z(x)$ 抵消掉，得到 DPO Loss：

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[ \log \sigma\left( \beta \cdot \left[ \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right] \right) \right]
$$

```python
def dpo_loss_fn(policy_chosen_logps: torch.Tensor,    # [B]
                policy_rejected_logps: torch.Tensor,  # [B]
                ref_chosen_logps: torch.Tensor,       # [B]
                ref_rejected_logps: torch.Tensor,     # [B]
                config: RLConfig) -> torch.Tensor:
    """
    DPO (Direct Preference Optimization):
      L_DPO = -log σ( β * [ (log π_θ(y_w) - log π_ref(y_w))
                           - (log π_θ(y_l) - log π_ref(y_l)) ] )

      其中 y_w 是 chosen（更好的）回答，y_l 是 rejected（更差的）回答。
    """
    policy_diff = policy_chosen_logps - policy_rejected_logps
    ref_diff = ref_chosen_logps - ref_rejected_logps

    logits = config.dpo_beta * (policy_diff - ref_diff)
    loss = -F.logsigmoid(logits).mean()
    return loss
```

### 8.3 DPO 的优缺点

**优点**：
- 不需要训练 Reward Model，流程简单
- 不需要在线采样，直接用静态偏好数据集训练
- 训练稳定（类似分类问题）

**缺点**：
- 依赖参考策略 $\pi_{\text{ref}}$ 的质量
- $\beta$ 的选择敏感：太大 → policy 不敢偏离 ref；太小 → policy 容易 collapse
- 无法利用 "prompt → N 条 response" 这种丰富的组内比较信号（而这些是 GRPO 的优势）


## 9. 六种算法全景对比

### 9.1 Loss 函数速查

| 算法 | Loss 形式（简化） |
|---|---|
| **REINFORCE** | $-\frac{1}{G}\sum_i \log\pi_i \cdot (r_i - \bar{r})$ |
| **PPO** | $-\min(r_t A_t,\; \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) A_t) + c_1(V-R)^2 - c_2 \mathcal{H}$ |
| **GRPO** | $-\frac{1}{G}\sum_i \frac{1}{\|\text{resp}_i\|}\sum_t \min(r_{i,t} A_i,\; \text{clip}(r_{i,t}) A_i) + \beta D_{KL}$ |
| **GSPO** | $-\frac{1}{G}\sum_i \min(r_i A_i,\; \text{clip}(r_i, 1/\mu, \mu) A_i) + \beta D_{KL}$ |
| **RLOO** | $-\frac{1}{G}\sum_i \frac{1}{\|\text{resp}_i\|}\sum_t \log\pi_{i,t} \cdot (r_i - \frac{1}{G-1}\sum_{j \neq i} r_j)$ |
| **DPO** | $-\log\sigma(\beta[(\log\frac{\pi_\theta^{w}}{\pi_{\text{ref}}^{w}}) - (\log\frac{\pi_\theta^{l}}{\pi_{\text{ref}}^{l}})])$ |

### 9.2 架构依赖对比

| 维度 | PPO | GRPO | GSPO | RLOO | DPO | REINFORCE |
|---|---|---|---|---|---|---|
| **需要 Value Net** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **需要 Reward Model** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **需要 Reference Model** | ❌ | 可选 | 可选 | ❌ | ✅ | ❌ |
| **需要偏好对数据** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Group 采样（一个 prompt 多条 response）** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌/✅ |
| **Clip 策略** | token 级 | token 级 | **序列级** | 无 | N/A | 无 |
| **在线采样** | 是 | 是 | 是 | 是 | **否** | 是 |

### 9.3 选择建议

```
你的场景：
├── 有偏好对数据 (chosen vs rejected)，不想训 Reward Model
│   └── → DPO
├── 传统 RL 环境（CartPole / Mujoco），每步有 reward
│   └── → PPO
├── LLM RLHF，已经有 Reward Model
│   ├── 追求稳定、成熟方案
│   │   └── → PPO（需要额外训练 Value Net）
│   ├── 不想训 Value Net，追求简单
│   │   ├── 最简方案
│   │   │   └── → RLOO
│   │   └── 需要 trust region 稳定训练
│   │       ├── 精细 token-level 控制
│   │       │   └── → GRPO
│   │       └── 粗粒度 sequence-level 控制
│   │           └── → GSPO
│   └── 最简 baseline
│       └── → REINFORCE
```


## 10. 代码导航

完整实现见 [code.py](./code.py)，代码结构如下：

```
算法文件 (code.py)
├── RLConfig              ← 统一超参（一行切算法）
├── Trajectory            ← PPO 的数据结构
├── GroupTrajectory       ← GRPO/GSPO/RLOO 的数据结构
├── PolicyValueNet        ← Actor-Critic 网络（PPO 用）
├── LLMPolicyNet          ← 类 LLM 策略网络（GRPO/GSPO/RLOO/DPO 用）
├── Advantage 估计器
│   ├── compute_gae              ← PPO 的 GAE
│   ├── compute_group_advantages ← GRPO/GSPO 的 group z-score
│   └── compute_rloo_advantages  ← RLOO 的 leave-one-out
├── 6 个 Loss 函数
│   ├── ppo_loss_fn
│   ├── grpo_loss_fn
│   ├── gspo_loss_fn
│   ├── rloo_loss_fn
│   ├── dpo_loss_fn
│   └── reinforce_loss_fn
├── UnifiedRLTrainer      ← 统一训练器（按 config.algorithm 路由）
└── Demo × 6              ← 每个算法的快速验证
```

核心用法 —— 切换算法只需要改一行配置：

```python
# 用 GRPO
config = RLConfig(algorithm="grpo", group_size=4, clip_epsilon=0.2)
trainer = UnifiedRLTrainer(config, vocab_size=50000)

# 切换到 DPO
config = RLConfig(algorithm="dpo", dpo_beta=0.1)
trainer = UnifiedRLTrainer(config, vocab_size=50000)

# 训练
trainer.train_step(groups=...)
# 或
trainer.train_step(chosen_ids=..., rejected_ids=..., ...)
```

所有算法共享同一个网络骨架和训练器，切换算法只影响 loss 函数的计算方式——这也体现了这些算法的本质差异全部凝聚在 **"怎么算 advantage、怎么算 ratio、怎么 clip"** 这三件事上。
