---
title: '这是一个测试'
publishDate: 2025-02-09
updatedDate: 2025-02-24
description: '和意为呢'
tags:
  - Example
  - 3D
language: 'English'
heroImage: { src: './u.jpg', color: '#D58388' }
---



## What is 3D Rendering?

Put simply, 3D rendering is the process of using a computer to generate a 2D image from a digital three-dimensional scene.

To generate an image, specific methodologies and special software and hardware are used. Therefore, we need to understand that 3D rendering is a process—the one that builds the image.

![alt text](./nikola-arsov-still-life-interior-design-vray-3ds-max-05-930px.jpg)

## Types of 3D rendering

We can create different types of rendered image; they can be realistic or non-realistic.

A realistic image could be an architectural interior that looks like a photograph, a product-design image such as a piece of furniture, or an automotive rendering of a car. On the other hand, we can create a non-realistic image such as an outline-type diagram or a cartoon-style image with a traditional 2D look. Technically, we can visualize anything we can imagine.

## How is 3D rendering used?

3D rendering is an essential technique for many industries including architecture, product design, advertising, video games and visual effects for film, TV and animation.

In design and architecture, renders allow creative people to communicate their ideas in a clear and transparent way. A render gives them the chance to evaluate their proposals, experiment with materials, conduct studies and contextualize their designs in the real world before they are built or manufactured.

For the media and entertainment industries, 3D rendering is fundamental to the creation of sequences and animations that tell stories, whether we’re watching an animated movie, a period drama, or an action sequence with explosions, ships from the future, exotic locales, or extraterrestrial creatures.

![alt text](./thanos-dd-single-image-004a.jpg)

Over the past few years, the evolution of computer graphics in these industries has replaced traditional techniques. For example, special effects are being replaced by visual effects, which means stunt people no longer risk their lives in car crashes.

In advertising, I would dare to say that 90% of automotive commercials are CG—or even more. In the architecture industry, many traditional techniques to create representations, such as scale models, have been replaced with photorealistic imagery to ensure we can see exactly how something will look once it’s built.

Accelerating processes, reducing costs and the demand for better quality results have helped technology evolve. Hardware is more powerful than ever and the switch to CG was inevitable.

## How is a 3D rendered image generated?

Two pieces of software, with different characteristics, are used to computer-generate images and animations: render engines and game engines. Render engines use a technique called ray tracing, while game engines use a technique called rasterization—and some engines mix both techniques, but we will talk about that later on.

行捏公式 $\alpha$. $a_{1}^{2}$



$$J_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \min \left( r_i(\theta) A_i, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) A_i \right) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right) \right]$$



## 这是代码

```python
import torch
import torch.nn.functional as F

def cal_grpo_loss(logits, old_logits, ref_logits, rewards, epsilon=0.2, beta=0.01):
    """
    计算 GRPO (Group Relative Policy Optimization) 的损失函数
    
    参数:
    - logits: 当前策略模型的输出 (Group_size, seq_len, vocab_size)
    - old_logits: 采样时旧策略模型的输出 (Group_size, seq_len, vocab_size)
    - ref_logits: 参考模型(如SFT模型)的输出 (Group_size, seq_len, vocab_size)
    - rewards: 组内每个回答的原始奖励得分 (Group_size,)
    - epsilon: PPO 裁剪超参数
    - beta: KL 散度惩罚系数
    """
    # -------------------------------------------------------------
    # 1. 核心：组内相对优势计算 (Group Relative Advantage)
    # -------------------------------------------------------------
    mean_reward = rewards.mean()
    std_reward = rewards.std() + 1e-8  # 防止除以 0
    # 组内归一化，得到相对优势 A_i
    advantages = (rewards - mean_reward) / std_reward  # 形状: (Group_size,)
    
    # -------------------------------------------------------------
    # 2. 计算重要性采样比率 r_i(theta)
    # -------------------------------------------------------------
    # 这里简化模拟，假设我们已经提取了对应 token 的 log 概率
    # 实际应用中需要根据 input_ids 进行 gather
    log_probs = F.log_softmax(logits, dim=-1).mean(dim=[1, 2])      # (Group_size,)
    old_log_probs = F.log_softmax(old_logits, dim=-1).mean(dim=[1, 2]) # (Group_size,)
    
    # 计算 r_i(theta) = pi_theta / pi_old
    ratio = torch.exp(log_probs - old_log_probs)
    
    # -------------------------------------------------------------
    # 3. 计算裁剪后的 PPO 目标
    # -------------------------------------------------------------
    # 将 advantages 维度对齐
    advantages = advantages.detach() # 梯度不回传给优势函数本身
    
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean() # 取负号用于梯度下降最小化
    
    # -------------------------------------------------------------
    # 4. 计算 KL 散度惩罚 (当前策略 vs 参考策略)
    # -------------------------------------------------------------
    log_probs_full = F.log_softmax(logits, dim=-1)
    ref_log_probs_full = F.log_softmax(ref_logits, dim=-1)
    
    # KL(pi_theta || pi_ref) = pi_theta * (log(pi_theta) - log(pi_ref))
    kl_div = F.kl_div(log_probs_full, ref_log_probs_full, log_target=True, reduction='batchmean')
    
    # -------------------------------------------------------------
    # 5. 总损失
    # -------------------------------------------------------------
    total_loss = policy_loss + beta * kl_div
    
    return total_loss, policy_loss.item(), kl_div.item()

# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟参数
    G = 4          # 组大小 (Group Size)
    S = 16         # 序列长度 (Sequence Length)
    V = 1000       # 词表大小 (Vocabulary Size)
    
    # 模拟模型输出的 Logits
    torch.manual_seed(42)
    logits = torch.randn(G, S, V, requires_grad=True)
    old_logits = torch.randn(G, S, V)
    ref_logits = torch.randn(G, S, V)
    
    # 模拟来自 Reward Model 的奖励评分 (比如 4 个回答的得分)
    rewards = torch.tensor([1.2, 0.5, 2.3, -0.8]) 
    
    # 计算损失
    loss, p_loss, kl = cal_grpo_loss(logits, old_logits, ref_logits, rewards)
    
    print(f"Total GRPO Loss: {loss.item():.4f}")
    print(f"Policy Loss:     {p_loss:.4f}")
    print(f"KL Penalty:      {kl:.4f}")
    
    # 反向传播
    loss.backward()
    print("Backward pass successful!")


```