---
title: 'PPO全解'
publishDate: 2026-06-30
updatedDate: 2026-06-30
description: ‘讲解PPO的各种内容'
tags:
  - RL
heroImage: { src: './image.png', color: '#fb0919' }
---


## PPO数学公式解读


### Importance Sampling


首先定义当前策略 $\pi_\theta$ 与旧策略 $\pi_{\theta_{old}}$ 在状态 $s_t$ 下采取动作 $a_t$ 的概率比率：$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$当 $r_t > 1$ 时，说明当前策略比旧策略更容易做出该动作。当 $r_t < 1$ 时，说明当前策略做该动作的概率降低了。

### Clipping Objective


### Adaptive KL Penality


### Generalized Advantage Estimation


### Loss Function







## TRL实现



## PPO常见疑惑解读


