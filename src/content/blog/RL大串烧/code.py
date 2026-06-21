"""
RL 大串烧 —— 统一策略优化框架（完整 LLM RL 流程版）
=================================================
支持的算法 / 目标函数：
  - PPO      (Proximal Policy Optimization)
  - GRPO     (Group Relative Policy Optimization)
  - GSPO     (Group-level Sequence-level Policy Optimization)
  - RLOO     (REINFORCE Leave-One-Out)
  - DPO      (Direct Preference Optimization)
  - REINFORCE (Vanilla Policy Gradient)

完整训练流程：从策略采样 → 获得 reward → 计算 advantage → 更新策略
所有算法共享同一个 Policy / Value 网络和训练骨架，仅通过切换 loss_fn 来区分。
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Callable
import numpy as np


# ═══════════════════════════════════════════════
# 0. 配置 & 数据容器
# ═══════════════════════════════════════════════

@dataclass
class RLConfig:
    """统一超参数"""
    # 算法选择
    algorithm: Literal["ppo", "grpo", "gspo", "rloo", "dpo", "reinforce"] = "ppo"

    # 网络
    hidden_dim: int = 128
    n_layers: int = 2

    # PPO / GRPO / GSPO 共用
    clip_epsilon: float = 0.2          # PPO clipping ε
    gspo_mu: float = 3.0               # GSPO sequence-level bound μ
    entropy_coef: float = 0.01         # 熵正则系数
    value_coef: float = 0.5            # value loss 权重（PPO 用）

    # GRPO / GSPO / RLOO 共用
    group_size: int = 4                # 每个 prompt 采样 G 条响应
    kl_coef: float = 0.04              # KL 散度惩罚系数（GRPO / GSPO）

    # DPO
    dpo_beta: float = 0.1              # DPO temperature β

    # 训练
    lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95           # GAE λ（PPO 用）
    epochs_per_batch: int = 4
    batch_size: int = 64


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
    """
    GRPO / GSPO / RLOO 的一条 group 数据（同一 prompt 的 G 条回答）。

    字段详解：
      prompt_ids:    prompt 的 token ids [G, L_prompt]
      response_ids:  prompt + response 拼接后的完整 token ids [G, L_full]
      action_masks:  标记哪些位置是 response token [G, L_full]（1=response, 0=prompt）
      log_probs:     旧策略（采样时）在每个位置的 log π_old(token_t | context) [G, L_full]
                     prompt 位置填 0，response 位置填真实采样 log_prob
      rewards:       Reward Model 对每条 response 的标量打分 [G]
      ref_log_probs: 参考策略的 log_probs（可选，用于 KL 惩罚）[G, L_full]
    """
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    action_masks: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    ref_log_probs: Optional[torch.Tensor] = None


# ═══════════════════════════════════════════════
# 1. 网络模块
# ═══════════════════════════════════════════════

def mlp(sizes: List[int], activation=nn.Tanh, output_activation=nn.Identity):
    """简单的 MLP 构造器"""
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class PolicyValueNet(nn.Module):
    """
    Actor-Critic 网络（PPO 用，传统 RL 环境）。
    - actor:  输出 action 的 logits
    - critic: 输出 state value V(s)
    """
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        super().__init__()
        self.shared = mlp([state_dim] + [config.hidden_dim] * config.n_layers)
        self.actor = nn.Linear(config.hidden_dim, action_dim)
        self.critic = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action(self, state: torch.Tensor):
        """采样动作并返回 action, log_prob, entropy, value"""
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class LLMPolicyNet(nn.Module):
    """
    类 LLM 的策略网络（GRPO / GSPO / RLOO / DPO / REINFORCE 用）。

    用一个 embedding + MLP 来模拟 token 级别的策略。
    实际 LLM 场景中这里会替换为 Transformer / MoE。

    核心方法：
      - forward(token_ids) → logits            [B, T, V]
      - get_log_probs(token_ids, shift=True) → [B, T-1]  当前策略的 token log prob
      - generate(prompt_ids, max_new_tokens)  → 自回归采样，返回 (full_ids, log_probs, masks)
    """

    def __init__(self, vocab_size: int, config: RLConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, config.hidden_dim)
        self.body = mlp([config.hidden_dim] + [config.hidden_dim] * config.n_layers)
        self.head = nn.Linear(config.hidden_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """返回 logits: [B, T, vocab_size]"""
        x = self.embedding(token_ids)
        x = self.body(x)
        return self.head(x)

    def get_log_probs(self, token_ids: torch.Tensor, shift: bool = True) -> torch.Tensor:
        """
        返回每个位置当前策略的 log π(token_{t+1} | token_{0:t})。

        当 shift=True 时：
          - logits 取 [:, :-1, :]（去掉最后一步预测）
          - targets 取 [:, 1:]（去掉第一个 token）
          返回 [B, T-1]

        当 shift=False 时：返回 [B, T]
        """
        logits = self.forward(token_ids)  # [B, T, V]
        if shift:
            logits = logits[:, :-1, :]    # 前 T-1 步的预测
            targets = token_ids[:, 1:]    # 预测目标是后 T-1 个 token
        else:
            targets = token_ids
        return Categorical(logits=logits).log_prob(targets)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,        # [B, L_prompt]
        max_new_tokens: int,
        temperature: float = 1.0,
    ):
        """
        自回归生成 response token，并记录每个生成 token 的 log π_old。

        这是 LLM RL 流程的第一步：用「当前策略」采样一批 response，
        同时冻结 log_prob 作为 old_log_probs（后续 loss 中的 anchor）。

        返回:
          full_ids:     [B, L_prompt + max_new_tokens]  完整的 prompt + response
          log_probs:    [B, L_prompt + max_new_tokens]  每个位置的采样 log_prob
                        （prompt 位置填 0，response 位置填真实值）
          action_masks: [B, L_prompt + max_new_tokens]  1=response token, 0=prompt token
        """
        B, L_prompt = prompt_ids.shape
        device = prompt_ids.device

        generated = prompt_ids.clone()      # [B, cur_len]
        resp_log_probs_list: List[torch.Tensor] = []

        for _ in range(max_new_tokens):
            logits = self.forward(generated)                 # [B, cur_len, V]
            next_logits = logits[:, -1, :] / temperature      # [B, V] 只取最后一个位置
            dist = Categorical(logits=next_logits)
            next_token = dist.sample()                        # [B]
            log_prob = dist.log_prob(next_token)              # [B]

            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            resp_log_probs_list.append(log_prob)

        # ---------- 组装成 GroupTrajectory 所需的格式 ----------
        T = L_prompt + max_new_tokens
        log_probs_full = torch.zeros(B, T, device=device)
        action_masks = torch.zeros(B, T, device=device)

        resp_log_probs = torch.stack(resp_log_probs_list, dim=-1)  # [B, max_new_tokens]
        log_probs_full[:, L_prompt:] = resp_log_probs
        action_masks[:, L_prompt:] = 1.0

        return generated, log_probs_full, action_masks


# ═══════════════════════════════════════════════
# 2. Advantage 估计器
# ═══════════════════════════════════════════════

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    GRPO / GSPO 的 group-based advantage：
    A_i = (r_i - mean(r_group)) / std(r_group)
    """
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    return (rewards - mean_r) / std_r


def compute_rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    RLOO baseline：对每条响应 i，
    baseline_i = mean(rewards of other K-1 samples)
    A_i = r_i - baseline_i
    """
    K = len(rewards)
    total = rewards.sum()
    advantages = torch.zeros(K)
    for i in range(K):
        baseline = (total - rewards[i]) / (K - 1)
        advantages[i] = rewards[i] - baseline
    return advantages


# ═══════════════════════════════════════════════
# 3. Reward 函数（规则型，用于 demo 模拟 Reward Model）
# ═══════════════════════════════════════════════

def make_target_match_reward(target_ids: List[int]):
    """
    构造一个「目标匹配」奖励函数。

    奖励 = response token 与 target 的匹配率 (0~1)。
    模型学会输出 target sequence 就能拿到满分 1.0。

    用法:
      reward_fn = make_target_match_reward([7, 7, 7, 7, 7])
      scores = reward_fn(response_ids)  # [B]

    真实 RLHF 中这里替换为 Reward Model 的 forward。
    """
    target = torch.tensor(target_ids)

    def reward_fn(response_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            response_ids: [B, L]  只有 response 部分的 token ids（不含 prompt）
        Returns:
            scores: [B]  每条 response 的得分 (0~1)
        """
        L = min(response_ids.shape[1], len(target))
        t = target[:L].unsqueeze(0).to(response_ids.device)  # [1, L]
        matches = (response_ids[:, :L] == t).float()          # [B, L]
        return matches.mean(dim=-1)                            # [B]

    return reward_fn


def make_length_reward(target_len: int, tolerance: int = 3):
    """
    构造一个「长度控制」奖励函数。

    奖励 = 1.0 - |实际长度 - 目标长度| / tolerance（clamp 到 [0, 1]）。

    用法:
      reward_fn = make_length_reward(10, tolerance=3)
    """

    def reward_fn(response_ids: torch.Tensor) -> torch.Tensor:
        actual_len = response_ids.shape[1]
        penalty = abs(actual_len - target_len) / tolerance
        return torch.full((response_ids.shape[0],), max(0.0, 1.0 - penalty))

    return reward_fn


def make_combined_reward(reward_fns: List[Callable], weights: List[float] = None):
    """
    组合多个 reward 函数（加权求和）。

    用法:
      combined = make_combined_reward([target_reward, length_reward], [0.8, 0.2])
      scores = combined(response_ids)  # [B]
    """
    if weights is None:
        weights = [1.0 / len(reward_fns)] * len(reward_fns)

    def reward_fn(response_ids: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(response_ids.shape[0])
        for fn, w in zip(reward_fns, weights):
            total = total + w * fn(response_ids)
        return total

    return reward_fn


# ═══════════════════════════════════════════════
# 4. 损失函数 —— 所有算法统一接口
# ═══════════════════════════════════════════════

def ppo_loss_fn(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    entropy: torch.Tensor,
    config: RLConfig,
) -> torch.Tensor:
    """
    PPO Clipped Objective:
      L^CLIP = min( r_t(θ) * A_t,  clip(r_t(θ), 1-ε, 1+ε) * A_t )
      L^VF   = (V_θ(s) - R_t)^2
      L^ENT  = entropy bonus
      L      = -L^CLIP + c1 * L^VF - c2 * L^ENT
    """
    ratio = torch.exp(log_probs - old_log_probs.detach())

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(values, returns)
    entropy_loss = entropy.mean()

    return policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_loss


def grpo_loss_fn(
    log_probs: torch.Tensor,              # [G, T]  当前策略的 log π_θ
    old_log_probs: torch.Tensor,          # [G, T]  采样时冻结策略的 log π_old
    action_masks: torch.Tensor,           # [G, T]  1=response token
    rewards: torch.Tensor,                # [G]
    ref_log_probs: Optional[torch.Tensor],# [G, T]  参考策略 log π_ref
    config: RLConfig,
) -> torch.Tensor:
    """
    GRPO Objective (token-level clipping):

      A_i = (r_i - mean(r_group)) / std(r_group)

      L_GRPO = - 1/G Σ_i [ 1/|resp_i| Σ_t
                 min( r_{i,t}(θ) * A_i,  clip(r_{i,t}(θ), 1-ε, 1+ε) * A_i )
                 - β * D_KL(π_old || π_ref) ]
    """
    G = log_probs.shape[0]
    advantages = compute_group_advantages(rewards)  # [G]

    ratio = torch.exp(log_probs - old_log_probs.detach())  # [G, T]

    adv = advantages.unsqueeze(-1)  # [G, 1]

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * adv
    clip_loss = torch.min(surr1, surr2)

    mask = action_masks.float()
    resp_lens = mask.sum(dim=-1, keepdim=True) + 1e-8
    per_resp_loss = (clip_loss * mask).sum(dim=-1) / resp_lens.squeeze(-1)
    policy_loss = -per_resp_loss.mean()

    kl_loss = torch.tensor(0.0)
    if ref_log_probs is not None:
        kl_per_token = old_log_probs - ref_log_probs
        kl_loss = (kl_per_token * mask).sum() / mask.sum()

    return policy_loss + config.kl_coef * kl_loss


def gspo_loss_fn(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    action_masks: torch.Tensor,
    rewards: torch.Tensor,
    ref_log_probs: Optional[torch.Tensor],
    config: RLConfig,
) -> torch.Tensor:
    """
    GSPO Objective (sequence-level ratio):

      r_i(θ) = exp( avg_{t∈resp} log π_θ - avg_{t∈resp} log π_old )

      A_i = (r_i - mean(r_group)) / std(r_group)

      L_GSPO = - 1/G Σ_i [ min( r_i(θ) * A_i,  clip(r_i(θ), 1/μ, μ) * A_i )
                            - β * D_KL(π_old || π_ref) ]
    """
    G = log_probs.shape[0]
    mask = action_masks.float()
    resp_lens = mask.sum(dim=-1) + 1e-8

    # 序列级 log prob（per-token 平均）
    seq_log_probs = (log_probs * mask).sum(dim=-1) / resp_lens      # [G]
    old_seq_log_probs = (old_log_probs * mask).sum(dim=-1) / resp_lens

    seq_ratio = torch.exp(seq_log_probs - old_seq_log_probs.detach())  # [G]

    advantages = compute_group_advantages(rewards)

    surr1 = seq_ratio * advantages
    surr2 = torch.clamp(seq_ratio, 1.0 / config.gspo_mu, config.gspo_mu) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    kl_loss = torch.tensor(0.0)
    if ref_log_probs is not None:
        kl_per_token = old_log_probs - ref_log_probs
        kl_loss = (kl_per_token * mask).sum() / mask.sum()

    return policy_loss + config.kl_coef * kl_loss


def rloo_loss_fn(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    action_masks: torch.Tensor,
    rewards: torch.Tensor,
    config: RLConfig,
) -> torch.Tensor:
    """
    RLOO (REINFORCE Leave-One-Out):

      baseline_i = mean(r_{j≠i})
      A_i = r_i - baseline_i

      L_RLOO = - 1/G Σ_i [ 1/|resp_i| Σ_t log π_θ(a_t|s_t) * A_i ]
    """
    G = log_probs.shape[0]
    mask = action_masks.float()
    resp_lens = mask.sum(dim=-1) + 1e-8

    advantages = compute_rloo_advantages(rewards)  # [G]

    seq_log_probs = (log_probs * mask).sum(dim=-1) / resp_lens  # [G]

    policy_loss = -(seq_log_probs * advantages).mean()
    return policy_loss


def dpo_loss_fn(
    policy_chosen_logps: torch.Tensor,     # [B]
    policy_rejected_logps: torch.Tensor,   # [B]
    ref_chosen_logps: torch.Tensor,        # [B]
    ref_rejected_logps: torch.Tensor,      # [B]
    config: RLConfig,
) -> torch.Tensor:
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


def reinforce_loss_fn(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    action_masks: torch.Tensor,
    rewards: torch.Tensor,
    config: RLConfig,
) -> torch.Tensor:
    """
    Vanilla REINFORCE:

      L = - 1/G Σ_i [ log π_θ(a_i|s_i) * (r_i - baseline) ]
      这里用 group mean 作为 baseline。
    """
    G = log_probs.shape[0]
    mask = action_masks.float()
    resp_lens = mask.sum(dim=-1) + 1e-8

    seq_log_probs = (log_probs * mask).sum(dim=-1) / resp_lens

    baseline = rewards.mean()
    advantages = rewards - baseline

    policy_loss = -(seq_log_probs * advantages).mean()
    return policy_loss


# ═══════════════════════════════════════════════
# 5. 统一 Trainer
# ═══════════════════════════════════════════════

class UnifiedRLTrainer:
    """
    统一的 RL 训练器。
    通过 config.algorithm 自动选择对应的 loss 函数。
    """

    def __init__(
        self,
        config: RLConfig,
        state_dim: int = None,
        action_dim: int = None,
        vocab_size: int = None,
    ):
        self.config = config
        self.algorithm = config.algorithm

        # 根据算法选择网络类型
        if self.algorithm in ("grpo", "gspo", "rloo", "dpo", "reinforce"):
            assert vocab_size is not None, "LLM-based algorithms need vocab_size"
            self.policy_net = LLMPolicyNet(vocab_size, config)
            self.value_net = None
            self.is_llm = True
        else:
            assert state_dim is not None and action_dim is not None
            self.policy_net = PolicyValueNet(state_dim, action_dim, config)
            self.value_net = None  # 包含在 PolicyValueNet 中
            self.is_llm = False

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr)

        # Loss 函数注册表
        self.loss_registry = {
            "ppo":       self._train_step_ppo,
            "grpo":      self._train_step_grpo,
            "gspo":      self._train_step_gspo,
            "rloo":      self._train_step_rloo,
            "dpo":       self._train_step_dpo,
            "reinforce": self._train_step_reinforce,
        }

    # ═══════════════════════════════════════
    # PPO 训练步骤（传统 RL 环境）
    # ═══════════════════════════════════════

    def _train_step_ppo(self, trajectories: List[Trajectory]) -> dict:
        config = self.config

        all_states = torch.cat([t.states for t in trajectories])
        all_actions = torch.cat([t.actions for t in trajectories])
        all_old_logps = torch.cat([t.log_probs for t in trajectories])
        all_advantages = torch.cat([t.advantages for t in trajectories])
        all_returns = torch.cat([t.returns for t in trajectories])

        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        N = len(all_states)
        total_loss = 0.0

        for _ in range(config.epochs_per_batch):
            indices = torch.randperm(N)
            for start in range(0, N, config.batch_size):
                idx = indices[start : start + config.batch_size]

                logits, values = self.policy_net(all_states[idx])
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(all_actions[idx])
                entropy = dist.entropy()

                loss = ppo_loss_fn(
                    log_probs=log_probs,
                    old_log_probs=all_old_logps[idx],
                    advantages=all_advantages[idx],
                    returns=all_returns[idx],
                    values=values,
                    entropy=entropy,
                    config=config,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        return {"loss": total_loss}

    # ═══════════════════════════════════════
    # GRPO 训练步骤
    # ═══════════════════════════════════════

    def _train_step_grpo(self, groups: List[GroupTrajectory]) -> dict:
        config = self.config
        total_loss = 0.0
        n_updates = 0

        for group in groups:
            # 对齐 get_log_probs(shift=True) 的 [G, T-1] 维度
            old_log_probs = group.log_probs[:, 1:]
            action_masks = group.action_masks[:, 1:]
            rewards = group.rewards
            ref_log_probs = (
                group.ref_log_probs[:, 1:] if group.ref_log_probs is not None else None
            )

            for _ in range(config.epochs_per_batch):
                log_probs = self.policy_net.get_log_probs(group.response_ids)
                loss = grpo_loss_fn(
                    log_probs=log_probs,
                    old_log_probs=old_log_probs,
                    action_masks=action_masks,
                    rewards=rewards,
                    ref_log_probs=ref_log_probs,
                    config=config,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_updates += 1

        return {"loss": total_loss / max(n_updates, 1)}

    # ═══════════════════════════════════════
    # GSPO 训练步骤
    # ═══════════════════════════════════════

    def _train_step_gspo(self, groups: List[GroupTrajectory]) -> dict:
        config = self.config
        total_loss = 0.0
        n_updates = 0

        for group in groups:
            old_log_probs = group.log_probs[:, 1:]
            action_masks = group.action_masks[:, 1:]
            rewards = group.rewards
            ref_log_probs = (
                group.ref_log_probs[:, 1:] if group.ref_log_probs is not None else None
            )

            for _ in range(config.epochs_per_batch):
                log_probs = self.policy_net.get_log_probs(group.response_ids)
                loss = gspo_loss_fn(
                    log_probs=log_probs,
                    old_log_probs=old_log_probs,
                    action_masks=action_masks,
                    rewards=rewards,
                    ref_log_probs=ref_log_probs,
                    config=config,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_updates += 1

        return {"loss": total_loss / max(n_updates, 1)}

    # ═══════════════════════════════════════
    # RLOO 训练步骤
    # ═══════════════════════════════════════

    def _train_step_rloo(self, groups: List[GroupTrajectory]) -> dict:
        config = self.config
        total_loss = 0.0
        n_updates = 0

        for group in groups:
            old_log_probs = group.log_probs[:, 1:]
            action_masks = group.action_masks[:, 1:]
            rewards = group.rewards

            for _ in range(config.epochs_per_batch):
                log_probs = self.policy_net.get_log_probs(group.response_ids)
                loss = rloo_loss_fn(
                    log_probs=log_probs,
                    old_log_probs=old_log_probs,
                    action_masks=action_masks,
                    rewards=rewards,
                    config=config,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_updates += 1

        return {"loss": total_loss / max(n_updates, 1)}

    # ═══════════════════════════════════════
    # DPO 训练步骤
    # ═══════════════════════════════════════

    def _train_step_dpo(
        self,
        chosen_ids: torch.Tensor,          # [B, L]
        rejected_ids: torch.Tensor,        # [B, L]
        ref_chosen_logps: torch.Tensor,    # [B]
        ref_rejected_logps: torch.Tensor,  # [B]
        chosen_mask: torch.Tensor = None,  # [B, L]
        rejected_mask: torch.Tensor = None,
    ) -> dict:
        config = self.config
        total_loss = 0.0
        n_updates = 0

        B = chosen_ids.shape[0]
        for start in range(0, B, config.batch_size):
            end = min(start + config.batch_size, B)
            c_ids = chosen_ids[start:end]
            r_ids = rejected_ids[start:end]

            c_log_probs = self.policy_net.get_log_probs(c_ids)
            r_log_probs = self.policy_net.get_log_probs(r_ids)

            if chosen_mask is not None:
                c_mask = chosen_mask[start:end, 1:]
                r_mask = rejected_mask[start:end, 1:]
                c_log_probs = (c_log_probs * c_mask).sum(-1) / (c_mask.sum(-1) + 1e-8)
                r_log_probs = (r_log_probs * r_mask).sum(-1) / (r_mask.sum(-1) + 1e-8)
            else:
                c_log_probs = c_log_probs.sum(-1)
                r_log_probs = r_log_probs.sum(-1)

            ref_c = ref_chosen_logps[start:end]
            ref_r = ref_rejected_logps[start:end]

            loss = dpo_loss_fn(c_log_probs, r_log_probs, ref_c, ref_r, config)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_updates += 1

        return {"loss": total_loss / max(n_updates, 1)}

    # ═══════════════════════════════════════
    # REINFORCE 训练步骤
    # ═══════════════════════════════════════

    def _train_step_reinforce(self, groups: List[GroupTrajectory]) -> dict:
        config = self.config
        total_loss = 0.0
        n_updates = 0

        for group in groups:
            old_log_probs = group.log_probs[:, 1:]
            action_masks = group.action_masks[:, 1:]
            rewards = group.rewards

            for _ in range(config.epochs_per_batch):
                log_probs = self.policy_net.get_log_probs(group.response_ids)
                loss = reinforce_loss_fn(
                    log_probs=log_probs,
                    old_log_probs=old_log_probs,
                    action_masks=action_masks,
                    rewards=rewards,
                    config=config,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_updates += 1

        return {"loss": total_loss / max(n_updates, 1)}

    # ═══════════════════════════════════════
    # 统一入口
    # ═══════════════════════════════════════

    def train_step(self, **kwargs) -> dict:
        """根据 self.algorithm 自动路由到对应训练步骤"""
        train_fn = self.loss_registry.get(self.algorithm)
        if train_fn is None:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        return train_fn(**kwargs)


# ═══════════════════════════════════════════════
# 6. 演示 / 示例（完整 RL 流程版）
# ═══════════════════════════════════════════════
#
# 每个 LLM demo 的流程：
#   for round in range(N_ROUNDS):
#       ❶ 用当前 policy 对每个 prompt 自回归采样 G 条 response，记录 log_probs
#       ❷ 用 reward 函数给每条 response 打分
#       ❸ 打包成 GroupTrajectory
#       ❹ trainer.train_step(groups=...)  执行 epochs_per_batch 次梯度更新
#       ❺ 打印 avg_reward, loss
#
# 这完整模拟了 LLM RLHF 的核心循环：
#   Policy → 采样 → Reward Model → 打包数据 → 策略更新 → 下一轮
# ═══════════════════════════════════════════════

# ── 公共参数 ──
DEMO_ROUNDS = 10          # 采样 + 训练 的轮数
DEMO_PROMPTS = 4          # 每轮的 prompt 数量
DEMO_RESP_LEN = 10        # 每条 response 生成多少个 token
DEMO_SEED = 42            # 固定随机种子，保证可复现


def _demo_header(name: str):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")


def _print_round(rnd: int, avg_reward: float, loss: float, extras: dict = None):
    parts = [f"  Round {rnd:2d} | avg_reward={avg_reward:+.4f} | loss={loss:.4f}"]
    if extras:
        for k, v in extras.items():
            parts.append(f" | {k}={v:.4f}")
    print("".join(parts))


# ─────────────────────────────────────
# 6.1  PPO（传统 RL 环境 — CartPole 风格）
# ─────────────────────────────────────

def demo_ppo():
    """PPO 在 CartPole-like 环境下的演示"""
    _demo_header("🔵 PPO Demo (传统 RL 环境)")

    state_dim = 4
    action_dim = 2
    config = RLConfig(algorithm="ppo", hidden_dim=64, n_layers=2,
                      clip_epsilon=0.2, lr=3e-4, epochs_per_batch=4)
    trainer = UnifiedRLTrainer(config, state_dim=state_dim, action_dim=action_dim)

    # 模拟收集轨迹（CartPole 环境下由 agent 与环境交互产生）
    trajectories = []
    for _ in range(8):
        T_ep = np.random.randint(10, 50)
        states = torch.randn(T_ep, state_dim)
        values = torch.randn(T_ep) * 0.5
        rewards = torch.randn(T_ep) * 0.1
        dones = torch.zeros(T_ep)
        dones[-1] = 1.0

        advantages, returns = compute_gae(rewards, values, dones, config.gamma, config.gae_lambda)

        actions = torch.randint(0, action_dim, (T_ep,))
        traj = Trajectory(
            states=states,
            actions=actions,
            log_probs=torch.log(torch.ones(T_ep) / action_dim),
            rewards=rewards,
            dones=dones,
            values=values,
            advantages=advantages,
            returns=returns,
        )
        trajectories.append(traj)

    for epoch in range(5):
        metrics = trainer.train_step(trajectories=trajectories)
        print(f"  Epoch {epoch}: loss={metrics['loss']:.4f}")

    print("  ✅ PPO 演示完成（传统 RL 每条 trajectory 包含 per-step reward）")


# ─────────────────────────────────────
# 6.2  GRPO（完整 LLM RL 流程）
# ─────────────────────────────────────

def demo_grpo():
    _demo_header("🟢 GRPO — Group Relative Policy Optimization")

    torch.manual_seed(DEMO_SEED)

    vocab_size = 100
    target = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]   # 最优回答：10 个 token 全是 7
    reward_fn = make_target_match_reward(target)

    config = RLConfig(algorithm="grpo", hidden_dim=64, n_layers=1,
                      clip_epsilon=0.2, group_size=4, kl_coef=0.04,
                      lr=5e-3, epochs_per_batch=2)
    trainer = UnifiedRLTrainer(config, vocab_size=vocab_size)

    # 固定的 prompt（每轮复用）
    prompts = torch.randint(0, vocab_size, (DEMO_PROMPTS, 5))

    for rnd in range(1, DEMO_ROUNDS + 1):
        groups = []

        for p in range(DEMO_PROMPTS):
            prompt = prompts[p:p+1]  # [1, L_prompt]
            prompt_g = prompt.expand(config.group_size, -1)  # [G, L_prompt]

            # ❶ 采样：用当前策略对同一个 prompt 生成 G 条 response
            full_ids, log_probs, masks = trainer.policy_net.generate(
                prompt_g, max_new_tokens=DEMO_RESP_LEN, temperature=1.0
            )

            # ❷ 奖励：提取 response 部分，用 reward 函数打分
            resp_only = full_ids[:, 5:]  # [G, L_resp]  去掉 prompt 前缀
            rewards = reward_fn(resp_only)

            # ❸ 打包
            groups.append(GroupTrajectory(
                prompt_ids=full_ids[:, :5],
                response_ids=full_ids,
                action_masks=masks,
                log_probs=log_probs,          # ← 真实的采样 log_probs！
                rewards=rewards,
            ))

        # ❹ 训练
        metrics = trainer.train_step(groups=groups)
        avg_r = sum(g.rewards.mean().item() for g in groups) / len(groups)
        _print_round(rnd, avg_r, metrics["loss"])

    print("  ✅ GRPO 完整流程演示完成（group z-score advantage + token-level clip）")


# ─────────────────────────────────────
# 6.3  GSPO（序列级 ratio）
# ─────────────────────────────────────

def demo_gspo():
    _demo_header("🟡 GSPO — Group-level Sequence-level Policy Optimization")

    torch.manual_seed(DEMO_SEED)

    vocab_size = 100
    target = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    reward_fn = make_target_match_reward(target)

    config = RLConfig(algorithm="gspo", hidden_dim=64, n_layers=1,
                      gspo_mu=3.0, group_size=4, kl_coef=0.04,
                      lr=5e-3, epochs_per_batch=2)
    trainer = UnifiedRLTrainer(config, vocab_size=vocab_size)

    prompts = torch.randint(0, vocab_size, (DEMO_PROMPTS, 5))

    for rnd in range(1, DEMO_ROUNDS + 1):
        groups = []

        for p in range(DEMO_PROMPTS):
            prompt = prompts[p:p+1]
            prompt_g = prompt.expand(config.group_size, -1)

            full_ids, log_probs, masks = trainer.policy_net.generate(
                prompt_g, max_new_tokens=DEMO_RESP_LEN, temperature=1.0
            )

            resp_only = full_ids[:, 5:]
            rewards = reward_fn(resp_only)

            groups.append(GroupTrajectory(
                prompt_ids=full_ids[:, :5],
                response_ids=full_ids,
                action_masks=masks,
                log_probs=log_probs,
                rewards=rewards,
            ))

        metrics = trainer.train_step(groups=groups)
        avg_r = sum(g.rewards.mean().item() for g in groups) / len(groups)
        _print_round(rnd, avg_r, metrics["loss"])

    print("  ✅ GSPO 完整流程演示完成（sequence-level ratio + 1/μ~μ clip）")


# ─────────────────────────────────────
# 6.4  RLOO（Leave-One-Out baseline）
# ─────────────────────────────────────

def demo_rloo():
    _demo_header("🟣 RLOO — REINFORCE Leave-One-Out")

    torch.manual_seed(DEMO_SEED)

    vocab_size = 100
    target = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    reward_fn = make_target_match_reward(target)

    config = RLConfig(algorithm="rloo", hidden_dim=64, n_layers=1,
                      group_size=4, lr=5e-3, epochs_per_batch=2)
    trainer = UnifiedRLTrainer(config, vocab_size=vocab_size)

    prompts = torch.randint(0, vocab_size, (DEMO_PROMPTS, 5))

    for rnd in range(1, DEMO_ROUNDS + 1):
        groups = []

        for p in range(DEMO_PROMPTS):
            prompt = prompts[p:p+1]
            prompt_g = prompt.expand(config.group_size, -1)

            full_ids, log_probs, masks = trainer.policy_net.generate(
                prompt_g, max_new_tokens=DEMO_RESP_LEN, temperature=1.0
            )

            resp_only = full_ids[:, 5:]
            rewards = reward_fn(resp_only)

            groups.append(GroupTrajectory(
                prompt_ids=full_ids[:, :5],
                response_ids=full_ids,
                action_masks=masks,
                log_probs=log_probs,
                rewards=rewards,
            ))

        metrics = trainer.train_step(groups=groups)
        avg_r = sum(g.rewards.mean().item() for g in groups) / len(groups)
        _print_round(rnd, avg_r, metrics["loss"])

    print("  ✅ RLOO 完整流程演示完成（leave-one-out baseline，无 clip 无 KL）")


# ─────────────────────────────────────
# 6.5  REINFORCE（Vanilla Policy Gradient）
# ─────────────────────────────────────

def demo_reinforce():
    _demo_header("⚪ Vanilla REINFORCE — 最朴素的 Policy Gradient")

    torch.manual_seed(DEMO_SEED)

    vocab_size = 100
    target = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    reward_fn = make_target_match_reward(target)

    config = RLConfig(algorithm="reinforce", hidden_dim=64, n_layers=1,
                      group_size=4, lr=5e-3, epochs_per_batch=2)
    trainer = UnifiedRLTrainer(config, vocab_size=vocab_size)

    prompts = torch.randint(0, vocab_size, (DEMO_PROMPTS, 5))

    for rnd in range(1, DEMO_ROUNDS + 1):
        groups = []

        for p in range(DEMO_PROMPTS):
            prompt = prompts[p:p+1]
            prompt_g = prompt.expand(config.group_size, -1)

            full_ids, log_probs, masks = trainer.policy_net.generate(
                prompt_g, max_new_tokens=DEMO_RESP_LEN, temperature=1.0
            )

            resp_only = full_ids[:, 5:]
            rewards = reward_fn(resp_only)

            groups.append(GroupTrajectory(
                prompt_ids=full_ids[:, :5],
                response_ids=full_ids,
                action_masks=masks,
                log_probs=log_probs,
                rewards=rewards,
            ))

        metrics = trainer.train_step(groups=groups)
        avg_r = sum(g.rewards.mean().item() for g in groups) / len(groups)
        _print_round(rnd, avg_r, metrics["loss"])

    print("  ✅ REINFORCE 完整流程演示完成（group-mean baseline，无 clip）")


# ─────────────────────────────────────
# 6.6  DPO（Direct Preference Optimization）
# ─────────────────────────────────────

def demo_dpo():
    _demo_header("🔴 DPO — Direct Preference Optimization")

    torch.manual_seed(DEMO_SEED)

    vocab_size = 100
    target = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    reward_fn = make_target_match_reward(target)

    config = RLConfig(algorithm="dpo", hidden_dim=64, n_layers=1,
                      dpo_beta=0.1, lr=5e-3, epochs_per_batch=2)
    trainer = UnifiedRLTrainer(config, vocab_size=vocab_size)

    # 冻结一份初始策略作为 π_ref
    ref_model = copy.deepcopy(trainer.policy_net)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    prompts = torch.randint(0, vocab_size, (DEMO_PROMPTS, 5))

    for rnd in range(1, DEMO_ROUNDS + 1):
        all_chosen_ids = []
        all_rejected_ids = []
        all_c_masks = []
        all_r_masks = []
        ref_chosen_logps_list = []
        ref_rejected_logps_list = []

        for p in range(DEMO_PROMPTS):
            prompt = prompts[p:p+1]

            # 每个 prompt 生成 2 条 response（用于构造偏好对）
            prompt_2 = prompt.expand(2, -1)  # [2, L_prompt]
            full_ids, _, masks = trainer.policy_net.generate(
                prompt_2, max_new_tokens=DEMO_RESP_LEN, temperature=1.0
            )

            resp_only = full_ids[:, 5:]
            rewards = reward_fn(resp_only)  # [2]

            # 高分 → chosen, 低分 → rejected
            if rewards[0] >= rewards[1]:
                chosen_idx, rejected_idx = 0, 1
            else:
                chosen_idx, rejected_idx = 1, 0

            chosen_full = full_ids[chosen_idx:chosen_idx+1]     # [1, L_full]
            rejected_full = full_ids[rejected_idx:rejected_idx+1]

            chosen_mask = masks[chosen_idx:chosen_idx+1]
            rejected_mask = masks[rejected_idx:rejected_idx+1]

            # 用冻结的 ref model 计算序列级 log prob
            with torch.no_grad():
                ref_c_lp = ref_model.get_log_probs(chosen_full)   # [1, L_full-1]
                ref_r_lp = ref_model.get_log_probs(rejected_full)

                # 只取 response 部分的 per-token 平均
                c_m = chosen_mask[:, 1:]
                r_m = rejected_mask[:, 1:]
                ref_c = (ref_c_lp * c_m).sum(-1) / (c_m.sum(-1) + 1e-8)  # [1]
                ref_r = (ref_r_lp * r_m).sum(-1) / (r_m.sum(-1) + 1e-8)

            all_chosen_ids.append(chosen_full)
            all_rejected_ids.append(rejected_full)
            all_c_masks.append(chosen_mask)
            all_r_masks.append(rejected_mask)
            ref_chosen_logps_list.append(ref_c)
            ref_rejected_logps_list.append(ref_r)

        # 拼接 batch
        chosen_batch = torch.cat(all_chosen_ids, dim=0)           # [P, L_full]
        rejected_batch = torch.cat(all_rejected_ids, dim=0)
        c_mask_batch = torch.cat(all_c_masks, dim=0)
        r_mask_batch = torch.cat(all_r_masks, dim=0)
        ref_chosen = torch.cat(ref_chosen_logps_list, dim=0)      # [P]
        ref_rejected = torch.cat(ref_rejected_logps_list, dim=0)

        metrics = trainer.train_step(
            chosen_ids=chosen_batch,
            rejected_ids=rejected_batch,
            ref_chosen_logps=ref_chosen,
            ref_rejected_logps=ref_rejected,
            chosen_mask=c_mask_batch,
            rejected_mask=r_mask_batch,
        )

        # 评估：当前 policy 在 chosen 和 rejected 上的 log prob 差距
        with torch.no_grad():
            c_lp = trainer.policy_net.get_log_probs(chosen_batch)
            r_lp = trainer.policy_net.get_log_probs(rejected_batch)
            c_avg = (c_lp * c_mask_batch[:, 1:]).sum(-1) / (c_mask_batch[:, 1:].sum(-1) + 1e-8)
            r_avg = (r_lp * r_mask_batch[:, 1:]).sum(-1) / (r_mask_batch[:, 1:].sum(-1) + 1e-8)
            gap = (c_avg - r_avg).mean().item()

        # 计算 chosen 的平均 reward
        chosen_rewards = []
        for p in range(DEMO_PROMPTS):
            c_ids = chosen_batch[p:p+1]
            cr = reward_fn(c_ids[:, 5:])
            chosen_rewards.append(cr.item())
        avg_reward = sum(chosen_rewards) / len(chosen_rewards)

        _print_round(rnd, avg_reward, metrics["loss"], {"chosen-rej_gap": gap})

    print("  ✅ DPO 完整流程演示完成（直接优化偏好对，无需 Reward Model 在线打分）")


# ═══════════════════════════════════════════════
# 7. 算法对比一览
# ═══════════════════════════════════════════════

def print_algorithm_summary():
    """打印所有算法对比"""
    summary = """
╔═══════════════╦══════════════════════════════════════════════════════╗
║  算法         ║  核心思路                                            ║
╠═══════════════╬══════════════════════════════════════════════════════╣
║  PPO          ║  clipped surrogate + GAE + value fn                 ║
║  GRPO         ║  group-based advantage, token-level clip, KL pen    ║
║  GSPO         ║  与 GRPO 类似但 ratio 在 sequence-level clip         ║
║  RLOO         ║  leave-one-out baseline, 无需 value fn              ║
║  DPO          ║  直接对偏好对 (chosen, rejected) 优化               ║
║  REINFORCE    ║  vanilla policy gradient                            ║
╠═══════════════╬══════════════════════════════════════════════════════╣
║  对比维度     ║  PPO    GRPO   GSPO   RLOO   DPO    REINFORCE      ║
╠═══════════════╬══════════════════════════════════════════════════════╣
║  需要 Value   ║   ✅     ❌     ❌     ❌     ❌       ❌            ║
║  需要 Reward  ║   ✅     ✅     ✅     ✅     ❌       ✅            ║
║  需要 Ref π   ║   ❌   可选   可选    ❌     ✅       ❌            ║
║  需要偏好对   ║   ❌     ❌     ❌     ❌     ✅       ❌            ║
║  Group采样    ║   ❌     ✅     ✅     ✅     ❌       ❌/✅         ║
║  Clip策略     ║  token  token  seq.   N/A    N/A      N/A          ║
╚═══════════════╩══════════════════════════════════════════════════════╝
"""
    print(summary)


# ═══════════════════════════════════════════════
# 8. 主入口
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print_algorithm_summary()

    # ── 传统 RL ──
    demo_ppo()

    # ── LLM 风格（完整采样→奖励→训练循环）──
    demo_grpo()
    demo_gspo()
    demo_rloo()
    demo_reinforce()
    demo_dpo()

    print(f"\n{'='*70}")
    print("  🎉 所有 RL 算法演示完成！")
    print(f"{'='*70}")
    print("""
  每个 LLM demo 完整执行了：
    ① 策略采样（Policy.generate）    → 记录 log π_old
    ② 奖励打分（Reward Model 模拟）  → 得到 scalar reward
    ③ 数据打包（GroupTrajectory）    → 组装训练数据
    ④ 策略更新（train_step × N）     → 梯度下降更新参数
    ⑤ 下一轮重新采样                  → 策略已更新，数据分布改变

  这就是 LLM RLHF 的核心训练循环。
  """)
