from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class MAPPOActor(nn.Module):

    def __init__(self, input_dim: int = 32, action_dim: int = 19,
                 hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.logits_head = nn.Linear(hidden_dim, action_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = 0.01 if m is self.logits_head else 1.0
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        return self.logits_head(feat)

    def get_action_and_log_prob(self, state: torch.Tensor,
                                  deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy

    def evaluate_actions(self, state: torch.Tensor,
                          action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy()


class MAPPOCritic(nn.Module):

    def __init__(self, input_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = 1.0 if m is self.value_head else 1.0
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.network(x)
        return self.value_head(feat)


class RunningMeanStd:

    def __init__(self):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray):
        batch_mean  = float(np.mean(x))
        batch_var   = float(np.var(x))
        batch_count = float(len(x))

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total
        new_var = m_2 / total

        self.mean  = new_mean
        self.var   = max(new_var, 1e-8)
        self.count = total

    def normalize(self, x):
        return (x - self.mean) / float(np.sqrt(self.var) + 1e-8)

    def denormalize(self, x):
        return x * float(np.sqrt(self.var) + 1e-8) + self.mean


class MAPPORolloutBuffer:

    def __init__(self, n_steps: int = 256, state_dim: int = 32, n_agents: int = 3):
        self.n_steps   = n_steps
        self.state_dim = state_dim
        self.n_agents  = n_agents
        self._reset()

    def _reset(self):
        self.states        : list = []
        self.joint_actions : list = []
        self.log_probs     : list = []
        self.values        : list = []
        self.joint_rewards : list = []
        self.dones         : list = []
        self.advantages    : Optional[np.ndarray] = None
        self.returns       : Optional[np.ndarray] = None
        self._size = 0

    def add(self, state, joint_actions, log_probs, values,
            joint_rewards, done):
        self.states.append(np.asarray(state, dtype='float32'))
        self.joint_actions.append(np.asarray(joint_actions, dtype=np.int64))
        self.log_probs.append(np.asarray(log_probs, dtype='float32'))
        self.values.append(np.asarray(values, dtype='float32'))
        self.joint_rewards.append(np.asarray(joint_rewards, dtype='float32'))
        self.dones.append(float(done))
        self._size += 1

    def is_full(self) -> bool:
        return self._size >= self.n_steps

    def __len__(self):
        return self._size

    def compute_advantages(self, last_value: float, gamma: float = 0.99,
                            gae_lambda: float = 0.95):
        T = self._size
        rewards = np.asarray([np.sum(r) for r in self.joint_rewards], dtype='float32')
        values  = np.asarray([np.mean(v) for v in self.values], dtype='float32')
        dones   = np.asarray(self.dones, dtype='float32')

        advantages = np.zeros(T, dtype='float32')
        last_gae = 0.0
        for t in range(T - 1, -1, -1):
            if t == T - 1:
                next_value   = last_value
                next_nondone = 1.0 - dones[t]
            else:
                next_value   = values[t + 1]
                next_nondone = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_nondone - values[t]
            last_gae = delta + gamma * gae_lambda * next_nondone * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        self.advantages = advantages
        self.returns    = returns

    def get_minibatches(self, batch_size: int, n_agents: int):
        T = self._size
        N = T * n_agents
        states_flat = np.repeat(np.asarray(self.states), n_agents, axis=0)
        actions_flat   = np.asarray(self.joint_actions).reshape(N)
        log_probs_flat = np.asarray(self.log_probs).reshape(N)
        adv_flat = np.repeat(self.advantages, n_agents)
        ret_flat = np.repeat(self.returns,    n_agents)

        if len(adv_flat) > 1:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        idx = np.random.permutation(N)
        for start in range(0, N, batch_size):
            end = start + batch_size
            mb_idx = idx[start:end]
            yield {
                "states":    states_flat[mb_idx],
                "actions":   actions_flat[mb_idx],
                "log_probs": log_probs_flat[mb_idx],
                "advantages": adv_flat[mb_idx],
                "returns":    ret_flat[mb_idx],
            }

    def clear(self):
        self._reset()


class ParameterSharedMAPPO:

    def __init__(self,
                 input_dim: int = 32,
                 action_dim: int = 19,
                 n_agents: int = 3,
                 lr_actor: float = 5e-4,
                 lr_critic: float = 5e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_eps: float = 0.2,
                 ppo_epochs: int = 4,
                 minibatch_size: int = 64,
                 entropy_coef: float = 0.01,
                 value_coef: float = 1.0,
                 max_grad_norm: float = 10.0,
                 rollout_length: int = 256,
                 reward_scale: float = 0.5,
                 use_cer: bool = False, cer_alpha: float = 0.6, cer_beta: float = 2.0,
                 min_buffer_size: int = 0, target_update_freq: int = 200):
        self.action_dim = action_dim
        self.n_agents   = n_agents
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps   = clip_eps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_length = rollout_length
        self.reward_scale   = reward_scale
        self._train_step_count = 0
        self.use_cer = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor  = MAPPOActor(input_dim, action_dim).to(self.device)
        self.critic = MAPPOCritic(input_dim).to(self.device)

        self.actor_optim  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.memory = MAPPORolloutBuffer(
            n_steps=rollout_length, state_dim=input_dim, n_agents=n_agents,
        )

        self.value_rms = RunningMeanStd()

        self._pending_log_probs: list = []
        self._pending_values:    list = []

        self.batch_size = minibatch_size
        self.min_buffer_size = min_buffer_size

        self.last_train_metrics: Optional[dict] = None

    def select_action(self, state, epsilon: float = 0.0) -> int:
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _ = self.actor.get_action_and_log_prob(s)
            value = self.critic(s)

        self._pending_log_probs.append(float(log_prob.item()))
        self._pending_values.append(float(value.item()))
        return int(action.item())

    def get_q_values(self, state):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.actor(s)
            return logits.squeeze().cpu().numpy()

    def add_experience(self, state, joint_actions, joint_rewards, next_state, done,
                       coop_priority: float = 0.0):
        if len(self._pending_log_probs) != self.n_agents:
            self._pending_log_probs = [0.0] * self.n_agents
            self._pending_values    = [0.0] * self.n_agents

        scaled_rewards = [r * self.reward_scale for r in joint_rewards]

        self.memory.add(
            state=state,
            joint_actions=joint_actions,
            log_probs=self._pending_log_probs,
            values=self._pending_values,
            joint_rewards=scaled_rewards,
            done=done,
        )

        self._pending_log_probs = []
        self._pending_values    = []

    def update_target_network(self):
        pass

    def train_step(self, joint_memory=None, smal_weight: float = 0.0,
                   role_tuple=None):
        if not self.memory.is_full():
            self.last_train_metrics = None
            return

        last_done = self.memory.dones[-1]
        if last_done:
            last_value = 0.0
        else:
            last_state = self.memory.states[-1]
            with torch.no_grad():
                s = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
                last_value = float(self.critic(s).item())

        self.memory.compute_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        self.value_rms.update(self.memory.returns)

        total_pol_loss = 0.0
        total_val_loss = 0.0
        total_entropy  = 0.0
        n_updates = 0
        clip_fraction = 0.0
        for epoch in range(self.ppo_epochs):
            for mb in self.memory.get_minibatches(self.minibatch_size, self.n_agents):
                states   = torch.FloatTensor(mb["states"]).to(self.device)
                actions  = torch.LongTensor(mb["actions"]).to(self.device)
                old_lp   = torch.FloatTensor(mb["log_probs"]).to(self.device)
                advs     = torch.FloatTensor(mb["advantages"]).to(self.device)
                returns  = torch.FloatTensor(mb["returns"]).to(self.device)

                new_lp, entropy = self.actor.evaluate_actions(states, actions)
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_bonus = entropy.mean()

                values_pred = self.critic(states).squeeze(-1)
                returns_norm = torch.from_numpy(
                    self.value_rms.normalize(returns.cpu().numpy())
                ).float().to(self.device)
                values_norm = (values_pred - self.value_rms.mean) / (
                    float(np.sqrt(self.value_rms.var)) + 1e-8
                )
                value_loss = F.mse_loss(values_norm, returns_norm)

                actor_total_loss = policy_loss - self.entropy_coef * entropy_bonus
                critic_total_loss = self.value_coef * value_loss

                self.actor_optim.zero_grad()
                actor_total_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm,
                )
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_total_loss.backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm,
                )
                self.critic_optim.step()

                total_pol_loss += policy_loss.detach().item()
                total_val_loss += value_loss.detach().item()
                total_entropy  += entropy_bonus.detach().item()
                with torch.no_grad():
                    clip_fraction += float(
                        ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                    )
                n_updates += 1
        self._train_step_count += 1

        self.memory.clear()
        self._pending_log_probs = []
        self._pending_values    = []

        avg_pol_loss = total_pol_loss / max(n_updates, 1)
        avg_val_loss = total_val_loss / max(n_updates, 1)
        avg_entropy  = total_entropy  / max(n_updates, 1)
        avg_clip_frac = clip_fraction / max(n_updates, 1)

        with torch.no_grad():
            sample_idx = np.random.choice(len(self.memory.states),
                                          size=min(32, len(self.memory.states)),
                                          replace=False) if len(self.memory.states) > 0 else None
            logits_stats_q_mean = 0.0
            logits_stats_q_std  = 0.0
            logits_stats_q_max  = 0.0
            logits_stats_q_min  = 0.0

        self.last_train_metrics = {
            "q_mean":   logits_stats_q_mean,
            "q_std":    logits_stats_q_std,
            "q_max":    logits_stats_q_max,
            "q_min":    logits_stats_q_min,
            "q_range":  0.0,
            "target_q_mean": float(self.value_rms.mean),
            "td_error_mean": avg_val_loss,
            "td_error_std":  0.0,
            "td_error_max":  0.0,
            "loss_total":     float(avg_pol_loss + self.value_coef * avg_val_loss),
            "loss_dqn":       None,
            "loss_mixer":     None,
            "loss_actor":     float(avg_pol_loss),
            "loss_critic":    float(avg_val_loss),
            "loss_smal_aux":  0.0,
            "grad_norm_actor":  float(actor_grad_norm.item()) if n_updates > 0 else 0.0,
            "grad_norm_critic": float(critic_grad_norm.item()) if n_updates > 0 else 0.0,
            "grad_norm_mixer":  None,
            "grad_clip_ratio":  avg_clip_frac,
            "buffer_size":         self.rollout_length,
            "buffer_avg_priority": 0.0,
            "buffer_priority_std": 0.0,
            "ppo_entropy":     avg_entropy,
            "ppo_clip_frac":   avg_clip_frac,
            "ppo_n_updates":   n_updates,
            "value_rms_mean":  float(self.value_rms.mean),
            "value_rms_std":   float(np.sqrt(self.value_rms.var)),
        }

    def state_dict(self) -> dict:
        return {
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optim":  self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "value_rms": {"mean": self.value_rms.mean,
                          "var":  self.value_rms.var,
                          "count": self.value_rms.count},
            "train_step": self._train_step_count,
        }

    def load_state_dict(self, state: dict):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])
        rms = state.get("value_rms", {})
        self.value_rms.mean  = float(rms.get("mean", 0.0))
        self.value_rms.var   = float(rms.get("var",  1.0))
        self.value_rms.count = float(rms.get("count", 1e-4))
        self._train_step_count = int(state.get("train_step", 0))
