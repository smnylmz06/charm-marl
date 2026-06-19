from __future__ import annotations

import os
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import faiss

from qplex_agents import QPLEXAgent, QPLEXMixer


class EMCCuriosityModule(nn.Module):
    def __init__(self, input_dim: int = 32, action_dim: int = 19,
                 hidden_dim: int = 64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        return self.predictor(state)


class EMCEpisodicMemory:
    def __init__(self, state_dim: int = 32, max_size: int = 50_000,
                 similarity_threshold: float = 0.85):
        self.state_dim = state_dim
        self.max_size  = max_size
        self.similarity_threshold = float(similarity_threshold)

        self.index   = faiss.IndexFlatL2(state_dim)
        self._states : list = []
        self._returns: list = []

        self._cum_queries = 0
        self._cum_hits    = 0
        self._cum_updates = 0

    def update_with_trajectory(self, states: List[np.ndarray],
                                rewards: List[float], gamma: float = 0.99):
        T = len(rewards)
        if T == 0 or len(states) != T:
            return

        returns = [0.0] * T
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = float(rewards[t]) + gamma * running
            returns[t] = running

        for s_arr, R in zip(states, returns):
            s_np = np.asarray(s_arr, dtype='float32').reshape(1, -1)
            if s_np.shape[1] != self.state_dim:
                continue

            self.index.add(s_np)
            self._states.append(s_np.flatten())
            self._returns.append(float(R))
            self._cum_updates += 1

        if len(self._states) > self.max_size:
            evict_n = self.max_size // 5
            self._states  = self._states[evict_n:]
            self._returns = self._returns[evict_n:]
            self.index = faiss.IndexFlatL2(self.state_dim)
            if self._states:
                self.index.add(np.vstack(self._states))

    def query_batch(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = states.size(0)
        device = states.device

        if self.index.ntotal == 0:
            return (torch.zeros(bs, device=device),
                    torch.zeros(bs, device=device, dtype=torch.bool))

        states_np = states.detach().cpu().numpy().astype('float32')
        distances, indices = self.index.search(states_np, 1)

        H_values = np.zeros(bs, dtype='float32')
        mask     = np.zeros(bs, dtype=bool)
        for b in range(bs):
            self._cum_queries += 1
            d   = float(distances[b, 0])
            sim = 1.0 / (1.0 + d)
            if sim >= self.similarity_threshold and indices[b, 0] < len(self._returns):
                H_values[b] = self._returns[indices[b, 0]]
                mask[b]     = True
                self._cum_hits += 1

        return (torch.from_numpy(H_values).to(device),
                torch.from_numpy(mask).to(device))

    def __len__(self):
        return len(self._states)

    def get_diagnostics(self) -> dict:
        q = max(self._cum_queries, 1)
        return {
            "emc_em_size"     : len(self),
            "emc_em_queries"  : int(self._cum_queries),
            "emc_em_hits"     : int(self._cum_hits),
            "emc_em_hit_rate" : round(self._cum_hits / q, 4),
            "emc_em_updates"  : int(self._cum_updates),
        }


class EMCReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer  = deque(maxlen=capacity)
        self._cur_traj_states  : list = []
        self._cur_traj_rewards : list = []

    def push(self, state, joint_actions, joint_rewards, next_state, done,
             coop_priority: float = 0.0):
        self.buffer.append((
            np.asarray(state, dtype='float32'),
            np.asarray(joint_actions, dtype=np.int64),
            np.asarray(joint_rewards, dtype='float32'),
            np.asarray(next_state, dtype='float32'),
            float(done),
        ))

        self._cur_traj_states.append(np.asarray(state, dtype='float32'))
        self._cur_traj_rewards.append(float(np.sum(joint_rewards)))

    def flush_trajectory(self) -> Tuple[List[np.ndarray], List[float]]:
        states  = self._cur_traj_states
        rewards = self._cur_traj_rewards
        self._cur_traj_states  = []
        self._cur_traj_rewards = []
        return states, rewards

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return None
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states), np.stack(actions), np.stack(rewards),
                np.stack(next_states), np.array(dones, dtype='float32'))

    def __len__(self):
        return len(self.buffer)

    def avg_priority(self):    return 0.0
    def priority_std(self):     return 0.0
    def priority_quantiles(self): return [0.0] * 5


class ParameterSharedEMC:

    def __init__(self,
                 input_dim: int = 32,
                 action_dim: int = 19,
                 n_agents: int = 3,
                 lr: float = 0.0005,
                 gamma: float = 0.99,
                 reward_scale: float = 0.5,
                 min_buffer_size: int = 512,
                 target_update_freq: int = 200,
                 curiosity_beta: float = 0.05,
                 em_lambda: float = 0.1,
                 predictor_alpha: float = 0.1,
                 em_max_size: int = 50_000,
                 em_similarity_threshold: float = 0.85,
                 use_cer: bool = False, cer_alpha: float = 0.6, cer_beta: float = 2.0):
        self.action_dim         = action_dim
        self.n_agents           = n_agents
        self.gamma              = gamma
        self.reward_scale       = reward_scale
        self.min_buffer_size    = min_buffer_size
        self.target_update_freq = target_update_freq
        self._train_step_count  = 0
        self.use_cer            = use_cer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_q = QPLEXAgent(input_dim, action_dim).to(self.device)
        self.target_q = QPLEXAgent(input_dim, action_dim).to(self.device)
        self.target_q.load_state_dict(self.policy_q.state_dict())
        self.target_q.eval()

        self.mixer        = QPLEXMixer(n_agents, input_dim).to(self.device)
        self.target_mixer = QPLEXMixer(n_agents, input_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_mixer.eval()

        self.curiosity = EMCCuriosityModule(input_dim, action_dim).to(self.device)
        self.episodic_memory = EMCEpisodicMemory(
            state_dim=input_dim, max_size=em_max_size,
            similarity_threshold=em_similarity_threshold,
        )

        self.optimizer = optim.Adam(
            list(self.policy_q.parameters())
            + list(self.mixer.parameters())
            + list(self.curiosity.parameters()),
            lr=lr,
        )

        if use_cer:
            from qmix_agents import QMIXCooperativeReplayBuffer
            self.memory = QMIXCooperativeReplayBuffer(
                capacity=10000, alpha=cer_alpha, coop_beta=cer_beta,
            )
            self._has_trajectory_tracking = False
        else:
            self.memory = EMCReplayBuffer(capacity=10000)
            self._has_trajectory_tracking = True

        self.curiosity_beta  = float(curiosity_beta)
        self.em_lambda       = float(em_lambda)
        self.predictor_alpha = float(predictor_alpha)

        self.batch_size = 64

        self.last_train_metrics: Optional[dict] = None

    def select_action(self, state, epsilon: float) -> int:
        if np.random.random() < epsilon:
            return int(np.random.randint(self.action_dim))
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_q(s)
            return int(q.argmax(1).item())

    def get_q_values(self, state):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_q(s).squeeze().cpu().numpy()

    def add_experience(self, state, joint_actions, joint_rewards, next_state, done,
                       coop_priority: float = 0.0):
        if self.use_cer:
            self.memory.push(state, joint_actions, joint_rewards, next_state, done,
                             coop_priority=coop_priority)
        else:
            self.memory.push(state, joint_actions, joint_rewards, next_state, done)

        if done and self._has_trajectory_tracking:
            states, rewards = self.memory.flush_trajectory()
            self.episodic_memory.update_with_trajectory(
                states, rewards, gamma=self.gamma,
            )

    def update_target_network(self):
        self.target_q.load_state_dict(self.policy_q.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def train_step(self, joint_memory=None, smal_weight: float = 0.0,
                   role_tuple=None):
        if len(self.memory) < self.min_buffer_size:
            self.last_train_metrics = None
            return

        sample_result = self.memory.sample(self.batch_size)
        if sample_result is None:
            self.last_train_metrics = None
            return

        if self.use_cer:
            if (isinstance(sample_result, tuple) and len(sample_result) == 2
                    and isinstance(sample_result[0], tuple)):
                (states_np, j_actions_np, j_rewards_np,
                 next_states_np, dones_np), is_weights_np = sample_result
                if states_np is None:
                    self.last_train_metrics = None
                    return
                is_weights = torch.FloatTensor(is_weights_np).unsqueeze(1).to(self.device)
            else:
                states_np, j_actions_np, j_rewards_np, next_states_np, dones_np = sample_result
                is_weights = torch.ones(self.batch_size, 1, device=self.device)
        else:
            states_np, j_actions_np, j_rewards_np, next_states_np, dones_np = sample_result
            is_weights = torch.ones(self.batch_size, 1, device=self.device)

        states  = torch.FloatTensor(states_np).to(self.device)
        joint_actions = torch.LongTensor(j_actions_np).to(self.device)
        team_rewards_raw = torch.FloatTensor(
            j_rewards_np.sum(axis=1) * self.reward_scale
        ).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states_np).to(self.device)
        dones = torch.FloatTensor(dones_np).unsqueeze(1).to(self.device)

        q_all = self.policy_q(states)
        agent_qs = (
            q_all.unsqueeze(1).expand(-1, self.n_agents, -1)
                 .gather(2, joint_actions.unsqueeze(-1)).squeeze(-1)
        )
        agent_vs = q_all.max(dim=1, keepdim=True).values.expand(-1, self.n_agents)
        q_tot = self.mixer(agent_qs, agent_vs, states)

        q_predicted = self.curiosity(states)
        with torch.no_grad():
            q_pred_chosen = (
                q_predicted.unsqueeze(1).expand(-1, self.n_agents, -1)
                           .gather(2, joint_actions.unsqueeze(-1)).squeeze(-1)
            )
            pred_err = ((agent_qs - q_pred_chosen) ** 2).mean(dim=1, keepdim=True)
            r_intrinsic = self.curiosity_beta * pred_err
        team_rewards = team_rewards_raw + r_intrinsic

        with torch.no_grad():
            next_q_policy = self.policy_q(next_states)
            next_actions  = next_q_policy.argmax(1, keepdim=True)
            next_actions_joint = next_actions.expand(-1, self.n_agents)
            next_q_target = self.target_q(next_states)
            next_agent_qs = (
                next_q_target.unsqueeze(1).expand(-1, self.n_agents, -1)
                             .gather(2, next_actions_joint.unsqueeze(-1)).squeeze(-1)
            )
            next_agent_vs = next_q_target.max(dim=1, keepdim=True).values.expand(-1, self.n_agents)
            next_q_tot   = self.target_mixer(next_agent_qs, next_agent_vs, next_states)
            target_q_tot = team_rewards + self.gamma * next_q_tot * (1 - dones)

        td_errors = (q_tot - target_q_tot).detach()
        huber = nn.SmoothL1Loss(reduction='none')(q_tot, target_q_tot)
        td_loss = (is_weights * huber).mean()

        em_loss = torch.tensor(0.0, device=self.device)
        em_hit_count = 0
        if self.em_lambda > 0 and len(self.episodic_memory) > 0:
            H_values, em_mask = self.episodic_memory.query_batch(states)
            if em_mask.any():
                H_target = H_values * self.reward_scale
                diff = (q_tot.squeeze(-1) - H_target) ** 2
                em_loss = (diff * em_mask.float()).sum() / max(em_mask.sum(), 1)
                em_hit_count = int(em_mask.sum().item())

        with torch.no_grad():
            q_target_for_pred = q_all.detach()
        pred_loss = F.mse_loss(q_predicted, q_target_for_pred)

        total_loss = td_loss + self.em_lambda * em_loss + self.predictor_alpha * pred_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.policy_q.parameters())
            + list(self.mixer.parameters())
            + list(self.curiosity.parameters()),
            max_norm=10.0,
        )
        self.optimizer.step()

        self._train_step_count += 1
        if self._train_step_count % self.target_update_freq == 0:
            self.update_target_network()

        with torch.no_grad():
            q_flat = q_all.detach().cpu().numpy().flatten()
            td_flat = td_errors.detach().cpu().numpy().flatten()
        cer_avg_priority = (
            self.memory.avg_priority() if hasattr(self.memory, "avg_priority") else 0.0
        )
        cer_priority_std = (
            self.memory.priority_std() if hasattr(self.memory, "priority_std") else 0.0
        )
        self.last_train_metrics = {
            "q_mean":   float(np.mean(q_flat)),
            "q_std":    float(np.std(q_flat)),
            "q_max":    float(np.max(q_flat)),
            "q_min":    float(np.min(q_flat)),
            "q_range":  float(np.max(q_flat) - np.min(q_flat)),
            "target_q_mean": float(target_q_tot.detach().mean().item()),
            "td_error_mean": float(np.mean(np.abs(td_flat))),
            "td_error_std":  float(np.std(td_flat)),
            "td_error_max":  float(np.max(np.abs(td_flat))),
            "loss_total":     float(total_loss.detach().item()),
            "loss_dqn":       None,
            "loss_mixer":     float(td_loss.detach().item()),
            "loss_actor":     None,
            "loss_critic":    None,
            "loss_smal_aux":  0.0,
            "loss_em":        float(em_loss.detach().item()) if isinstance(em_loss, torch.Tensor) else 0.0,
            "loss_pred":      float(pred_loss.detach().item()),
            "em_hit_count":   em_hit_count,
            "curiosity_mean": float(r_intrinsic.detach().mean().item()),
            "grad_norm_actor":  float(grad_norm.item()),
            "grad_norm_critic": None,
            "grad_norm_mixer":  None,
            "grad_clip_ratio":  1.0 if grad_norm.item() > 10.0 else 0.0,
            "buffer_size":         len(self.memory),
            "buffer_avg_priority": float(cer_avg_priority),
            "buffer_priority_std": float(cer_priority_std),
        }

    def state_dict(self) -> dict:
        return {
            "policy_q":     self.policy_q.state_dict(),
            "target_q":     self.target_q.state_dict(),
            "mixer":        self.mixer.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "curiosity":    self.curiosity.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "train_step":   self._train_step_count,
        }

    def load_state_dict(self, state: dict):
        self.policy_q.load_state_dict(state["policy_q"])
        self.target_q.load_state_dict(state["target_q"])
        self.mixer.load_state_dict(state["mixer"])
        self.target_mixer.load_state_dict(state["target_mixer"])
        self.curiosity.load_state_dict(state["curiosity"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._train_step_count = int(state.get("train_step", 0))
