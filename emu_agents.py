from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import faiss

from qplex_agents import QPLEXAgent, QPLEXMixer


class EMUEncoderDecoder(nn.Module):

    def __init__(self, input_dim: int = 32, embedding_dim: int = 32,
                 hidden_dim: int = 64):
        super().__init__()
        self.input_dim     = input_dim
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, e: torch.Tensor) -> torch.Tensor:
        return self.decoder(e)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.encode(x)
        s_hat = self.decode(e)
        return e, s_hat

    def recon_loss(self, x: torch.Tensor) -> torch.Tensor:
        _, s_hat = self.forward(x)
        return F.mse_loss(s_hat, x)


class EMUEpisodicMemory:
    def __init__(self, embedding_dim: int = 32, max_size: int = 50_000,
                 similarity_threshold: float = 0.85,
                 desirability_quantile: float = 0.80):
        self.embedding_dim = embedding_dim
        self.max_size = max_size
        self.similarity_threshold = float(similarity_threshold)
        self.desirability_quantile = float(desirability_quantile)

        self.index = faiss.IndexFlatL2(embedding_dim)
        self._embeddings: list = []
        self._returns:    list = []

        self._desirability_threshold_value = 0.0

        self._cum_queries  = 0
        self._cum_hits     = 0
        self._cum_desirable_hits = 0

    def update_with_trajectory(self, embeddings: torch.Tensor,
                                rewards: List[float], gamma: float = 0.99):
        T = len(rewards)
        if T == 0 or embeddings.size(0) != T:
            return

        returns = [0.0] * T
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = float(rewards[t]) + gamma * running
            returns[t] = running

        emb_np = embeddings.detach().cpu().numpy().astype('float32')
        if emb_np.shape[1] != self.embedding_dim:
            return

        self.index.add(emb_np)
        for t in range(T):
            self._embeddings.append(emb_np[t])
            self._returns.append(returns[t])

        if len(self._embeddings) > self.max_size:
            evict_n = self.max_size // 5
            self._embeddings = self._embeddings[evict_n:]
            self._returns    = self._returns[evict_n:]
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            if self._embeddings:
                self.index.add(np.vstack(self._embeddings))

        if self._returns:
            self._desirability_threshold_value = float(
                np.quantile(np.asarray(self._returns), self.desirability_quantile)
            )

    def query_batch(self, embeddings: torch.Tensor) -> torch.Tensor:
        bs = embeddings.size(0)
        device = embeddings.device

        if self.index.ntotal == 0:
            return torch.zeros(bs, device=device, dtype=torch.bool)

        emb_np = embeddings.detach().cpu().numpy().astype('float32')
        distances, indices = self.index.search(emb_np, 1)

        mask = np.zeros(bs, dtype=bool)
        for b in range(bs):
            self._cum_queries += 1
            d   = float(distances[b, 0])
            sim = 1.0 / (1.0 + d)
            if sim < self.similarity_threshold or indices[b, 0] >= len(self._returns):
                continue
            self._cum_hits += 1
            R_match = self._returns[indices[b, 0]]
            if R_match >= self._desirability_threshold_value:
                mask[b] = True
                self._cum_desirable_hits += 1

        return torch.from_numpy(mask).to(device)

    def __len__(self):
        return len(self._embeddings)

    def get_diagnostics(self) -> dict:
        q = max(self._cum_queries, 1)
        return {
            "emu_em_size"            : len(self),
            "emu_em_queries"         : int(self._cum_queries),
            "emu_em_hits"            : int(self._cum_hits),
            "emu_em_desirable_hits"  : int(self._cum_desirable_hits),
            "emu_em_hit_rate"        : round(self._cum_hits / q, 4),
            "emu_em_desirability_rate": round(self._cum_desirable_hits / q, 4),
            "emu_em_threshold_value" : round(self._desirability_threshold_value, 4),
        }


class EMUReplayBuffer:

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)
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

    def avg_priority(self):       return 0.0
    def priority_std(self):       return 0.0
    def priority_quantiles(self): return [0.0] * 5


class ParameterSharedEMU:

    def __init__(self,
                 input_dim: int = 32,
                 action_dim: int = 19,
                 n_agents: int = 3,
                 lr: float = 0.0005,
                 gamma: float = 0.99,
                 reward_scale: float = 0.5,
                 min_buffer_size: int = 512,
                 target_update_freq: int = 200,
                 embedding_dim: int = 32,
                 incentive_coef: float = 0.05,
                 recon_alpha: float = 0.1,
                 em_max_size: int = 50_000,
                 em_similarity_threshold: float = 0.85,
                 em_desirability_quantile: float = 0.80,
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

        self.autoencoder = EMUEncoderDecoder(
            input_dim=input_dim, embedding_dim=embedding_dim,
        ).to(self.device)
        self.episodic_memory = EMUEpisodicMemory(
            embedding_dim=embedding_dim,
            max_size=em_max_size,
            similarity_threshold=em_similarity_threshold,
            desirability_quantile=em_desirability_quantile,
        )
        self.embedding_dim = embedding_dim

        self.optimizer = optim.Adam(
            list(self.policy_q.parameters())
            + list(self.mixer.parameters())
            + list(self.autoencoder.parameters()),
            lr=lr,
        )

        if use_cer:
            from qmix_agents import QMIXCooperativeReplayBuffer
            self.memory = QMIXCooperativeReplayBuffer(
                capacity=10000, alpha=cer_alpha, coop_beta=cer_beta,
            )
            self._has_trajectory_tracking = False
        else:
            self.memory = EMUReplayBuffer(capacity=10000)
            self._has_trajectory_tracking = True

        self.incentive_coef = float(incentive_coef)
        self.recon_alpha    = float(recon_alpha)

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
            if states:
                with torch.no_grad():
                    states_t = torch.FloatTensor(np.stack(states)).to(self.device)
                    embeddings = self.autoencoder.encode(states_t)
                self.episodic_memory.update_with_trajectory(
                    embeddings, rewards, gamma=self.gamma,
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

        with torch.no_grad():
            embeddings = self.autoencoder.encode(states)
            incentive_mask = self.episodic_memory.query_batch(embeddings)
            r_intrinsic = (
                self.incentive_coef * incentive_mask.float()
            ).unsqueeze(-1)
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

        recon_loss = self.autoencoder.recon_loss(states)

        total_loss = td_loss + self.recon_alpha * recon_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.policy_q.parameters())
            + list(self.mixer.parameters())
            + list(self.autoencoder.parameters()),
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
            "loss_recon":     float(recon_loss.detach().item()),
            "incentive_rate": float(incentive_mask.float().mean().item()),
            "intrinsic_mean": float(r_intrinsic.detach().mean().item()),
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
            "autoencoder":  self.autoencoder.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "train_step":   self._train_step_count,
        }

    def load_state_dict(self, state: dict):
        self.policy_q.load_state_dict(state["policy_q"])
        self.target_q.load_state_dict(state["target_q"])
        self.mixer.load_state_dict(state["mixer"])
        self.target_mixer.load_state_dict(state["target_mixer"])
        self.autoencoder.load_state_dict(state["autoencoder"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._train_step_count = int(state.get("train_step", 0))
