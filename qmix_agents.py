import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


class QMIXAgent(nn.Module):
    def __init__(self, input_dim=32, action_dim=19, hidden_dim=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head     = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, action_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = 0.5 if m is self.advantage_head else 1.0
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.feature(x)
        v    = self.value_head(feat)
        a    = self.advantage_head(feat)
        q = v + (a - a.mean(dim=-1, keepdim=True))
        return q


class QMixingNetwork(nn.Module):
    def __init__(self, n_agents=3, state_dim=32, embed_dim=32):
        super().__init__()
        self.n_agents  = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs, state):
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(bs, 1, self.n_agents)

        w1 = torch.abs(self.hyper_w1(state)).view(bs, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(bs, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w2 = torch.abs(self.hyper_w2(state)).view(bs, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(bs, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(bs, 1)


class QMIXReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, joint_actions, joint_rewards, next_state, done, coop_priority=0.0):
        self.buffer.append((state, joint_actions, joint_rewards, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, j_actions, j_rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(j_actions),
            np.array(j_rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class QMIXCooperativeReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, coop_beta=2.0, eps=1e-3,
                 is_beta_start=0.4, is_beta_end=1.0, is_beta_steps=100_000):
        self.capacity      = capacity
        self.alpha         = alpha
        self.coop_beta     = coop_beta
        self.eps           = eps
        self.buffer        = []
        self.priorities    = []
        self.is_beta_start = is_beta_start
        self.is_beta_end   = is_beta_end
        self.is_beta_steps = is_beta_steps
        self._sample_step  = 0

    def push(self, state, joint_actions, joint_rewards, next_state, done, coop_priority=0.0):
        team_reward = float(np.sum(joint_rewards))
        priority = (
            abs(team_reward) ** self.alpha
            + self.coop_beta * float(coop_priority)
            + self.eps
        )
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        self.buffer.append((state, joint_actions, joint_rewards, next_state, done))
        self.priorities.append(priority)

    def _current_beta(self):
        progress = min(self._sample_step / self.is_beta_steps, 1.0)
        return self.is_beta_start + progress * (self.is_beta_end - self.is_beta_start)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None
        self._sample_step += 1
        priorities = np.array(self.priorities, dtype=np.float64)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        N = len(self.buffer)
        beta = self._current_beta()
        is_weights = (N * probs[indices]) ** (-beta)
        is_weights = is_weights / (is_weights.max() + 1e-8)
        batch = [self.buffer[i] for i in indices]
        states, j_actions, j_rewards, next_states, dones = zip(*batch)
        return (
            (np.array(states), np.array(j_actions), np.array(j_rewards),
             np.array(next_states), np.array(dones)),
            is_weights.astype(np.float32),
        )

    def avg_priority(self) -> float:
        if not self.priorities:
            return 0.0
        return float(np.mean(self.priorities))

    def priority_std(self) -> float:
        if not self.priorities:
            return 0.0
        return float(np.std(self.priorities))

    def priority_quantiles(self, qs=(0.10, 0.25, 0.50, 0.75, 0.90)) -> list:
        if not self.priorities:
            return [0.0 for _ in qs]
        arr = np.array(self.priorities, dtype=np.float64)
        return [float(np.quantile(arr, q)) for q in qs]

    def __len__(self):
        return len(self.buffer)


class ParameterSharedQMIX:
    def __init__(self, input_dim=32, action_dim=19, n_agents=3,
                 lr=0.0005, gamma=0.99,
                 reward_scale=0.5,
                 min_buffer_size=512,
                 target_update_freq=200,
                 use_cer=True, cer_alpha=0.6, cer_beta=2.0):
        self.action_dim         = action_dim
        self.n_agents           = n_agents
        self.gamma              = gamma
        self.reward_scale       = reward_scale
        self.min_buffer_size    = min_buffer_size
        self.target_update_freq = target_update_freq
        self._train_step_count  = 0
        self.use_cer            = use_cer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_q = QMIXAgent(input_dim, action_dim).to(self.device)
        self.target_q = QMIXAgent(input_dim, action_dim).to(self.device)
        self.target_q.load_state_dict(self.policy_q.state_dict())
        self.target_q.eval()

        self.mixer        = QMixingNetwork(n_agents, input_dim).to(self.device)
        self.target_mixer = QMixingNetwork(n_agents, input_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_mixer.eval()

        self.optimizer = optim.Adam(
            list(self.policy_q.parameters()) + list(self.mixer.parameters()),
            lr=lr,
        )

        if use_cer:
            self.memory = QMIXCooperativeReplayBuffer(
                capacity=10000, alpha=cer_alpha, coop_beta=cer_beta,
            )
        else:
            self.memory = QMIXReplayBuffer(capacity=10000)

        self.batch_size = 64
        self.last_train_metrics: dict | None = None

    def select_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_q(s)
            return int(q.argmax(1).item())

    def get_q_values(self, state):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_q(s).squeeze().cpu().numpy()
            return q

    def add_experience(self, state, joint_actions, joint_rewards, next_state, done,
                       coop_priority=0.0):
        self.memory.push(state, joint_actions, joint_rewards, next_state, done,
                         coop_priority=coop_priority)

    def update_target_network(self):
        self.target_q.load_state_dict(self.policy_q.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def train_step(self, joint_memory=None, smal_weight=0.1, role_tuple=None):
        if len(self.memory) < self.min_buffer_size:
            self.last_train_metrics = None
            return

        sample_result = self.memory.sample(self.batch_size)
        if (isinstance(sample_result, tuple) and len(sample_result) == 2
                and isinstance(sample_result[0], tuple)):
            (states_np, j_actions_np, j_rewards_np,
             next_states_np, dones_np), is_weights_np = sample_result
            if states_np is None:
                return
            is_weights = torch.FloatTensor(is_weights_np).unsqueeze(1).to(self.device)
        else:
            if sample_result is None:
                return
            states_np, j_actions_np, j_rewards_np, next_states_np, dones_np = sample_result
            is_weights = torch.ones(self.batch_size, 1, device=self.device)

        states  = torch.FloatTensor(states_np).to(self.device)
        joint_actions = torch.LongTensor(j_actions_np).to(self.device)
        team_rewards = torch.FloatTensor(
            j_rewards_np.sum(axis=1) * self.reward_scale
        ).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states_np).to(self.device)
        dones = torch.FloatTensor(dones_np).unsqueeze(1).to(self.device)

        q_all = self.policy_q(states)
        agent_qs = (
            q_all.unsqueeze(1)
                 .expand(-1, self.n_agents, -1)
                 .gather(2, joint_actions.unsqueeze(-1))
                 .squeeze(-1)
        )
        q_tot = self.mixer(agent_qs, states)

        with torch.no_grad():
            next_q_policy   = self.policy_q(next_states)
            next_actions    = next_q_policy.argmax(1, keepdim=True)
            next_actions_joint = next_actions.expand(-1, self.n_agents)
            next_q_target   = self.target_q(next_states)
            next_agent_qs   = (
                next_q_target.unsqueeze(1)
                             .expand(-1, self.n_agents, -1)
                             .gather(2, next_actions_joint.unsqueeze(-1))
                             .squeeze(-1)
            )
            next_q_tot = self.target_mixer(next_agent_qs, next_states)
            target_q_tot = team_rewards + self.gamma * next_q_tot * (1 - dones)

        td_errors_qmix = (q_tot - target_q_tot).detach()
        huber = nn.SmoothL1Loss(reduction='none')(q_tot, target_q_tot)
        loss  = (is_weights * huber).mean()

        aux_loss = torch.tensor(0.0, device=self.device)
        if joint_memory is not None and smal_weight > 0:
            n_hits = 0
            rt = role_tuple if role_tuple is not None else ['DF', 'MF', 'FW']
            for i in range(self.batch_size):
                state_np = states[i].detach().cpu().numpy()
                results = joint_memory.query(state_np, role_tuple=rt,
                                             k=1, min_similarity=0.50,
                                             for_smal=True)
                if not results:
                    continue
                best  = results[0]
                q_row = self.policy_q(states[i].unsqueeze(0)).squeeze()
                max_q = q_row.max()
                rec_qs = []
                for a_rec in best['joint_actions']:
                    if 0 <= a_rec < q_row.shape[0]:
                        rec_qs.append(q_row[a_rec])
                if rec_qs:
                    rec_q_mean = torch.stack(rec_qs).mean()
                    aux_loss = aux_loss + best['weighted_score'] * (max_q - rec_q_mean)
                    n_hits += 1
            if n_hits > 0:
                aux_loss = aux_loss / n_hits

        total_loss = loss + smal_weight * aux_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.policy_q.parameters()) + list(self.mixer.parameters()),
            max_norm=10.0,
        )
        self.optimizer.step()

        self._train_step_count += 1
        if self._train_step_count % self.target_update_freq == 0:
            self.update_target_network()

        with torch.no_grad():
            q_all_batch = self.policy_q(states)
            td_abs      = td_errors_qmix.abs()
            grad_norm_f = float(grad_norm) if grad_norm is not None else 0.0
        self.last_train_metrics = {
            "q_mean"        : round(float(q_all_batch.mean().item()), 6),
            "q_std"         : round(float(q_all_batch.std().item()), 6),
            "q_max"         : round(float(q_all_batch.max().item()), 6),
            "q_min"         : round(float(q_all_batch.min().item()), 6),
            "q_range"       : round(float(q_all_batch.max().item() - q_all_batch.min().item()), 6),
            "target_q_mean" : round(float(target_q_tot.detach().mean().item()), 6),
            "td_error_mean" : round(float(td_abs.mean().item()), 6),
            "td_error_std"  : round(float(td_abs.std().item()), 6),
            "td_error_max"  : round(float(td_abs.max().item()), 6),
            "loss_total"    : round(float(total_loss.item()), 6),
            "loss_dqn"      : None,
            "loss_mixer"    : round(float(loss.item()), 6),
            "loss_actor"    : None,
            "loss_critic"   : None,
            "loss_smal_aux" : round(float((smal_weight * aux_loss).item()), 6),
            "grad_norm_actor" : None,
            "grad_norm_critic": None,
            "grad_norm_mixer" : round(grad_norm_f, 6),
            "grad_clip_ratio" : 1.0 if grad_norm_f > 10.0 else 0.0,
            "buffer_size"          : int(len(self.memory)),
            "buffer_avg_priority"  : round(float(self.memory.avg_priority()), 6)
                                     if hasattr(self.memory, "avg_priority") else 0.0,
            "buffer_priority_std"  : round(float(self.memory.priority_std()), 6)
                                     if hasattr(self.memory, "priority_std") else 0.0,
        }

    def state_dict(self):
        return {
            'policy_q' : self.policy_q.state_dict(),
            'mixer'    : self.mixer.state_dict(),
        }

    def load_state_dict(self, state):
        self.policy_q.load_state_dict(state['policy_q'])
        self.target_q.load_state_dict(state['policy_q'])
        self.mixer.load_state_dict(state['mixer'])
        self.target_mixer.load_state_dict(state['mixer'])
