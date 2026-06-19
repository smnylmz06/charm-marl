import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


class MADDPGActor(nn.Module):
    def __init__(self, input_dim=32, action_dim=19, hidden_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
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

    def forward(self, x):
        return self.net(x)

    def sample_categorical(self, x):
        logits = self.forward(x)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        return dist.sample()

    def gumbel_softmax_sample(self, x, temperature=1.0, hard=False):
        logits = self.forward(x)
        return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)


class MADDPGCritic(nn.Module):
    def __init__(self, state_dim=32, action_dim=19, n_agents=3, hidden_dim=128):
        super().__init__()
        self.n_agents   = n_agents
        self.action_dim = action_dim
        joint_input_dim = state_dim + n_agents * action_dim
        self.net = nn.Sequential(
            nn.Linear(joint_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, state, joint_action_vectors):
        x = torch.cat([state, joint_action_vectors], dim=-1)
        return self.net(x)


class MADDPGReplayBuffer:
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


class MADDPGCooperativeReplayBuffer:
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
        N    = len(self.buffer)
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


class ParameterSharedMADDPG:
    def __init__(self, input_dim=32, action_dim=19, n_agents=3,
                 actor_lr=0.0005, critic_lr=0.001, gamma=0.99,
                 reward_scale=0.5,
                 min_buffer_size=512,
                 tau=0.01,
                 gumbel_temperature=1.0,
                 use_cer=True, cer_alpha=0.6, cer_beta=2.0):
        self.input_dim          = input_dim
        self.action_dim         = action_dim
        self.n_agents           = n_agents
        self.gamma              = gamma
        self.reward_scale       = reward_scale
        self.min_buffer_size    = min_buffer_size
        self.tau                = tau
        self.gumbel_temperature = gumbel_temperature
        self.use_cer            = use_cer
        self._train_step_count  = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor        = MADDPGActor(input_dim, action_dim).to(self.device)
        self.target_actor = MADDPGActor(input_dim, action_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()

        self.critic        = MADDPGCritic(input_dim, action_dim, n_agents).to(self.device)
        self.target_critic = MADDPGCritic(input_dim, action_dim, n_agents).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        if use_cer:
            self.memory = MADDPGCooperativeReplayBuffer(
                capacity=10000, alpha=cer_alpha, coop_beta=cer_beta,
            )
        else:
            self.memory = MADDPGReplayBuffer(capacity=10000)

        self.batch_size = 64
        self.last_train_metrics: dict | None = None

    def select_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.actor(s)
            probs  = F.softmax(logits, dim=-1)
            dist   = torch.distributions.Categorical(probs)
            return int(dist.sample().item())

    def get_q_values(self, state):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.actor(s).squeeze().cpu().numpy()
            return logits

    def add_experience(self, state, joint_actions, joint_rewards, next_state, done,
                       coop_priority=0.0):
        self.memory.push(state, joint_actions, joint_rewards, next_state, done,
                         coop_priority=coop_priority)

    def soft_update_targets(self):
        for tp, sp in zip(self.target_actor.parameters(), self.actor.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        for tp, sp in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    @staticmethod
    def _to_one_hot(actions_long, n_classes):
        return F.one_hot(actions_long, num_classes=n_classes).float()

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

        states        = torch.FloatTensor(states_np).to(self.device)
        joint_actions = torch.LongTensor(j_actions_np).to(self.device)
        team_rewards  = torch.FloatTensor(
            j_rewards_np.sum(axis=1) * self.reward_scale
        ).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states_np).to(self.device)
        dones = torch.FloatTensor(dones_np).unsqueeze(1).to(self.device)

        joint_actions_oh = self._to_one_hot(joint_actions, self.action_dim)
        joint_actions_flat = joint_actions_oh.view(self.batch_size, -1)

        with torch.no_grad():
            next_logits = self.target_actor(next_states)
            next_probs  = F.softmax(next_logits, dim=-1)
            next_dist   = torch.distributions.Categorical(next_probs)
            next_action_single = next_dist.sample()
            next_action_joint  = next_action_single.unsqueeze(1).expand(-1, self.n_agents)
            next_actions_oh   = self._to_one_hot(next_action_joint, self.action_dim)
            next_actions_flat = next_actions_oh.view(self.batch_size, -1)

            target_q = self.target_critic(next_states, next_actions_flat)
            y = team_rewards + self.gamma * target_q * (1 - dones)

        current_q = self.critic(states, joint_actions_flat)
        critic_huber = nn.SmoothL1Loss(reduction='none')(current_q, y)
        critic_loss  = (is_weights * critic_huber).mean()
        td_errors_maddpg = (y - current_q).detach()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), max_norm=10.0
        )
        self.critic_optimizer.step()

        gumbel_actions = self.actor.gumbel_softmax_sample(
            states, temperature=self.gumbel_temperature, hard=False
        )
        gumbel_joint = gumbel_actions.unsqueeze(1).expand(-1, self.n_agents, -1)
        gumbel_flat  = gumbel_joint.contiguous().view(self.batch_size, -1)

        actor_q = self.critic(states, gumbel_flat)
        actor_loss_main = -(is_weights * actor_q).mean()

        smal_loss = torch.tensor(0.0, device=self.device)
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
                best   = results[0]
                logits = self.actor(states[i].unsqueeze(0))
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                rec_log_probs = []
                for a_rec in best['joint_actions']:
                    if 0 <= a_rec < self.action_dim:
                        rec_log_probs.append(log_probs[a_rec])
                if rec_log_probs:
                    rec_log_p_mean = torch.stack(rec_log_probs).mean()
                    smal_loss = smal_loss - best['weighted_score'] * rec_log_p_mean
                    n_hits += 1
            if n_hits > 0:
                smal_loss = smal_loss / n_hits

        actor_total_loss = actor_loss_main + smal_weight * smal_loss

        self.actor_optimizer.zero_grad()
        actor_total_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=10.0
        )
        self.actor_optimizer.step()

        self._train_step_count += 1
        self.soft_update_targets()

        with torch.no_grad():
            actor_logits = self.actor(states)
            td_abs       = td_errors_maddpg.abs()
            a_gn = float(actor_grad_norm) if actor_grad_norm is not None else 0.0
            c_gn = float(critic_grad_norm) if critic_grad_norm is not None else 0.0
            total_loss_val = float(actor_total_loss.item()) + float(critic_loss.item())
        self.last_train_metrics = {
            "q_mean"        : round(float(actor_logits.mean().item()), 6),
            "q_std"         : round(float(actor_logits.std().item()), 6),
            "q_max"         : round(float(actor_logits.max().item()), 6),
            "q_min"         : round(float(actor_logits.min().item()), 6),
            "q_range"       : round(float(actor_logits.max().item() - actor_logits.min().item()), 6),
            "target_q_mean" : round(float(y.detach().mean().item()), 6),
            "td_error_mean" : round(float(td_abs.mean().item()), 6),
            "td_error_std"  : round(float(td_abs.std().item()), 6),
            "td_error_max"  : round(float(td_abs.max().item()), 6),
            "loss_total"    : round(total_loss_val, 6),
            "loss_dqn"      : None,
            "loss_mixer"    : None,
            "loss_actor"    : round(float(actor_loss_main.item()), 6),
            "loss_critic"   : round(float(critic_loss.item()), 6),
            "loss_smal_aux" : round(float((smal_weight * smal_loss).item()), 6),
            "grad_norm_actor" : round(a_gn, 6),
            "grad_norm_critic": round(c_gn, 6),
            "grad_norm_mixer" : None,
            "grad_clip_ratio" : 1.0 if (a_gn > 10.0 or c_gn > 10.0) else 0.0,
            "buffer_size"          : int(len(self.memory)),
            "buffer_avg_priority"  : round(float(self.memory.avg_priority()), 6)
                                     if hasattr(self.memory, "avg_priority") else 0.0,
            "buffer_priority_std"  : round(float(self.memory.priority_std()), 6)
                                     if hasattr(self.memory, "priority_std") else 0.0,
        }

    def state_dict(self):
        return {
            'actor'  : self.actor.state_dict(),
            'critic' : self.critic.state_dict(),
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state['actor'])
        self.target_actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.target_critic.load_state_dict(state['critic'])
