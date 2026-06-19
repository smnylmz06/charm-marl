import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class SharedDQN(nn.Module):
    def __init__(self, input_dim=32, output_dim=19):
        super(SharedDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.value_head     = nn.Linear(64, 1)
        self.advantage_head = nn.Linear(64, output_dim)

        nn.init.orthogonal_(self.value_head.weight,     gain=1.0)
        nn.init.constant_(self.value_head.bias,         0.0)
        nn.init.orthogonal_(self.advantage_head.weight, gain=0.5)
        nn.init.constant_(self.advantage_head.bias,     0.0)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        f = self.feature(x)
        v = self.value_head(f)
        a = self.advantage_head(f)
        q = v + (a - a.mean(dim=-1, keepdim=True))

        if squeeze_back:
            q = q.squeeze(0)
        return q

class DQNReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, coop_priority=0.0):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class CooperativeReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, coop_beta=2.0, eps=1e-3,
                 is_beta_start=0.4, is_beta_end=1.0, is_beta_steps=100_000):
        self.capacity   = capacity
        self.alpha      = alpha
        self.coop_beta  = coop_beta
        self.eps        = eps
        self.buffer     = []
        self.priorities = []
        self.is_beta_start = is_beta_start
        self.is_beta_end   = is_beta_end
        self.is_beta_steps = is_beta_steps
        self._sample_step  = 0

    def push(self, state, action, reward, next_state, done, coop_priority=0.0):
        priority = (abs(float(reward)) ** self.alpha
                    + self.coop_beta * float(coop_priority)
                    + self.eps)

        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)

        self.buffer.append((state, action, reward, next_state, done))
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
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            (np.array(states), np.array(actions), np.array(rewards),
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

class ParameterSharedMARL:
    def __init__(self, input_dim=26, action_dim=19, lr=0.001, gamma=0.99,
                 reward_scale=0.5,
                 min_buffer_size=512,
                 target_update_freq=200,
                 use_cer=False,
                 cer_alpha=0.6, cer_beta=2.0,
                 weight_decay=0.0,
                 rand_aux=False):
        self.action_dim = action_dim
        self.gamma = gamma
        self.reward_scale       = reward_scale
        self.min_buffer_size    = min_buffer_size
        self.target_update_freq = target_update_freq
        self._train_step_count  = 0
        self.use_cer            = use_cer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = SharedDQN(input_dim, action_dim).to(self.device)
        self.target_net = SharedDQN(input_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.rand_aux = rand_aux
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr,
                                    weight_decay=weight_decay)
        if use_cer:
            self.memory = CooperativeReplayBuffer(
                capacity=10000, alpha=cer_alpha, coop_beta=cer_beta
            )
        else:
            self.memory = DQNReplayBuffer()
        self.batch_size = 64
        self.last_train_metrics: dict | None = None

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def add_experience(self, state, action, reward, next_state, done, coop_priority=0.0):
        self.memory.push(state, action, reward, next_state, done, coop_priority=coop_priority)

    def train_step(self, joint_memory=None, smal_weight=0.1, role_tuple=None):
        if len(self.memory) < self.min_buffer_size:
            self.last_train_metrics = None
            return

        sample_result = self.memory.sample(self.batch_size)
        if isinstance(sample_result, tuple) and len(sample_result) == 2 and \
           isinstance(sample_result[0], tuple):
            (states_np, actions_np, rewards_np, next_states_np, dones_np), is_weights_np = sample_result
            if states_np is None:
                return
            is_weights = torch.FloatTensor(is_weights_np).unsqueeze(1).to(self.device)
        else:
            states_np, actions_np, rewards_np, next_states_np, dones_np = sample_result
            is_weights = torch.ones(self.batch_size, 1, device=self.device)

        states = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor(actions_np).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards_np * self.reward_scale).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states_np).to(self.device)
        dones = torch.FloatTensor(dones_np).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            max_next_q   = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        td_errors = current_q - target_q
        huber = nn.SmoothL1Loss(reduction='none')(current_q, target_q)
        loss = (is_weights * huber).mean()

        aux_loss = torch.tensor(0.0, device=self.device)
        if joint_memory is not None and smal_weight > 0:
            n_hits = 0
            rt = role_tuple if role_tuple is not None else ['DF','MF','FW']
            for i in range(self.batch_size):
                state_np = states[i].detach().cpu().numpy()
                results = joint_memory.query(state_np, role_tuple=rt,
                                              k=1, min_similarity=0.50,
                                              for_smal=True)
                if not results:
                    continue
                best = results[0]
                q_row = self.policy_net(states[i].unsqueeze(0)).squeeze()
                max_q = q_row.max()
                recommended_qs = []
                for a_rec in best['joint_actions']:
                    if 0 <= a_rec < q_row.shape[0]:
                        recommended_qs.append(q_row[a_rec])
                if recommended_qs:
                    rec_q_mean = torch.stack(recommended_qs).mean()
                    aux_loss = aux_loss + best['weighted_score'] * (max_q - rec_q_mean)
                    n_hits += 1
            if n_hits > 0:
                aux_loss = aux_loss / n_hits

        if joint_memory is None and getattr(self, "rand_aux", False) and smal_weight > 0:
            q_batch = self.policy_net(states)
            max_q   = q_batch.max(dim=1, keepdim=True).values
            rand_a  = torch.randint(0, self.action_dim,
                                    (states.shape[0], 1), device=self.device)
            rand_q  = q_batch.gather(1, rand_a)
            aux_loss = 0.5 * (max_q - rand_q).mean()

        total_loss = loss + smal_weight * aux_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=10.0
        )
        self.optimizer.step()

        self._train_step_count += 1
        if self._train_step_count % self.target_update_freq == 0:
            self.update_target_network()

        with torch.no_grad():
            q_all_batch = self.policy_net(states)
            td_abs      = td_errors.detach().abs()
            grad_norm_f = float(grad_norm) if grad_norm is not None else 0.0
        self.last_train_metrics = {
            "q_mean"        : round(float(q_all_batch.mean().item()), 6),
            "q_std"         : round(float(q_all_batch.std().item()), 6),
            "q_max"         : round(float(q_all_batch.max().item()), 6),
            "q_min"         : round(float(q_all_batch.min().item()), 6),
            "q_range"       : round(float(q_all_batch.max().item() - q_all_batch.min().item()), 6),
            "target_q_mean" : round(float(target_q.detach().mean().item()), 6),
            "td_error_mean" : round(float(td_abs.mean().item()), 6),
            "td_error_std"  : round(float(td_abs.std().item()), 6),
            "td_error_max"  : round(float(td_abs.max().item()), 6),
            "loss_total"    : round(float(total_loss.item()), 6),
            "loss_dqn"      : round(float(loss.item()), 6),
            "loss_mixer"    : None,
            "loss_actor"    : None,
            "loss_critic"   : None,
            "loss_smal_aux" : round(float((smal_weight * aux_loss).item()), 6),
            "grad_norm_actor" : round(grad_norm_f, 6),
            "grad_norm_critic": None,
            "grad_norm_mixer" : None,
            "grad_clip_ratio" : 1.0 if grad_norm_f > 10.0 else 0.0,
            "buffer_size"          : int(len(self.memory)),
            "buffer_avg_priority"  : round(float(self.memory.avg_priority()), 6)
                                     if hasattr(self.memory, "avg_priority") else 0.0,
            "buffer_priority_std"  : round(float(self.memory.priority_std()), 6)
                                     if hasattr(self.memory, "priority_std") else 0.0,
        }

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
