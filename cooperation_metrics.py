from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from math import factorial
from itertools import combinations
from typing import Callable, Dict, List, Optional


def compute_policy_entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits.detach(), dim=-1)
    return float(-(probs * (probs + 1e-8).log()).sum().item())


def compute_td_error(q_current: torch.Tensor, q_target: torch.Tensor) -> float:
    return float((q_target.detach() - q_current.detach()).abs().mean().item())


def compute_coma_advantage(
    policy_net,
    state: torch.Tensor,
    actions: List[int],
    agent_i: int,
    action_dim: int,
) -> float:
    with torch.no_grad():
        logits = policy_net(state.unsqueeze(0)).squeeze()
        probs  = F.softmax(logits, dim=-1)

        q_actual = logits[actions[agent_i]].item()

        baseline = float((probs * logits).sum().item())

    return round(q_actual - baseline, 4)


def compute_shapley_values(
    reward_fn: Callable[[List[int], dict], float],
    n_agents: int,
    obs: dict,
) -> List[float]:
    agents       = list(range(n_agents))
    shapley_vals = []

    for agent_i in agents:
        others = [a for a in agents if a != agent_i]
        phi    = 0.0

        for r in range(len(others) + 1):
            weight = (factorial(r) * factorial(n_agents - r - 1)) / factorial(n_agents)
            for S in combinations(others, r):
                S_list     = list(S)
                v_with     = reward_fn(S_list + [agent_i], obs)
                v_without  = reward_fn(S_list, obs)
                phi       += weight * (v_with - v_without)

        shapley_vals.append(round(phi, 4))

    return shapley_vals


def compute_social_influence(
    action_log_i: List[int],
    action_log_j: List[int],
    lag: int = 1,
) -> float:
    min_len = max(lag, 1)
    if len(action_log_i) < min_len + 2:
        return 0.0

    if lag == 0:
        a_i = np.array(action_log_i)
        a_j = np.array(action_log_j)
    else:
        a_i = np.array(action_log_i[:-lag])
        a_j = np.array(action_log_j[lag:])
    min_len = min(len(a_i), len(a_j))
    a_i, a_j = a_i[:min_len], a_j[:min_len]

    def _mi(x: np.ndarray, y: np.ndarray) -> float:
        n = len(x)
        joint: dict = {}
        px: dict    = {}
        py: dict    = {}
        for xi, yi in zip(x, y):
            joint[(xi, yi)] = joint.get((xi, yi), 0) + 1
            px[xi]          = px.get(xi, 0) + 1
            py[yi]          = py.get(yi, 0) + 1
        mi = 0.0
        for (xi, yi), cnt in joint.items():
            p_xy = cnt / n
            p_x  = px[xi] / n
            p_y  = py[yi] / n
            mi  += p_xy * np.log(p_xy / (p_x * p_y) + 1e-10)
        return float(mi)

    mi  = _mi(a_i, a_j)
    h_j = _mi(a_j, a_j)

    return float(round(mi / max(h_j, 1e-8), 4))


def compute_joint_entropy_and_coordination(
    action_logs: Dict[int, List[int]],
    n_actions: int = 19,
) -> Dict[str, object]:
    individual_entropies = []

    for log in action_logs.values():
        if not log:
            individual_entropies.append(float(np.log(n_actions)))
            continue
        counts = np.bincount(log, minlength=n_actions).astype(float)
        probs  = counts / max(counts.sum(), 1)
        h      = float(-(probs * np.log(probs + 1e-8)).sum())
        individual_entropies.append(h)

    return {
        "joint_entropy"       : round(float(sum(individual_entropies)), 4),
        "coordination_score"  : round(float(np.std(individual_entropies)), 4),
        "individual_entropies": [round(h, 4) for h in individual_entropies],
    }


def compute_team_spirit_reward(
    individual_rewards: List[float],
    team_spirit: float,
) -> List[float]:
    team_avg = float(np.mean(individual_rewards))
    return [
        (1.0 - team_spirit) * r + team_spirit * team_avg
        for r in individual_rewards
    ]


def compute_off_ball_quality(
    obs: dict,
    agent_i: int,
    actions: List[int],
    move_action_ids: tuple = tuple(range(1, 9)),
) -> float:
    is_offball = (
        obs['ball_owned_team'] != 0
        or obs['ball_owned_player'] != agent_i
    )
    if not (is_offball and actions[agent_i] in move_action_ids):
        return 0.0

    my_pos      = obs['left_team'][agent_i][:2]
    nearby_opp  = sum(
        np.linalg.norm(my_pos - opp[:2]) < 0.15
        for opp in obs['right_team']
    )
    return float(max(0.0, 1.0 - nearby_opp * 0.3))


def compute_formation_compactness(obs: dict, team: str = "left") -> float:
    pos    = obs[f"{team}_team"][:, :2]
    length = float(pos[:, 0].max() - pos[:, 0].min())
    width  = float(pos[:, 1].max() - pos[:, 1].min())
    area   = max(length * width, 1e-6)
    return float(min(1.0 / (area * 10.0), 1.0))


def compute_team_spread(obs: dict, team: str = "left") -> float:
    pos = obs[f"{team}_team"][:, :2]
    n   = len(pos)
    if n < 2:
        return 0.0

    total = sum(
        np.linalg.norm(pos[i] - pos[j])
        for i in range(n)
        for j in range(i + 1, n)
    )
    return float(total / (n * (n - 1) / 2))


def compute_role_specialization(
    action_history: List[List[str]],
    n_actions: int = 19,
) -> Dict[str, object]:
    H_uniform = float(np.log(n_actions))
    result    = {}

    for agent_id, log in enumerate(action_history):
        if not log:
            result[str(agent_id)] = {
                "specialization": 0.0,
                "entropy"       : round(H_uniform, 4),
                "dominant_action": "N/A",
            }
            continue

        unique  = sorted(set(log))
        a2i     = {a: i for i, a in enumerate(unique)}
        ids     = [a2i[a] for a in log]
        counts  = np.bincount(ids, minlength=len(unique)).astype(float)
        probs   = counts / counts.sum()
        H       = float(-(probs * np.log(probs + 1e-8)).sum())

        result[str(agent_id)] = {
            "specialization" : round(1.0 - H / H_uniform, 4),
            "entropy"        : round(H, 4),
            "dominant_action": unique[int(np.argmax(counts))],
        }

    return result


def compute_protocol_score(
    action_log_i: List[str],
    action_log_j: List[str],
    lag: int = 1,
) -> float:
    if len(action_log_i) < lag + 2 or len(action_log_j) < lag + 2:
        return 0.0

    all_actions = sorted(set(action_log_i) | set(action_log_j))
    a2i         = {a: i for i, a in enumerate(all_actions)}

    a_i = np.array([a2i[a] for a in action_log_i[:-lag]])
    a_j = np.array([a2i[a] for a in action_log_j[lag:]])
    n   = min(len(a_i), len(a_j))
    if n < 2:
        return 0.0
    a_i, a_j = a_i[:n], a_j[:n]

    def _mi(x: np.ndarray, y: np.ndarray) -> float:
        total = len(x)
        joint: dict = {}
        px: dict    = {}
        py: dict    = {}
        for xi, yi in zip(x, y):
            joint[(int(xi), int(yi))] = joint.get((int(xi), int(yi)), 0) + 1
            px[int(xi)] = px.get(int(xi), 0) + 1
            py[int(yi)] = py.get(int(yi), 0) + 1
        mi = 0.0
        for (xi, yi), cnt in joint.items():
            p_xy = cnt / total
            p_x  = px[xi] / total
            p_y  = py[yi] / total
            mi  += p_xy * np.log(p_xy / (p_x * p_y) + 1e-10)
        return float(mi)

    mi  = _mi(a_i, a_j)
    h_j = _mi(a_j, a_j)
    return float(round(mi / max(h_j, 1e-8), 4))


def compute_q_value_stats(
    policy_net,
    state: torch.Tensor,
) -> dict:
    with torch.no_grad():
        q = policy_net(state.unsqueeze(0)).squeeze()
    return {
        "q_mean" : float(q.mean().item()),
        "q_std"  : float(q.std().item()),
        "q_max"  : float(q.max().item()),
        "q_min"  : float(q.min().item()),
        "q_range": float((q.max() - q.min()).item()),
    }


def triangle_formation_bonus(
    positions: np.ndarray,
    optimal_min: float = 0.15,
    optimal_max: float = 0.40,
) -> float:
    n = len(positions)
    if n < 2:
        return 0.0

    pairs_in_range = 0
    total_pairs    = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            total_pairs += 1
            if optimal_min <= d <= optimal_max:
                pairs_in_range += 1

    return float(pairs_in_range / max(total_pairs, 1))


def off_ball_movement_reward(
    obs: dict,
    agent_i: int,
    actions: list,
    move_action_ids: tuple = tuple(range(1, 9)),
) -> float:
    return compute_off_ball_quality(obs, agent_i, actions, move_action_ids)


def wall_pass_bonus(
    recent_passes: list,
    new_pass_from: int,
    new_pass_to: int,
    current_step: int = 0,
    max_lag: int = 8,
) -> float:
    if not recent_passes:
        return 0.0

    last = recent_passes[-1]
    if (last.get("to")   == new_pass_from
            and last.get("from") == new_pass_to
            and (current_step - last.get("step", 0)) <= max_lag):
        return 1.0
    return 0.0


def pass_network_diversity_bonus(
    pass_pairs: set,
    n_agents: int,
) -> float:
    max_pairs = n_agents * (n_agents - 1)
    return float(len(pass_pairs) / max(max_pairs, 1))


def compute_cooperation_reward(
    obs            : dict,
    actions        : list,
    agent_i        : int,
    recent_passes  : list,
    current_pass   : tuple = None,
    current_step   : int   = 0,
    n_agents       : int   = 3,
    w_triangle     : float = 0.20,
    w_offball      : float = 0.30,
    w_wallpass     : float = 5.00,
) -> float:
    r = 0.0

    positions = obs['left_team'][:n_agents, :2]
    r += w_triangle * triangle_formation_bonus(positions)

    r += w_offball * off_ball_movement_reward(obs, agent_i, actions)

    if current_pass is not None and agent_i in current_pass:
        from_p, to_p = current_pass
        r += w_wallpass * wall_pass_bonus(
            recent_passes, from_p, to_p, current_step
        )

    return r


def compute_behavioral_entropy(
    epsilon: float,
    action_dim: int = 19,
    policy_entropy: float = 0.0,
) -> float:
    uniform_h = float(np.log(action_dim))
    return float(round(
        epsilon * uniform_h + (1.0 - epsilon) * policy_entropy, 4
    ))


def compute_cooperation_index(metrics: dict) -> float:
    pcr = float(metrics.get("pass_completion_rate", 0.0))

    counts = list(metrics.get("pass_counts_per_agent", {}).values())
    total  = sum(counts)
    if total > 0:
        max_share     = max(counts) / total
        pass_equality = 1.0 - max_share
    else:
        pass_equality = 0.0

    si = list(metrics.get("social_influence", {}).values())
    si_avg = float(sum(si) / len(si)) if si else 0.0

    coord = min(float(metrics.get("coordination_score", 0.0)) / 0.5, 1.0)

    offball = float(metrics.get("avg_offball_quality", 0.0))

    compact = float(metrics.get("avg_compactness", 0.0))

    coop_idx = (
        0.30 * pcr
        + 0.20 * pass_equality
        + 0.15 * si_avg
        + 0.15 * coord
        + 0.10 * offball
        + 0.10 * compact
    )
    return float(round(min(coop_idx, 1.0), 4))


def hindsight_relabel(
    team_snapshots: list,
    top_k: int = 3,
    fallback_quality: float = 0.4,
    pass_quality: float = 0.65,
    goal_quality: float = 1.0,
) -> list:
    if not team_snapshots:
        return []

    def _score(s):
        if s.get("goal_scored"):       return (2, s.get("step_reward", 0))
        elif s.get("had_pass"):        return (1, s.get("step_reward", 0))
        else:                          return (0, s.get("step_reward", 0))

    sorted_snaps = sorted(team_snapshots, key=_score, reverse=True)
    selected     = sorted_snaps[:top_k]

    relabeled = []
    for snap in selected:
        if snap.get("goal_scored"):
            quality = goal_quality
            label   = "goal_actual"
        elif snap.get("had_pass"):
            quality = pass_quality
            label   = "pass_actual"
        else:
            quality = fallback_quality
            label   = "hindsight_imagined"
        relabeled.append({
            "team_state"         : snap["team_state"],
            "joint_actions"      : snap["joint_actions"],
            "role_tuple"         : snap.get("role_tuple", []),
            "cooperation_quality": quality,
            "reward"             : snap.get("step_reward", 0.0),
            "hindsight_label"    : label,
        })
    return relabeled


def compute_charm_summary(metrics: dict) -> dict:
    return {
        "jacm_size"      : metrics.get("jacm_size", 0),
        "jacm_hit_rate"  : metrics.get("jacm_hit_rate", 0.0),
        "rcmp_partitions": metrics.get("jacm_partition_count", 0),
        "hcm_relabels"   : metrics.get("hcm_relabels_per_episode", 0),
        "cer_priority"   : metrics.get("cer_avg_priority", 0.0),
        "smal_aux_loss"  : metrics.get("smal_aux_loss_avg", 0.0),
        "cooperation_index": metrics.get("cooperation_index", 0.0),
    }


def analyze_memory_impact(rows: List[dict], window: int = 50) -> dict:
    if len(rows) < 10:
        return {"error": "Insufficient data (< 10 matches)"}

    rows = [r for r in rows if not r.get("memory_disabled", False)]
    if not rows:
        return {"error": "Memory disabled in all matches"}

    use_rates = [float(r.get("memory_use_rate", 0.0)) for r in rows]
    coop_idxs = [float(r.get("cooperation_index", 0.0)) for r in rows]

    sorted_pairs = sorted(zip(use_rates, coop_idxs))
    n = len(sorted_pairs)
    mid = n // 2
    low_half  = sorted_pairs[:mid]
    high_half = sorted_pairs[mid:]

    coop_low  = sum(c for _, c in low_half)  / max(len(low_half), 1)
    coop_high = sum(c for _, c in high_half) / max(len(high_half), 1)

    mean_u = sum(use_rates) / n
    mean_c = sum(coop_idxs) / n
    num    = sum((u - mean_u) * (c - mean_c) for u, c in zip(use_rates, coop_idxs))
    den_u  = (sum((u - mean_u) ** 2 for u in use_rates)) ** 0.5
    den_c  = (sum((c - mean_c) ** 2 for c in coop_idxs)) ** 0.5
    pearson_r = num / (den_u * den_c) if (den_u * den_c) > 0 else 0.0

    layer_totals = {"coach": 0, "selected": 0, "agent_self": 0}
    for r in rows:
        for k, v in r.get("memory_layer_usage", {}).items():
            if k in layer_totals:
                layer_totals[k] += v
    total_layer = sum(layer_totals.values())
    layer_dist = {
        k: round(v / max(total_layer, 1), 4)
        for k, v in layer_totals.items()
    }

    return {
        "n_episodes_analyzed": len(rows),
        "memory_use_rate_avg": round(sum(use_rates) / n, 4),
        "coop_high_use"      : round(coop_high, 4),
        "coop_low_use"       : round(coop_low, 4),
        "memory_lift"        : round(coop_high - coop_low, 4),
        "correlation"        : round(pearson_r, 4),
        "layer_dominance"    : layer_dist,
    }


def compute_all_episode_metrics(
    action_history     : List[List[str]],
    step_spread_log    : List[float],
    step_compactness_log: List[float],
    step_offball_log   : List[float],
    pass_details       : List[dict],
    n_agents           : int,
    pass_attempts      : int = 0,
    compute_expensive  : bool = False,
    obs_for_shapley    : Optional[dict] = None,
    reward_fn_for_shapley: Optional[Callable] = None,
    epsilon            : float = 0.0,
) -> dict:
    total_passes = len(pass_details)
    pass_counts  = {str(i): 0 for i in range(n_agents)}
    for p in pass_details:
        key = str(p.get("from", -1))
        if key in pass_counts:
            pass_counts[key] += 1
    real_pcr = round(total_passes / pass_attempts, 4) if pass_attempts > 0 else 0.0

    avg_spread       = float(np.mean(step_spread_log))       if step_spread_log       else 0.0
    avg_compactness  = float(np.mean(step_compactness_log))  if step_compactness_log  else 0.0
    avg_offball      = float(np.mean(step_offball_log))       if step_offball_log      else 0.0

    int_logs = {}
    for agent_id, log in enumerate(action_history):
        unique   = sorted(set(log)) if log else ["_"]
        a2i      = {a: i for i, a in enumerate(unique)}
        int_logs[agent_id] = [a2i[a] for a in log] if log else [0]

    entropy_data = compute_joint_entropy_and_coordination(int_logs)

    specialization = compute_role_specialization(action_history)

    protocol_scores: dict = {}
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            protocol_scores[f"{i}→{j}"] = compute_social_influence(
                int_logs[i], int_logs[j], lag=1
            )

    influence_scores: dict = {}
    for i in range(min(n_agents, 4)):
        for j in range(min(n_agents, 4)):
            if i != j:
                influence_scores[f"{i}→{j}"] = compute_social_influence(
                    int_logs[i], int_logs[j], lag=1
                )

    shapley_vals = None
    if compute_expensive and obs_for_shapley and reward_fn_for_shapley:
        shapley_vals = compute_shapley_values(
            reward_fn_for_shapley, n_agents, obs_for_shapley
        )

    output = {
        "total_passes"         : total_passes,
        "pass_attempts"        : pass_attempts,
        "pass_completion_rate" : real_pcr,
        "pass_counts_per_agent": pass_counts,
        "avg_team_spread"      : round(avg_spread, 4),
        "avg_compactness"      : round(avg_compactness, 4),
        "avg_offball_quality"  : round(avg_offball, 4),
        "joint_entropy"        : entropy_data["joint_entropy"],
        "coordination_score"   : entropy_data["coordination_score"],
        "individual_entropies" : entropy_data["individual_entropies"],
        "role_specialization"  : specialization,
        "protocol_scores"      : protocol_scores,
        "social_influence"     : influence_scores,
        "shapley_values"       : shapley_vals,
    }

    output["cooperation_index"]  = compute_cooperation_index(output)
    output["behavioral_entropy"] = compute_behavioral_entropy(
        epsilon=epsilon, action_dim=19
    )

    return output
