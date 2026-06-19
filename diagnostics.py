from __future__ import annotations

import os
import json
import math
import time
import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Any


def _gini(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.array([float(v) for v in values], dtype=np.float64)
    if arr.sum() <= 1e-9:
        return 0.0
    arr = np.sort(np.abs(arr))
    n = len(arr)
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def _pass_network_metrics(pass_details: List[dict], n_agents: int) -> Dict[str, float]:
    M = np.zeros((n_agents, n_agents), dtype=np.float64)
    for p in pass_details:
        u = p.get("from", -1)
        v = p.get("to", -1)
        if 0 <= u < n_agents and 0 <= v < n_agents and u != v:
            M[u, v] += 1.0

    total_edges = float((M > 0).sum())
    max_edges = n_agents * (n_agents - 1)
    density = total_edges / max_edges if max_edges > 0 else 0.0

    out_counts = M.sum(axis=1).tolist()
    centralization = _gini(out_counts)

    M_und = ((M + M.T) > 0).astype(np.float64)
    triangles = 0
    triples = 0
    for i in range(n_agents):
        nbrs = [j for j in range(n_agents) if i != j and M_und[i, j] > 0]
        if len(nbrs) < 2:
            continue
        for a, b in combinations(nbrs, 2):
            triples += 1
            if M_und[a, b] > 0:
                triangles += 1
    clustering = (triangles / triples) if triples > 0 else 0.0

    return {
        "density": round(density, 4),
        "centralization": round(centralization, 4),
        "clustering_coef": round(clustering, 4),
    }


def _convex_hull_area(points: List[List[float]]) -> float:
    if len(points) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(np.array(points))
        return float(hull.volume)
    except Exception:
        pts = np.array(points)
        w = float(pts[:, 0].max() - pts[:, 0].min())
        h = float(pts[:, 1].max() - pts[:, 1].min())
        return 0.5 * w * h


def _team_centroid(points: List[List[float]]) -> tuple:
    if not points:
        return (0.0, 0.0)
    arr = np.array(points)
    return (float(arr[:, 0].mean()), float(arr[:, 1].mean()))


def _possession_share(possession_time: List[int]) -> float:
    total = sum(possession_time)
    if total <= 0:
        return 0.0
    return float(total / max(total + 1, 1))


def _dynamic_role(player_pos_x: float) -> str:
    if player_pos_x < -0.3:
        return "DF"
    elif player_pos_x > 0.3:
        return "FW"
    return "MF"


def _count_role_switches(role_sequence: List[List[str]]) -> int:
    if len(role_sequence) < 2:
        return 0
    switches = 0
    for t in range(1, len(role_sequence)):
        prev = role_sequence[t - 1]
        curr = role_sequence[t]
        for a in range(min(len(prev), len(curr))):
            if prev[a] != curr[a]:
                switches += 1
    return switches


def _action_responsiveness(action_logs: Dict[int, List[int]],
                            state_change_logs: Optional[List[float]] = None) -> float:
    agents = sorted(action_logs.keys())
    if len(agents) < 2:
        return 0.0
    scores = []
    for i in agents:
        for j in agents:
            if i == j:
                continue
            log_i = action_logs[i]
            log_j = action_logs[j]
            L = min(len(log_i), len(log_j)) - 1
            if L <= 0:
                continue
            j_changed = [k for k in range(L) if log_j[k] != log_j[k + 1]]
            if not j_changed:
                continue
            i_responses = sum(1 for k in j_changed if log_i[k] != log_i[k + 1])
            scores.append(i_responses / len(j_changed))
    return float(round(np.mean(scores), 4)) if scores else 0.0


def _percentile_quantiles(values: List[float],
                           qs=(0.10, 0.25, 0.50, 0.75, 0.90)) -> List[float]:
    if not values:
        return [0.0 for _ in qs]
    arr = np.array(values, dtype=np.float64)
    return [float(round(np.quantile(arr, q), 6)) for q in qs]


class TrainingDiagnostics:

    def __init__(self, output_path: str):
        self.output_path = output_path
        self._buf: List[dict] = []
        self._global_step = 0

    def collect(self, marl: Any):
        m = getattr(marl, "last_train_metrics", None)
        if not m:
            return
        self._global_step += 1
        snapshot = {**m, "step_count": self._global_step}
        self._buf.append(snapshot)

    def flush_to_disk(self, match: int):
        if not self._buf:
            return
        n = len(self._buf)
        agg: Dict[str, Any] = {"match": match}
        all_keys: set = set()
        for snap in self._buf:
            all_keys.update(snap.keys())

        if "step_count" in all_keys:
            agg["step_count"] = int(self._buf[-1]["step_count"])
            all_keys.discard("step_count")

        if "grad_clip_ratio" in all_keys:
            vals = [snap.get("grad_clip_ratio", 0.0) for snap in self._buf
                    if snap.get("grad_clip_ratio") is not None]
            agg["grad_clip_ratio"] = round(float(np.mean(vals)), 4) if vals else 0.0
            all_keys.discard("grad_clip_ratio")

        if "buffer_size" in all_keys:
            agg["buffer_size"] = int(self._buf[-1].get("buffer_size", 0))
            all_keys.discard("buffer_size")

        for k in sorted(all_keys):
            vals = [snap[k] for snap in self._buf
                    if k in snap and snap[k] is not None]
            if not vals:
                agg[k] = None
                continue
            try:
                agg[k] = round(float(np.mean(vals)), 6)
            except (TypeError, ValueError):
                agg[k] = vals[-1]

        _append_jsonl(self.output_path, agg)
        self._buf.clear()


class MemoryDiagnostics:

    def __init__(self, output_path: str, flush_interval: int = 1):
        self.output_path = output_path
        self.flush_interval = flush_interval
        self._hcm_cum = {"goal": 0, "pass": 0, "fallback": 0, "total": 0}
        self._hcm_episodes = 0
        self._smal_aux_sum = 0.0
        self._smal_aux_count = 0

    def update_smal_aux(self, aux_loss: Optional[float]):
        if aux_loss is None:
            return
        self._smal_aux_sum += float(aux_loss)
        self._smal_aux_count += 1

    def flush_episode(self, match: int, trainer: Any, hcm_breakdown: Dict[str, int]):
        if (match % self.flush_interval) != 0:
            return

        for k in ("goal", "pass", "fallback"):
            self._hcm_cum[k] += int(hcm_breakdown.get(k, 0))
        self._hcm_cum["total"] += int(hcm_breakdown.get("total", 0))
        self._hcm_episodes += 1

        mm = trainer.mem
        joint = mm.joint
        marl_buf = trainer.marl.memory

        jacm_size = len(joint)
        jacm_n_part = len(joint.partitions)
        jacm_q_total = int(getattr(joint, "_cum_query_count", 0))
        jacm_h_total = int(getattr(joint, "_cum_hit_count", 0))
        jacm_hit_rate = round(jacm_h_total / max(jacm_q_total, 1), 4)
        jacm_avg_sim = round(float(getattr(joint, "_cum_sim_mean", 0.0)), 4)
        jacm_avg_qual = round(float(getattr(joint, "_cum_qual_mean", 0.0)), 4)
        jacm_part_dist = joint.partition_summary()

        smal_phase = "warmup"
        if hasattr(trainer, "_current_smal_weight"):
            ep = match - 1
            w_eps = trainer.smal_warmup_episodes
            f_eps = trainer.smal_full_episodes
            if ep < w_eps:
                smal_phase = "warmup"
            elif ep < f_eps:
                smal_phase = "ramp"
            else:
                smal_phase = "full"
            smal_w = round(float(trainer._current_smal_weight(ep)), 4)
        else:
            smal_w = 0.0
        smal_aux_avg = (round(self._smal_aux_sum / self._smal_aux_count, 6)
                        if self._smal_aux_count > 0 else 0.0)
        smal_active = self._smal_aux_count
        self._smal_aux_sum = 0.0

        hcm_per_ep = round(self._hcm_cum["total"] / max(self._hcm_episodes, 1), 4)

        cer_avg_p = 0.0
        cer_quantiles = [0.0, 0.0, 0.0, 0.0, 0.0]
        cer_alpha = None
        cer_beta = None
        if hasattr(marl_buf, "priorities"):
            priorities = list(marl_buf.priorities)
            if priorities:
                cer_avg_p = round(float(np.mean(priorities)), 6)
                cer_quantiles = _percentile_quantiles(priorities)
            cer_alpha = float(getattr(marl_buf, "alpha", 0.0))
            cer_beta = float(getattr(marl_buf, "coop_beta", 0.0))

        hier_enabled = bool(mm.enabled)
        coach_size = len(mm.coach.experiences)
        selected_size = len(mm.selected.experiences)
        agent_self_sizes = [len(s.experiences) for s in mm.agent_self[:3]]

        rec = {
            "match": match,
            "jacm_size": jacm_size,
            "jacm_num_partitions": jacm_n_part,
            "jacm_queries_total": jacm_q_total,
            "jacm_hits_total": jacm_h_total,
            "jacm_hit_rate": jacm_hit_rate,
            "jacm_avg_similarity": jacm_avg_sim,
            "jacm_avg_coop_quality": jacm_avg_qual,
            "jacm_partitions_distribution": jacm_part_dist,
            "smal_phase": smal_phase,
            "smal_weight_current": smal_w,
            "smal_active_steps": smal_active,
            "smal_aux_loss_avg": smal_aux_avg,
            "hcm_relabels_total": int(self._hcm_cum["total"]),
            "hcm_relabels_per_episode": hcm_per_ep,
            "hcm_goal_replays": int(self._hcm_cum["goal"]),
            "hcm_pass_replays": int(self._hcm_cum["pass"]),
            "hcm_fallback_replays": int(self._hcm_cum["fallback"]),
            "cer_avg_priority": cer_avg_p,
            "cer_priority_quantiles": cer_quantiles,
            "cer_alpha": cer_alpha,
            "cer_beta": cer_beta,
            "hier_enabled": hier_enabled,
            "coach_size": coach_size,
            "selected_size": selected_size,
            "agent_self_sizes": agent_self_sizes,
        }

        if hasattr(mm.joint, "get_hybrid_diagnostics"):
            try:
                rec.update(mm.joint.get_hybrid_diagnostics())
                if hasattr(mm.joint, "get_t3_cer_diagnostics"):
                    rec.update(mm.joint.get_t3_cer_diagnostics())
            except Exception as e:
                print(f"  ⚠️  hybrid diagnostics read error: {e}")
                rec["hybrid_enabled"] = False
        else:
            rec["hybrid_enabled"] = False
        _append_jsonl(self.output_path, rec)


class Cooperation5DimDiagnostics:

    def __init__(self, output_path: str):
        self.output_path = output_path

    def flush_episode(self,
                       match: int,
                       env: Any,
                       trainer: Any,
                       legacy_coop_metrics: dict):
        n_agents = env.num_agents
        score = list(env.prev_score) if hasattr(env.prev_score, "__iter__") else [0, 0]
        goals_for = int(score[0])
        goals_against = int(score[1]) if len(score) > 1 else 0

        shots_arr = env.match_metrics.get("shots", [])
        opp_shots_arr = env.match_metrics.get("opp_shots", [])
        perf_shots = len(shots_arr)
        perf_shots_on_t = sum(1 for s in shots_arr if s.get("xG", 0.0) > 0.1)
        perf_xg_for = round(sum(s.get("xG", 0.0) for s in shots_arr), 4)
        perf_xg_against = round(sum(s.get("xG", 0.0) for s in opp_shots_arr), 4)

        from cooperation_metrics import compute_social_influence
        action_history = env.action_history
        int_logs: Dict[int, List[int]] = {}
        for agent_id, log in enumerate(action_history):
            unique = sorted(set(log)) if log else ["_"]
            a2i = {a: i for i, a in enumerate(unique)}
            int_logs[agent_id] = [a2i[a] for a in log] if log else [0]

        si_dict = {}
        for i in range(n_agents):
            others_si = []
            for j in range(n_agents):
                if i == j:
                    continue
                others_si.append(compute_social_influence(int_logs[i], int_logs[j], lag=1))
            si_dict[f"agent_{i}_to_others"] = round(float(np.mean(others_si)), 4) \
                if others_si else 0.0
        si_dict["team_avg"] = round(float(np.mean(list(si_dict.values()))), 4) \
            if si_dict else 0.0

        mi_dict = {}
        if n_agents >= 2:
            for (i, j) in combinations(range(n_agents), 2):
                mi_dict[f"MI_{i}_{j}"] = compute_social_influence(
                    int_logs[i], int_logs[j], lag=0
                )
            mi_vals = [v for k, v in mi_dict.items() if k.startswith("MI_")]
            mi_dict["MI_team"] = round(float(np.mean(mi_vals)), 4) if mi_vals else 0.0
        else:
            mi_dict = {"MI_team": 0.0}

        joint_entropy = float(legacy_coop_metrics.get("joint_entropy", 0.0))
        indiv_entropies = legacy_coop_metrics.get("individual_entropies", [0.0] * n_agents)
        entropy_diff = round(float(np.std(indiv_entropies)), 4) if indiv_entropies else 0.0

        resp = _action_responsiveness(int_logs)

        pass_details = env.match_metrics.get("passes", [])
        pass_attempts = env.match_metrics.get("pass_attempts", 0)
        n_passes = len(pass_details)
        pcr = round(n_passes / max(pass_attempts, 1), 4)

        pn = _pass_network_metrics(pass_details, n_agents)

        avg_spread = float(legacy_coop_metrics.get("avg_team_spread", 0.0))
        avg_compactness = float(legacy_coop_metrics.get("avg_compactness", 0.0))
        avg_offball = float(legacy_coop_metrics.get("avg_offball_quality", 0.0))

        centroid_x, centroid_y, surface_area = 0.0, 0.0, 0.0
        try:
            last_obs = getattr(env, "_last_obs_cache", None)
            if last_obs is not None:
                teammates = np.array(last_obs.get("left_team", []))[:n_agents]
                if len(teammates) >= 1:
                    centroid_x = float(teammates[:, 0].mean())
                    centroid_y = float(teammates[:, 1].mean())
                if len(teammates) >= 3:
                    surface_area = _convex_hull_area(teammates[:, :2].tolist())
        except Exception:
            pass

        role_spec_data = legacy_coop_metrics.get("role_specialization", {})
        if isinstance(role_spec_data, dict) and role_spec_data:
            role_spec_vals = []
            for v in role_spec_data.values():
                if isinstance(v, dict):
                    role_spec_vals.append(float(v.get("specialization", 0.0)))
                elif isinstance(v, (int, float)):
                    role_spec_vals.append(float(v))
            role_spec = float(np.mean(role_spec_vals)) if role_spec_vals else 0.0
        elif isinstance(role_spec_data, (int, float)):
            role_spec = float(role_spec_data)
        else:
            role_spec = 0.0

        role_seq = getattr(env, "_dynamic_role_log", [])
        role_switches = _count_role_switches(role_seq)

        poss_time_list = env.match_metrics.get("possession_time", [0] * n_agents)
        total_poss = sum(poss_time_list)
        steps_total = max(getattr(env, "_step_count", 1), 1)
        poss_share = round(total_poss / steps_total, 4) if steps_total > 0 else 0.0

        shapley_vals = self._compute_shapley_proxy(env, n_agents)
        shapley_var = round(float(np.var(shapley_vals)), 4) if shapley_vals else 0.0

        gini_contrib = _gini(shapley_vals)
        pass_counts_per_agent = [
            sum(1 for p in pass_details if p.get("from") == i)
            for i in range(n_agents)
        ]
        gini_passes = _gini(pass_counts_per_agent)
        action_diversity = [len(set(log)) for log in action_history]
        gini_actions = _gini(action_diversity)

        per_agent_rewards = self._per_agent_episode_rewards(env, n_agents)
        team_avg = float(np.mean(per_agent_rewards)) if per_agent_rewards else 0.0
        cf_adv = [round(float(r - team_avg), 4) for r in per_agent_rewards]

        legacy_idx = float(legacy_coop_metrics.get("cooperation_index", 0.0))

        win_norm = 1.0 if goals_for > goals_against else 0.0
        perf_score = round(
            0.6 * win_norm + 0.2 * (perf_xg_for / max(perf_xg_for + perf_xg_against, 1e-6))
            + 0.2 * min(perf_shots_on_t / 5.0, 1.0),
            4,
        )
        coord_score = round(
            0.4 * si_dict["team_avg"]
            + 0.3 * mi_dict["MI_team"]
            + 0.3 * resp,
            4,
        )
        behav_score = round(
            0.3 * pcr
            + 0.2 * pn["density"]
            + 0.2 * (1.0 - pn["centralization"])
            + 0.15 * min(avg_compactness, 1.0)
            + 0.15 * min(avg_offball, 1.0),
            4,
        )
        credit_eq = round(1.0 - gini_contrib, 4)

        composite = round((perf_score + coord_score + behav_score + credit_eq) / 4.0, 4)

        rec = {
            "match": match,
            "epsilon": round(float(trainer.epsilon), 6),
            "roles": list(env.roles),

            "perf_win": int(win_norm),
            "perf_goals_for": goals_for,
            "perf_goals_against": goals_against,
            "perf_score_differential": goals_for - goals_against,
            "perf_shots": perf_shots,
            "perf_shots_on_target": perf_shots_on_t,
            "perf_xg_for": perf_xg_for,
            "perf_xg_against": perf_xg_against,

            "coord_social_influence": si_dict,
            "coord_mutual_information": mi_dict,
            "coord_joint_action_entropy": round(joint_entropy, 4),
            "coord_individual_entropies": [round(float(h), 4) for h in indiv_entropies],
            "coord_entropy_diff": entropy_diff,
            "coord_action_responsiveness": resp,

            "behav_total_passes": n_passes,
            "behav_pass_completion_rate": pcr,
            "behav_pass_network_density": pn["density"],
            "behav_pass_centralization": pn["centralization"],
            "behav_pass_clustering_coef": pn["clustering_coef"],
            "behav_team_compactness": round(avg_compactness, 4),
            "behav_team_spread": round(avg_spread, 4),
            "behav_team_centroid_x": round(centroid_x, 4),
            "behav_team_centroid_y": round(centroid_y, 4),
            "behav_surface_area": round(surface_area, 4),
            "behav_offball_quality": round(avg_offball, 4),
            "behav_role_specialization": round(role_spec, 4),
            "behav_role_switches": int(role_switches),
            "behav_possession_time": int(total_poss),
            "behav_possession_share": poss_share,

            "credit_shapley_values": shapley_vals,
            "credit_shapley_variance": shapley_var,
            "credit_gini_contribution": round(gini_contrib, 4),
            "credit_gini_passes": round(gini_passes, 4),
            "credit_gini_actions": round(gini_actions, 4),
            "credit_counterfactual_advantage": cf_adv,

            "cooperation_index_legacy": round(legacy_idx, 4),
            "cooperation_index_v2": {
                "performance": perf_score,
                "coordination": coord_score,
                "behavioral": behav_score,
                "credit_equity": credit_eq,
                "composite": composite,
            },
        }
        _append_jsonl(self.output_path, rec)

    def _compute_shapley_proxy(self, env: Any, n_agents: int) -> List[float]:
        passes = env.match_metrics.get("passes", [])
        shots = env.match_metrics.get("shots", [])
        possession = env.match_metrics.get("possession_time", [0] * n_agents)
        steps_total = max(getattr(env, "_step_count", 1), 1)

        contrib = np.zeros(n_agents, dtype=np.float64)
        for p in passes:
            u = p.get("from", -1)
            if 0 <= u < n_agents:
                contrib[u] += 1.0
        for s in shots:
            a = s.get("agent", -1)
            if 0 <= a < n_agents:
                contrib[a] += 2.0 * float(s.get("xG", 0.0))
        for i in range(n_agents):
            contrib[i] += 0.5 * (possession[i] / steps_total)

        total = float(contrib.sum())
        if total <= 1e-9:
            return [0.0] * n_agents
        shap = (contrib / total).tolist()
        return [round(float(v), 4) for v in shap]

    def _per_agent_episode_rewards(self, env: Any, n_agents: int) -> List[float]:
        accum = getattr(env, "_per_agent_episode_reward", None)
        if accum is None or len(accum) != n_agents:
            return [0.0] * n_agents
        return [round(float(r), 4) for r in accum]


class CharmDiagnostics:

    def __init__(self,
                 output_dir: str = "checkpoints/outputs",
                 train_flush_every: int = 10,
                 mem_flush_every: int = 1,
                 enabled: bool = True):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.train_flush_every = train_flush_every
        self.enabled = enabled

        self.training_diag = TrainingDiagnostics(
            os.path.join(output_dir, "training_diagnostic.jsonl")
        )
        self.memory_diag = MemoryDiagnostics(
            os.path.join(output_dir, "memory_diagnostic.jsonl"),
            flush_interval=mem_flush_every,
        )
        self.coop5_diag = Cooperation5DimDiagnostics(
            os.path.join(output_dir, "cooperation_metric_5boyut.jsonl"),
        )

    def collect_train_step(self, marl: Any):
        if not self.enabled:
            return
        self.training_diag.collect(marl)
        m = getattr(marl, "last_train_metrics", None)
        if m:
            self.memory_diag.update_smal_aux(m.get("loss_smal_aux"))

    def flush_episode(self,
                      match: int,
                      trainer: Any,
                      env: Any,
                      legacy_coop_metrics: dict,
                      hcm_breakdown: Dict[str, int]):
        if not self.enabled:
            return
        try:
            self.coop5_diag.flush_episode(match, env, trainer, legacy_coop_metrics)
        except Exception as e:
            print(f"  ⚠️  coop5_diag flush error (match {match}): {e}")
            if match <= 1:
                import traceback
                traceback.print_exc()

        try:
            self.memory_diag.flush_episode(match, trainer, hcm_breakdown)
        except Exception as e:
            print(f"  ⚠️  memory_diag flush error (match {match}): {e}")
            if match <= 1:
                import traceback
                traceback.print_exc()

        if (match % self.train_flush_every) == 0:
            try:
                self.training_diag.flush_to_disk(match)
            except Exception as e:
                print(f"  ⚠️  training_diag flush error (match {match}): {e}")


def _append_jsonl(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, default=_json_safe) + "\n")


def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    return str(o)
