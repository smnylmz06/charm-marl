from __future__ import annotations
import gym
import numpy as np
import os
import json
import gfootball.env as football_env

from cooperation_metrics import (
    compute_team_spread,
    compute_formation_compactness,
    compute_off_ball_quality,
    compute_all_episode_metrics,
    compute_cooperation_reward,
    pass_network_diversity_bonus,
)


class ParametricFootballEnv(gym.Wrapper):

    def __init__(self, mode: str = "3v3"):
        self.configs = {
            "3v3":   {"name": "academy_3_vs_1_with_keeper",  "agents": 3,
                      "roles": ["DF", "MF", "FW"]},
            "5v5":   {"name": "5_vs_5",                       "agents": 4,
                      "roles": ["DF", "MF", "MF", "FW"]},
            "11v11": {"name": "11_vs_11_stochastic",          "agents": 10,
                      "roles": ["DF","DF","DF","DF","MF","MF","MF","FW","FW","FW"]},
        }
        cfg = self.configs[mode]
        env = football_env.create_environment(
            env_name=cfg["name"],
            representation="raw",
            number_of_left_players_agent_controls=cfg["agents"],
            write_video=False,
            logdir="./match_videos",
        )
        super().__init__(env)
        self.mode       = mode
        self.num_agents = cfg["agents"]
        self.roles      = cfg["roles"]

        self.agent_names = [
            "Turing","Einstein","Johnson","DaVinci",
            "Newton","Tesla","Curie","Galileo","Bohr","Faraday",
        ]
        self.action_names = {
            0:"Idle",     1:"Left",     2:"Top-Left",  3:"Top",     4:"Top-Right",
            5:"Right",    6:"Bottom-Right", 7:"Bottom", 8:"Bottom-Left",
            9:"Long-Pass",10:"High-Pass",11:"Short-Pass",12:"Shot",
            13:"Sprint", 14:"Release-Direction",15:"Slow-Down",16:"Sliding",
            17:"Dribble",18:"Release-Dribble",
        }

        self.plot_dir = "checkpoints/outputs"
        os.makedirs(self.plot_dir, exist_ok=True)

        self.match_metrics = {
            "passes"        : [],
            "shots"         : [],
            "possession_time": [0] * self.num_agents,
        }

        self._init_episode_state()

    def _init_episode_state(self):
        self.prev_score          = [0, 0]
        self.current_trajectory  = []
        self.action_history      = [[] for _ in range(self.num_agents)]
        self.prev_ball_to_goal   = None
        self.prev_owned_player   = -1
        self.prev_owned_team     = -1
        self.prev_ball_pos       = np.zeros(2, dtype=np.float32)
        self.consecutive_counts  = [0] * self.num_agents
        self.possession_steps    = 0

        self.last_owner_team0_player   = -1
        self.last_owner_team0_ball_pos = None

        self.last_stable_owner = -1

        self._step_spread_sum      = 0.0
        self._step_compactness_sum = 0.0
        self._step_offball_sum     = 0.0
        self._step_count           = 0

        self.recent_passes: list = []
        self.pass_pairs   : set  = set()
        self._team_snapshots: list = []
        self._coord_log_step       = 0

        self._prev_ball_dist = {}

        self._pass_chain               = []
        self._team_possession_streak   = 0

        self._per_agent_episode_reward = [0.0] * self.num_agents
        self._dynamic_role_log: list = []
        self._last_obs_cache = None

    def reset(self):
        self._init_episode_state()
        self.match_metrics = {
            "passes"        : [],
            "shots"         : [],
            "possession_time": [0] * self.num_agents,
            "pass_attempts" : 0,
            "opp_shots"     : [],
        }
        return self.env.reset()

    def get_enhanced_state(self, obs_list) -> np.ndarray:
        raw   = obs_list[0]
        state = np.zeros(32, dtype=np.float32)
        state[0:2] = raw["ball"][:2]
        for i, pos in enumerate(raw["left_team"][:10]):
            state[2 + 2*i : 4 + 2*i] = pos
        state[22] = np.linalg.norm(raw["ball"][:2] - np.array([1.0, 0.0]))
        state[23] = 1.0 if raw["ball_owned_team"] == 0 else 0.0
        for i, opp_pos in enumerate(raw["right_team"][:3]):
            state[24 + 2*i : 26 + 2*i] = opp_pos
        return state

    def step(self, actions):
        prev_raw = self.env.unwrapped.observation()[0]

        obs_list, reward, done, info = self.env.step(actions)
        state_32d = self.get_enhanced_state(obs_list)
        raw       = obs_list[0]

        ball_pos         = raw["ball"][:2]
        goal_pos         = np.array([1.0, 0.0])
        goal_scored      = raw["score"][0] > self.prev_score[0]
        curr_owned_team  = raw["ball_owned_team"]
        curr_owned_player= raw["ball_owned_player"]
        teammates        = raw["left_team"]

        current_ball_to_goal = np.linalg.norm(ball_pos - goal_pos)
        if self.prev_ball_to_goal is None:
            self.prev_ball_to_goal = current_ball_to_goal
        progress = self.prev_ball_to_goal - current_ball_to_goal

        _is_agent = lambda p: 0 <= p < self.num_agents

        is_valid_pass = False
        current_pass_tuple = None
        if curr_owned_team == 0 and _is_agent(curr_owned_player):
            if (self.last_owner_team0_player != -1
                    and _is_agent(self.last_owner_team0_player)
                    and self.last_owner_team0_player != curr_owned_player):
                ball_travel = (
                    np.linalg.norm(ball_pos - self.last_owner_team0_ball_pos)
                    if self.last_owner_team0_ball_pos is not None else 1.0
                )
                if ball_travel > 0.05:
                    is_valid_pass = True
                    self.match_metrics["passes"].append({
                        "from"  : int(self.last_owner_team0_player),
                        "to"    : int(curr_owned_player),
                        "coords": ball_pos.tolist(),
                    })
                    current_pass_tuple = (
                        int(self.last_owner_team0_player),
                        int(curr_owned_player),
                    )
                    self.pass_pairs.add(current_pass_tuple)
            self.last_owner_team0_player   = curr_owned_player
            self.last_owner_team0_ball_pos = ball_pos.copy()

        elif curr_owned_team == 0 and not _is_agent(curr_owned_player):
            self.last_owner_team0_player   = -1
            self.last_owner_team0_ball_pos = None

        elif curr_owned_team == 1:
            self.last_owner_team0_player   = -1
            self.last_owner_team0_ball_pos = None

        coop_bonus_per_agent = 0.0

        if is_valid_pass and _is_agent(curr_owned_player):
            if not self._pass_chain or self._pass_chain[-1] != curr_owned_player:
                self._pass_chain.append(int(curr_owned_player))
            chain_len = len(set(self._pass_chain))
            if chain_len >= 3:
                coop_bonus_per_agent += 5.0 * chain_len

        if curr_owned_team == 1:
            self._pass_chain = []

        if curr_owned_team == 0:
            self._team_possession_streak += 1
            if self._team_possession_streak % 30 == 0:
                coop_bonus_per_agent += 2.0
        else:
            self._team_possession_streak = 0

        PASS_ACTIONS = {9, 10, 11}
        for i in range(self.num_agents):
            if (actions[i] in PASS_ACTIONS
                    and prev_raw["ball_owned_team"] == 0
                    and prev_raw["ball_owned_player"] == i):
                self.match_metrics["pass_attempts"] += 1

        SHOT_ACTION = 12
        shot_xg_per_agent: dict = {}
        for i in range(self.num_agents):
            if (actions[i] == SHOT_ACTION
                    and prev_raw["ball_owned_team"] == 0
                    and prev_raw["ball_owned_player"] == i):
                shot_pos      = prev_raw["ball"][:2]
                dist_to_goal  = np.linalg.norm(shot_pos - goal_pos)
                angle         = abs(np.arctan2(shot_pos[1], 1.0 - shot_pos[0]))
                xg            = max(0.02, 0.35 * np.exp(-dist_to_goal * 1.5)
                                    * (1 - angle / np.pi))
                shot_xg_per_agent[i] = float(xg)
                self.match_metrics["shots"].append({
                    "agent" : i,
                    "coords": shot_pos.tolist(),
                    "xG"    : round(float(xg), 3),
                })

        own_goal_pos = np.array([-1.0, 0.0], dtype=np.float32)
        if (prev_raw["ball_owned_team"] == 1
                and curr_owned_team != 1
                and ball_pos[0] < 0.0):
            opp_shot_pos = prev_raw["ball"][:2]
            opp_dist     = np.linalg.norm(opp_shot_pos - own_goal_pos)
            opp_angle    = abs(np.arctan2(opp_shot_pos[1], -1.0 - opp_shot_pos[0]))
            opp_xg       = max(0.02, 0.35 * np.exp(-opp_dist * 1.5)
                               * (1 - opp_angle / np.pi))
            self.match_metrics["opp_shots"].append({
                "coords": opp_shot_pos.tolist(),
                "xG"    : round(float(opp_xg), 3),
            })

        teammate_pos = teammates[:self.num_agents, :2]
        diff         = (teammate_pos[:, None, :]
                        - teammate_pos[None, :, :])
        pairwise_dist = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(pairwise_dist, np.inf)
        crowding_count = (pairwise_dist < 0.08).sum(axis=1)

        custom_rewards = []
        for i in range(self.num_agents):
            act_name  = self.action_names.get(actions[i], str(actions[i]))
            agent_r   = float(reward[i])
            is_holder = (curr_owned_team == 0 and curr_owned_player == i)

            if (self.action_history[i]
                    and self.action_history[i][-1] == act_name):
                self.consecutive_counts[i] += 1
            else:
                self.consecutive_counts[i] = 1
            if self.consecutive_counts[i] > 2:
                agent_r -= 0.5 * self.consecutive_counts[i]

            agent_r -= 0.15 * float(crowding_count[i])

            if not done:
                agent_r -= 0.01
                if progress > 0:
                    agent_r += progress * (2.0 if is_holder else 1.0)

                if is_holder and i == 1 and ball_pos[0] > 0.5:
                    self.possession_steps += 1
                    if self.possession_steps > 15:
                        agent_r -= 5.0
                elif not is_holder and i == 1:
                    self.possession_steps = 0

                if is_valid_pass:
                    if i == curr_owned_player:
                        agent_r += 20.0
                    if i == self.last_owner_team0_player:
                        agent_r += 15.0

                _had_ball = (prev_raw["ball_owned_team"] == 0
                             and prev_raw["ball_owned_player"] == i)
                if _had_ball:
                    if actions[i] in {9, 10, 11}:
                        agent_r += 1.5
                    elif actions[i] == 12:
                        agent_r += 0.8
                    elif actions[i] == 17:
                        agent_r += 0.3
                    elif actions[i] == 13:
                        agent_r += 0.2

                if i in shot_xg_per_agent:
                    xg = shot_xg_per_agent[i]
                    if xg >= 0.20:
                        agent_r += 8.0 * xg
                    else:
                        agent_r -= 3.0 * (0.20 - xg)

                agent_r += compute_cooperation_reward(
                    obs           = raw,
                    actions       = actions,
                    agent_i       = i,
                    recent_passes = self.recent_passes,
                    current_pass  = current_pass_tuple,
                    current_step  = self._coord_log_step,
                    n_agents      = self.num_agents,
                )

                if not is_holder:
                    my_pos        = teammates[i][:2]
                    dist_to_ball  = float(np.linalg.norm(my_pos - ball_pos))
                    prev_dist     = self._prev_ball_dist.get(i, dist_to_ball)
                    if dist_to_ball < prev_dist:
                        agent_r += 0.1
                    self._prev_ball_dist[i] = dist_to_ball

                agent_r += coop_bonus_per_agent

            if goal_scored:
                agent_r += (2000.0 if self.match_metrics["passes"]
                            else 100.0)

            self.action_history[i].append(act_name)
            custom_rewards.append(agent_r)
            self.current_trajectory.append({"s": state_32d, "a": actions[i], "id": i})

            if curr_owned_team == 0 and curr_owned_player == i:
                self.match_metrics["possession_time"][i] += 1

        had_pass = is_valid_pass
        self._team_snapshots.append({
            'team_state'  : state_32d.copy(),
            'joint_actions': list(actions),
            'step_reward' : float(sum(custom_rewards)),
            'had_pass'    : had_pass,
            'goal_scored' : goal_scored,
            'role_tuple'  : list(self.roles),
        })

        self._step_spread_sum      += compute_team_spread(raw, "left")
        self._step_compactness_sum += compute_formation_compactness(raw, "left")
        offball_vals = [compute_off_ball_quality(raw, i, actions)
                        for i in range(self.num_agents)]
        self._step_offball_sum     += float(np.mean(offball_vals))
        self._step_count           += 1

        self._coord_log_step += 1
        if self._coord_log_step % 10 == 0:
            self.match_metrics.setdefault("steps", []).append({
                "t"      : self._coord_log_step,
                "ball"   : ball_pos.tolist(),
                "players": [p.tolist() for p in teammates[:self.num_agents]],
            })

        if current_pass_tuple is not None:
            self.recent_passes.append({
                "from": current_pass_tuple[0],
                "to"  : current_pass_tuple[1],
                "step": self._coord_log_step,
            })
            self.recent_passes = self.recent_passes[-8:]

        if done and self.pass_pairs:
            diversity = pass_network_diversity_bonus(
                self.pass_pairs, self.num_agents
            )
            diversity_bonus = 15.0 * diversity
            for j in range(self.num_agents):
                custom_rewards[j] += diversity_bonus

        self.prev_ball_to_goal = current_ball_to_goal
        self.prev_owned_team   = curr_owned_team
        self.prev_owned_player = curr_owned_player
        self.prev_ball_pos     = ball_pos.copy()

        self._last_obs_cache = raw
        dyn_roles = []
        for i in range(self.num_agents):
            x_pos = float(teammates[i, 0]) if i < len(teammates) else 0.0
            if x_pos < -0.3:
                dyn_roles.append("DF")
            elif x_pos > 0.3:
                dyn_roles.append("FW")
            else:
                dyn_roles.append("MF")
        self._dynamic_role_log.append(dyn_roles)

        if done and goal_scored:
            self._log_goal_path()

        return obs_list, np.array(custom_rewards), done, info

    def compute_episode_cooperation_metrics(
        self,
        compute_expensive: bool = False,
        reward_fn_for_shapley=None,
        last_obs: dict = None,
        epsilon: float = 0.0,
    ) -> dict:
        n = self._step_count or 1
        return compute_all_episode_metrics(
            action_history       = self.action_history,
            step_spread_log      = [self._step_spread_sum / n],
            step_compactness_log = [self._step_compactness_sum / n],
            step_offball_log     = [self._step_offball_sum / n],
            pass_details         = self.match_metrics["passes"],
            n_agents             = self.num_agents,
            pass_attempts        = self.match_metrics["pass_attempts"],
            compute_expensive    = compute_expensive,
            obs_for_shapley      = last_obs,
            reward_fn_for_shapley= reward_fn_for_shapley,
            epsilon              = epsilon,
        )

    def hindsight_populate_jacm(self, mem_manager, top_k=3, require_cooperation=True):
        empty = {"total": 0, "goal": 0, "pass": 0, "fallback": 0}
        if not mem_manager.joint_enabled:
            return empty
        if not self._team_snapshots:
            return empty

        if require_cooperation:
            has_cooperation = any(
                s.get("had_pass") or s.get("goal_scored")
                for s in self._team_snapshots
            )
            if not has_cooperation:
                return empty

        from cooperation_metrics import hindsight_relabel
        relabeled = hindsight_relabel(self._team_snapshots, top_k=top_k)

        counts = {"total": 0, "goal": 0, "pass": 0, "fallback": 0}
        for snap in relabeled:
            label = snap.get("hindsight_label", "")
            mem_manager.joint.add(
                state               = snap["team_state"],
                joint_actions       = snap["joint_actions"],
                role_tuple          = snap["role_tuple"] if snap["role_tuple"] else self.roles,
                cooperation_quality = snap["cooperation_quality"],
                reward              = snap["reward"],
                hindsight_label     = label,
            )
            counts["total"] += 1
            if label == "goal_actual":
                counts["goal"] += 1
            elif label == "pass_actual":
                counts["pass"] += 1
            elif label == "hindsight_imagined":
                counts["fallback"] += 1
        return counts

    def save_plot_data(
        self,
        match_no: int,
        epsilon: float,
        total_rewards: float,
        memory_size: int,
        cooperation_metrics: dict = None,
    ):
        base = {
            "match"      : match_no,
            "epsilon"    : round(float(epsilon), 6),
            "roles"      : self.roles,
            "agent_names": self.agent_names[:self.num_agents],
        }

        _score = list(self.prev_score) if hasattr(self.prev_score,"__iter__") else [0,0]
        self._append_jsonl("rewards_and_winrate.jsonl", {
            **base,
            "total_reward" : round(float(total_rewards), 2),
            "win"          : 1 if _score[0] > 0 else 0,
            "score"        : _score,
            "goals_scored" : int(_score[0]),
            "goals_conceded": int(_score[1]) if len(_score) > 1 else 0,
        })

        coach_c    = cooperation_metrics.get("_coach_count", 0)    if cooperation_metrics else 0
        selected_c = cooperation_metrics.get("_selected_count", 0) if cooperation_metrics else 0
        self._append_jsonl("memory_growth.jsonl", {
            **base,
            "memory_count"        : memory_size,
            "coach_experiences"   : coach_c,
            "selected_experiences": selected_c,
        })

        _successful = len(self.match_metrics["passes"])
        _attempts   = max(self.match_metrics["pass_attempts"], 1)
        self._append_jsonl("passing_analysis.jsonl", {
            **base,
            "total_passes"     : _successful,
            "pass_attempts"    : self.match_metrics["pass_attempts"],
            "pass_completion_rate": round(_successful / _attempts, 4),
            "possession_time"  : self.match_metrics["possession_time"],
            "pass_details"     : self.match_metrics["passes"],
        })

        self._append_jsonl("shot_and_xg_metrics.jsonl", {
            **base,
            "shots"         : self.match_metrics["shots"],
            "expected_goals": round(
                sum(s["xG"] for s in self.match_metrics["shots"]), 3
            ),
        })

        if cooperation_metrics:
            self._append_jsonl("cooperation_metrics.jsonl", {
                **base, **cooperation_metrics,
            })

        self._append_jsonl("spatial_dominance.jsonl", {
            **base,
            "trajectories": self.match_metrics.get("steps", []),
        })

    def _append_jsonl(self, filename: str, data: dict):
        path = os.path.join(self.plot_dir, filename)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _append_to_json(self, filename: str, data: dict):
        self._append_jsonl(filename.replace(".json", ".jsonl"), data)

    def finalize_event_based(self, obs_list, rewards, mem_manager):
        raw           = obs_list[0]
        current_score = raw["score"][0]
        goal_scored   = current_score > self.prev_score[0]
        total_step_r  = sum(rewards)

        MAX_SELECTED = 15_000
        selected_full = len(mem_manager.selected.experiences) >= MAX_SELECTED

        SELECTED_THRESHOLD       = 25.0
        high_reward_event        = total_step_r > SELECTED_THRESHOLD

        RECENT_PASS_WINDOW = 5
        had_recent_pass_hier = False
        if self.recent_passes:
            last_step = self.recent_passes[-1].get("step", 0)
            if (self._coord_log_step - last_step) <= RECENT_PASS_WINDOW:
                had_recent_pass_hier = True
        should_add_hier = (had_recent_pass_hier or high_reward_event) and not selected_full

        had_pass_episode_jacm = len(self.recent_passes) > 0
        should_add_jacm = had_pass_episode_jacm or high_reward_event

        write_hierarchical = mem_manager.enabled

        if write_hierarchical and (goal_scored or should_add_hier):
            decay = 1.0
            recent = self.current_trajectory[-12:]
            for s_data in reversed(recent):
                agent_role = self.roles[s_data["id"]]
                if goal_scored:
                    mem_manager.coach.add(s_data["s"], s_data["a"], agent_role, 1.0 * decay)
                else:
                    dynamic_r = min(max(total_step_r * 0.1 * decay, 0.1), 0.9)
                    mem_manager.selected.add(s_data["s"], s_data["a"], agent_role, dynamic_r)
                decay *= 0.96

        if mem_manager.joint_enabled and (goal_scored or should_add_jacm) and self._team_snapshots:
            window_size = min(5, len(self._team_snapshots))
            window = self._team_snapshots[-window_size:]
            best_snap = max(window, key=lambda s: s['step_reward'])

            if best_snap['goal_scored']:
                quality = 1.0
            elif best_snap['had_pass']:
                quality = 0.7
            else:
                quality = 0.5

            mem_manager.joint.add(
                state               = best_snap['team_state'],
                joint_actions       = best_snap['joint_actions'],
                role_tuple          = self.roles,
                cooperation_quality = quality,
                reward              = best_snap['step_reward'],
            )

        max_buffer = self.num_agents * 300
        if goal_scored or len(self.current_trajectory) > max_buffer:
            self.current_trajectory = []

        self.prev_score = raw["score"]

    def _log_goal_path(self):
        log_dir  = "checkpoints"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "Agent_Path.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'⚽'*20}\n"
                    f"🌟 REAL GOAL ANALYSIS (Duration: {len(self.action_history[0])} steps)\n")
            for j in range(self.num_agents):
                actions_to_show = self.action_history[j][-20:]
                f.write(f"👤 {self.agent_names[j]}: {' -> '.join(actions_to_show)}\n")
