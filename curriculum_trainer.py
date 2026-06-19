from __future__ import annotations

import os
import json
import time

import faiss
import numpy as np
import torch

from env_parametric import ParametricFootballEnv
from dqn_agents import ParameterSharedMARL
from qmix_agents import ParameterSharedQMIX
from qplex_agents import ParameterSharedQPLEX
from maddpg_agents import ParameterSharedMADDPG
from emc_agents   import ParameterSharedEMC
from emu_agents   import ParameterSharedEMU
from mappo_agents import ParameterSharedMAPPO
from memory_system import MemoryManager
from tactical_quality import TacticalQualityCalculator, TacticWeights
from cooperation_metrics import (
    compute_policy_entropy,
    compute_team_spirit_reward,
    compute_q_value_stats,
)
from diagnostics import CharmDiagnostics


class CurriculumTrainer:

    def __init__(self, mode: str = "11v11", load_from: str = None,
                 use_cer: bool = True,
                 algorithm: str = "dqn",
                 reg_l2: float = 0.0,
                 rand_aux: bool = False,
                 use_hybrid: bool = False,
                 hybrid_config: dict = None):
        self.algorithm = algorithm.lower()
        if self.algorithm not in ("dqn", "qmix", "qplex", "maddpg", "emc", "emu", "mappo"):
            raise ValueError(
                f"algorithm must be 'dqn'|'qmix'|'qplex'|'maddpg'|'emc'|'emu'|'mappo' "
                f"— got {algorithm}"
            )

        self.env  = ParametricFootballEnv(mode=mode)
        n_agents  = self.env.num_agents

        if self.algorithm == "dqn":
            self.marl = ParameterSharedMARL(
                input_dim=32, action_dim=19,
                use_cer=use_cer,
                cer_alpha=0.6, cer_beta=2.0,
                weight_decay=reg_l2, rand_aux=rand_aux,
            )
        elif self.algorithm == "qmix":
            self.marl = ParameterSharedQMIX(
                input_dim=32, action_dim=19, n_agents=n_agents,
                use_cer=use_cer,
                cer_alpha=0.6, cer_beta=2.0,
            )
        elif self.algorithm == "qplex":
            self.marl = ParameterSharedQPLEX(
                input_dim=32, action_dim=19, n_agents=n_agents,
                use_cer=use_cer,
                cer_alpha=0.6, cer_beta=2.0,
            )
        elif self.algorithm == "maddpg":
            self.marl = ParameterSharedMADDPG(
                input_dim=32, action_dim=19, n_agents=n_agents,
                use_cer=use_cer,
                cer_alpha=0.6, cer_beta=2.0,
            )
        elif self.algorithm == "emc":
            self.marl = ParameterSharedEMC(
                input_dim=32, action_dim=19, n_agents=n_agents,
                use_cer=False,
            )
        elif self.algorithm == "emu":
            self.marl = ParameterSharedEMU(
                input_dim=32, action_dim=19, n_agents=n_agents,
                use_cer=False,
            )
        else:
            self.marl = ParameterSharedMAPPO(
                input_dim=32, action_dim=19, n_agents=n_agents,
            )

        self.mem  = MemoryManager(
            state_dim=32, n_agents=n_agents,
            enabled=True, joint_enabled=False,
            use_hybrid=use_hybrid,
            hybrid_config=hybrid_config,
        )
        self._last_step_coop_priority = 0.0

        self.epsilon     = 1.0
        self.prev_ball_x = 0.0

        self.smal_warmup_episodes = 500
        self.smal_full_episodes   = 1000
        self.smal_max_weight      = 0.05
        self.hcr_enabled          = True
        self.rand_aux             = rand_aux

        self.team_spirit = 0.0

        self.last_avg_entropy = 0.0

        if mode != "11v11":
            w = TacticWeights(tiki_taka=3.0, progression=2.5, spacing=4.0)
        else:
            w = TacticWeights()
        self.calcs = [
            TacticalQualityCalculator(weights=w)
            for _ in range(self.env.num_agents)
        ]

        self._load_checkpoint(mode)
        if load_from:
            self.mem.load_phase_memory(phase_prefix=f"checkpoints/{load_from}")
            self._load_checkpoint(load_from, is_transfer=True)

        self.diag = CharmDiagnostics(
            output_dir=self.env.plot_dir,
            train_flush_every=10,
            mem_flush_every=1,
            enabled=True,
        )

        print(f"🎯 {self.env.num_agents} agents | mode: {mode} | "
              f"{'transfer: ' + load_from if load_from else 'from scratch'}")

    def _shape_rewards(
        self,
        obs,
        actions: list,
        rewards,
        episode: int = 0,
    ) -> list:
        ball_x        = obs["ball"][0]
        shaped        = []

        for i in range(self.env.num_agents):
            r = float(rewards[i])

            if actions[i] == 0:
                r -= 0.5

            if ball_x > self.prev_ball_x:
                r += 0.5
            else:
                r -= 0.2

            r += self.calcs[i].calculate(obs, actions, i, self.env.roles[i])

            shaped.append(r)

        self.prev_ball_x = ball_x

        if self.team_spirit > 0.0:
            shaped = compute_team_spirit_reward(shaped, self.team_spirit)

        return shaped

    def _current_smal_weight(self, ep: int) -> float:
        if ep < self.smal_warmup_episodes:
            return 0.0
        elif ep < self.smal_full_episodes:
            ramp_range = self.smal_full_episodes - self.smal_warmup_episodes
            progress   = (ep - self.smal_warmup_episodes) / max(ramp_range, 1)
            return self.smal_max_weight * progress
        else:
            return self.smal_max_weight

    def train(self, episodes: int = 2000):
        print(f"🚀 {self.env.mode} training starting | {episodes} matches")
        timer_10 = time.time()

        ts_ramp_episodes = min(episodes, 5000)

        for ep in range(episodes):

            obs_list            = self.env.reset()
            done                = False
            step                = 0
            total_r             = 0.0
            ep_entropy_sum      = 0.0
            ep_entropy_count    = 0
            ep_q_mean_sum       = 0.0
            ep_q_std_sum        = 0.0
            ep_q_range_sum      = 0.0
            self.mem.reset_episode_stats()
            self.mem.reset_joint_stats()

            while not done:
                state   = self.env.get_enhanced_state(obs_list)
                state_t = torch.FloatTensor(state)
                actions = []

                for i in range(self.env.num_agents):
                    guidance = self.mem.get_guidance(
                        i, self.env.roles[i], state, step
                    )
                    if guidance and np.random.random() < guidance[1]:
                        actions.append(int(guidance[0]))
                        self.mem.mark_guidance_used()
                    else:
                        actions.append(
                            self.marl.select_action(state, self.epsilon)
                        )

                if step % 10 == 0:
                    with torch.no_grad():
                        if self.algorithm == "dqn":
                            logits = self.marl.policy_net(state_t)
                        elif self.algorithm in ("qmix", "qplex", "emc", "emu"):
                            logits = self.marl.policy_q(state_t)
                        else:
                            logits = self.marl.actor(state_t)
                    ep_entropy_sum   += compute_policy_entropy(logits)
                    ep_entropy_count += 1

                    if self.algorithm == "dqn":
                        q_stats = compute_q_value_stats(self.marl.policy_net, state_t)
                    elif self.algorithm in ("qmix", "qplex", "emc", "emu"):
                        q_stats = compute_q_value_stats(self.marl.policy_q, state_t)
                    else:
                        q_stats = compute_q_value_stats(self.marl.actor, state_t)
                    ep_q_mean_sum  += q_stats["q_mean"]
                    ep_q_std_sum   += q_stats["q_std"]
                    ep_q_range_sum += q_stats["q_range"]

                next_obs, raw_r, done, _ = self.env.step(actions)
                shaped_r = self._shape_rewards(
                    obs_list[0], actions, raw_r, episode=ep
                )

                self.env.finalize_event_based(next_obs, shaped_r, self.mem)

                next_state = self.env.get_enhanced_state(next_obs)
                coop_step = 0.0
                if self.env.recent_passes:
                    last_pass = self.env.recent_passes[-1]
                    if (self.env._coord_log_step - last_pass.get("step", 0)) <= 3:
                        coop_step = 1.0
                if next_obs[0]["score"][0] > obs_list[0]["score"][0]:
                    coop_step = 2.0

                t3_score = self.mem.query_t3_for_cer(state)
                if t3_score is not None:
                    coop_step = min(2.3, coop_step + 0.3 * t3_score)

                if self.algorithm == "dqn":
                    for i in range(self.env.num_agents):
                        self.marl.add_experience(
                            state, actions[i], shaped_r[i], next_state, float(done),
                            coop_priority=coop_step,
                        )
                else:
                    self.marl.add_experience(
                        state, list(actions), list(shaped_r), next_state, float(done),
                        coop_priority=coop_step,
                    )

                obs_list = next_obs
                total_r += sum(shaped_r)
                step    += 1
                joint_mem  = self.mem.joint if self.mem.joint_enabled else None
                if self.mem.joint_enabled or self.rand_aux:
                    smal_weight = self._current_smal_weight(ep)
                else:
                    smal_weight = 0.0
                self.marl.train_step(
                    joint_memory=joint_mem,
                    smal_weight=smal_weight,
                    role_tuple=self.env.roles,
                )

                self.diag.collect_train_step(self.marl)

                for ai in range(self.env.num_agents):
                    self.env._per_agent_episode_reward[ai] += float(shaped_r[ai])


            self.team_spirit = min(0.7, 0.7 * ep / ts_ramp_episodes)

            self.last_avg_entropy = (
                ep_entropy_sum / ep_entropy_count
                if ep_entropy_count > 0 else 0.0
            )

            compute_expensive = (ep + 1) % 50 == 0
            coop_metrics = self.env.compute_episode_cooperation_metrics(
                compute_expensive=compute_expensive,
                last_obs=obs_list[0] if compute_expensive else None,
                epsilon=self.epsilon,
            )

            hcm_breakdown = {"total": 0, "goal": 0, "pass": 0, "fallback": 0}


            if self.mem.joint_enabled and self.hcr_enabled:
                hcm_breakdown = self.env.hindsight_populate_jacm(self.mem, top_k=3)

            coop_metrics["hcm_relabels"] = int(hcm_breakdown.get("total", 0))

            if hasattr(self.marl.memory, "avg_priority"):
                coop_metrics["cer_avg_priority"] = round(self.marl.memory.avg_priority(), 4)

            if self.mem.joint_enabled:
                coop_metrics["jacm_partitions"] = self.mem.joint.partition_summary()

            coop_metrics["smal_weight_current"] = round(self._current_smal_weight(ep), 4)
            coop_metrics["smal_phase"] = (
                "warmup" if ep < self.smal_warmup_episodes
                else ("ramp" if ep < self.smal_full_episodes else "full")
            )

            coop_metrics["avg_policy_entropy"] = round(self.last_avg_entropy, 4)
            coop_metrics["team_spirit"]        = round(self.team_spirit, 4)
            n_samples = max(1, ep_entropy_count)
            coop_metrics["q_value_mean"]  = round(ep_q_mean_sum  / n_samples, 4)
            coop_metrics["q_value_std"]   = round(ep_q_std_sum   / n_samples, 4)
            coop_metrics["q_value_range"] = round(ep_q_range_sum / n_samples, 4)
            coop_metrics["_coach_count"]    = len(self.mem.coach.experiences)
            coop_metrics["_selected_count"] = len(self.mem.selected.experiences)

            coop_metrics.update(self.mem.get_episode_stats())

            self.env.save_plot_data(
                match_no           = ep + 1,
                epsilon            = self.epsilon,
                total_rewards      = total_r,
                memory_size        = len(self.mem.coach.experiences) + len(self.mem.selected.experiences),
                cooperation_metrics= coop_metrics,
            )

            self.diag.flush_episode(
                match               = ep + 1,
                trainer             = self,
                env                 = self.env,
                legacy_coop_metrics = coop_metrics,
                hcm_breakdown       = hcm_breakdown,
            )

            self.epsilon = max(0.40, self.epsilon * 0.996)

            if (ep + 1) % 10 == 0:
                duration = time.time() - timer_10
                q_std_disp = coop_metrics.get("q_value_std", 0.0)
                coop_idx   = coop_metrics.get("cooperation_index", 0.0)
                print(
                    f"Mode: {self.env.mode} | Match {ep+1:5d} | "
                    f"Reward: {total_r:8.1f} | "
                    f"Coop: {coop_idx:.3f} | "
                    f"Qstd: {q_std_disp:.3f} | "
                    f"TS: {self.team_spirit:.2f} | "
                    f"Pas: {len(self.env.match_metrics['passes']):3d} | "
                    f"ε: {self.epsilon:.3f} | "
                    f"{duration:.1f}s"
                )
                timer_10 = time.time()
                self._save_checkpoint(ep + 1)

    def _save_checkpoint(self, ep: int):
        os.makedirs("checkpoints", exist_ok=True)
        prefix = f"checkpoints/{self.env.mode}_{self.algorithm}"

        if self.algorithm == "dqn":
            model_state = self.marl.policy_net.state_dict()
        else:
            model_state = self.marl.state_dict()

        torch.save({
            "model_state_dict": model_state,
            "algorithm"       : self.algorithm,
            "epsilon"         : self.epsilon,
            "team_spirit"     : self.team_spirit,
            "episode"         : ep,
        }, f"{prefix}_brain.pth")

        faiss.write_index(self.mem.coach.index,    f"{prefix}_coach.index")
        faiss.write_index(self.mem.selected.index, f"{prefix}_selected.index")

        meta = {
            "coach"   : self.mem.coach.experiences,
            "selected": self.mem.selected.experiences,
        }
        with open(f"{prefix}_metadata.json", "w") as f:
            json.dump(meta, f)
        with open(f"{prefix}_coach_exp.json", "w") as f:
            json.dump(self.mem.coach.experiences, f)
        with open(f"{prefix}_selected_exp.json", "w") as f:
            json.dump(self.mem.selected.experiences, f)

        print(f"  💾 Checkpoint [{self.algorithm}]: Match {ep} | ε={self.epsilon:.3f} | "
              f"ts={self.team_spirit:.2f}")

    def _load_checkpoint(self, mode_name: str, is_transfer: bool = False) -> int:
        path      = f"checkpoints/{mode_name}_{self.algorithm}_brain.pth"
        meta_path = f"checkpoints/{mode_name}_{self.algorithm}_metadata.json"

        if not os.path.exists(path):
            legacy_path = f"checkpoints/{mode_name}_brain.pth"
            legacy_meta = f"checkpoints/{mode_name}_metadata.json"
            if os.path.exists(legacy_path):
                path, meta_path = legacy_path, legacy_meta
            else:
                print(f"ℹ️  No checkpoint found for {mode_name} ({self.algorithm}).")
                return 0

        try:
            ckpt = torch.load(path, weights_only=False)

            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
                if self.algorithm == "dqn":
                    self.marl.policy_net.load_state_dict(state)
                else:
                    self.marl.load_state_dict(state)
                if not is_transfer:
                    self.epsilon     = ckpt.get("epsilon", 1.0)
                    self.team_spirit = ckpt.get("team_spirit", 0.0)
                    start_ep         = ckpt.get("episode", 0)
                else:
                    start_ep = 0
            else:
                if self.algorithm == "dqn":
                    self.marl.policy_net.load_state_dict(ckpt)
                start_ep = 0

            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                self.mem.coach.experiences    = meta["coach"]
                self.mem.selected.experiences = meta["selected"]

                for attr, fname in [
                    ("coach",    f"{mode_name}_{self.algorithm}_coach.index"),
                    ("selected", f"{mode_name}_{self.algorithm}_selected.index"),
                ]:
                    idx_path = f"checkpoints/{fname}"
                    if not os.path.exists(idx_path):
                        legacy_idx = f"checkpoints/{fname.replace(f'_{self.algorithm}', '')}"
                        if os.path.exists(legacy_idx):
                            idx_path = legacy_idx
                    if os.path.exists(idx_path):
                        getattr(self.mem, attr).index = faiss.read_index(idx_path)

            tag = "TRANSFER" if is_transfer else "RESUME"
            print(f"🔄 [{tag}] {mode_name} ({self.algorithm}) loaded | "
                  f"Start match: {start_ep}")
            return start_ep

        except Exception as e:
            print(f"⚠️  Load error: {e}. Starting from scratch.")
            return 0


if __name__ == "__main__":


    trainer = CurriculumTrainer(
    mode="3v3",
    algorithm="maddpg",
    use_cer=True,
    )
    trainer.mem.enabled       = False
    trainer.mem.joint_enabled = True
    trainer.epsilon = 1.0
    trainer.train(episodes=30)
