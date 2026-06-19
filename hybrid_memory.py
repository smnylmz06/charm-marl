from __future__ import annotations

import numpy as np
import faiss
from typing import Optional, List, Dict, Any


class JointEpisodicMemory:

    def __init__(self, state_dim: int = 32, n_agents: int = 3,
                 max_size: int = 2000, name: str = "tier"):
        self.state_dim = state_dim
        self.n_agents  = n_agents
        self.max_size  = max_size
        self.name      = name

        self.index       = faiss.IndexFlatL2(state_dim)
        self.experiences: list = []
        self._states:     list = []

        self._cum_query_count = 0
        self._cum_hit_count   = 0
        self._cum_sim_mean    = 0.0
        self._cum_sim_n       = 0
        self._cum_qual_mean   = 0.0
        self._cum_qual_n      = 0

    def add(self, state, joint_actions, role_tuple,
            cooperation_quality, reward, source_label: str = "direct"):
        if len(joint_actions) != self.n_agents:
            return
        s_arr = state.astype('float32').reshape(1, -1)
        if s_arr.shape[1] != self.state_dim:
            return

        self.index.add(s_arr)
        self.experiences.append({
            'joint_actions'      : [int(a) for a in joint_actions],
            'role_tuple'         : list(role_tuple),
            'cooperation_quality': float(cooperation_quality),
            'reward'             : float(reward),
            'source_label'       : source_label,
        })
        self._states.append(s_arr.flatten())

        if len(self.experiences) > self.max_size:
            evict_n = self.max_size // 5
            self.experiences = self.experiences[evict_n:]
            self._states     = self._states[evict_n:]
            self.index       = faiss.IndexFlatL2(self.state_dim)
            if self._states:
                self.index.add(np.vstack(self._states))

    def query(self, state, role_tuple=None, k: int = 3,
              min_similarity: float = 0.75):
        self._cum_query_count += 1

        if self.index.ntotal == 0:
            return None
        s_arr = state.astype('float32').reshape(1, -1)
        if s_arr.shape[1] != self.state_dim:
            return None

        k_eff = min(k, self.index.ntotal)
        distances, indices = self.index.search(s_arr, k_eff)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            sim = 1.0 / (1.0 + float(dist))
            if sim < min_similarity:
                continue
            exp = self.experiences[idx]
            results.append({
                'joint_actions'      : exp['joint_actions'],
                'similarity'         : sim,
                'cooperation_quality': exp['cooperation_quality'],
                'weighted_score'     : sim * exp['cooperation_quality'],
                'role_tuple'         : exp['role_tuple'],
                'source_label'       : exp.get('source_label', 'unknown'),
            })
        if results:
            self._cum_hit_count += 1
            for r in results:
                self._cum_sim_n += 1
                self._cum_sim_mean += (r['similarity'] - self._cum_sim_mean) / self._cum_sim_n
                self._cum_qual_n += 1
                self._cum_qual_mean += (
                    (r['cooperation_quality'] - self._cum_qual_mean) / self._cum_qual_n
                )
        return results if results else None

    def __len__(self):
        return len(self.experiences)


class HybridMemorySystem:

    def __init__(self,
                 state_dim: int = 32,
                 n_agents: int = 3,
                 tier1_enabled: bool = True,
                 tier2_enabled: bool = True,
                 tier3_enabled: bool = True,
                 tier1_max_size: int = 2000,
                 tier1_threshold: float = 0.95,
                 t1_coop_quality_insert_thresh: float = 0.40,
                 tier2_max_size: int = 2000,
                 tier2_threshold: float = 0.85,
                 tier3_max_size: int = 15000,
                 tier3_threshold: float = 0.80,
                 t3_coop_quality_insert_thresh: float = 0.50,
                 t3_accepts_hindsight_imagined: bool = False):
        self.state_dim = state_dim
        self.n_agents  = n_agents

        self.tier1_enabled = tier1_enabled
        self.tier2_enabled = tier2_enabled
        self.tier3_enabled = tier3_enabled

        self.tier1 = JointEpisodicMemory(
            state_dim=state_dim, n_agents=n_agents,
            max_size=tier1_max_size, name="T1",
        )
        self.tier1_threshold = float(tier1_threshold)
        self.t1_coop_quality_insert_thresh = float(t1_coop_quality_insert_thresh)

        from memory_system import JointActionMemory
        self.tier2 = JointActionMemory(
            state_dim=state_dim, n_agents=n_agents,
            max_size=tier2_max_size, role_partitioned=True,
        )
        self.tier2_threshold = float(tier2_threshold)

        self.tier3 = JointEpisodicMemory(
            state_dim=state_dim, n_agents=n_agents,
            max_size=tier3_max_size, name="T3",
        )
        self.tier3_threshold = float(tier3_threshold)
        self.t3_coop_quality_insert_thresh = float(t3_coop_quality_insert_thresh)

        self.t3_accepts_hindsight_imagined = bool(t3_accepts_hindsight_imagined)

        self._cum_query_count = 0
        self._cum_hit_count   = 0
        self._cum_hit_t1 = 0
        self._cum_hit_t2 = 0
        self._cum_hit_t3 = 0
        self._cum_sim_mean  = 0.0
        self._cum_sim_n     = 0
        self._cum_qual_mean = 0.0
        self._cum_qual_n    = 0

        self._query_count = 0
        self._hit_count   = 0

        self._insert_t1_direct   = 0
        self._insert_t2_direct   = 0
        self._insert_t3_direct   = 0
        self._insert_t1_hindsight = 0
        self._insert_t3_hindsight = 0

    def add(self, state, joint_actions, role_tuple,
            cooperation_quality, reward, hindsight_label: Optional[str] = None):

        if hindsight_label is not None:
            if hindsight_label == "goal_actual":
                if self.tier1_enabled:
                    self.tier1.add(state, joint_actions, role_tuple,
                                   cooperation_quality, reward,
                                   source_label="hcm_goal")
                    self._insert_t1_hindsight += 1

            elif hindsight_label == "pass_actual":
                if self.tier3_enabled:
                    self.tier3.add(state, joint_actions, role_tuple,
                                   cooperation_quality, reward,
                                   source_label="hcm_pass")
                    self._insert_t3_hindsight += 1

            elif hindsight_label == "hindsight_imagined":
                if self.tier3_enabled and self.t3_accepts_hindsight_imagined:
                    self.tier3.add(state, joint_actions, role_tuple,
                                   cooperation_quality, reward,
                                   source_label="hcm_imagined")
                    self._insert_t3_hindsight += 1
            return

        if self.tier1_enabled and (
            reward > 0 or cooperation_quality > self.t1_coop_quality_insert_thresh
        ):
            self.tier1.add(state, joint_actions, role_tuple,
                           cooperation_quality, reward, source_label="direct")
            self._insert_t1_direct += 1

        if self.tier2_enabled:
            self.tier2.add(state, joint_actions, role_tuple,
                           cooperation_quality, reward)
            self._insert_t2_direct += 1

        if self.tier3_enabled and cooperation_quality > self.t3_coop_quality_insert_thresh:
            self.tier3.add(state, joint_actions, role_tuple,
                           cooperation_quality, reward, source_label="direct")
            self._insert_t3_direct += 1

    def query(self, state, role_tuple, k: int = 3,
              min_similarity: Optional[float] = None,
              for_smal: bool = False) -> Optional[List[dict]]:
        self._cum_query_count += 1
        self._query_count     += 1

        def _eff(tier_t):
            if min_similarity is None:
                return tier_t
            return max(tier_t, min_similarity)

        if self.tier1_enabled:
            results = self.tier1.query(
                state, role_tuple=None, k=k,
                min_similarity=_eff(self.tier1_threshold),
            )
            if results:
                self._cum_hit_count += 1
                self._cum_hit_t1    += 1
                self._hit_count     += 1
                self._update_cum_sim_qual(results)
                for r in results:
                    r['tier_origin'] = 'T1'
                return results

        if self.tier2_enabled:
            results = self.tier2.query(
                state, role_tuple=role_tuple, k=k,
                min_similarity=_eff(self.tier2_threshold),
            )
            if results:
                self._cum_hit_count += 1
                self._cum_hit_t2    += 1
                self._hit_count     += 1
                self._update_cum_sim_qual(results)
                for r in results:
                    r['tier_origin'] = 'T2'
                return results

        if self.tier3_enabled and not for_smal:
            results = self.tier3.query(
                state, role_tuple=None, k=k,
                min_similarity=_eff(self.tier3_threshold),
            )
            if results:
                self._cum_hit_count += 1
                self._cum_hit_t3    += 1
                self._hit_count     += 1
                self._update_cum_sim_qual(results)
                for r in results:
                    r['tier_origin'] = 'T3'
                return results

        return None

    def query_t3_standalone(self, state) -> Optional[float]:
        if not hasattr(self, '_cum_t3_cer_queries'):
            self._cum_t3_cer_queries = 0
            self._cum_t3_cer_hits    = 0
        self._cum_t3_cer_queries += 1

        if not self.tier3_enabled:
            return None

        results = self.tier3.query(
            state, role_tuple=None, k=1,
            min_similarity=self.tier3_threshold,
        )
        if not results:
            return None

        self._cum_t3_cer_hits += 1
        return float(results[0].get("cooperation_quality", 0.5))

    def get_t3_cer_diagnostics(self) -> dict:
        q = getattr(self, '_cum_t3_cer_queries', 0)
        h = getattr(self, '_cum_t3_cer_hits', 0)
        return {
            "t3_cer_queries_total" : int(q),
            "t3_cer_hits_total"    : int(h),
            "t3_cer_hit_rate"      : round(h / max(q, 1), 4),
        }

    def _update_cum_sim_qual(self, results: List[dict]):
        for r in results:
            self._cum_sim_n += 1
            self._cum_sim_mean += (r['similarity'] - self._cum_sim_mean) / self._cum_sim_n
            self._cum_qual_n += 1
            self._cum_qual_mean += (
                (r['cooperation_quality'] - self._cum_qual_mean) / self._cum_qual_n
            )

    def __len__(self):
        return len(self.tier1) + len(self.tier2) + len(self.tier3)

    def partition_summary(self) -> dict:
        return self.tier2.partition_summary()

    @property
    def partitions(self) -> dict:
        return self.tier2.partitions

    def get_hybrid_diagnostics(self) -> dict:
        cum_q = max(self._cum_query_count, 1)
        miss = self._cum_query_count - self._cum_hit_count

        return {
            "hybrid_enabled": True,
            "tier1_size": len(self.tier1),
            "tier2_size": len(self.tier2),
            "tier3_size": len(self.tier3),
            "tier1_hits_total": int(self._cum_hit_t1),
            "tier2_hits_total": int(self._cum_hit_t2),
            "tier3_hits_total": int(self._cum_hit_t3),
            "tier1_hit_rate":   round(self._cum_hit_t1 / cum_q, 4),
            "tier2_hit_rate":   round(self._cum_hit_t2 / cum_q, 4),
            "tier3_hit_rate":   round(self._cum_hit_t3 / cum_q, 4),
            "tier1_avg_similarity":   round(self.tier1._cum_sim_mean, 4),
            "tier1_avg_coop_quality": round(self.tier1._cum_qual_mean, 4),
            "tier3_avg_similarity":   round(self.tier3._cum_sim_mean, 4),
            "tier3_avg_coop_quality": round(self.tier3._cum_qual_mean, 4),
            "cascade_exit_distribution": {
                "T1":   round(self._cum_hit_t1 / cum_q, 4),
                "T2":   round(self._cum_hit_t2 / cum_q, 4),
                "T3":   round(self._cum_hit_t3 / cum_q, 4),
                "miss": round(miss            / cum_q, 4),
            },
            "insert_counts": {
                "t1_direct":    int(self._insert_t1_direct),
                "t2_direct":    int(self._insert_t2_direct),
                "t3_direct":    int(self._insert_t3_direct),
                "t1_hindsight": int(self._insert_t1_hindsight),
                "t3_hindsight": int(self._insert_t3_hindsight),
            },
            "tier_enabled_flags": {
                "T1": bool(self.tier1_enabled),
                "T2": bool(self.tier2_enabled),
                "T3": bool(self.tier3_enabled),
            },
        }

    def add_hindsight(self, state, joint_actions, role_tuple,
                       cooperation_quality, reward, hindsight_label: str):
        self.add(state, joint_actions, role_tuple,
                 cooperation_quality, reward, hindsight_label=hindsight_label)
