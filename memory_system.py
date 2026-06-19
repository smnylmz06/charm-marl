import numpy as np
import faiss
import os
import json
import torch
import faiss

class HierarchicalMemory:
    def __init__(self, state_dim=32, confidence=0.5, threshold=0.85,
                 max_size=10000):
        self.state_dim   = state_dim
        self.max_size    = max_size
        self.index       = faiss.IndexFlatL2(state_dim)
        self.experiences = []
        self._states     = []
        self.confidence, self.threshold = confidence, threshold

    def add(self, state, action, role, reward):
        state_arr = state.reshape(1, -1).astype('float32')
        self.index.add(state_arr)
        self.experiences.append({'action': action, 'role': role, 'reward': reward})
        self._states.append(state.astype('float32'))

        if len(self.experiences) > self.max_size:
            evict_n          = self.max_size // 5
            self.experiences = self.experiences[evict_n:]
            self._states     = self._states[evict_n:]
            self.index       = faiss.IndexFlatL2(self.state_dim)
            if self._states:
                self.index.add(np.vstack(self._states))

    def search(self, state, role):
        if self.index.ntotal == 0: return None, 0
        dist, idx = self.index.search(state.reshape(1, -1).astype('float32'), 1)
        sim = 1 / (1 + dist[0][0])
        res = self.experiences[idx[0][0]]
        return (res['action'], sim) if res['role'] == role else (None, 0)


class JointActionMemory:
    def __init__(self, state_dim=32, n_agents=3, max_size=2000, role_partitioned=True):
        self.state_dim         = state_dim
        self.n_agents          = n_agents
        self.max_size          = max_size
        self.role_partitioned  = role_partitioned
        self.partitions: dict  = {}
        self._query_count = 0
        self._hit_count   = 0
        self._cum_query_count = 0
        self._cum_hit_count   = 0
        self._cum_sim_mean    = 0.0
        self._cum_sim_n       = 0
        self._cum_qual_mean   = 0.0
        self._cum_qual_n      = 0

    def _get_or_create_partition(self, role_tuple):
        key = tuple(sorted(role_tuple)) if self.role_partitioned else "_all"
        if key not in self.partitions:
            self.partitions[key] = {
                "index"      : faiss.IndexFlatL2(self.state_dim),
                "experiences": [],
                "states_buf" : [],
            }
        return key, self.partitions[key]

    def add(self, state, joint_actions, role_tuple, cooperation_quality, reward,
            hindsight_label=None):
        if hindsight_label == "hindsight_imagined":
            return
        if len(joint_actions) != self.n_agents:
            return
        s_arr = state.astype('float32').reshape(1, -1)
        if s_arr.shape[1] != self.state_dim:
            return

        _, part = self._get_or_create_partition(role_tuple)
        part["index"].add(s_arr)
        part["experiences"].append({
            'joint_actions'      : [int(a) for a in joint_actions],
            'role_tuple'         : list(role_tuple),
            'cooperation_quality': float(cooperation_quality),
            'reward'             : float(reward),
        })
        part["states_buf"].append(s_arr.flatten())

        if len(part["experiences"]) > self.max_size:
            evict_n = self.max_size // 5
            part["experiences"] = part["experiences"][evict_n:]
            part["states_buf"]  = part["states_buf"][evict_n:]
            part["index"]       = faiss.IndexFlatL2(self.state_dim)
            if part["states_buf"]:
                part["index"].add(np.vstack(part["states_buf"]))

    def query(self, state, role_tuple, k=3, min_similarity=0.75, for_smal=False):
        self._query_count     += 1
        self._cum_query_count += 1

        key = tuple(sorted(role_tuple)) if self.role_partitioned else "_all"
        if key not in self.partitions:
            return None
        part = self.partitions[key]
        if part["index"].ntotal == 0:
            return None

        s_arr = state.astype('float32').reshape(1, -1)
        if s_arr.shape[1] != self.state_dim:
            return None

        k = min(k, part["index"].ntotal)
        distances, indices = part["index"].search(s_arr, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            sim = 1.0 / (1.0 + float(dist))
            if sim < min_similarity:
                continue
            exp = part["experiences"][idx]
            results.append({
                'joint_actions'      : exp['joint_actions'],
                'similarity'         : sim,
                'cooperation_quality': exp['cooperation_quality'],
                'weighted_score'     : sim * exp['cooperation_quality'],
                'role_tuple'         : exp['role_tuple'],
            })
        if results:
            self._hit_count     += 1
            self._cum_hit_count += 1
            for r in results:
                self._cum_sim_n += 1
                self._cum_sim_mean += (r['similarity'] - self._cum_sim_mean) / self._cum_sim_n
                self._cum_qual_n += 1
                self._cum_qual_mean += (r['cooperation_quality'] - self._cum_qual_mean) / self._cum_qual_n
        return results if results else None

    def __len__(self):
        return sum(len(p["experiences"]) for p in self.partitions.values())

    def partition_summary(self) -> dict:
        return {
            (",".join(map(str, k)) if isinstance(k, tuple) else str(k))
            : len(p["experiences"])
            for k, p in self.partitions.items()
        }

class MemoryManager:
    def __init__(self, state_dim=32, query_interval=2, enabled=True,
                 joint_enabled=False, n_agents=3,
                 use_hybrid: bool = False,
                 hybrid_config: dict = None):
        self.query_interval = query_interval
        self.enabled = enabled
        self.joint_enabled = joint_enabled
        self.coach      = HierarchicalMemory(state_dim, 0.90, 0.92, max_size=5_000)
        self.selected   = HierarchicalMemory(state_dim, 0.40, 0.93, max_size=15_000)
        self.agent_self = [HierarchicalMemory(state_dim, 0.2, 0.75, max_size=3_000)
                          for _ in range(10)]
        self._stats = self._fresh_stats()
        self.use_hybrid = bool(use_hybrid)
        if self.use_hybrid:
            from hybrid_memory import HybridMemorySystem
            cfg = dict(hybrid_config or {})
            cfg.setdefault("state_dim", state_dim)
            cfg.setdefault("n_agents",  n_agents)
            self.joint = HybridMemorySystem(**cfg)
        else:
            self.joint = JointActionMemory(
                state_dim=state_dim, n_agents=n_agents, max_size=2_000
            )
        self._joint_stats = {'queries': 0, 'hits': 0, 'avg_quality': 0.0}

    def _fresh_stats(self) -> dict:
        return {
            "offered" : 0,
            "used"    : 0,
            "by_layer": {"coach": 0, "selected": 0, "agent_self": 0},
        }

    def reset_episode_stats(self):
        self._stats = self._fresh_stats()

    def mark_guidance_used(self):
        self._stats["used"] += 1

    def get_episode_stats(self) -> dict:
        s = self._stats
        offered = s["offered"]
        jacm_q  = self.joint._query_count
        jacm_h  = self.joint._hit_count
        return {
            "memory_disabled"        : not self.enabled,
            "memory_guidance_offered": offered,
            "memory_guidance_used"   : s["used"],
            "memory_use_rate"        : round(s["used"] / max(offered, 1), 4),
            "memory_layer_usage"     : dict(s["by_layer"]),
            "jacm_enabled"           : self.joint_enabled,
            "jacm_size"              : len(self.joint),
            "jacm_queries"           : jacm_q,
            "jacm_hits"              : jacm_h,
            "jacm_hit_rate"          : round(jacm_h / max(jacm_q, 1), 4),
        }

    def query_joint(self, agent_states, k=3, min_similarity=0.75):
        if not self.joint_enabled:
            return None
        self._joint_stats["queries"] += 1
        results = self.joint.query(agent_states, k=k, min_similarity=min_similarity)
        if results:
            self._joint_stats["hits"] += 1
        return results

    def query_t3_for_cer(self, state):
        if not self.joint_enabled:
            return None
        if not self.use_hybrid:
            return None
        if not hasattr(self.joint, 'query_t3_standalone'):
            return None
        return self.joint.query_t3_standalone(state)

    def reset_joint_stats(self):
        self._joint_stats = {'queries': 0, 'hits': 0, 'avg_quality': 0.0}
        self.joint._query_count = 0
        self.joint._hit_count   = 0

    def get_guidance(self, agent_id, role, state, step):
        if not self.enabled:
            return None
        if step % self.query_interval != 0:
            return None

        layers = [
            ("coach",      self.coach),
            ("selected",   self.selected),
            ("agent_self", self.agent_self[agent_id]),
        ]
        for layer_name, mem in layers:
            action, sim = mem.search(state, role)
            if action is not None and sim >= mem.threshold:
                self._stats["offered"]              += 1
                self._stats["by_layer"][layer_name] += 1
                return action, mem.confidence
        return None

    def load_phase_memory(self, phase_prefix, target_layer="coach"):

        idx_path = f"{phase_prefix}_selected.index"
        exp_path = f"{phase_prefix}_selected_exp.json"

        if os.path.exists(idx_path) and os.path.exists(exp_path):
            target = self.coach if target_layer == "coach" else self.selected
            target.index = faiss.read_index(idx_path)
            with open(exp_path, "r") as f:
                target.experiences = json.load(f)
            print(f"✅ {phase_prefix} experiences transferred to the {target_layer} layer.")
