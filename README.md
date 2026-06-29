# CHARM-MARL

Code for the paper **"Seed Variance and Evaluation Rigor in Memory-Augmented
Cooperative MARL: A Study with CHARM"**.

CHARM (Cooperative Hindsight-Augmented Role-conditioned Memory) is a memory
layer for cooperative multi-agent reinforcement learning. This repository runs
the full study: it trains several backbones with and without CHARM on a Google
Research Football task, then runs the statistical analysis used in the paper.

The task is the `academy_3_vs_1_with_keeper` scenario from Google Research
Football. A "win" is an episode where the controlled team scores at least one
goal.

> **Note on results.** This is an honest negative-results study. CHARM does not
> improve the mean win count, the variance it reduces comes from generic
> regularization rather than the memory mechanism, and no CHARM comparison
> survives Benjamini-Hochberg correction. The code is published so the runs and
> the numbers can be checked.

---

## Repository layout

**Training and run control**

| File | What it does |
|------|--------------|
| `run_one_experiment.py` | Runs a single (seed, experiment) combination and saves the logs. |
| `run_parallel.py` | Runs many (seed, experiment) combinations, a few at a time. |
| `curriculum_trainer.py` | The training loop that ties the env, the agent, and the memory together. |
| `env_parametric.py` | Wraps the Google Research Football scenario and adds reward shaping and metric logging. |

**Backbones**

| File | What it does |
|------|--------------|
| `dqn_agents.py` | DQN backbone. |
| `qmix_agents.py` | QMIX backbone. |
| `maddpg_agents.py` | MADDPG backbone. |
| `qplex_agents.py` | QPLEX backbone. |

**Memory**

| File | What it does |
|------|--------------|
| `memory_system.py` | `MemoryManager`, the entry point for the CHARM memory layer. |
| `hybrid_memory.py` | The three-tier hybrid memory variant. |

**Literature baselines**

| File | What it does |
|------|--------------|
| `emc_agents.py` | EMC episodic-memory baseline. |
| `emu_agents.py` | EMU episodic-memory baseline. |
| `mappo_agents.py` | MAPPO baseline. |
| `random_baseline.py` | Random-action floor for the win count. |

**Metrics and analysis**

| File | What it does |
|------|--------------|
| `cooperation_metrics.py` | Cooperation index and its sub-metrics. |
| `tactical_quality.py` | Tactical quality scores. |
| `diagnostics.py` | Per-run diagnostic logging for CHARM. |
| `analyze_all.py` | Reads all run logs and writes `results_all.json` plus the paper tables. |
| `analyze_multiseed.py` | Multi-seed summary. |
| `analyze_ablation.py` | Component ablation summary. |
| `analyze_full_matrix.py` | Full configuration matrix summary. |

---

## Install

Google Research Football builds a C++ engine and needs several system
libraries, so Docker is the easier path.

### Option A: Docker (recommended)

```bash
# Build the image (this also installs Google Research Football).
docker build -t charm-marl .

# Open a shell inside the container.
docker run -it --rm -v "$(pwd)/experiments:/app/experiments" charm-marl bash
```

The `-v` flag keeps your experiment outputs on the host machine, so they stay
after the container stops.

### Option B: Local install (Linux)

You need Python 3.9. Other versions may not work with Google Research Football.

```bash
# 1. System libraries for the Google Research Football engine.
sudo apt-get install git cmake build-essential \
  libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev \
  libsdl2-gfx-dev libboost-all-dev libdirectfb-dev libst-dev \
  mesa-utils xvfb x11vnc

# 2. A virtual environment.
python3.9 -m venv .venv
source .venv/bin/activate

# 3. gym 0.21.0 needs older build tools, so pin them before installing.
pip install "pip==23.3.2"
pip install "setuptools==65.5.0" "wheel==0.38.4" psutil

# 4. Project dependencies (this installs Google Research Football too).
pip install -r requirements.txt
```

Check the install:

```bash
python -c "import torch, faiss, gym; import gfootball.env; print('OK')"
```

---

## How to run

### 1. A single experiment

```bash
# Pattern: python3 run_one_experiment.py <experiment_name> --seed <seed>
python3 run_one_experiment.py A2_dqn_charm --seed 42
```

Run with no valid name to print the full list of experiment names:

```bash
python3 run_one_experiment.py
```

Useful flags:

- `--episodes N` runs a short test instead of the full 3000 episodes.
- `--resume` continues from saved checkpoints.

Outputs go to `experiments/seed_<seed>/<experiment_name>/`.

### 2. Many experiments in parallel

```bash
# A named group, for example the main comparison.
python3 run_parallel.py main

# Pick seeds and how many run at once.
SEEDS="0 1 7 42 123" PARALLEL_COUNT=2 python3 run_parallel.py main
```

Run `python3 run_parallel.py` and read the file header for the group names
(`main`, `baselines`, `ablation`, `vanilla`, `hybrid`, `charm`, `all`, and the
ablation groups).

### 3. The random baseline

```bash
python3 random_baseline.py
```

### 4. Analysis

After the runs finish, build the result file and the paper tables:

```bash
python3 analyze_all.py
```

This reads everything under `experiments/` and writes `results_all.json`,
`paper_tables.md`, and `paper_tables.tex`. `results_all.json` is the single
source for every number in the paper.

---

## Saved runs and data

`results_all.json` is in the repository root, so every number in the paper can
be checked without running anything.

The raw runs under `experiments/` are too large for GitHub, so they are on
Google Drive:

https://drive.google.com/drive/folders/128YjR5NAcbrr1-DGd6i9YLaEF82u3N-h?usp=drive_link

The folder holds one directory per seed (`seed_0`, `seed_1`, `seed_7`,
`seed_42`, `seed_123`), each with the 17 configurations of the study. The
`seed_42` folder also contains `A6_dqn_hybrid`, the three-tier hybrid run used
for the hybrid-memory result (Table 18).

To use it, download `experiments.zip` from Google Drive and unzip it in
the repository root. This creates the `experiments/` folder. Then
`python3 analyze_all.py` rebuilds `results_all.json` and the paper tables.

## Experiment names

Names follow a simple pattern:

- `A*` DQN family, `B*` QMIX, `C*` MADDPG, `E*` QPLEX.
- `D*` literature baselines (EMC, EMU, MAPPO).
- `R*` non-memory regularizer controls.
- Suffix `_vanilla` is the base algorithm, `_charm` adds CHARM, `_hybrid` adds
  the three-tier memory.
- `A2a`–`A2d` are the component ablations (drop one CHARM part at a time).

The full setup is 17 configurations times 5 seeds, which is 85 training runs.

---

## Notes

- The study uses 5 seeds: `0, 1, 7, 42, 123`, 3000 episodes each.
- faiss runs on CPU here (`IndexFlatL2`), so `faiss-cpu` is enough.
- Training is heavy. A full sweep takes many hours. Start with `--episodes`
  set to a small number to confirm the pipeline works end to end.
- The `gym==0.21.0` install fails with new setuptools. The pinned versions in
  the steps above and in the Dockerfile avoid this.

---

## License

This project is released under the MIT License. See the `LICENSE` file for the
full text.
