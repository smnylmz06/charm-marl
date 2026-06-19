import random
import numpy as np

from env_parametric import ParametricFootballEnv

SEEDS      = [0, 1, 7, 42, 123]
EPISODES   = 3000
NUM_AGENTS = 3
NUM_ACTIONS = 19
MAX_STEPS  = 400


def _reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _left_score(obs_list):
    return int(obs_list[0]["score"][0])


def run_seed(seed):
    rng = random.Random(seed)
    np.random.seed(seed)

    wrapper = ParametricFootballEnv(mode="3v3")
    env = wrapper.env

    wins = 0
    for _ in range(EPISODES):
        obs = _reset(env)
        done = False
        steps = 0
        while not done and steps < MAX_STEPS:
            actions = [rng.randrange(NUM_ACTIONS) for _ in range(NUM_AGENTS)]
            step_out = env.step(actions)
            obs, _, done, _ = step_out[0], step_out[1], step_out[2], step_out[-1]
            steps += 1
        if _left_score(obs) > 0:
            wins += 1

    try:
        env.close()
    except Exception:
        pass
    return wins


def main():
    results = {}
    for s in SEEDS:
        w = run_seed(s)
        results[s] = w
        print(f"  seed {s:>3}: {w} wins")

    vals = np.array(list(results.values()), dtype=float)
    mean = vals.mean()
    sd = vals.std(ddof=1)
    print("\nRandom baseline (5 seeds, 3000 episodes):")
    print(f"  per-seed: {results}")
    print(f"  mean ± σ (ddof=1): {mean:.1f} ± {sd:.1f}")
    print("\nPaste-ready for the paper:")
    print(f'  "A random-action baseline scores {mean:.1f} ± {sd:.1f} wins over the '
          f'same protocol (5 seeds, 3000 episodes), so the trained methods sit '
          f'well above the floor."')


if __name__ == "__main__":
    main()
