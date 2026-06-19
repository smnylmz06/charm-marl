import os
import sys
import time
import json
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


DEFAULT_SEEDS = [42, 7, 123]

if "SEEDS" in os.environ:
    SEEDS = [int(s) for s in os.environ["SEEDS"].split()]
else:
    SEEDS = DEFAULT_SEEDS


EXPERIMENT_GROUPS = {
    "main"     : ["A2_dqn_charm", "A6_dqn_hybrid"],

    "baselines": ["D1_emc", "D2_emu", "D3_mappo"],

    "ablation" : ["B2_qmix_charm", "B6_qmix_hybrid",
                  "C2_maddpg_charm", "C6_maddpg_hybrid"],

    "vanilla"  : ["A1_dqn_vanilla", "B1_qmix_vanilla", "C1_maddpg_vanilla"],

    "hybrid"   : ["A6_dqn_hybrid", "B6_qmix_hybrid", "C6_maddpg_hybrid"],

    "charm"    : ["A2_dqn_charm", "B2_qmix_charm", "C2_maddpg_charm"],

    "qplex_matched": ["E2_qplex_charm", "D1_emc", "D2_emu", "E1_qplex_vanilla"],

    "ablation1"    : ["A2a_dqn_charm_noCER",  "A2b_dqn_charm_noSMAL"],
    "ablation2"    : ["A2c_dqn_charm_noHCR",  "A2d_dqn_charm_noJACM"],
    "ablation_full": ["A2a_dqn_charm_noCER",  "A2b_dqn_charm_noSMAL",
                      "A2c_dqn_charm_noHCR",  "A2d_dqn_charm_noJACM"],
}

EXPERIMENT_GROUPS["all"] = []
_seen = set()
for grp_name in ("vanilla", "charm", "hybrid", "baselines"):
    for exp in EXPERIMENT_GROUPS[grp_name]:
        if exp not in _seen:
            EXPERIMENT_GROUPS["all"].append(exp)
            _seen.add(exp)


if os.environ.get("EXPS"):
    EXPERIMENTS = os.environ["EXPS"].split()
    GROUP_NAME = "custom (EXPS env)"
elif len(sys.argv) > 1 and sys.argv[1] in EXPERIMENT_GROUPS:
    GROUP_NAME = sys.argv[1]
    EXPERIMENTS = EXPERIMENT_GROUPS[GROUP_NAME]
elif len(sys.argv) > 1:
    print(f"❌ Unknown group: {sys.argv[1]}")
    print(f"   Available groups: {', '.join(EXPERIMENT_GROUPS.keys())}")
    sys.exit(1)
else:
    GROUP_NAME = "main"
    EXPERIMENTS = EXPERIMENT_GROUPS["main"]


PARALLEL_COUNT = int(os.environ.get("PARALLEL_COUNT", "2"))

EXPERIMENTS_ROOT = SCRIPT_DIR / "experiments"
MASTER_LOG       = EXPERIMENTS_ROOT / "master_summary.jsonl"


def log_master(record: dict):
    EXPERIMENTS_ROOT.mkdir(exist_ok=True)
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def is_complete(seed: int, exp_name: str, episodes: int = 3000) -> bool:
    exp_dir = EXPERIMENTS_ROOT / f"seed_{seed}" / exp_name
    coop = exp_dir / f"seed_{seed}__{exp_name}__cooperation_metrics.jsonl"
    if not coop.exists():
        return False
    try:
        with open(coop) as f:
            count = sum(1 for _ in f)
        return count >= int(episodes * 0.9)
    except Exception:
        return False


def launch(seed: int, exp_name: str):
    exp_dir = EXPERIMENTS_ROOT / f"seed_{seed}" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "run.log"
    log_file = open(log_path, "w", buffering=1)

    cmd = ["xvfb-run", "-a", "python3",
           str(SCRIPT_DIR / "run_one_experiment.py"),
           exp_name, "--seed", str(seed)]

    env = os.environ.copy()
    env["SEED"] = str(seed)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"   ▶️  Started: seed_{seed}/{exp_name}  (log: {log_path.relative_to(SCRIPT_DIR)})")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(SCRIPT_DIR),
        env=env,
    )
    return proc, log_file


def make_runs():
    pending = []
    skipped = []
    for seed in SEEDS:
        for exp in EXPERIMENTS:
            if is_complete(seed, exp):
                skipped.append((seed, exp))
            else:
                pending.append((seed, exp))
    return pending, skipped


HEAVY_CONFIGS = {
    "A2_dqn_charm", "B2_qmix_charm", "C2_maddpg_charm",
    "A6_dqn_hybrid", "B6_qmix_hybrid", "C6_maddpg_hybrid",
    "E2_qplex_charm", "E6_qplex_hybrid",
    "A2a_dqn_charm_noCER",
    "A2b_dqn_charm_noSMAL",
    "A2c_dqn_charm_noHCR",
}
MEDIUM_CONFIGS = {
    "D1_emc", "D2_emu", "D3_mappo",
}


def config_weight(exp_name: str) -> str:
    if exp_name in HEAVY_CONFIGS:
        return "heavy"
    if exp_name in MEDIUM_CONFIGS:
        return "medium"
    return "light"


def memory_aware_batches(pending, parallel):
    heavy = [r for r in pending if config_weight(r[1]) == "heavy"]
    rest  = [r for r in pending if config_weight(r[1]) != "heavy"]
    rest.sort(key=lambda r: 0 if config_weight(r[1]) == "medium" else 1)

    batches = []

    ri = 0
    for h in heavy:
        batch = [h]
        while len(batch) < parallel and ri < len(rest):
            batch.append(rest[ri])
            ri += 1
        batches.append(batch)

    remaining = rest[ri:]
    for i in range(0, len(remaining), parallel):
        batches.append(remaining[i:i + parallel])

    return batches


def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def estimate_hours(n_runs: int, parallel: int, avg_per_run: float = 1.0) -> float:
    return n_runs * avg_per_run / parallel


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("═" * 78)
    print(f"🏆 MULTI-SEED PARALLEL RUNNER — Group: {GROUP_NAME.upper()}")
    print("═" * 78)
    print(f"  Seeds               : {SEEDS}  ({len(SEEDS)} total)")
    print(f"  Experiments         : {len(EXPERIMENTS)} total — {EXPERIMENTS}")
    print(f"  Total combinations  : {len(SEEDS) * len(EXPERIMENTS)} runs")
    print(f"  Parallel            : {PARALLEL_COUNT}")
    print(f"  Master log          : {MASTER_LOG}")
    print("═" * 78)

    pending, skipped = make_runs()
    print(f"\n  Total: {len(SEEDS) * len(EXPERIMENTS)}")
    print(f"  ✓ Already complete: {len(skipped)}")
    print(f"  ⏳ To run     : {len(pending)}")

    if skipped:
        print("\n  To skip (already complete):")
        for seed, exp in skipped[:6]:
            print(f"    • seed_{seed}/{exp}")
        if len(skipped) > 6:
            print(f"    ... and {len(skipped)-6} more")

    if not pending:
        print("\n🎉 All runs already complete, nothing to do.")
        return

    est_hours = estimate_hours(len(pending), PARALLEL_COUNT)
    print(f"\n  Estimated time: ~{est_hours:.1f} hours ({len(pending)} runs / {PARALLEL_COUNT} parallel)")
    print(f"  NOTE: CHARM/Hybrid variants ~1h, vanilla ~45min, baselines ~50min")
    print()

    grand_start = time.time()
    log_master({
        "event": "RUN_STARTED",
        "group": GROUP_NAME,
        "experiments": EXPERIMENTS,
        "n_pending": len(pending),
        "n_skipped": len(skipped),
        "seeds": SEEDS,
        "parallel_count": PARALLEL_COUNT,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    mem_aware = os.environ.get("MEM_AWARE", "1") != "0"
    if mem_aware:
        batches = memory_aware_batches(pending, PARALLEL_COUNT)
        n_heavy = sum(1 for r in pending if config_weight(r[1]) == "heavy")
        print(f"  Batch count: {len(batches)}  "
              f"(memory-aware: {n_heavy} heavy configs isolated)")
    else:
        batches = list(chunk(pending, PARALLEL_COUNT))
        print(f"  Batch count: {len(batches)}  (naive chunk)")
    print()

    try:
        for batch_idx, batch in enumerate(batches, 1):
            batch_start = time.time()
            print(f"\n{'━' * 78}")
            print(f"BATCH {batch_idx}/{len(batches)}: {len(batch)} runs in parallel")
            for seed, exp in batch:
                w = config_weight(exp)
                tag = {"heavy": "🔴 HEAVY", "medium": "🟡 MEDIUM",
                       "light": "🟢 LIGHT"}[w]
                print(f"   • seed_{seed}/{exp}  [{tag}]")
            print(f"{'━' * 78}")

            procs_logs = []
            for seed, exp in batch:
                proc, log_file = launch(seed, exp)
                procs_logs.append((proc, log_file, seed, exp))
                time.sleep(2)

            print(f"\n   ⏳ {len(batch)} processes running...")
            print(f"      Watch: tail -f experiments/seed_<S>/<exp>/run.log")

            for proc, log_file, seed, exp in procs_logs:
                rc = proc.wait()
                log_file.close()
                status = "✅" if rc == 0 else "❌"
                log_master({
                    "event": "RUN_COMPLETED",
                    "group": GROUP_NAME,
                    "seed": seed, "exp_name": exp,
                    "return_code": rc,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                print(f"   {status} seed_{seed}/{exp} (rc={rc})")

            elapsed = time.time() - batch_start
            print(f"\n   ⏱  Batch time: {elapsed/60:.1f} minutes")

    except KeyboardInterrupt:
        print(f"\n\n⛔ STOPPED. Completed runs are kept, you can resume from here.")
        log_master({
            "event": "INTERRUPTED",
            "group": GROUP_NAME,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        sys.exit(130)

    grand_elapsed = time.time() - grand_start
    log_master({
        "event": "RUN_FULLY_DONE",
        "group": GROUP_NAME,
        "total_hours": round(grand_elapsed/3600, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    print(f"\n{'═' * 78}")
    print(f"🎉 {GROUP_NAME.upper()} GROUP DONE ({grand_elapsed/3600:.2f} hours)")
    print(f"{'═' * 78}")
    print(f"\n📤 Prepare for upload:")
    print(f"   mkdir -p uploads_ready")
    print(f"   cp experiments/seed_*/*/seed_*__*__*.jsonl uploads_ready/")
    expected_files = len(SEEDS) * len(EXPERIMENTS) * 4
    print(f"   ls uploads_ready/ | wc -l    # beklenen: ~{expected_files}+ dosya")


if __name__ == "__main__":
    main()
