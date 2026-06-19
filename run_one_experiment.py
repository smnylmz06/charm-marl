import os
import sys
import gc
import json
import time
import shutil
import random
import argparse
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


CONFIGS = {
    "A1_dqn_vanilla"    : {"algorithm":"dqn",    "use_cer":False, "joint_enabled":False, "use_hybrid":False, "hier_enabled":False},
    "A2_dqn_charm"      : {"algorithm":"dqn",    "use_cer":True,  "joint_enabled":True,  "use_hybrid":False, "hier_enabled":False},
    "A6_dqn_hybrid"     : {"algorithm":"dqn",    "use_cer":True,  "joint_enabled":True,  "use_hybrid":True,  "hier_enabled":False},

    "B1_qmix_vanilla"   : {"algorithm":"qmix",   "use_cer":False, "joint_enabled":False, "use_hybrid":False, "hier_enabled":False},
    "B2_qmix_charm"     : {"algorithm":"qmix",   "use_cer":True,  "joint_enabled":True,  "use_hybrid":False, "hier_enabled":False},
    "B6_qmix_hybrid"    : {"algorithm":"qmix",   "use_cer":True,  "joint_enabled":True,  "use_hybrid":True,  "hier_enabled":False},

    "C1_maddpg_vanilla" : {"algorithm":"maddpg", "use_cer":False, "joint_enabled":False, "use_hybrid":False, "hier_enabled":False},
    "C2_maddpg_charm"   : {"algorithm":"maddpg", "use_cer":True,  "joint_enabled":True,  "use_hybrid":False, "hier_enabled":False},
    "C6_maddpg_hybrid"  : {"algorithm":"maddpg", "use_cer":True,  "joint_enabled":True,  "use_hybrid":True,  "hier_enabled":False},

    "D1_emc"            : {"algorithm":"emc",    "use_cer":False, "joint_enabled":False, "use_hybrid":False, "hier_enabled":False},
    "D2_emu"            : {"algorithm":"emu",    "use_cer":False, "joint_enabled":False, "use_hybrid":False, "hier_enabled":False},
    "D3_mappo"          : {"algorithm":"mappo",  "use_cer":False, "joint_enabled":False, "use_hybrid":False, "hier_enabled":False},

    "E1_qplex_vanilla"  : {"algorithm":"qplex",  "use_cer":False, "joint_enabled":False, "use_hybrid":False, "hier_enabled":False},
    "E2_qplex_charm"    : {"algorithm":"qplex",  "use_cer":True,  "joint_enabled":True,  "use_hybrid":False, "hier_enabled":False},
    "E6_qplex_hybrid"   : {"algorithm":"qplex",  "use_cer":True,  "joint_enabled":True,  "use_hybrid":True,  "hier_enabled":False},

    "A2a_dqn_charm_noCER"  : {"algorithm":"dqn", "use_cer":False, "joint_enabled":True,  "use_hybrid":False, "hier_enabled":False},
    "A2b_dqn_charm_noSMAL" : {"algorithm":"dqn", "use_cer":True,  "joint_enabled":True,  "use_hybrid":False, "hier_enabled":False, "smal_max_weight":0.0},
    "A2c_dqn_charm_noHCR"  : {"algorithm":"dqn", "use_cer":True,  "joint_enabled":True,  "use_hybrid":False, "hier_enabled":False, "hcr_enabled":False},
    "A2d_dqn_charm_noJACM" : {"algorithm":"dqn", "use_cer":True,  "joint_enabled":False, "use_hybrid":False, "hier_enabled":False},

    "R1_dqn_l2"      : {"algorithm":"dqn", "use_cer":False, "joint_enabled":False, "use_hybrid":False, "hier_enabled":False, "reg_l2":1e-4},
    "R2_dqn_randaux" : {"algorithm":"dqn", "use_cer":False, "joint_enabled":False, "use_hybrid":False, "hier_enabled":False, "rand_aux":True},

}

COMMON = {
    "mode"                 : "3v3",
    "episodes"             : 3000,
    "epsilon_start"        : 1.0,
    "smal_warmup_episodes" : 500,
    "smal_full_episodes"   : 1000,
}


def set_all_seeds(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def is_already_complete(exp_dir: Path, exp_name: str, seed: int) -> bool:
    coop = exp_dir / f"seed_{seed}__{exp_name}__cooperation_metrics.jsonl"
    if not coop.exists():
        return False
    try:
        with open(coop) as f:
            count = sum(1 for _ in f)
        return count >= int(COMMON["episodes"] * 0.9)
    except Exception:
        return False


def archive_outputs(work_dir: Path, exp_dir: Path, exp_name: str,
                    seed: int, mode: str, algorithm: str) -> int:
    n = 0
    prefix = f"seed_{seed}__{exp_name}"

    possible_jsonl_dirs = [
        work_dir / "outputs",
        work_dir / "checkpoints" / "outputs",
        Path("outputs"),
    ]

    for out_dir in possible_jsonl_dirs:
        if not out_dir.exists():
            continue
        for f in out_dir.glob("*.jsonl"):
            target = exp_dir / f"{prefix}__{f.name}"
            try:
                if target.exists():
                    backup = exp_dir / f"{prefix}__{f.name}.bak"
                    target.rename(backup)
                shutil.copy2(str(f), str(target))
                n += 1
                print(f"  ✓ JSONL: {f.relative_to(work_dir.parent.parent) if work_dir.parent.parent in f.parents else f.name} → {target.name}")
            except Exception as e:
                print(f"  ⚠️  JSONL error ({f.name}): {e}")

    ckpt_dir = work_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpt_pattern = f"{mode}_{algorithm}_"
        for f in ckpt_dir.glob(f"{ckpt_pattern}*"):
            suffix = f.name.replace(ckpt_pattern, "", 1)
            target = exp_dir / f"{prefix}__{suffix}"
            try:
                shutil.copy2(str(f), str(target))
                n += 1
            except Exception as e:
                print(f"  ⚠️  CKPT error ({f.name}): {e}")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="Experiment name (defined in CONFIGS)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override COMMON['episodes'] (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoints "
                             "(work/checkpoints/ kept, trainer resumes from there)")
    args = parser.parse_args()

    if args.seed is not None:
        seed = args.seed
    elif "SEED" in os.environ:
        seed = int(os.environ["SEED"])
    else:
        seed = 42

    exp_name = args.exp_name
    if exp_name not in CONFIGS:
        print(f"❌ Unknown experiment: {exp_name}")
        print(f"   Available experiments:")
        for name in sorted(CONFIGS.keys()):
            spec = CONFIGS[name]
            features = []
            if spec.get("use_cer"): features.append("CER")
            if spec.get("joint_enabled"): features.append("JACM")
            if spec.get("use_hybrid"): features.append("Hybrid")
            features_str = "+".join(features) if features else "vanilla"
            print(f"     {name:<25} → {spec['algorithm']:<7} ({features_str})")
        sys.exit(1)
    spec = CONFIGS[exp_name]

    episodes_to_run = args.episodes if args.episodes else COMMON["episodes"]

    set_all_seeds(seed)

    exp_dir  = SCRIPT_DIR / "experiments" / f"seed_{seed}" / exp_name
    work_dir = exp_dir / "work"
    exp_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.episodes is None and is_already_complete(exp_dir, exp_name, seed):
        print(f"⏭️  SKIPPING: seed_{seed} / {exp_name} (already complete)")
        return 0

    os.chdir(str(work_dir))

    if args.resume:
        ckpt_dir = work_dir / "checkpoints"
        if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
            print(f"📂 RESUME mode: {len(list(ckpt_dir.iterdir()))} checkpoints found, continuing")
        else:
            print(f"⚠️  --resume given but no checkpoint found, starting from scratch")
    else:
        for sub in ("outputs", "checkpoints"):
            sub_path = work_dir / sub
            if sub_path.exists():
                for f in sub_path.iterdir():
                    try:
                        f.unlink()
                    except Exception:
                        pass

    features = []
    if spec.get("use_cer"):        features.append("CER")
    if spec.get("joint_enabled"):  features.append("JACM/T2")
    if spec.get("use_hybrid"):     features.append("Hybrid/T1+T2+T3")
    if spec.get("hier_enabled"):   features.append("Hierarchical")
    features_str = " + ".join(features) if features else "vanilla (no memory)"

    is_lit_baseline = spec["algorithm"] in ("emc", "emu", "mappo")
    if is_lit_baseline:
        features_str = "paper-faithful (own memory mechanism)"

    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🚀 STARTING (PID {os.getpid()}): seed_{seed} / {exp_name}")
    print(f"   Algorithm        : {spec['algorithm']}")
    print(f"   Features         : {features_str}")
    print(f"   Episodes / Seed  : {episodes_to_run} / {seed}")
    print(f"   Work dir         : {work_dir}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    start = time.time()
    status, error = "FAILED", None
    n_archived = 0

    try:
        set_all_seeds(seed)
        from curriculum_trainer import CurriculumTrainer

        trainer = CurriculumTrainer(
            mode       = COMMON["mode"],
            algorithm  = spec["algorithm"],
            use_cer    = spec["use_cer"],
            use_hybrid = spec["use_hybrid"],
            reg_l2     = spec.get("reg_l2", 0.0),
            rand_aux   = spec.get("rand_aux", False),
        )

        trainer.epsilon              = COMMON["epsilon_start"]
        trainer.mem.enabled          = spec["hier_enabled"]
        trainer.mem.joint_enabled    = spec["joint_enabled"]
        trainer.smal_warmup_episodes = COMMON["smal_warmup_episodes"]
        trainer.smal_full_episodes   = COMMON["smal_full_episodes"]
        if "smal_max_weight" in spec:
            trainer.smal_max_weight = spec["smal_max_weight"]
        if "hcr_enabled" in spec:
            trainer.hcr_enabled = spec["hcr_enabled"]

        trainer.train(episodes=episodes_to_run)

        n_archived = archive_outputs(
            work_dir, exp_dir, exp_name, seed,
            COMMON["mode"], spec["algorithm"],
        )
        status = "SUCCESS"
    except Exception as e:
        error = str(e)
        traceback.print_exc()
    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    elapsed = time.time() - start
    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if status == "SUCCESS":
        print(f"✅ DONE: seed_{seed} / {exp_name} | {elapsed/60:.1f} min | {n_archived} files")
    else:
        print(f"❌ ERROR: seed_{seed} / {exp_name} | {elapsed/60:.1f} min | {error}")

    summary = exp_dir / "summary.json"
    with open(summary, "w") as f:
        json.dump({
            "seed": seed, "exp_name": exp_name,
            "spec": spec, "common": COMMON,
            "episodes_run": episodes_to_run,
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_minutes": round(elapsed/60, 2),
            "status": status, "error": error, "n_archived": n_archived,
            "pid": os.getpid(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    return 0 if status == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
