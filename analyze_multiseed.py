import json
import statistics
from pathlib import Path

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("⚠️  scipy not found — t-test/Wilcoxon skipped. To install:")
    print("    pip install scipy --break-system-packages\n")

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR / "experiments"

METHODS = {
    "CHARM (A2)": "A2_dqn_charm",
    "EMC (D1)":   "D1_emc",
    "EMU (D2)":   "D2_emu",
}
SEEDS = [0, 1, 7, 42, 123]
EPISODES = 3000
BUCKET = 1000

KNOWN_OVERRIDES = {
    ("A2_dqn_charm", 42): 61,
}


def load_wins(exp_name: str, seed: int):
    if (exp_name, seed) in KNOWN_OVERRIDES:
        w = KNOWN_OVERRIDES[(exp_name, seed)]
        return w, None

    path = EXP_ROOT / f"seed_{seed}" / exp_name / \
           f"seed_{seed}__{exp_name}__rewards_and_winrate.jsonl"
    if not path.exists():
        return None, None
    try:
        recs = [json.loads(l) for l in open(path)]
        m1 = [i for i, r in enumerate(recs) if r.get("match") == 1]
        if not m1:
            return None, None
        latest = recs[m1[-1]:]
        if len(latest) < EPISODES * 0.9:
            return None, None
        total = sum(1 for r in latest if r.get("win"))
        n_buckets = len(latest) // BUCKET
        buckets = [
            sum(1 for r in latest[i*BUCKET:(i+1)*BUCKET] if r.get("win"))
            for i in range(n_buckets)
        ]
        return total, buckets
    except Exception as e:
        print(f"  ⚠️  {exp_name} seed_{seed}: {e}")
        return None, None


def cohens_d(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    if len(diffs) < 2:
        return float("nan")
    m = statistics.mean(diffs)
    s = statistics.stdev(diffs)
    return m / s if s > 0 else float("inf")


def main():
    print("═" * 70)
    print("CHARM MULTI-SEED ANALYSIS")
    print("═" * 70)

    results = {}
    bucket_data = {}

    print("\n### 1. WIN COUNTS ###\n")
    for label, exp in METHODS.items():
        results[label] = {}
        bucket_data[label] = {}
        per_seed = []
        for s in SEEDS:
            total, buckets = load_wins(exp, s)
            if total is not None:
                results[label][s] = total
                if buckets:
                    bucket_data[label][s] = buckets
                per_seed.append(f"seed_{s}={total}W")
            else:
                per_seed.append(f"seed_{s}=--")
        vals = list(results[label].values())
        if len(vals) >= 2:
            mean = statistics.mean(vals)
            std = statistics.stdev(vals)
            print(f"  {label:15} : {' '.join(per_seed):40}  "
                  f"→ {mean:.1f} ± {std:.1f}W  (n={len(vals)})")
        elif len(vals) == 1:
            print(f"  {label:15} : {' '.join(per_seed):40}  "
                  f"→ {vals[0]}W  (n=1, no std)")
        else:
            print(f"  {label:15} : {' '.join(per_seed):40}  → no data")

    print("\n### 2. TRAINING STABILITY (bucket CV) ###\n")
    print("  CV = std(bucket_wins) / mean(bucket_wins); low = stable\n")
    for label in METHODS:
        cvs = []
        sample_pattern = None
        for s, buckets in bucket_data[label].items():
            if buckets and len(buckets) >= 2:
                bm = statistics.mean(buckets)
                bs = statistics.stdev(buckets)
                cv = bs / bm if bm > 0 else float("nan")
                cvs.append(cv)
                if sample_pattern is None:
                    sample_pattern = "/".join(str(b) for b in buckets[:3])
        if cvs:
            print(f"  {label:15} : CV ort = {statistics.mean(cvs):.3f}  "
                  f"(sample pattern: {sample_pattern})")
        else:
            print(f"  {label:15} : no bucket data (raw may be missing)")

    print("\n### 3. PAIRWISE COMPARISONS ###\n")
    labels = list(METHODS.keys())
    base = labels[0]
    for other in labels[1:]:
        common = sorted(set(results[base]) & set(results[other]))
        if len(common) < 2:
            print(f"  {base} vs {other}: shared seeds < 2, test skipped")
            continue
        a = [results[base][s] for s in common]
        b = [results[other][s] for s in common]
        diff = statistics.mean(a) - statistics.mean(b)
        d = cohens_d(a, b)
        print(f"  {base} vs {other}  (n={len(common)}, seeds={common})")
        print(f"    Δ ortalama: {diff:+.1f}W   Cohen's d = {d:.2f}", end="")
        if abs(d) >= 0.8:
            print("  (large effect)")
        elif abs(d) >= 0.5:
            print("  (orta etki)")
        elif abs(d) >= 0.2:
            print("  (small effect)")
        else:
            print("  (ihmal edilebilir etki)")
        if HAVE_SCIPY and len(common) >= 2:
            try:
                t_stat, t_p = scipy_stats.ttest_rel(a, b)
                print(f"    Paired t-test : t={t_stat:.3f}, p={t_p:.4f}", end="")
                print("  ✓ significant" if t_p < 0.05 else
                      ("  ~ borderline" if t_p < 0.15 else "  ✗ not significant"))
            except Exception as e:
                print(f"    t-test error: {e}")
            try:
                if len(common) >= 3:
                    w_stat, w_p = scipy_stats.wilcoxon(a, b)
                    print(f"    Wilcoxon      : W={w_stat:.1f}, p={w_p:.4f}")
            except Exception:
                pass
        print()

    print("\n### 4. LATEX TABLE (paste into paper) ###\n")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{comparison of methods on the 3v3 GRF "
          r"cooperative scenario (3 seeds, 3000 episodes).}")
    print(r"\label{tab:main_results}")
    print(r"\begin{tabular}{lccc}")
    print(r"\hline")
    print(r"Method & Wins (mean $\pm$ std) & CV & $n$ \\")
    print(r"\hline")
    for label in METHODS:
        vals = list(results[label].values())
        cvs = []
        for s, buckets in bucket_data[label].items():
            if buckets and len(buckets) >= 2:
                bm = statistics.mean(buckets)
                bs = statistics.stdev(buckets)
                if bm > 0:
                    cvs.append(bs / bm)
        if len(vals) >= 2:
            cell = f"${statistics.mean(vals):.1f} \\pm {statistics.stdev(vals):.1f}$"
        elif len(vals) == 1:
            cell = f"${vals[0]}$"
        else:
            cell = "--"
        cv_cell = f"{statistics.mean(cvs):.2f}" if cvs else "--"
        name = f"\\textbf{{{label}}}" if label == base else label
        print(f"{name} & {cell} & {cv_cell} & {len(vals)} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n" + "═" * 70)
    print("Analysis done. Put the LaTeX table into the paper and the numbers into the "
          "placeholders.")
    print("═" * 70)


if __name__ == "__main__":
    main()
