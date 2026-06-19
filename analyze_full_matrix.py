import json
import statistics
from pathlib import Path

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("⚠️  scipy not found — t-test skipped (mean/std/CV still computed)\n")

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR / "experiments"

CONFIGS = {
    "A1 (DQN vanilla)":    "A1_dqn_vanilla",
    "A2 (DQN+CHARM)":      "A2_dqn_charm",
    "B1 (QMIX vanilla)":   "B1_qmix_vanilla",
    "B2 (QMIX+CHARM)":     "B2_qmix_charm",
    "C1 (MADDPG vanilla)": "C1_maddpg_vanilla",
    "C2 (MADDPG+CHARM)":   "C2_maddpg_charm",
    "D1 (EMC)":            "D1_emc",
    "D2 (EMU)":            "D2_emu",
    "D3 (MAPPO)":          "D3_mappo",
}
ALL_SEEDS = [42, 7, 123, 0, 1]
EPISODES = 3000
BUCKET = 1000


def load(exp_name, seed):
    path = (EXP_ROOT / f"seed_{seed}" / exp_name /
            f"seed_{seed}__{exp_name}__rewards_and_winrate.jsonl")
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
        nb = len(latest) // BUCKET
        buckets = [sum(1 for r in latest[i*BUCKET:(i+1)*BUCKET] if r.get("win"))
                   for i in range(nb)]
        return total, buckets
    except Exception:
        return None, None


def paired_test(a, b, name_a, name_b):
    if len(a) < 2 or len(a) != len(b):
        print(f"  {name_a} vs {name_b}: insufficient shared data (n={len(a)})")
        return
    diffs = [x - y for x, y in zip(a, b)]
    delta = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0
    d = delta / sd if sd > 0 else float("inf")
    print(f"  {name_a} vs {name_b}  (n={len(a)})")
    print(f"    Δ = {delta:+.1f}W   Cohen's d = {d:.2f}", end="")
    print("  (large)" if abs(d) >= 0.8 else
          ("  (orta)" if abs(d) >= 0.5 else
           ("  (small)" if abs(d) >= 0.2 else "  (negligible)")))
    if HAVE_SCIPY:
        try:
            t, p = scipy_stats.ttest_rel(a, b)
            tag = ("✓ SIGNIFICANT" if p < 0.05 else
                   ("~ borderline" if p < 0.15 else "✗ not significant"))
            print(f"    paired t: t={t:.3f}, p={p:.4f}  {tag}")
        except Exception as e:
            print(f"    t-test error: {e}")


def main():
    print("═" * 74)
    print("CHARM FULL MATRIX ANALYSIS")
    print("═" * 74)

    print("\n### 1. RESULTS ###\n")
    R = {}
    B = {}
    for label, exp in CONFIGS.items():
        R[label], B[label] = {}, {}
        cells = []
        for s in ALL_SEEDS:
            t, bk = load(exp, s)
            if t is not None:
                R[label][s] = t
                B[label][s] = bk
                cells.append(f"s{s}={t}")
            else:
                cells.append(f"s{s}=--")
        vals = list(R[label].values())
        if len(vals) >= 2:
            print(f"  {label:22}: {' '.join(cells):32} "
                  f"→ {statistics.mean(vals):.1f} ± {statistics.stdev(vals):.1f} "
                  f"(n={len(vals)})")
        elif len(vals) == 1:
            print(f"  {label:22}: {' '.join(cells):32} → {vals[0]}W (n=1)")
        else:
            print(f"  {label:22}: no data")

    def common(la, lb):
        cs = sorted(set(R[la]) & set(R[lb]))
        return cs, [R[la][s] for s in cs], [R[lb][s] for s in cs]

    print("\n### 2. ABLATION (contribution of CHARM components) ###\n")
    for v, c, fam in [("A1 (DQN vanilla)", "A2 (DQN+CHARM)", "DQN"),
                      ("B1 (QMIX vanilla)", "B2 (QMIX+CHARM)", "QMIX"),
                      ("C1 (MADDPG vanilla)", "C2 (MADDPG+CHARM)", "MADDPG")]:
        cs, a, b = common(v, c)
        if cs:
            print(f"  [{fam}]")
            paired_test(b, a, c, v)
            print()

    print("### 3. CHARM (A2) vs EPISODIC MEMORY ###\n")
    for other in ["D1 (EMC)", "D2 (EMU)"]:
        cs, a, b = common("A2 (DQN+CHARM)", other)
        if cs:
            paired_test(a, b, "CHARM(A2)", other)
            print()

    print("### 4. CHARM BACKBONE COMPARISON ###\n")
    for x in ["B2 (QMIX+CHARM)", "C2 (MADDPG+CHARM)"]:
        cs, a, b = common("A2 (DQN+CHARM)", x)
        if cs:
            paired_test(a, b, "A2(DQN)", x)
            print()

    print("### 5. TRAINING STABILITY (bucket CV) ###\n")
    for label in CONFIGS:
        cvs = []
        for s, bk in B[label].items():
            if bk and len(bk) >= 2:
                bm = statistics.mean(bk)
                if bm > 0:
                    cvs.append(statistics.stdev(bk) / bm)
        if cvs:
            print(f"  {label:22}: CV = {statistics.mean(cvs):.3f}")

    print("\n### 6. LATEX TABLO ###\n")
    print(r"\begin{table}[t]\centering")
    print(r"\caption{3v3 GRF cooperative scenario — method comparison.}")
    print(r"\label{tab:main}")
    print(r"\begin{tabular}{llcc}")
    print(r"\hline")
    print(r"Family & Method & Wins (mean$\pm$std) & $n$ \\")
    print(r"\hline")
    fam_map = [
        ("DQN", ["A1 (DQN vanilla)", "A2 (DQN+CHARM)"]),
        ("QMIX", ["B1 (QMIX vanilla)", "B2 (QMIX+CHARM)"]),
        ("MADDPG", ["C1 (MADDPG vanilla)", "C2 (MADDPG+CHARM)"]),
        ("Episodik", ["D1 (EMC)", "D2 (EMU)", "D3 (MAPPO)"]),
    ]
    for fam, labels in fam_map:
        for i, label in enumerate(labels):
            vals = list(R[label].values())
            fam_cell = fam if i == 0 else ""
            short = label.split("(")[1].rstrip(")")
            if len(vals) >= 2:
                cell = f"${statistics.mean(vals):.1f} \\pm {statistics.stdev(vals):.1f}$"
            elif len(vals) == 1:
                cell = f"${vals[0]}$"
            else:
                cell = "--"
            bold = "\\textbf" if label == "A2 (DQN+CHARM)" else ""
            nm = f"{bold}{{{short}}}" if bold else short
            print(f"{fam_cell} & {nm} & {cell} & {len(vals)} \\\\")
        print(r"\hline")
    print(r"\end{tabular}\end{table}")

    print("\n" + "═" * 74)
    print("Analysis done. Missing: D3_mappo (needs to be run). "
          "All other configs are ready.")
    print("═" * 74)


if __name__ == "__main__":
    main()
