import json
from pathlib import Path
import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR / "experiments"
SEEDS = [0, 1, 7, 42, 123]
EPISODES = 3000
BUCKET = 1000

CONFIGS = [
    ("A2",  "A2_dqn_charm",          "Tam CHARM (referans)",  False),
    ("A2a", "A2a_dqn_charm_noCER",   "no CER",                True),
    ("A2b", "A2b_dqn_charm_noSMAL",  "no SMAL",               True),
    ("A2c", "A2c_dqn_charm_noHCR",   "no HCR",                True),
    ("A2d", "A2d_dqn_charm_noJACM",  "no JACM",               True),
]


def load(exp_name, seed):
    path = (EXP_ROOT / f"seed_{seed}" / exp_name /
            f"seed_{seed}__{exp_name}__rewards_and_winrate.jsonl")
    if not path.exists():
        return None
    try:
        recs = [json.loads(l) for l in open(path)]
        m1 = [i for i, r in enumerate(recs) if r.get("match") == 1]
        if not m1:
            return None
        latest = recs[m1[-1]:]
        if len(latest) < EPISODES * 0.9:
            return None
        return sum(1 for r in latest if r.get("win"))
    except Exception:
        return None


def cohens_d_paired(a, b):
    diff = a - b
    sd = diff.std(ddof=1)
    return 0.0 if sd == 0 else diff.mean() / sd


def fmt_sig(p):
    return "✓ SIGNIFICANT" if p < 0.05 else ("~ borderline" if p < 0.10 else "✗ not significant")


def main():
    print("═" * 78)
    print("CHARM FULL COMPONENT ABLATION — A2 vs A2a/b/c/d")
    print(f"  n = {len(SEEDS)} seeds, {EPISODES} episodes, paired")
    print("═" * 78)

    data = {}
    missing = []
    for code, exp, label, _ in CONFIGS:
        ws = []
        for s in SEEDS:
            w = load(exp, s)
            if w is None:
                missing.append((code, s, exp))
                ws.append(None)
            else:
                ws.append(w)
        data[code] = ws

    if missing:
        print("\n⚠️  Missing:")
        for code, s, exp in missing:
            print(f"   • seed_{s}/{exp}")

    print("\n### 1. RESULTS ###")
    for code, exp, label, _ in CONFIGS:
        ws = data[code]
        ws_clean = [w for w in ws if w is not None]
        if not ws_clean:
            print(f"  {code} ({label}): NO DATA")
            continue
        ws_arr = np.array(ws_clean)
        seed_strs = [f"s{s}={w}" if w is not None else f"s{s}=?" for s,w in zip(SEEDS, ws)]
        print(f"  {code:<4} ({label:<23}): " + " ".join(seed_strs)
              + f" → {ws_arr.mean():.1f} ± {ws_arr.std(ddof=1):.1f} (n={len(ws_arr)})")

    print("\n### 2. PAIRED COMPARISONS (each ablation vs A2) ###")
    baseline = data["A2"]
    if any(w is None for w in baseline):
        print("  ❌ A2 reference data is incomplete.")
        return
    baseline_arr = np.array(baseline)

    summary_rows = [("A2 (Tam CHARM)", baseline_arr, None, None, None, None)]
    component_effects = {}

    for code, exp, label, is_abl in CONFIGS:
        if not is_abl:
            continue
        if any(w is None for w in data[code]):
            print(f"\n  {code} ({label}): MISSING seeds, skipped.")
            continue
        abl_arr = np.array(data[code])
        delta = (abl_arr - baseline_arr).mean()
        t, p = stats.ttest_rel(abl_arr, baseline_arr)
        d = cohens_d_paired(abl_arr, baseline_arr)
        sigma_diff = abl_arr.std(ddof=1) - baseline_arr.std(ddof=1)

        print(f"\n  {code} ({label}) vs A2:")
        print(f"    Δ mean = {delta:+.1f}W   d = {d:+.2f}   p = {p:.4f}   {fmt_sig(p)}")
        print(f"    σ change = {sigma_diff:+.2f}  (A2: {baseline_arr.std(ddof=1):.1f}, {code}: {abl_arr.std(ddof=1):.1f})")

        comp_name = {"A2a":"CER", "A2b":"SMAL", "A2c":"HCR", "A2d":"JACM"}[code]
        component_effects[comp_name] = {
            "mean_drop": baseline_arr.mean() - abl_arr.mean(),
            "sigma_increase": sigma_diff,
            "p": p
        }
        summary_rows.append((f"{code} ({label})", abl_arr, delta, d, p, sigma_diff))

    print("\n### 3. COMPONENT CONTRIBUTIONS (sorted) ###")
    print("    'Mean drop': how much the mean dropped when removing the component (+ = component helped)")
    print("    'σ increase': how much σ rose when removing the component (+ = component aided stability)")
    print()
    sorted_by_sigma = sorted(component_effects.items(), key=lambda x: -x[1]["sigma_increase"])
    print("  σ-reducing contribution (largest to smallest):")
    for name, eff in sorted_by_sigma:
        print(f"    {name:<6} σ rise: {eff['sigma_increase']:+.2f}  |  mean drop: {eff['mean_drop']:+.1f}W")

    print("\n### 4. LATEX TABLO 8 ###")
    print(r"\begin{table}[t]\centering")
    print(r"\caption{CHARM component ablation: A2 (Full CHARM) reference; paired $t$-test, $n=5$, 3000 ep.}")
    print(r"\label{tab:ablation}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\hline")
    print(r"Config & Wins (mean$\pm\sigma$) & $\Delta$ vs A2 & Cohen $d$ & $p$ & Sig.? \\")
    print(r"\hline")
    for label, arr, delta, d, p, _ in summary_rows:
        if delta is None:
            print(rf"\textbf{{{label}}} & ${arr.mean():.1f} \pm {arr.std(ddof=1):.1f}$ & --- & --- & --- & (ref) \\")
        else:
            sig = r"\textbf{Yes}" if p < 0.05 else "No"
            print(f"{label} & ${arr.mean():.1f} \\pm {arr.std(ddof=1):.1f}$ & "
                  f"${delta:+.1f}$ & ${d:.2f}$ & ${p:.3f}$ & {sig} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n" + "═" * 78)


if __name__ == "__main__":
    main()
