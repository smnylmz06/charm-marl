import json
import argparse
from pathlib import Path

import numpy as np

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("WARNING: scipy not found -> t-test/F/Levene skipped. "
          "pip install scipy --break-system-packages\n")

SEEDS = [0, 1, 7, 42, 123]
EPISODES = 3000

CONFIG_MAP = {
    "DQN Vanilla":    "A1_dqn_vanilla",
    "DQN+CHARM":      "A2_dqn_charm",
    "QMIX Vanilla":   "B1_qmix_vanilla",
    "QMIX+CHARM":     "B2_qmix_charm",
    "MADDPG Vanilla": "C1_maddpg_vanilla",
    "MADDPG+CHARM":   "C2_maddpg_charm",
    "QPLEX Vanilla":  "E1_qplex_vanilla",
    "QPLEX+CHARM":    "E2_qplex_charm",
    "EMC":            "D1_emc",
    "EMU":            "D2_emu",
    "MAPPO":          "D3_mappo",
}

ABLATION_MAP = {
    "A2 Full CHARM": "A2_dqn_charm",
    "A2a no-CER":    "A2a_dqn_charm_no_cer",
    "A2b no-SMAL":   "A2b_dqn_charm_no_smal",
    "A2c no-HCR":    "A2c_dqn_charm_no_hcr",
    "A2d no-JACM":   "A2d_dqn_charm_no_jacm",
}

BACKBONES = [
    ("DQN",    "DQN Vanilla",    "DQN+CHARM"),
    ("QMIX",   "QMIX Vanilla",   "QMIX+CHARM"),
    ("MADDPG", "MADDPG Vanilla", "MADDPG+CHARM"),
    ("QPLEX",  "QPLEX Vanilla",  "QPLEX+CHARM"),
]

REPORTED_RAW = {
    "A2_dqn_charm":              [55, 52, 58, 61, 63],
    "A2a_dqn_charm_no_cer":      [65, 56, 54, 54, 56],
    "A2b_dqn_charm_no_smal":     [58, 57, 60, 67, 59],
    "A2c_dqn_charm_no_hcr":      [51, 48, 51, 60, 55],
    "A2d_dqn_charm_no_jacm":     [58, 57, 60, 67, 59],
}

REPORTED_SUMMARY = {
    "DQN Vanilla":    {"mean": 55.6, "std": 10.9, "min": 47, "max": 74},
    "DQN+CHARM":      {"mean": 57.8, "std":  4.4, "min": 52, "max": 63},
    "QMIX Vanilla":   {"mean": 46.4, "std": 12.1, "min": 35, "max": 64},
    "QMIX+CHARM":     {"mean": 49.6, "std":  7.6, "min": 42, "max": 58},
    "MADDPG Vanilla": {"mean": 42.8, "std": 24.8, "min":  6, "max": 68},
    "MADDPG+CHARM":   {"mean": 44.8, "std":  7.3, "min": 39, "max": 57},
    "QPLEX Vanilla":  {"mean": 54.2, "std": 15.0, "min": 43, "max": 80},
    "QPLEX+CHARM":    {"mean": 54.0, "std": 12.5, "min": 39, "max": 70},
    "EMC":            {"mean": 63.6, "std":  5.6, "min": 57, "max": 70},
    "EMU":            {"mean": 47.8, "std":  6.7, "min": 40, "max": 56},
    "MAPPO":          {"mean": 53.6, "std": 22.9, "min": 28, "max": 76},
}

RNG = np.random.default_rng(12345)
P_REGISTRY = []


def _read_jsonl_wins(path, episodes=EPISODES, window=None):
    try:
        recs = [json.loads(l) for l in open(path) if l.strip()]
    except Exception:
        return None, None
    if not recs:
        return None, None
    m1 = [i for i, r in enumerate(recs) if r.get("match") == 1]
    latest = recs[m1[-1]:] if m1 else recs
    if not m1:
        latest = [r for r in recs if 1 <= r.get("match", 0) <= episodes]
    if len(latest) < episodes * 0.9:
        return None, None
    total = sum(1 for r in latest if r.get("win"))
    buckets = None
    if window:
        nb = len(latest) // window
        buckets = [sum(1 for r in latest[i*window:(i+1)*window] if r.get("win"))
                   for i in range(nb)]
    return total, buckets


def find_jsonl(exp_dir, seed, config):
    base = Path(exp_dir) / f"seed_{seed}" / config
    if not base.exists():
        return None
    p = base / f"seed_{seed}__{config}__rewards_and_winrate.jsonl"
    if p.exists():
        return p
    cands = list(base.glob("*rewards_and_winrate*.jsonl")) + list(base.glob("*rewards*.jsonl"))
    return cands[0] if cands else None


def load_config(exp_dir, config, window=500):
    totals, buckets = {}, {}
    got_raw = False
    for s in SEEDS:
        path = find_jsonl(exp_dir, s, config)
        if path:
            t, bk = _read_jsonl_wins(path, window=window)
            if t is not None:
                totals[s] = t
                if bk:
                    buckets[s] = bk
                got_raw = True
    if got_raw:
        return totals, buckets, "raw"
    if config in REPORTED_RAW:
        for s, v in zip(SEEDS, REPORTED_RAW[config]):
            totals[s] = v
        return totals, {}, "reported_raw"
    return {}, {}, "none"


def arr(d, seeds=None):
    seeds = seeds or sorted(d)
    return np.array([d[s] for s in seeds], dtype=float)


def std1(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1)) if len(x) > 1 else 0.0


def boot_ci(stat_fn, *samples, B=10000, lo=2.5, hi=97.5):
    n = len(samples[0])
    vals = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        rs = [s[idx] for s in samples]
        v = stat_fn(*rs)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, lo)), float(np.percentile(vals, hi)))


def hedges_g_paired(charm, van):
    diff = np.asarray(charm, float) - np.asarray(van, float)
    n = len(diff)
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0 if diff.mean() == 0 else float("inf")
    dz = diff.mean() / sd
    J = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
    return float(J * dz)


def prob_improvement(x, y):
    x = np.asarray(x, float)[:, None]
    y = np.asarray(y, float)[None, :]
    return float((np.sum(x > y) + 0.5 * np.sum(x == y)) / (x.shape[0] * y.shape[1]))


def paired_block(charm, van):
    charm = np.asarray(charm, float); van = np.asarray(van, float)
    delta = float(charm.mean() - van.mean())
    g = hedges_g_paired(charm, van)
    g_ci = boot_ci(lambda a, b: hedges_g_paired(a, b), charm, van)
    d_ci = boot_ci(lambda a, b: float(a.mean() - b.mean()), charm, van)
    poi = prob_improvement(charm, van)
    p = float("nan"); t = float("nan")
    if HAVE_SCIPY and len(charm) >= 2:
        if np.allclose(charm, van):
            t, p = 0.0, 1.0
        else:
            t, p = sps.ttest_rel(charm, van)
            t, p = float(t), float(p)
    return dict(delta=delta, g=g, g_ci=g_ci, d_ci=d_ci, poi=poi, t=t, p=p, n=len(charm))


def variance_block(van, charm):
    van = np.asarray(van, float); charm = np.asarray(charm, float)
    sv, sc = std1(van), std1(charm)
    ratio = sv / sc if sc > 0 else float("inf")

    def ratio_fn(a, b):
        sb = b.std(ddof=1)
        return a.std(ddof=1) / sb if sb > 0 else None
    ci = boot_ci(ratio_fn, van, charm)

    F = (sv**2 / sc**2) if sc > 0 else float("inf")
    pF = float("nan"); pLev = float("nan"); pBF = float("nan")
    if HAVE_SCIPY:
        nv, nc = len(van), len(charm)
        if sc > 0 and nv > 1 and nc > 1:
            pF = float(sps.f.sf(F, nv - 1, nc - 1))
        try:
            pLev = float(sps.levene(van, charm, center="mean").pvalue)
            pBF = float(sps.levene(van, charm, center="median").pvalue)
        except Exception:
            pass
    return dict(sv=sv, sc=sc, ratio=ratio, ratio_ci=ci, F=F, pF=pF, pLev=pLev, pBF=pBF)


def bh_adjust(pairs):
    valid = [(lbl, p) for lbl, p in pairs if p is not None and np.isfinite(p)]
    m = len(valid)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: valid[i][1])
    adj = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        p = valid[i][1]
        val = min(prev, p * m / k)
        adj[i] = val
        prev = val
    return [(valid[i][0], valid[i][1], adj[i], adj[i] < 0.05) for i in range(m)]


def fmt_ci(ci):
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-dir", default="experiments")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out-json", default="results_all.json")
    ap.add_argument("--out-md", default="paper_tables.md")
    ap.add_argument("--out-tex", default="paper_tables.tex")
    args = ap.parse_args()

    bar = "=" * 78
    print(bar); print(" CHARM — FULL ANALYSIS DUMP"); print(bar)

    print("\n### 0. CONFIG LOADING ###\n")
    data, bdata, src = {}, {}, {}
    for label, cfg in CONFIG_MAP.items():
        t, bk, s = load_config(args.experiments_dir, cfg)
        data[label], bdata[label], src[label] = t, bk, s
        vals = arr(t) if t else np.array([])
        if len(vals) >= 2:
            tag = {"raw": "[raw]", "reported_raw": "[rep-raw]"}.get(s, "[?]")
            print(f"  {label:16s}: {[int(v) for v in vals]}  "
                  f"mean={vals.mean():5.1f}  σ={std1(vals):5.2f}  n={len(vals)} {tag}")
        else:
            r = REPORTED_SUMMARY.get(label)
            if r:
                print(f"  {label:16s}: mean={r['mean']:5.1f}  σ={r['std']:5.2f}  "
                      f"[reported-summary, no raw -> advanced statistics skipped]")
            else:
                print(f"  {label:16s}: no data")

    def have_raw(label):
        return src.get(label) in ("raw", "reported_raw") and len(data.get(label, {})) >= 2

    def common(la, lb):
        cs = sorted(set(data.get(la, {})) & set(data.get(lb, {})))
        return cs, arr(data[la], cs), arr(data[lb], cs)

    results = {"summary": {}, "vanilla_vs_charm": {}, "variance": {},
               "vs_episodic": {}, "backbone": {}, "ablation": {},
               "n_effect": {}, "cumulation": {}, "bh": []}

    for label in CONFIG_MAP:
        if have_raw(label):
            v = arr(data[label])
            results["summary"][label] = dict(
                mean=float(v.mean()), std=std1(v), n=int(len(v)),
                raw=[int(x) for x in v], source=src[label])

    print("\n### 1. VANILLA vs +CHARM  (Δ, Hedges g [95% CI], paired-t p, P(improve)) ###\n")
    for fam, van, charm in BACKBONES:
        if not (have_raw(van) and have_raw(charm)):
            print(f"  [{fam}] raw missing -> skipped")
            continue
        cs, a_charm, a_van = arr(data[charm], sorted(set(data[charm]) & set(data[van]))), None, None
        cs = sorted(set(data[charm]) & set(data[van]))
        a_charm = arr(data[charm], cs); a_van = arr(data[van], cs)
        blk = paired_block(a_charm, a_van)
        P_REGISTRY.append((f"{fam}: van vs CHARM", blk["p"]))
        results["vanilla_vs_charm"][fam] = blk
        print(f"  [{fam}]  Δ={blk['delta']:+.1f}  g={blk['g']:+.2f} CI{fmt_ci(blk['g_ci'])}  "
              f"p={blk['p']:.3f}  POI={blk['poi']:.2f}  (n={blk['n']})")

    print("\n### 2. VARIANCE REDUCTION  (ratio [95% CI], F one-sided, Levene, Brown-Forsythe) ###\n")
    for fam, van, charm in BACKBONES:
        if not (have_raw(van) and have_raw(charm)):
            print(f"  [{fam}] raw missing -> skipped")
            continue
        cs = sorted(set(data[charm]) & set(data[van]))
        a_van = arr(data[van], cs); a_charm = arr(data[charm], cs)
        vb = variance_block(a_van, a_charm)
        results["variance"][fam] = vb
        print(f"  [{fam}]  σ {vb['sv']:.1f}->{vb['sc']:.1f}  ratio={vb['ratio']:.2f}x "
              f"CI{fmt_ci(vb['ratio_ci'])}  F p={vb['pF']:.3f}  "
              f"Levene p={vb['pLev']:.3f}  BF p={vb['pBF']:.3f}")

    print("\n### 3. CHARM vs EPISODIC MEMORY  (matched=QPLEX first) ###\n")
    vs_pairs = [
        ("QPLEX+CHARM", "EMU"), ("QPLEX+CHARM", "EMC"),
        ("DQN+CHARM", "EMU"),   ("DQN+CHARM", "EMC"),
        ("QMIX+CHARM", "EMU"),  ("QMIX+CHARM", "EMC"),
    ]
    for ch, base in vs_pairs:
        if not (have_raw(ch) and have_raw(base)):
            print(f"  {ch:12s} vs {base:4s}: raw missing -> skipped")
            continue
        cs = sorted(set(data[ch]) & set(data[base]))
        a, b = arr(data[ch], cs), arr(data[base], cs)
        blk = paired_block(a, b)
        P_REGISTRY.append((f"{ch} vs {base}", blk["p"]))
        results["vs_episodic"][f"{ch} vs {base}"] = blk
        flag = "  <-- matched" if ch == "QPLEX+CHARM" else ("  (confounded)" if ch == "DQN+CHARM" else "")
        print(f"  {ch:12s} vs {base:4s}: Δ={blk['delta']:+.1f}  g={blk['g']:+.2f} "
              f"CI{fmt_ci(blk['g_ci'])}  p={blk['p']:.3f}  POI={blk['poi']:.2f}{flag}")

    print("\n### 4. BACKBONE COMPARISON (CHARM, DQN reference) ###\n")
    ref = "DQN+CHARM"
    if have_raw(ref):
        for other in ["QMIX+CHARM", "MADDPG+CHARM", "QPLEX+CHARM"]:
            if not have_raw(other):
                print(f"  DQN vs {other}: raw missing -> skipped"); continue
            cs = sorted(set(data[ref]) & set(data[other]))
            a, b = arr(data[ref], cs), arr(data[other], cs)
            blk = paired_block(a, b)
            P_REGISTRY.append((f"DQN vs {other}", blk["p"]))
            results["backbone"][other] = blk
            print(f"  DQN vs {other:12s}: Δ={blk['delta']:+.1f}  g={blk['g']:+.2f} "
                  f"CI{fmt_ci(blk['g_ci'])}  p={blk['p']:.3f}")

    print("\n### 5. COMPONENT ABLATION (DQN, vs A2 full) ###\n")
    abl = {}
    for label, cfg in ABLATION_MAP.items():
        t, _, s = load_config(args.experiments_dir, cfg)
        if len(t) >= 2:
            abl[label] = arr(t, sorted(t))
            print(f"  {label:16s}: {[int(x) for x in abl[label]]}  "
                  f"mean={abl[label].mean():.1f}  σ={std1(abl[label]):.2f}  [{s}]")
    full = abl.get("A2 Full CHARM")
    if full is not None:
        for label, vals in abl.items():
            if label == "A2 Full CHARM":
                continue
            blk = paired_block(vals, full)
            P_REGISTRY.append((f"ablation {label}", blk["p"]))
            results["ablation"][label] = blk
            print(f"    {label:16s}: Δ={blk['delta']:+.1f}  g={blk['g']:+.2f}  p={blk['p']:.3f}")
        a2b, a2d = abl.get("A2b no-SMAL"), abl.get("A2d no-JACM")
        if a2b is not None and a2d is not None:
            ident = np.array_equal(a2b, a2d)
            print(f"\n    A2b==A2d byte-identical? {ident}  "
                  f"(replay-not-recall: σ does not change when SMAL/JACM is removed)")
            results["ablation"]["A2b_equals_A2d"] = bool(ident)
        print("\n    σ across ablations:",
              {k: round(std1(v), 2) for k, v in abl.items()})

    print("\n### 6. n=3 vs n=5 DEMONSTRATION (DQN vanilla vs +CHARM) ###\n")
    if have_raw("DQN Vanilla") and have_raw("DQN+CHARM"):
        for tag, sub in [("n=3 (seeds 42,7,123)", [42, 7, 123]),
                         ("n=5 (all seeds)", SEEDS)]:
            cs = [s for s in sub if s in data["DQN+CHARM"] and s in data["DQN Vanilla"]]
            if len(cs) < 2:
                print(f"  {tag}: insufficient seeds"); continue
            a = arr(data["DQN+CHARM"], cs); b = arr(data["DQN Vanilla"], cs)
            blk = paired_block(a, b)
            results["n_effect"][tag] = blk
            print(f"  {tag:24s}: Δ={blk['delta']:+.1f}  g={blk['g']:+.2f}  p={blk['p']:.3f}")
        print("  -> Henderson: if an effect that looks large/significant at n=3 disappears at n=5, "
              "the small-sample evaluation is unreliable.")
    else:
        print("  DQN vanilla/charm raw missing -> skipped")

    print("\n### 7. CUMULATION ARTIFACT CHECK (cumulative σ vs windowed σ) ###\n")
    print("  500-episode windows; if final-window σ << cumulative σ, the effect mostly "
          "comes from the cumulative metric (B1).\n")
    for fam, van, charm in BACKBONES:
        for label in (van, charm):
            bk = bdata.get(label, {})
            if len(bk) < 2:
                continue
            cs = sorted(bk)
            totals = arr(data[label], cs)
            cum_sigma = std1(totals)
            minlen = min(len(bk[s]) for s in cs)
            if minlen < 1:
                continue
            final = np.array([bk[s][minlen - 1] for s in cs], float)
            per_window_sigma = np.mean([std1([bk[s][w] for s in cs]) for w in range(minlen)])
            results["cumulation"][label] = dict(
                cumulative_sigma=cum_sigma, final_window_sigma=std1(final),
                mean_window_sigma=float(per_window_sigma))
            print(f"  {label:16s}: cumulative σ={cum_sigma:5.2f}  "
                  f"final-window σ={std1(final):5.2f}  mean-window σ={per_window_sigma:5.2f}")
    if not results["cumulation"]:
        print("  no bucket data (raw JSONL needed) -> skipped")

    print("\n### 8. BENJAMINI-HOCHBERG (all paired p-values) ###\n")
    bh = bh_adjust(P_REGISTRY)
    results["bh"] = [dict(label=l, p=p, p_adj=pa, sig=sig) for l, p, pa, sig in bh]
    for l, p, pa, sig in bh:
        print(f"  {l:28s}: p={p:.3f}  p_BH={pa:.3f}  {'SIG' if sig else 'ns'}")

    print("\n### 9. VARIANCE RANKING (low σ = stable) ###\n")
    rank = []
    for label in CONFIG_MAP:
        if have_raw(label):
            rank.append((label, std1(arr(data[label])), arr(data[label]).mean()))
        elif label in REPORTED_SUMMARY:
            r = REPORTED_SUMMARY[label]
            rank.append((label + " (rep)", r["std"], r["mean"]))
    for i, (m, s, mu) in enumerate(sorted(rank, key=lambda x: x[1]), 1):
        mark = "*" if m.startswith("DQN+CHARM") else " "
        print(f"  {i:2d}. {mark} {m:20s}  σ={s:5.2f}  mean={mu:5.1f}")

    write_outputs(args, results, data, src)
    print("\n" + bar)
    print(f" Written: {args.out_json}, {args.out_md}, {args.out_tex}")
    print(" Note: 'reported-summary' rows are summary only; real bootstrap/Hedges "
          "needs the JSONL of those configs.")
    print(bar)


def write_outputs(args, results, data, src):
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    lines = []
    lines.append("% --- tab:vanilla_charm (Hedges g + bootstrap CI) ---")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Backbone & $\Delta$ & Hedges $g$ & 95\% CI & $p$ & P(improve) \\")
    lines.append(r"\midrule")
    for fam in ["DQN", "QMIX", "MADDPG", "QPLEX"]:
        b = results["vanilla_vs_charm"].get(fam)
        if not b:
            continue
        lines.append(f"{fam} & ${b['delta']:+.1f}$ & ${b['g']:+.2f}$ & "
                     f"$[{b['g_ci'][0]:.2f},\\,{b['g_ci'][1]:.2f}]$ & "
                     f"${b['p']:.3f}$ & ${b['poi']:.2f}$ \\\\")
    lines.append(r"\bottomrule"); lines.append(r"\end{tabular}"); lines.append("")

    lines.append("% --- tab:variance_reduction (ratio CI + variance tests) ---")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Backbone & Vanilla $\sigma$ & +CHARM $\sigma$ & Ratio & 95\% CI & $F$-test $p$ & Levene $p$ \\")
    lines.append(r"\midrule")
    for fam in ["DQN", "QMIX", "MADDPG", "QPLEX"]:
        v = results["variance"].get(fam)
        if not v:
            continue
        lines.append(f"{fam} & ${v['sv']:.1f}$ & ${v['sc']:.1f}$ & ${v['ratio']:.1f}\\times$ & "
                     f"$[{v['ratio_ci'][0]:.2f},\\,{v['ratio_ci'][1]:.2f}]$ & "
                     f"${v['pF']:.3f}$ & ${v['pLev']:.3f}$ \\\\")
    lines.append(r"\bottomrule"); lines.append(r"\end{tabular}")
    with open(args.out_tex, "w") as f:
        f.write("\n".join(lines))

    md = ["# CHARM — recomputed stats\n", "## Vanilla vs +CHARM\n",
          "| Backbone | Δ | Hedges g | 95% CI | p | P(improve) |",
          "|---|---|---|---|---|---|"]
    for fam in ["DQN", "QMIX", "MADDPG", "QPLEX"]:
        b = results["vanilla_vs_charm"].get(fam)
        if b:
            md.append(f"| {fam} | {b['delta']:+.1f} | {b['g']:+.2f} | "
                      f"[{b['g_ci'][0]:.2f}, {b['g_ci'][1]:.2f}] | {b['p']:.3f} | {b['poi']:.2f} |")
    md.append("\n## Variance reduction\n")
    md.append("| Backbone | Vanilla σ | +CHARM σ | Ratio | 95% CI | F p | Levene p | BF p |")
    md.append("|---|---|---|---|---|---|---|---|")
    for fam in ["DQN", "QMIX", "MADDPG", "QPLEX"]:
        v = results["variance"].get(fam)
        if v:
            md.append(f"| {fam} | {v['sv']:.1f} | {v['sc']:.1f} | {v['ratio']:.1f}× | "
                      f"[{v['ratio_ci'][0]:.2f}, {v['ratio_ci'][1]:.2f}] | "
                      f"{v['pF']:.3f} | {v['pLev']:.3f} | {v['pBF']:.3f} |")
    with open(args.out_md, "w") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()
