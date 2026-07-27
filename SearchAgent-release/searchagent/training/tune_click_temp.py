#!/usr/bin/env python3
"""Offline click_temperature tuner for SearchAgent++.

Records click traces (pre-temperature raw ClickModel probs + candidate ranks)
during ONE replay pass, then offline-sweeps click_temperature to minimise the
click_position_jsd against each dataset's REAL click-position distribution.
Temperature only re-weights click POSITION (not count, not session trajectory),
so the sweep is exact and needs ZERO additional API calls.

Usage:
  # 1) record a trace (uses API once):
  python -m searchagent.training.tune_click_temp record --dataset trec2014
  # 2) sweep offline (no API):
  python -m searchagent.training.tune_click_temp sweep --dataset trec2014
"""
from __future__ import annotations
import argparse, json, os, glob
import numpy as np

# Real click-position distributions (rank1..rank10), from full-pool reference.
REAL_CLICKDIST = {
    "trec2012": [0.474, 0.265, 0.143, 0.058, 0.030, 0.015, 0.008, 0.004, 0.002, 0.001],
    "trec2013": [0.425, 0.237, 0.135, 0.085, 0.050, 0.030, 0.018, 0.010, 0.006, 0.004],
    "trec2014": [0.618, 0.231, 0.088, 0.036, 0.017, 0.007, 0.003, 0.001, 0.0, 0.0],
    "yandex":   [0.383, 0.166, 0.114, 0.080, 0.060, 0.050, 0.045, 0.040, 0.035, 0.027],
    "orcas":    [0.664, 0.218, 0.069, 0.025, 0.012, 0.006, 0.003, 0.002, 0.001, 0.0],
}
DATA_PATHS = {
    "trec2012": "data/search_logs/trec_session2012_fixed.xml",
    "trec2013": "data/search_logs/trec_session2013.xml",
    "trec2014": "data/search_logs/trec_session2014.xml",
    "yandex":   "data/yandex/train.gz",
    "orcas":    "/root/orcas.tsv.gz",
}


def _normalize(v):
    v = np.asarray(v, dtype=float)
    s = v.sum()
    return v / s if s > 0 else np.ones_like(v) / len(v)


def _jsd(p, q):
    p, q = _normalize(p), _normalize(q)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def record(args):
    """Run one replay pass with click-trace recording on, dump traces to disk."""
    import scripts.run_experiment as R
    from searchagent.agent.search_agent import SearchUserAgent
    from searchagent.config import SearchAgentConfig
    SearchUserAgent._CLICK_TRACE = []   # turn recording ON
    cfg = f"configs/{args.dataset}.yaml"
    argv = ["run_experiment.py", "--config", cfg, "--agents", "full_pp",
            "--mode", "replay", "--trec_data", DATA_PATHS[args.dataset],
            "--num_sessions", str(args.num_sessions), "--seeds", "42",
            "--calibrator", f"output/calib_{args.dataset}/calibrator.pt",
            "--output_dir", f"output/tunetrace_{args.dataset}"]
    import sys
    sys.argv = argv
    R.main()
    trace = SearchUserAgent._CLICK_TRACE
    out = f"output/clicktrace_{args.dataset}.json"
    json.dump(trace, open(out, "w"))
    print(f"recorded {len(trace)} click decisions -> {out}")


def sweep(args):
    """Offline sweep click_temperature over the recorded trace; report best JSD.

    Uses the SAMPLING estimator with N Monte-Carlo repeats (NOT the expectation
    estimator): real eval samples one click per decision, so we must match that
    semantics. Expectation-based JSD is systematically optimistic and picks the
    wrong temperature on small-sample datasets (verified on orcas: expectation
    says T=2.1, sampling says T=2.0). N repeats drive the standard error down so
    the U-shaped JSD(T) curve is cleanly resolved.
    """
    trace = json.load(open(f"output/clicktrace_{args.dataset}.json"))
    real = REAL_CLICKDIST[args.dataset]
    max_rank = len(real)
    n_mc = args.mc

    def jsd_at(temp):
        rng = np.random.RandomState(0)
        js = []
        for _ in range(n_mc):
            counts = np.zeros(max_rank)
            for step in trace:
                p = np.asarray(step["probs"], dtype=float)
                if temp != 1.0:
                    p = p ** (1.0 / temp)
                p = p / p.sum()
                rk = step["ranks"][rng.choice(len(p), p=p)]
                counts[min(rk - 1, max_rank - 1)] += 1
            js.append(_jsd(counts, real))
        return float(np.mean(js)), float(np.std(js) / np.sqrt(n_mc))

    grid = np.arange(args.lo, args.hi + 1e-9, args.step)
    results = [(round(float(t), 2),) + jsd_at(t) for t in grid]
    results.sort(key=lambda x: x[1])
    cur_m, cur_se = jsd_at(args.current)
    print(f"=== {args.dataset}: click_temperature sweep ({len(trace)} decisions, {n_mc} MC) ===")
    print(f"  current T={args.current}: JSD={cur_m:.4f} (SE={cur_se:.4f})")
    print("  best 5:")
    for t, m, se in results[:5]:
        print(f"    T={t:<5} JSD={m:.4f} (SE={se:.4f})")
    print(f"  >>> recommended T={results[0][0]} (JSD={results[0][1]:.4f})")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("record"); pr.add_argument("--dataset", required=True)
    pr.add_argument("--num_sessions", type=int, default=20); pr.set_defaults(func=record)
    ps = sub.add_parser("sweep"); ps.add_argument("--dataset", required=True)
    ps.add_argument("--lo", type=float, default=1.0); ps.add_argument("--hi", type=float, default=6.0)
    ps.add_argument("--step", type=float, default=0.1); ps.add_argument("--current", type=float, default=1.9)
    ps.add_argument("--mc", type=int, default=50, help="Monte-Carlo repeats (drives down sampling SE)")
    ps.set_defaults(func=sweep)
    args = p.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
