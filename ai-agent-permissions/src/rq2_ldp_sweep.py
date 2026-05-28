"""RQ2: Selective LDP sweep for utility-privacy tradeoff.

Runs federated MoE under two LDP noise positions:
    1) user_stats   - perturb user statistics/profile features
    2) gating_input - perturb gate input each forward

Epsilon candidates:
    {0.5, 1, 2, 4, 8}

Outputs:
    ../results/rq2_ldp_sweep_results.json
    ../results/rq2_ldp_accuracy_vs_epsilon.png
    ../results/rq2_ldp_coverage_vs_epsilon.png
    ../results/rq2_ldp_tradeoff_score_vs_epsilon.png
"""

import json
import os
import subprocess
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required for rq2_ldp_sweep.py.\n"
        "Install with: pip install matplotlib"
    ) from exc


RESULTS_DIR = "../results"
RUN_SCRIPT = "permission_federated_moe.py"

EPSILONS = [0.5, 1.0, 2.0, 4.0, 8.0]
MODES = ["user_stats", "gating_input"]

# Keep runtime manageable for sweeps; override with env if needed.
SWEEP_ROUNDS = int(os.environ.get("RQ2_FED_ROUNDS", "20"))

RESULT_JSON = os.path.join(RESULTS_DIR, "rq2_ldp_sweep_results.json")
ACC_FIG = os.path.join(RESULTS_DIR, "rq2_ldp_accuracy_vs_epsilon.png")
COV_FIG = os.path.join(RESULTS_DIR, "rq2_ldp_coverage_vs_epsilon.png")
SCORE_FIG = os.path.join(RESULTS_DIR, "rq2_ldp_tradeoff_score_vs_epsilon.png")


def fmt_eps_tag(eps):
    if float(eps).is_integer():
        return str(int(eps))
    return str(eps).replace(".", "p")


def run_one(mode, epsilon):
    tag = f"rq2_{mode}_eps{fmt_eps_tag(epsilon)}"
    metrics_path = os.path.join(RESULTS_DIR, f"federated_moe_metrics_{tag}.json")

    env = os.environ.copy()
    env["FED_MOE_LDP_MODE"] = mode
    env["FED_MOE_LDP_EPSILON"] = str(epsilon)
    env["FED_MOE_OUTPUT_TAG"] = tag
    env["FED_MOE_ROUNDS"] = str(SWEEP_ROUNDS)
    env["FED_MOE_USE_LLM_SEMANTIC"] = env.get("FED_MOE_USE_LLM_SEMANTIC", "0")

    print(f"\n[run] mode={mode:<12} epsilon={epsilon:<4} tag={tag}")
    subprocess.run([sys.executable, RUN_SCRIPT], check=True, env=env)

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Expected metrics not found: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return tag, metrics


def normalize(values):
    arr = np.array(values, dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.ones_like(arr) * 0.5
    return (arr - lo) / (hi - lo)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 68)
    print("RQ2 Sweep: Selective LDP utility-privacy tradeoff")
    print("=" * 68)
    print(f"Federated rounds per run: {SWEEP_ROUNDS}")
    print(f"Epsilons: {EPSILONS}")
    print(f"Modes: {MODES}")

    records = []
    for mode in MODES:
        for eps in EPSILONS:
            tag, metrics = run_one(mode, eps)
            records.append(
                {
                    "tag": tag,
                    "mode": mode,
                    "epsilon": float(eps),
                    "accuracy": float(metrics["accuracy"]),
                    "high_conf_accuracy": float(metrics["high_conf_accuracy"]),
                    "coverage": float(metrics["coverage"]),
                    "auc": float(metrics.get("auc", float("nan"))),
                    "n_predictions": int(metrics["n_predictions"]),
                }
            )

    # Utility combines overall + high-confidence behavior.
    util_raw = [
        0.5 * r["accuracy"] + 0.25 * r["high_conf_accuracy"] + 0.25 * r["coverage"]
        for r in records
    ]
    privacy_risk_raw = [1.0 / r["epsilon"] for r in records]  # smaller epsilon => higher risk proxy
    util_norm = normalize(util_raw)
    risk_norm = normalize(privacy_risk_raw)

    for i, r in enumerate(records):
        r["utility_score"] = float(util_raw[i])
        r["privacy_risk_proxy"] = float(privacy_risk_raw[i])
        r["tradeoff_score"] = float(util_norm[i] - 0.5 * risk_norm[i])

    # Best per mode + global best
    best_by_mode = {}
    for mode in MODES:
        mode_rows = [r for r in records if r["mode"] == mode]
        best_by_mode[mode] = max(mode_rows, key=lambda x: x["tradeoff_score"])
    best_overall = max(records, key=lambda x: x["tradeoff_score"])

    payload = {
        "epsilons": EPSILONS,
        "modes": MODES,
        "federated_rounds_per_run": SWEEP_ROUNDS,
        "records": records,
        "best_by_mode": best_by_mode,
        "best_overall": best_overall,
        "tradeoff_definition": {
            "utility_score": "0.5*accuracy + 0.25*high_conf_accuracy + 0.25*coverage",
            "privacy_risk_proxy": "1/epsilon",
            "tradeoff_score": "normalize(utility_score) - 0.5*normalize(privacy_risk_proxy)",
        },
    }
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved sweep results -> {RESULT_JSON}")

    # -------- Accuracy plot --------
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, color, marker in [("user_stats", "#1f77b4", "o"), ("gating_input", "#d62728", "D")]:
        rows = sorted([r for r in records if r["mode"] == mode], key=lambda x: x["epsilon"])
        ax.plot(
            [r["epsilon"] for r in rows],
            [r["accuracy"] for r in rows],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=6,
            label=mode,
        )
    ax.set_xlabel("LDP epsilon")
    ax.set_ylabel("Accuracy")
    ax.set_title("RQ2: Accuracy vs epsilon")
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(ACC_FIG, dpi=300)
    plt.close(fig)
    print(f"Saved accuracy figure -> {ACC_FIG}")

    # -------- Coverage plot --------
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, color, marker in [("user_stats", "#1f77b4", "o"), ("gating_input", "#d62728", "D")]:
        rows = sorted([r for r in records if r["mode"] == mode], key=lambda x: x["epsilon"])
        ax.plot(
            [r["epsilon"] for r in rows],
            [r["coverage"] for r in rows],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=6,
            label=mode,
        )
    ax.set_xlabel("LDP epsilon")
    ax.set_ylabel("Coverage @ tau=0.8")
    ax.set_title("RQ2: Coverage vs epsilon")
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(COV_FIG, dpi=300)
    plt.close(fig)
    print(f"Saved coverage figure -> {COV_FIG}")

    # -------- Tradeoff score plot --------
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, color, marker in [("user_stats", "#1f77b4", "o"), ("gating_input", "#d62728", "D")]:
        rows = sorted([r for r in records if r["mode"] == mode], key=lambda x: x["epsilon"])
        ax.plot(
            [r["epsilon"] for r in rows],
            [r["tradeoff_score"] for r in rows],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=6,
            label=mode,
        )
    ax.set_xlabel("LDP epsilon")
    ax.set_ylabel("Utility-privacy tradeoff score")
    ax.set_title("RQ2: Utility-privacy tradeoff vs epsilon")
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(SCORE_FIG, dpi=300)
    plt.close(fig)
    print(f"Saved tradeoff figure -> {SCORE_FIG}")

    print("\nBest by mode:")
    for mode in MODES:
        b = best_by_mode[mode]
        print(
            f"  {mode:<12} eps={b['epsilon']:<4} "
            f"acc={b['accuracy']:.3f} cov={b['coverage']:.3f} score={b['tradeoff_score']:.3f}"
        )
    print(
        f"Best overall: mode={best_overall['mode']} eps={best_overall['epsilon']} "
        f"score={best_overall['tradeoff_score']:.3f}"
    )


if __name__ == "__main__":
    main()
