"""RQ2 visualization: Fed-MoE + selective LDP under varying epsilon.

For each epsilon in {0.5, 1, 2, 4, 8} we load the personalized federated MoE
predictions trained with Randomized-Response-based LDP:
    - Binary RR on each 0/1 permission decision (history statistics)
    - Categorical RR on user survey features (ordinal demographics)
    - Semantic context is never perturbed
A no-LDP federated MoE curve is drawn as a dashed reference upper bound.

We sweep a confidence threshold tau in [0.50, 0.95] and report, exactly like
the RQ1 plots:
    - Accuracy on samples with confidence >= tau
    - Coverage (fraction of samples that pass the threshold)

Outputs:
    ../results/rq2_ldp_accuracy_vs_threshold.png
    ../results/rq2_ldp_coverage_vs_threshold.png
    ../results/rq2_ldp_threshold_curves.json
"""

import json
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required for rq2_ldp_threshold_curves.py.\n"
        "Install with: pip install matplotlib"
    ) from exc


RESULTS_DIR = "../results"

ACCURACY_FIG = os.path.join(RESULTS_DIR, "rq2_ldp_accuracy_vs_threshold.png")
COVERAGE_FIG = os.path.join(RESULTS_DIR, "rq2_ldp_coverage_vs_threshold.png")
CURVES_JSON = os.path.join(RESULTS_DIR, "rq2_ldp_threshold_curves.json")

THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)

LDP_MODE = "user_stats"


def eps_tag(eps):
    return str(int(eps)) if float(eps).is_integer() else str(eps).replace(".", "p")


# Reference (no LDP) drawn as a dashed upper bound, then the 5 epsilon curves.
METHODS = [
    {
        "name": "Fed-MoE (no LDP)",
        "path": os.path.join(RESULTS_DIR, "federated_moe_predictions_rq2_rr_prior_no_ldp.json"),
        "color": "#7f7f7f",
        "marker": "o",
        "linestyle": "--",
    },
]

_EPS_STYLE = [
    (0.5, "#d62728", "D"),
    (1.0, "#ff7f0e", "s"),
    (2.0, "#2ca02c", "^"),
    (4.0, "#1f77b4", "v"),
    (8.0, "#9467bd", "P"),
]
for _eps, _color, _marker in _EPS_STYLE:
    METHODS.append({
        "name": f"RR eps={_eps:g}",
        "path": os.path.join(
            RESULTS_DIR,
            f"federated_moe_predictions_rq2_rr_prior_eps{eps_tag(_eps)}.json",
        ),
        "color": _color,
        "marker": _marker,
        "linestyle": "-",
    })


def load_simple_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    y_true = np.array([p["y_true"] for p in preds], dtype=int)
    p_allow = np.array([p["p_allow"] for p in preds], dtype=float)
    pred = (p_allow >= 0.5).astype(int)
    confidence = np.maximum(p_allow, 1.0 - p_allow)
    return y_true, pred, confidence


def threshold_curve(y_true, y_pred, confidence, thresholds):
    accs, covs = [], []
    for tau in thresholds:
        mask = confidence >= tau
        cov = float(mask.mean()) if len(mask) > 0 else 0.0
        if mask.sum() > 0:
            acc = float((y_pred[mask] == y_true[mask]).mean())
        else:
            acc = float("nan")
        accs.append(acc)
        covs.append(cov)
    return accs, covs


def plot_metric(metric_key, ylabel, title, ylim, legend_loc, out_path, curves):
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in METHODS:
        if method["name"] not in curves:
            continue
        c = curves[method["name"]]
        ax.plot(
            c["thresholds"],
            c[metric_key],
            marker=method["marker"],
            linestyle=method["linestyle"],
            color=method["color"],
            label=method["name"],
            linewidth=1.8,
            markersize=6,
        )
    ax.set_xlabel("Confidence threshold $\\tau$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.set_xlim(0.48, 0.97)
    ax.grid(alpha=0.3)
    ax.legend(loc=legend_loc, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


def main():
    print("=" * 60)
    print("RQ2 Visualization: Fed-MoE + selective LDP, threshold curves")
    print("=" * 60)

    curves = {}
    for method in METHODS:
        if not os.path.exists(method["path"]):
            print(f"  [skip]    {method['name']:<18}  missing: {method['path']}")
            continue
        y_true, y_pred, conf = load_simple_predictions(method["path"])
        if len(y_true) == 0:
            print(f"  [empty]   {method['name']:<18}  no predictions")
            continue
        accs, covs = threshold_curve(y_true, y_pred, conf, THRESHOLDS)
        curves[method["name"]] = {
            "thresholds": [float(t) for t in THRESHOLDS],
            "accuracy": accs,
            "coverage": covs,
            "n_samples": int(len(y_true)),
            "overall_accuracy": float((y_pred == y_true).mean()),
        }
        print(
            f"  [loaded]  {method['name']:<18}  n={len(y_true):>4d}  "
            f"acc(@0.5)={accs[0]:.3f}  acc(@0.8)={accs[6]:.3f}  "
            f"cov(@0.8)={covs[6]:.3f}"
        )

    if not curves:
        raise RuntimeError(
            "No prediction files found. Run the LDP sweep first, e.g.:\n"
            "  FED_MOE_LDP_MODE=user_stats FED_MOE_LDP_EPSILON=2 "
            "FED_MOE_OUTPUT_TAG=ldp_user_stats_eps2 python permission_federated_moe.py"
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CURVES_JSON, "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)
    print(f"\nSaved curves -> {CURVES_JSON}")

    plot_metric(
        "accuracy",
        "Accuracy on covered samples",
        "RQ2: Accuracy vs. Confidence threshold (Fed-MoE + LDP)",
        (0.5, 1.02),
        "lower right",
        ACCURACY_FIG,
        curves,
    )
    plot_metric(
        "coverage",
        "High-confidence coverage",
        "RQ2: High-confidence coverage vs. Confidence threshold (Fed-MoE + LDP)",
        (0.0, 1.02),
        "lower left",
        COVERAGE_FIG,
        curves,
    )


if __name__ == "__main__":
    main()
