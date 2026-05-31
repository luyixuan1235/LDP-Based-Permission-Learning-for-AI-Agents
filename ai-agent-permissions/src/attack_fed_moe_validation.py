"""Gating-leakage attack: centralized MoE vs personalized Federated MoE.

Goal
----
Test whether the gating-leakage / preference-inference attack that works on the
centralized MoE still works once we move to the personalized Federated MoE
(where each user owns a private gate that is never sent to the server).

Threat model (black-box, deployment-time)
------------------------------------------
The attacker repeatedly queries a target user's deployed agent and observes only
per-permission outputs: p_allow, and quantities derived from it
(confidence, pred, margin). The attacker NEVER sees gate weights. Gate weights /
dominant expert are used here only as evaluation ground truth.

Three attackers are compared:
    1. MoE  (reused attack)            -- run the MoE-style attack on MoE outputs
    2. Fed-MoE (reused attack)         -- run the SAME attack on Fed-MoE outputs
    3. Fed-MoE (tailored attack)       -- Fed-MoE-aware attacker that adds
                                          order-invariant per-user routing
                                          fingerprint features, exploiting that a
                                          personalized gate makes each user's
                                          output behaviour highly self-consistent.

Outputs (../results):
    attack_fed_moe_comparison.json
    attack_fed_moe_expert_asr_vs_queries.png
    attack_fed_moe_preference_asr_vs_queries.png
"""

import json
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = "../results"
MOE_PRED = os.path.join(RESULTS_DIR, "moe_rq1_predictions.json")
FED_PRED = os.path.join(RESULTS_DIR, "federated_moe_predictions.json")

OUT_JSON = os.path.join(RESULTS_DIR, "attack_fed_moe_comparison.json")
FIG_EXPERT = os.path.join(RESULTS_DIR, "attack_fed_moe_expert_asr_vs_queries.png")
FIG_PREF = os.path.join(RESULTS_DIR, "attack_fed_moe_preference_asr_vs_queries.png")

BUDGETS = [1, 2, 3, 5, 8, 10, 15, 20]
SEED = 42


def load_train_preference_labels():
    """3-class user preference label from real training allow-rate terciles."""
    from rq1_data_utils import build_records

    train_records, _test, _bio = build_records()
    by_user = defaultdict(list)
    for rec in train_records:
        by_user[rec["user_id"]].append(int(rec["label"]))
    rates = {u: float(np.mean(v)) for u, v in by_user.items() if v}
    values = np.array(list(rates.values()), dtype=float)
    q33, q67 = np.quantile(values, [0.33, 0.67])
    labels = {}
    for u, r in rates.items():
        if r <= q33:
            labels[u] = 0           # privacy-sensitive
        elif r >= q67:
            labels[u] = 2           # utility-driven
        else:
            labels[u] = 1           # balanced
    return labels, float(q33), float(q67)


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_user_table(preds):
    """Per-user rows (sorted), and per-user dominant expert (eval ground truth)."""
    rows_by_user = defaultdict(list)
    for r in preds:
        rows_by_user[r["participant_id"]].append(r)
    users = sorted(rows_by_user)
    n_experts = len(preds[0]["gate_weights"])
    y_expert = []
    for u in users:
        rows = rows_by_user[u]
        rows.sort(key=lambda r: (int(r["query_id"]), r["permission_key"]))
        avg_gate = np.array([r["gate_weights"] for r in rows], dtype=float).mean(axis=0)
        y_expert.append(int(np.argmax(avg_gate)))
    return users, rows_by_user, np.array(y_expert, dtype=int), n_experts


def make_features_basic(users, rows_by_user, m):
    """MoE-style attacker features: per-query [p, conf, pred, margin] for first m."""
    X = []
    for u in users:
        rows = rows_by_user[u][:m]
        feat = []
        for r in rows:
            p = float(r["p_allow"])
            conf = max(p, 1.0 - p)
            pred = 1.0 if p >= 0.5 else 0.0
            margin = abs(p - 0.5)
            feat.extend([p, conf, pred, margin])
        while len(feat) < 4 * m:
            feat.extend([0.5, 0.5, 0.0, 0.0])
        X.append(feat)
    return np.array(X, dtype=float)


def make_features_tailored(users, rows_by_user, m):
    """Fed-MoE-aware attacker: basic features + order-invariant routing fingerprint.

    A personalized gate makes a user's output behaviour highly self-consistent,
    so distributional summaries of the output trajectory (spread, quantiles,
    allow/high-confidence ratios, prediction entropy) act as a stable per-user
    routing fingerprint even though the gate itself is never observed.
    """
    X = []
    for u in users:
        rows = rows_by_user[u][:m]
        p = np.array([float(r["p_allow"]) for r in rows], dtype=float)
        if len(p) == 0:
            p = np.array([0.5])
        conf = np.maximum(p, 1.0 - p)
        pred = (p >= 0.5).astype(float)
        margin = np.abs(p - 0.5)

        basic = make_features_basic([u], {u: rows_by_user[u]}, m)[0].tolist()

        allow_ratio = float(pred.mean())
        entropy_pred = 0.0
        for c in (allow_ratio, 1.0 - allow_ratio):
            if c > 0:
                entropy_pred -= c * np.log(c + 1e-12)
        fingerprint = [
            float(p.mean()), float(p.std()), float(p.min()), float(p.max()),
            float(np.quantile(p, 0.25)), float(np.quantile(p, 0.5)),
            float(np.quantile(p, 0.75)),
            float(conf.mean()), float(conf.std()),
            float(margin.mean()), float(margin.std()),
            allow_ratio, float((conf >= 0.8).mean()), entropy_pred,
        ]
        X.append(basic + fingerprint)
    return np.array(X, dtype=float)


def oof_predict_proba(clf_factory, X, y):
    """Out-of-fold predicted class + posterior, robust to missing classes."""
    y = np.asarray(y, dtype=int)
    n_classes = int(y.max()) + 1
    out = np.zeros((len(y), n_classes), dtype=float)
    n_splits = min(5, min(Counter(y).values()))
    if n_splits < 2:
        raise RuntimeError("Not enough samples per class for CV")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    pred = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(X, y):
        clf = clf_factory()
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])
        cols = clf.classes_.astype(int)
        full = np.zeros((len(te), n_classes), dtype=float)
        full[:, cols] = proba
        out[te] = full
        pred[te] = np.argmax(full, axis=1)
    return pred, out


def run_attack(preds, feature_fn, pref_labels):
    """Return expert ASR + preference (direct / routing-augmented) ASR over budgets."""
    users, rows_by_user, y_expert, n_experts = build_user_table(preds)
    y_pref = np.array([pref_labels[u] for u in users], dtype=int)

    expert_asr, pref_direct, pref_routing = [], [], []
    for m in BUDGETS:
        X = feature_fn(users, rows_by_user, m)

        exp_factory = lambda: RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=SEED, class_weight="balanced"
        )
        e_pred, e_prob = oof_predict_proba(exp_factory, X, y_expert)
        expert_asr.append(float(np.mean(e_pred == y_expert)))

        pref_factory = lambda: LogisticRegression(
            max_iter=2000, random_state=SEED, class_weight="balanced"
        )
        d_pred, _ = oof_predict_proba(pref_factory, X, y_pref)
        pref_direct.append(float(np.mean(d_pred == y_pref)))

        X_route = np.concatenate([X, e_prob], axis=1)
        r_pred, _ = oof_predict_proba(pref_factory, X_route, y_pref)
        pref_routing.append(float(np.mean(r_pred == y_pref)))

    dom_expert_dist = Counter(int(e) for e in y_expert)
    majority = max(dom_expert_dist.values()) / len(y_expert)
    return {
        "n_users": len(users),
        "n_experts": n_experts,
        "expert_asr": expert_asr,
        "preference_direct_asr": pref_direct,
        "preference_routing_asr": pref_routing,
        "dominant_expert_distribution": dict(dom_expert_dist),
        "expert_majority_baseline": float(majority),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 72)
    print("Gating-leakage attack: centralized MoE vs Federated MoE")
    print("=" * 72)

    pref_labels, q33, q67 = load_train_preference_labels()
    moe_preds = load_predictions(MOE_PRED)
    fed_preds = load_predictions(FED_PRED)

    print("\n[1] MoE (reused attack)")
    moe = run_attack(moe_preds, make_features_basic, pref_labels)
    print(f"    dominant-expert dist={moe['dominant_expert_distribution']} "
          f"majority={moe['expert_majority_baseline']:.3f}")

    print("[2] Fed-MoE (reused attack)")
    fed_basic = run_attack(fed_preds, make_features_basic, pref_labels)
    print(f"    dominant-expert dist={fed_basic['dominant_expert_distribution']} "
          f"majority={fed_basic['expert_majority_baseline']:.3f}")

    print("[3] Fed-MoE (tailored attack)")
    fed_tailored = run_attack(fed_preds, make_features_tailored, pref_labels)

    K = moe["n_experts"]

    # ---- Figure 1: expert inference ASR ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(BUDGETS, moe["expert_asr"], marker="o", linewidth=2.0, color="#d62728",
            label="MoE (reused attack)")
    ax.plot(BUDGETS, fed_basic["expert_asr"], marker="s", linewidth=2.0, color="#1f77b4",
            label="Fed-MoE (reused attack)")
    ax.plot(BUDGETS, fed_tailored["expert_asr"], marker="D", linewidth=2.0, color="#2ca02c",
            label="Fed-MoE (tailored attack)")
    ax.axhline(1.0 / K, linestyle="--", color="gray", linewidth=1.0,
               label=f"random=1/K={1.0/K:.2f}")
    ax.axhline(moe["expert_majority_baseline"], linestyle=":", color="#d62728",
               linewidth=1.0, label="MoE majority-class")
    ax.axhline(fed_basic["expert_majority_baseline"], linestyle=":", color="#1f77b4",
               linewidth=1.0, label="Fed-MoE majority-class")
    ax.set_xlabel("Number of crafted queries m")
    ax.set_ylabel("Expert attack success rate")
    ax.set_title("Gating leakage: expert inference, MoE vs Fed-MoE")
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_EXPERT, dpi=300)
    plt.close(fig)
    print(f"\nSaved figure -> {FIG_EXPERT}")

    # ---- Figure 2: preference inference ASR (routing-augmented) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(BUDGETS, moe["preference_routing_asr"], marker="o", linewidth=2.0,
            color="#d62728", label="MoE (reused attack)")
    ax.plot(BUDGETS, fed_basic["preference_routing_asr"], marker="s", linewidth=2.0,
            color="#1f77b4", label="Fed-MoE (reused attack)")
    ax.plot(BUDGETS, fed_tailored["preference_routing_asr"], marker="D", linewidth=2.0,
            color="#2ca02c", label="Fed-MoE (tailored attack)")
    ax.axhline(1.0 / 3.0, linestyle="--", color="gray", linewidth=1.0,
               label="random=1/3")
    ax.set_xlabel("Number of crafted queries m")
    ax.set_ylabel("Preference attack success rate")
    ax.set_title("Gating leakage: preference inference, MoE vs Fed-MoE")
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_PREF, dpi=300)
    plt.close(fig)
    print(f"Saved figure -> {FIG_PREF}")

    report = {
        "setting": {
            "moe_prediction_file": MOE_PRED,
            "fed_prediction_file": FED_PRED,
            "query_budgets": BUDGETS,
            "attacker_observes": ["p_allow", "confidence", "pred", "margin"],
            "tailored_extra_features": "order-invariant per-user routing fingerprint",
            "evaluation_only_ground_truth": [
                "dominant expert from gate_weights",
                "preference class from real training history",
            ],
            "preference_thresholds": {"q33": q33, "q67": q67},
        },
        "moe_reused_attack": moe,
        "fed_moe_reused_attack": fed_basic,
        "fed_moe_tailored_attack": fed_tailored,
        "figures": {
            "expert_asr_vs_queries": FIG_EXPERT,
            "preference_asr_vs_queries": FIG_PREF,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report -> {OUT_JSON}")

    def fmt(v):
        return [round(x, 3) for x in v]

    print("\nExpert ASR:")
    print(f"  MoE          : {fmt(moe['expert_asr'])}")
    print(f"  Fed (reused) : {fmt(fed_basic['expert_asr'])}")
    print(f"  Fed (tailored): {fmt(fed_tailored['expert_asr'])}")
    print("Preference ASR (routing-augmented):")
    print(f"  MoE          : {fmt(moe['preference_routing_asr'])}")
    print(f"  Fed (reused) : {fmt(fed_basic['preference_routing_asr'])}")
    print(f"  Fed (tailored): {fmt(fed_tailored['preference_routing_asr'])}")


if __name__ == "__main__":
    main()
