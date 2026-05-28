"""Attack validation against the centralized MoE-only model (no LDP, no defense).

Threat model:
    Black-box API adversary. Attacker can submit queries to the deployed agent
    and observe per-permission outputs (p_allow, pred, confidence). All other
    quantities used here -- entropy, margin, top_expert, gate_weights -- are
    either derived from p_allow, or used ONLY as evaluation ground truth and
    NEVER exposed as attacker inputs.

We do not run prompt-injection here because LDP cannot mitigate it.

Outputs (under ../results):
    attack_moe_metrics.json
    attack_moe_mia_roc.png
    attack_moe_mia_auc_vs_features.png
    attack_moe_expert_gain_vs_queries.png
    attack_moe_preference_amplification_vs_queries.png
    attack_moe_mi_leakage.png
"""

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required for attack_moe_validation.py.\n"
        "Install with: pip install matplotlib"
    ) from exc


RESULTS_DIR = "../results"
QUERY_BUDGETS = [1, 3, 5, 10, 15, 20]
ATTACKER_FEATURES = ["p_allow", "confidence", "margin", "entropy"]

ATTACK_JSON = os.path.join(RESULTS_DIR, "attack_moe_metrics.json")
MIA_ROC_FIG = os.path.join(RESULTS_DIR, "attack_moe_mia_roc.png")
MIA_FEAT_FIG = os.path.join(RESULTS_DIR, "attack_moe_mia_auc_vs_features.png")
EXPERT_FIG = os.path.join(RESULTS_DIR, "attack_moe_expert_gain_vs_queries.png")
PREF_FIG = os.path.join(RESULTS_DIR, "attack_moe_preference_amplification_vs_queries.png")
MI_FIG = os.path.join(RESULTS_DIR, "attack_moe_mi_leakage.png")


def bce_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def entropy_from_p(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def safe_auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def gain_from_auc(auc):
    return float(2.0 * (auc - 0.5)) if not np.isnan(auc) else float("nan")


def mutual_information_histogram(x, y, x_bins=8):
    """Simple histogram-based MI estimator for I(X; Y) with discrete Y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(x) == 0:
        return float("nan")
    if x.ndim == 1:
        bins = np.linspace(x.min(), x.max() + 1e-9, x_bins + 1)
        x_disc = np.digitize(x, bins) - 1
    else:
        x_disc = x.astype(int)
    n = len(y)
    mi = 0.0
    y_values = np.unique(y)
    x_values = np.unique(x_disc)
    p_y = {int(v): float(np.mean(y == v)) for v in y_values}
    p_x = {int(v): float(np.mean(x_disc == v)) for v in x_values}
    for vx in x_values:
        for vy in y_values:
            joint = float(np.mean((x_disc == vx) & (y == vy)))
            if joint <= 0:
                continue
            mi += joint * np.log(joint / (p_x[int(vx)] * p_y[int(vy)] + 1e-12) + 1e-12)
    return float(mi)


def train_moe_and_collect():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    from rq1_data_utils import (
        build_records,
        build_user_profiles,
        build_vocab,
        records_to_arrays,
    )
    from permission_moe import (
        SEED,
        EPOCHS,
        train_moe,
        predict_moe,
    )

    if "ATTACK_MOE_EPOCHS" in os.environ:
        import permission_moe as pm
        pm.EPOCHS = int(os.environ["ATTACK_MOE_EPOCHS"])
        eff_epochs = pm.EPOCHS
    else:
        eff_epochs = EPOCHS

    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_records, test_records, bio = build_records()
    vocab = build_vocab(train_records)
    user_profiles, user_to_idx = build_user_profiles(train_records, bio, vocab)

    train_arrays = records_to_arrays(train_records, vocab, user_to_idx)
    test_arrays = records_to_arrays(test_records, vocab, user_to_idx)
    model = train_moe(train_arrays, user_profiles, vocab, device)

    train_p, train_gw = predict_moe(model, train_arrays, user_profiles, device)
    test_p, test_gw = predict_moe(model, test_arrays, user_profiles, device)
    return {
        "epochs": int(eff_epochs),
        "seed": int(SEED),
        "train_records": train_records,
        "test_records": test_records,
        "train_arrays": train_arrays,
        "test_arrays": test_arrays,
        "train_probs": train_p.astype(np.float64),
        "test_probs": test_p.astype(np.float64),
        "train_gate_w": train_gw.astype(np.float64),
        "test_gate_w": test_gw.astype(np.float64),
    }


def attack_mia(data):
    """Attack A: Membership inference using only black-box outputs."""
    p_m = data["train_probs"]
    p_n = data["test_probs"]
    feats_m = np.stack([
        p_m,
        np.maximum(p_m, 1 - p_m),
        np.abs(p_m - 0.5),
        entropy_from_p(p_m),
    ], axis=1)
    feats_n = np.stack([
        p_n,
        np.maximum(p_n, 1 - p_n),
        np.abs(p_n - 0.5),
        entropy_from_p(p_n),
    ], axis=1)
    X = np.concatenate([feats_m, feats_n], axis=0)
    y = np.concatenate(
        [np.ones(len(feats_m), dtype=int), np.zeros(len(feats_n), dtype=int)], axis=0
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.35, random_state=data["seed"], stratify=y
    )

    # 1) Full-feature attack -> ROC curve
    clf_full = LogisticRegression(max_iter=2000, random_state=data["seed"])
    clf_full.fit(X_tr, y_tr)
    score_full = clf_full.predict_proba(X_te)[:, 1]
    auc_full = safe_auc(y_te, score_full)
    fpr, tpr, _ = roc_curve(y_te, score_full)
    advantage_full = float(np.max(tpr - fpr))

    # 2) Cumulative feature sets for line chart
    auc_curve = []
    adv_curve = []
    for k in range(1, len(ATTACKER_FEATURES) + 1):
        clf_k = LogisticRegression(max_iter=2000, random_state=data["seed"])
        clf_k.fit(X_tr[:, :k], y_tr)
        s = clf_k.predict_proba(X_te[:, :k])[:, 1]
        auc_k = safe_auc(y_te, s)
        fpr_k, tpr_k, _ = roc_curve(y_te, s)
        auc_curve.append(auc_k)
        adv_curve.append(float(np.max(tpr_k - fpr_k)))

    # Plot ROC
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2.0,
            label=f"MoE-only MIA (AUC={auc_full:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Attack A: Membership inference ROC (centralized MoE)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(MIA_ROC_FIG, dpi=300)
    plt.close(fig)

    # Plot AUC vs cumulative features (line chart)
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = list(range(1, len(ATTACKER_FEATURES) + 1))
    ax.plot(xs, auc_curve, marker="o", color="#1f77b4", linewidth=1.8, label="MIA AUC")
    ax.plot(xs, adv_curve, marker="s", color="#d62728", linewidth=1.8, label="MIA Advantage")
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1.0, label="random AUC=0.5")
    ax.set_xticks(xs)
    ax.set_xticklabels(["+p_allow", "+confidence", "+margin", "+entropy"])
    ax.set_xlabel("Cumulative attacker features")
    ax.set_ylabel("Score")
    ax.set_title("Attack A: MIA strength grows with attacker features")
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(MIA_FEAT_FIG, dpi=300)
    plt.close(fig)

    return {
        "attacker_features": ATTACKER_FEATURES,
        "auc_overall": auc_full,
        "gain_overall": gain_from_auc(auc_full),
        "advantage_overall": advantage_full,
        "auc_by_cumulative_features": auc_curve,
        "advantage_by_cumulative_features": adv_curve,
    }


def build_user_pool_features(user_ids, query_ids, probs, query_pool):
    """Aggregate (mean p_allow, mean confidence) per user x query_id in pool."""
    user_to_idx = defaultdict(list)
    for i, uid in enumerate(user_ids):
        user_to_idx[uid].append(i)
    users = sorted(user_to_idx.keys())
    conf = np.maximum(probs, 1 - probs)
    X = []
    for u in users:
        idxs = np.array(user_to_idx[u], dtype=int)
        uq = query_ids[idxs]
        up = probs[idxs]
        uc = conf[idxs]
        row = []
        for q in query_pool:
            m = uq == q
            if np.any(m):
                row.extend([float(up[m].mean()), float(uc[m].mean())])
            else:
                row.extend([0.5, 0.5])
        X.append(row)
    return users, np.array(X, dtype=np.float64), user_to_idx


def attack_gating_leakage(data):
    """Attack B1: expert inference. Attack B2: preference inference + amplification."""
    user_ids = np.array(data["test_arrays"]["user_ids"])
    query_ids = np.array(data["test_arrays"]["query_ids"], dtype=int)
    test_probs = data["test_probs"]
    top_expert_test = np.argmax(data["test_gate_w"], axis=1).astype(int)
    K = int(data["test_gate_w"].shape[1])

    # Real training allow rate -> preference label
    train_by_user = defaultdict(list)
    for rec in data["train_records"]:
        train_by_user[rec["user_id"]].append(int(rec["label"]))
    allow_rate = {u: float(np.mean(v)) for u, v in train_by_user.items() if v}
    vals = np.array(list(allow_rate.values()), dtype=np.float64)
    q33 = float(np.quantile(vals, 0.33))
    q67 = float(np.quantile(vals, 0.67))
    pref_label = {
        u: 0 if r <= q33 else (1 if r >= q67 else -1)
        for u, r in allow_rate.items()
    }

    # User-level dominant expert (evaluation ground truth, NOT attacker input)
    user_to_dom_expert = {}
    user_to_indices = defaultdict(list)
    for i, uid in enumerate(user_ids):
        user_to_indices[uid].append(i)
    for u, idxs in user_to_indices.items():
        c = Counter(top_expert_test[np.array(idxs, dtype=int)].tolist())
        user_to_dom_expert[u] = c.most_common(1)[0][0]

    # Crafted query pool = most frequent test query_ids
    freq_q = [q for q, _ in Counter(query_ids.tolist()).most_common()]
    budgets = [m for m in QUERY_BUDGETS if m <= len(freq_q)]

    expert_acc = []
    expert_bal_acc = []
    expert_macro_f1 = []
    expert_gain = []
    pref_baseline_auc = []
    pref_moe_auc = []
    amplification = []
    mi_true = []
    mi_hat = []

    for m in budgets:
        pool = freq_q[:m]
        users, X, _ = build_user_pool_features(user_ids, query_ids, test_probs, pool)
        users = np.array(users)
        y_exp = np.array([user_to_dom_expert[u] for u in users], dtype=int)

        if len(np.unique(y_exp)) < 2 or len(users) < 30:
            for arr in (expert_acc, expert_bal_acc, expert_macro_f1, expert_gain,
                        pref_baseline_auc, pref_moe_auc, amplification, mi_true, mi_hat):
                arr.append(float("nan"))
            continue

        # ---- B1: expert inference ----
        X_tr, X_te, y_tr, y_te, u_tr, u_te = train_test_split(
            X, y_exp, users, test_size=0.35, random_state=data["seed"], stratify=y_exp
        )
        exp_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=data["seed"],
        )
        exp_clf.fit(X_tr, y_tr)
        y_hat = exp_clf.predict(X_te)
        acc = float(accuracy_score(y_te, y_hat))
        bal = float(balanced_accuracy_score(y_te, y_hat))
        f1m = float(f1_score(y_te, y_hat, average="macro"))
        random_acc = 1.0 / K
        gain = float((acc - random_acc) / max(1e-8, 1.0 - random_acc))
        expert_acc.append(acc)
        expert_bal_acc.append(bal)
        expert_macro_f1.append(f1m)
        expert_gain.append(gain)

        # ---- B2: preference inference on users with valid pref label ----
        pref_idx = np.array(
            [i for i, u in enumerate(users) if pref_label.get(u, -1) != -1], dtype=int
        )
        if len(pref_idx) < 40:
            pref_baseline_auc.append(float("nan"))
            pref_moe_auc.append(float("nan"))
            amplification.append(float("nan"))
            mi_true.append(float("nan"))
            mi_hat.append(float("nan"))
            continue

        Xp = X[pref_idx]
        yp = np.array([pref_label[users[i]] for i in pref_idx], dtype=int)
        if len(np.unique(yp)) < 2:
            pref_baseline_auc.append(float("nan"))
            pref_moe_auc.append(float("nan"))
            amplification.append(float("nan"))
            mi_true.append(float("nan"))
            mi_hat.append(float("nan"))
            continue

        # Baseline attacker: output-only summary stats per user
        baseline = np.stack([
            Xp.mean(axis=1),
            Xp.std(axis=1),
            Xp[:, ::2].mean(axis=1),  # mean p_allow across pool
            (Xp[:, ::2] >= 0.5).mean(axis=1),  # allow ratio across pool
        ], axis=1)

        # MoE-leakage attacker: baseline + inferred expert posterior
        prob_te = exp_clf.predict_proba(Xp)
        moe_feats = np.concatenate([baseline, prob_te], axis=1)

        Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
            baseline, yp, test_size=0.35, random_state=data["seed"], stratify=yp
        )
        Xm_tr, Xm_te, _, _ = train_test_split(
            moe_feats, yp, test_size=0.35, random_state=data["seed"], stratify=yp
        )

        base_clf = LogisticRegression(max_iter=2000, random_state=data["seed"])
        base_clf.fit(Xb_tr, yb_tr)
        moe_clf = LogisticRegression(max_iter=2000, random_state=data["seed"])
        moe_clf.fit(Xm_tr, yb_tr)

        s_base = base_clf.predict_proba(Xb_te)[:, 1]
        s_moe = moe_clf.predict_proba(Xm_te)[:, 1]
        auc_b = safe_auc(yb_te, s_base)
        auc_m = safe_auc(yb_te, s_moe)
        pref_baseline_auc.append(auc_b)
        pref_moe_auc.append(auc_m)
        amplification.append(float(auc_m - auc_b))

        # Mutual information: true expert vs preference; predicted expert vs preference
        g_true = np.array(
            [user_to_dom_expert[u] for u in users[pref_idx]], dtype=int
        )
        g_hat = np.argmax(prob_te, axis=1).astype(int)
        mi_true.append(mutual_information_histogram(g_true, yp))
        mi_hat.append(mutual_information_histogram(g_hat, yp))

    # ---- Expert inference plot ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(budgets, expert_acc, marker="o", color="#d62728", linewidth=1.8, label="Expert_Acc")
    ax.plot(budgets, expert_bal_acc, marker="s", color="#9467bd", linewidth=1.8, label="Expert_BalancedAcc")
    ax.plot(budgets, expert_gain, marker="^", color="#2ca02c", linewidth=1.8, label="Gain_gate")
    ax.axhline(1.0 / K, linestyle="--", color="gray", linewidth=1.0, label=f"random Acc=1/K={1.0/K:.2f}")
    ax.set_xlabel("Number of crafted queries m")
    ax.set_ylabel("Score")
    ax.set_title(f"Attack B1: Expert inference (K={K})")
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(EXPERT_FIG, dpi=300)
    plt.close(fig)

    # ---- Preference inference + amplification plot ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(budgets, pref_baseline_auc, marker="s", color="#1f77b4", linewidth=1.8,
            label="Direct-output AUC (baseline)")
    ax.plot(budgets, pref_moe_auc, marker="D", color="#9467bd", linewidth=1.8,
            label="MoE-leakage AUC")
    ax.plot(budgets, amplification, marker="^", color="#2ca02c", linewidth=1.8,
            label="Amplification = AUC_MoE - AUC_baseline")
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1.0, label="random AUC=0.5")
    ax.axhline(0.0, linestyle=":", color="gray", linewidth=1.0)
    ax.set_xlabel("Number of crafted queries m")
    ax.set_ylabel("Score")
    ax.set_title("Attack B2: Preference inference vs baseline")
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(PREF_FIG, dpi=300)
    plt.close(fig)

    # ---- MI leakage plot ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(budgets, mi_true, marker="o", color="#d62728", linewidth=1.8,
            label="I(G_true; Y_pref)")
    ax.plot(budgets, mi_hat, marker="s", color="#1f77b4", linewidth=1.8,
            label="I(G_hat; Y_pref) attacker-recovered")
    ax.set_xlabel("Number of crafted queries m")
    ax.set_ylabel("Mutual information (nats)")
    ax.set_title("Attack B: Expert routing carries preference information")
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(MI_FIG, dpi=300)
    plt.close(fig)

    return {
        "query_budgets": budgets,
        "K_experts": K,
        "preference_label_thresholds": {"q33": q33, "q67": q67},
        "expert_acc": expert_acc,
        "expert_balanced_acc": expert_bal_acc,
        "expert_macro_f1": expert_macro_f1,
        "gain_gate": expert_gain,
        "preference_baseline_auc": pref_baseline_auc,
        "preference_moe_auc": pref_moe_auc,
        "amplification_gain": amplification,
        "mi_g_true_pref": mi_true,
        "mi_g_hat_pref": mi_hat,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 72)
    print("Attack Validation on Centralized MoE-only (No LDP, No Defense)")
    print("=" * 72)
    data = train_moe_and_collect()
    print(f"  epochs={data['epochs']}  seed={data['seed']}")
    print(f"  train_n={len(data['train_probs'])}  test_n={len(data['test_probs'])}")

    print("\n[A] Membership inference attack...")
    mia = attack_mia(data)
    print(f"  AUC={mia['auc_overall']:.3f}  Gain={mia['gain_overall']:.3f}  "
          f"Advantage={mia['advantage_overall']:.3f}")

    print("[B] Gating leakage + preference inference...")
    gate = attack_gating_leakage(data)
    print(f"  Expert_Acc curve: {[round(x, 3) if isinstance(x, float) else x for x in gate['expert_acc']]}")
    print(f"  Gain_gate curve : {[round(x, 3) if isinstance(x, float) else x for x in gate['gain_gate']]}")
    print(f"  Amplification   : {[round(x, 3) if isinstance(x, float) else x for x in gate['amplification_gain']]}")

    report = {
        "setting": {
            "model": "Centralized MoE-only (RQ1)",
            "ldp_enabled": False,
            "epochs": data["epochs"],
            "seed": data["seed"],
            "attacker_observable_features": ["p_allow"],
            "derived_features": ["confidence", "margin", "entropy"],
            "evaluation_only_labels": ["top_expert", "gate_weights"],
            "query_budgets": QUERY_BUDGETS,
        },
        "attack_A_membership": mia,
        "attack_B_gating_leakage": gate,
        "figures": {
            "mia_roc": MIA_ROC_FIG,
            "mia_auc_vs_features": MIA_FEAT_FIG,
            "expert_gain_vs_queries": EXPERT_FIG,
            "preference_amplification_vs_queries": PREF_FIG,
            "mi_leakage": MI_FIG,
        },
    }
    with open(ATTACK_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report -> {ATTACK_JSON}")
    for name, path in report["figures"].items():
        print(f"Saved figure -> {path}")


if __name__ == "__main__":
    main()
