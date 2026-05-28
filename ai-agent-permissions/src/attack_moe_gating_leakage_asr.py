import json
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = '../results'
PRED_PATH = os.path.join(RESULTS_DIR, 'moe_rq1_predictions.json')
OUT_JSON = os.path.join(RESULTS_DIR, 'attack_moe_gating_leakage_asr.json')
FIG_EXPERT = os.path.join(RESULTS_DIR, 'attack_moe_expert_asr_vs_queries.png')
FIG_PREF = os.path.join(RESULTS_DIR, 'attack_moe_preference_asr_vs_queries.png')
FIG_AMP = os.path.join(RESULTS_DIR, 'attack_moe_preference_amplification_vs_queries.png')
BUDGETS = [1, 2, 3, 5, 8, 10, 15, 20]
SEED = 42


def load_train_preference_labels():
    from rq1_data_utils import build_records
    train_records, _test_records, _bio = build_records()
    by_user = defaultdict(list)
    for rec in train_records:
        by_user[rec['user_id']].append(int(rec['label']))
    rates = {u: float(np.mean(v)) for u, v in by_user.items() if v}
    values = np.array(list(rates.values()), dtype=float)
    q33, q67 = np.quantile(values, [0.33, 0.67])
    labels = {}
    for u, r in rates.items():
        if r <= q33:
            labels[u] = 0  # privacy-sensitive
        elif r >= q67:
            labels[u] = 2  # utility-driven
        else:
            labels[u] = 1  # balanced
    return labels, float(q33), float(q67)


def build_user_table(preds):
    rows_by_user = defaultdict(list)
    for r in preds:
        rows_by_user[r['participant_id']].append(r)
    users = sorted(rows_by_user)
    n_experts = len(preds[0]['gate_weights'])
    y_expert = []
    for u in users:
        rows = rows_by_user[u]
        avg_gate = np.array([r['gate_weights'] for r in rows], dtype=float).mean(axis=0)
        y_expert.append(int(np.argmax(avg_gate)))
        rows.sort(key=lambda r: (int(r['query_id']), r['permission_key']))
    return users, rows_by_user, np.array(y_expert, dtype=int), n_experts


def make_features(users, rows_by_user, m):
    X = []
    for u in users:
        rows = rows_by_user[u][:m]
        feat = []
        for r in rows:
            p = float(r['p_allow'])
            conf = max(p, 1.0 - p)
            pred = 1.0 if p >= 0.5 else 0.0
            margin = abs(p - 0.5)
            feat.extend([p, conf, pred, margin])
        while len(feat) < 4 * m:
            feat.extend([0.5, 0.5, 0.0, 0.0])
        X.append(feat)
    return np.array(X, dtype=float)


def oof_predict_proba(clf_factory, X, y):
    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    out = np.zeros((len(y), int(classes.max()) + 1), dtype=float)
    n_splits = min(5, min(Counter(y).values()))
    if n_splits < 2:
        raise RuntimeError('Not enough samples per class for CV')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    pred = np.zeros(len(y), dtype=int)
    conf = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        clf = clf_factory()
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        cols = clf.classes_.astype(int)
        full = np.zeros((len(te), out.shape[1]), dtype=float)
        full[:, cols] = p
        out[te] = full
        pred[te] = np.argmax(full, axis=1)
        conf[te] = np.max(full, axis=1)
    return pred, conf, out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(PRED_PATH, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    pref_labels, q33, q67 = load_train_preference_labels()
    users, rows_by_user, y_expert, n_experts = build_user_table(preds)
    y_pref = np.array([pref_labels[u] for u in users], dtype=int)

    expert_asr, expert_cond_acc, expert_coverage = [], [], []
    pref_direct_asr, pref_routing_asr, amp_asr = [], [], []
    pref_direct_cond, pref_routing_cond = [], []

    for m in BUDGETS:
        X = make_features(users, rows_by_user, m)
        exp_factory = lambda: RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED, class_weight='balanced')
        e_pred, e_conf, e_prob = oof_predict_proba(exp_factory, X, y_expert)
        expert_asr.append(float(np.mean(e_pred == y_expert)))
        # conditional high-confidence accuracy at attacker's median confidence, plus coverage
        tau = float(np.median(e_conf))
        mask = e_conf >= tau
        expert_coverage.append(float(np.mean(mask)))
        expert_cond_acc.append(float(np.mean(e_pred[mask] == y_expert[mask])))

        # preference direct attack: only black-box output trajectory
        pref_factory = lambda: LogisticRegression(max_iter=2000, random_state=SEED, class_weight='balanced')
        p_pred_d, p_conf_d, _ = oof_predict_proba(pref_factory, X, y_pref)
        pref_direct_asr.append(float(np.mean(p_pred_d == y_pref)))
        tau_d = float(np.median(p_conf_d)); mask_d = p_conf_d >= tau_d
        pref_direct_cond.append(float(np.mean(p_pred_d[mask_d] == y_pref[mask_d])))

        # routing-augmented preference attack: black-box trajectory + inferred expert posterior
        X_route = np.concatenate([X, e_prob], axis=1)
        p_pred_r, p_conf_r, _ = oof_predict_proba(pref_factory, X_route, y_pref)
        pref_routing_asr.append(float(np.mean(p_pred_r == y_pref)))
        tau_r = float(np.median(p_conf_r)); mask_r = p_conf_r >= tau_r
        pref_routing_cond.append(float(np.mean(p_pred_r[mask_r] == y_pref[mask_r])))
        amp_asr.append(pref_routing_asr[-1] - pref_direct_asr[-1])

    # Figure 1: expert attack success rate
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(BUDGETS, expert_asr, marker='o', linewidth=2.0, color='#d62728', label='Expert attack success rate')
    ax.plot(BUDGETS, expert_cond_acc, marker='s', linewidth=1.8, color='#ff9896', label='Conditional accuracy (top 50% confidence)')
    ax.axhline(1.0 / n_experts, linestyle='--', color='gray', linewidth=1.0, label=f'random baseline=1/K={1.0/n_experts:.2f}')
    ax.set_xlabel('Number of crafted queries m')
    ax.set_ylabel('Successful users / all users')
    ax.set_title('Gating Leakage: Expert prediction success vs crafted queries')
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_EXPERT, dpi=300)
    plt.close(fig)

    # Figure 2: preference attack success rate
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(BUDGETS, pref_direct_asr, marker='s', linewidth=2.0, color='#1f77b4', label='Direct-output attack')
    ax.plot(BUDGETS, pref_routing_asr, marker='D', linewidth=2.0, color='#9467bd', label='Routing-augmented attack')
    ax.axhline(1.0 / 3.0, linestyle='--', color='gray', linewidth=1.0, label='random baseline=1/3')
    ax.set_xlabel('Number of crafted queries m')
    ax.set_ylabel('Successful users / all users')
    ax.set_title('Gating Leakage: Preference prediction success vs crafted queries')
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_PREF, dpi=300)
    plt.close(fig)

    # Figure 3: amplification
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(BUDGETS, amp_asr, marker='^', linewidth=2.0, color='#2ca02c', label='Routing amplification')
    ax.axhline(0.0, linestyle='--', color='gray', linewidth=1.0)
    ax.set_xlabel('Number of crafted queries m')
    ax.set_ylabel('Delta success rate')
    ax.set_title('Routing amplification: Preference ASR gain from inferred expert')
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_AMP, dpi=300)
    plt.close(fig)

    report = {
        'setting': {
            'model': 'MoE-only centralized',
            'prediction_file': PRED_PATH,
            'n_users': len(users),
            'n_experts': n_experts,
            'query_budgets': BUDGETS,
            'attacker_observes': ['p_allow', 'confidence', 'pred', 'margin'],
            'evaluation_only_ground_truth': ['dominant expert from gate_weights', 'preference class from real training history'],
            'preference_thresholds': {'q33': q33, 'q67': q67},
        },
        'formula': {
            'dominant_expert': 'E_i*=argmax_k mean_q g_{i,q,k}',
            'expert_asr': 'mean_i 1[hat_E_i == E_i*]',
            'preference_asr': 'mean_i 1[hat_Y_i == Y_i*]',
            'amplification': 'PreferenceASR_routing - PreferenceASR_direct',
        },
        'results': {
            'expert_asr': expert_asr,
            'expert_conditional_accuracy_top50_confidence': expert_cond_acc,
            'expert_top50_confidence_coverage': expert_coverage,
            'preference_direct_asr': pref_direct_asr,
            'preference_routing_asr': pref_routing_asr,
            'preference_direct_conditional_accuracy_top50_confidence': pref_direct_cond,
            'preference_routing_conditional_accuracy_top50_confidence': pref_routing_cond,
            'preference_amplification': amp_asr,
        },
        'figures': {
            'expert_asr_vs_queries': FIG_EXPERT,
            'preference_asr_vs_queries': FIG_PREF,
            'preference_amplification_vs_queries': FIG_AMP,
        }
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report['results'], indent=2))
    print('Saved', OUT_JSON)
    print('Saved', FIG_EXPERT)
    print('Saved', FIG_PREF)
    print('Saved', FIG_AMP)


if __name__ == '__main__':
    main()
