"""RQ1 Baseline 4: Mixture-of-Experts (MoE-only, centralized, no privacy).

Architecture:
    z          = ContextEncoder(query_id, receiver, generic_data_type, domain)
    gate_in    = concat(z, user_profile)
    gate       = softmax(GateMLP(gate_in))     # shape (B, K)
    expert_k   = sigmoid(Expert_k(z))           # shape (B,)
    p_allow    = sum_k gate_k * expert_k        # shape (B,)

Optional sparse Top-K gating: only keep the K' largest gate weights per
sample, renormalize, and use them as gating weights (mimics Switch / Top-K
MoE used in NLP). Set TOP_K = None to disable and use dense gating.

This script implements the centralized version: server has all data, no LDP
and no federation. It is the upper-bound for RQ1 -- if even centralized MoE
cannot beat clustering / single, the architecture is not worth the privacy
overhead.

Outputs:
    ../results/moe_rq1_predictions.json
    ../results/moe_rq1_metrics.json
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq1_data_utils import (
    RESULTS_DIR,
    ContextEncoder,
    build_records,
    build_user_profiles,
    build_vocab,
    compute_full_metrics,
    make_batches,
    print_metric_summary,
    records_to_arrays,
    save_metrics,
    save_predictions,
)


N_EXPERTS = 4
TOP_K = 2                    # sparse gating; None disables (dense softmax over K)
EMB_DIM = 16
EXPERT_HIDDEN = 64
GATE_HIDDEN = 64
EPOCHS = 60
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.2
LOAD_BALANCE_COEF = 0.01     # prevents expert collapse; 0 disables
SEED = 42

OUTPUT_PRED = os.path.join(RESULTS_DIR, "moe_rq1_predictions.json")
OUTPUT_METRICS = os.path.join(RESULTS_DIR, "moe_rq1_metrics.json")


class Expert(nn.Module):
    def __init__(self, in_dim, hidden_dim=EXPERT_HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class MoEModel(nn.Module):
    def __init__(
        self,
        vocab,
        user_profile_dim,
        n_experts=N_EXPERTS,
        top_k=TOP_K,
        emb_dim=EMB_DIM,
        gate_hidden=GATE_HIDDEN,
        expert_hidden=EXPERT_HIDDEN,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.encoder = ContextEncoder(vocab, emb_dim=emb_dim)
        self.n_experts = n_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            Expert(self.encoder.out_dim, expert_hidden, dropout)
            for _ in range(n_experts)
        ])

        gate_in_dim = self.encoder.out_dim + user_profile_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, n_experts),
        )

    def forward(
        self,
        query_idx,
        receiver_idx,
        datatype_idx,
        domain_idx,
        user_profile,
    ):
        z = self.encoder(query_idx, receiver_idx, datatype_idx, domain_idx)
        gate_in = torch.cat([z, user_profile], dim=-1)
        gate_logits = self.gate(gate_in)
        gate_weights = torch.softmax(gate_logits, dim=-1)

        if self.top_k is not None and self.top_k < self.n_experts:
            top_vals, top_idx = torch.topk(gate_weights, self.top_k, dim=-1)
            sparse = torch.zeros_like(gate_weights)
            sparse.scatter_(1, top_idx, top_vals)
            denom = sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            gate_weights = sparse / denom

        expert_probs = torch.stack(
            [torch.sigmoid(expert(z)) for expert in self.experts],
            dim=-1,
        )  # (B, K)
        p_allow = (gate_weights * expert_probs).sum(dim=-1)
        return p_allow, gate_weights


def to_tensor_dict(arrays, user_profiles, device):
    return {
        "query_idx": torch.from_numpy(arrays["query_idx"]).to(device),
        "receiver_idx": torch.from_numpy(arrays["receiver_idx"]).to(device),
        "datatype_idx": torch.from_numpy(arrays["datatype_idx"]).to(device),
        "domain_idx": torch.from_numpy(arrays["domain_idx"]).to(device),
        "user_profile_idx": torch.from_numpy(arrays["user_profile_idx"]).to(device),
        "labels": torch.from_numpy(arrays["labels"]).to(device),
        "user_profiles": torch.from_numpy(user_profiles).to(device),
    }


def load_balance_loss(gate_weights):
    """Penalize uneven expert usage (encourages all experts to be used)."""
    mean_per_expert = gate_weights.mean(dim=0)
    target = 1.0 / gate_weights.size(-1)
    return ((mean_per_expert - target) ** 2).sum()


def train_moe(train_arrays, user_profiles, vocab, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    model = MoEModel(
        vocab=vocab,
        user_profile_dim=int(user_profiles.shape[1]),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCELoss()

    tensors = to_tensor_dict(train_arrays, user_profiles, device)
    n_samples = int(len(train_arrays["labels"]))

    print(f"  Training samples : {n_samples}")
    print(f"  Experts          : {N_EXPERTS}  (top_k = {TOP_K})")
    print(f"  Epochs           : {EPOCHS}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  LB coef          : {LOAD_BALANCE_COEF}")

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_bce, total_lb, n_batches = 0.0, 0.0, 0
        for idx in make_batches(n_samples, BATCH_SIZE, rng):
            idx_t = torch.from_numpy(idx).to(device)
            user_prof = tensors["user_profiles"][tensors["user_profile_idx"][idx_t]]
            p_allow, gate_w = model(
                tensors["query_idx"][idx_t],
                tensors["receiver_idx"][idx_t],
                tensors["datatype_idx"][idx_t],
                tensors["domain_idx"][idx_t],
                user_prof,
            )
            p_allow = p_allow.clamp(1e-6, 1.0 - 1e-6)
            bce = loss_fn(p_allow, tensors["labels"][idx_t])
            lb = load_balance_loss(gate_w)
            loss = bce + LOAD_BALANCE_COEF * lb

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_bce += float(bce.item())
            total_lb += float(lb.item())
            n_batches += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            avg_bce = total_bce / max(n_batches, 1)
            avg_lb = total_lb / max(n_batches, 1)
            print(f"    epoch {epoch:3d}/{EPOCHS}  bce={avg_bce:.4f}  lb={avg_lb:.4f}")
    return model


@torch.no_grad()
def predict_moe(model, arrays, user_profiles, device):
    model.eval()
    tensors = to_tensor_dict(arrays, user_profiles, device)
    user_prof = tensors["user_profiles"][tensors["user_profile_idx"]]
    p_allow, gate_w = model(
        tensors["query_idx"],
        tensors["receiver_idx"],
        tensors["datatype_idx"],
        tensors["domain_idx"],
        user_prof,
    )
    return p_allow.cpu().numpy(), gate_w.cpu().numpy()


def main():
    print("=" * 60)
    print("RQ1 Baseline: MoE-only (Centralized)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading data...")
    train_records, test_records, bio_features = build_records()
    print(f"  Train records : {len(train_records)}")
    print(f"  Test  records : {len(test_records)}")

    vocab = build_vocab(train_records)
    user_profiles, user_to_idx = build_user_profiles(
        train_records, bio_features, vocab
    )
    print(f"  Users         : {len(user_to_idx)}")
    print(f"  Profile dim   : {user_profiles.shape[1]}")

    train_arrays = records_to_arrays(train_records, vocab, user_to_idx)
    test_arrays = records_to_arrays(test_records, vocab, user_to_idx)

    print("\nTraining MoE...")
    model = train_moe(train_arrays, user_profiles, vocab, device)

    print("\nPredicting on test set...")
    probs, gate_weights = predict_moe(model, test_arrays, user_profiles, device)

    expert_assignments = gate_weights.argmax(axis=-1)
    expert_usage = {
        int(k): int((expert_assignments == k).sum()) for k in range(N_EXPERTS)
    }

    predictions = []
    n_test = int(len(test_arrays["labels"]))
    for i in range(n_test):
        p = float(probs[i])
        pred = int(p >= 0.5)
        conf = max(p, 1.0 - p)
        predictions.append({
            "participant_id": test_arrays["user_ids"][i],
            "query_id": int(test_arrays["query_ids"][i]),
            "permission_key": test_arrays["permission_keys"][i],
            "y_true": int(test_arrays["labels"][i]),
            "p_allow": p,
            "pred": pred,
            "confidence": float(conf),
            "top_expert": int(expert_assignments[i]),
            "gate_weights": [float(w) for w in gate_weights[i]],
        })

    metrics = compute_full_metrics(predictions, method_name="MoE Only (RQ1)")
    metrics["n_experts"] = N_EXPERTS
    metrics["top_k"] = TOP_K
    metrics["expert_usage"] = expert_usage
    metrics["hyperparameters"] = {
        "emb_dim": EMB_DIM,
        "expert_hidden": EXPERT_HIDDEN,
        "gate_hidden": GATE_HIDDEN,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "load_balance_coef": LOAD_BALANCE_COEF,
        "seed": SEED,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_predictions(predictions, OUTPUT_PRED)
    save_metrics(metrics, OUTPUT_METRICS)

    print_metric_summary(metrics)
    print(f"  Expert usage : {expert_usage}")
    print(f"\nPredictions -> {OUTPUT_PRED}")
    print(f"Metrics     -> {OUTPUT_METRICS}")


if __name__ == "__main__":
    main()
