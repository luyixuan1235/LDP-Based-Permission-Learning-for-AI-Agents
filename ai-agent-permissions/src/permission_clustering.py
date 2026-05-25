"""RQ1 Baseline 3: Clustering-based Personalization.

Steps:
1. Build per-user profile vectors from training history + demographics.
2. KMeans cluster users into K groups (default K=4).
3. Train one MLP per cluster on that cluster's training records.
4. At test time, route a sample to its user's cluster model.
5. A global fallback model is also trained; used when a cluster has too
   few samples to train its own model (< MIN_CLUSTER_SAMPLES).

This is "coarse-grained" personalization: every user is rigidly assigned to
one cluster, and all their requests use the same cluster model. It sits in
between "no personalization" and "MoE" in the ablation ladder.

Outputs:
    ../results/clustering_personalization_predictions.json
    ../results/clustering_personalization_metrics.json
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans

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


N_CLUSTERS = 4
MIN_CLUSTER_SAMPLES = 100

EMB_DIM = 16
HIDDEN_DIM = 128
EPOCHS = 40
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.2
SEED = 42

OUTPUT_PRED = os.path.join(RESULTS_DIR, "clustering_personalization_predictions.json")
OUTPUT_METRICS = os.path.join(RESULTS_DIR, "clustering_personalization_metrics.json")


class ClusterModel(nn.Module):
    """Same MLP head used by every cluster (parameters trained separately)."""

    def __init__(self, vocab, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.encoder = ContextEncoder(vocab, emb_dim=emb_dim)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_idx, receiver_idx, datatype_idx, domain_idx):
        z = self.encoder(query_idx, receiver_idx, datatype_idx, domain_idx)
        return self.head(z).squeeze(-1)


def to_tensor_dict(arrays, device):
    return {
        "query_idx": torch.from_numpy(arrays["query_idx"]).to(device),
        "receiver_idx": torch.from_numpy(arrays["receiver_idx"]).to(device),
        "datatype_idx": torch.from_numpy(arrays["datatype_idx"]).to(device),
        "domain_idx": torch.from_numpy(arrays["domain_idx"]).to(device),
        "labels": torch.from_numpy(arrays["labels"]).to(device),
    }


def select_arrays(arrays, mask):
    """Slice a records_to_arrays dict by a boolean mask."""
    out = {}
    mask = np.asarray(mask, dtype=bool)
    indices = np.where(mask)[0]
    for k, v in arrays.items():
        if isinstance(v, np.ndarray):
            out[k] = v[mask]
        else:
            out[k] = [v[i] for i in indices]
    return out


def train_one_model(vocab, sub_arrays, device, label="model", seed_offset=0):
    torch.manual_seed(SEED + seed_offset)
    rng = np.random.default_rng(SEED + seed_offset)

    model = ClusterModel(vocab).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    tensors = to_tensor_dict(sub_arrays, device)
    n_samples = int(len(sub_arrays["labels"]))

    print(f"  [{label}] training samples: {n_samples}")

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss, n_batches = 0.0, 0
        for idx in make_batches(n_samples, BATCH_SIZE, rng):
            idx_t = torch.from_numpy(idx).to(device)
            logits = model(
                tensors["query_idx"][idx_t],
                tensors["receiver_idx"][idx_t],
                tensors["datatype_idx"][idx_t],
                tensors["domain_idx"][idx_t],
            )
            loss = loss_fn(logits, tensors["labels"][idx_t])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            avg = total_loss / max(n_batches, 1)
            print(f"    [{label}] epoch {epoch:3d}/{EPOCHS}  loss={avg:.4f}")
    return model


@torch.no_grad()
def predict_with_model(model, arrays, device):
    model.eval()
    tensors = to_tensor_dict(arrays, device)
    logits = model(
        tensors["query_idx"],
        tensors["receiver_idx"],
        tensors["datatype_idx"],
        tensors["domain_idx"],
    )
    return torch.sigmoid(logits).cpu().numpy()


def main():
    print("=" * 60)
    print("RQ1 Baseline: Clustering-based Personalization")
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

    print(f"\nClustering users with KMeans (k={N_CLUSTERS})...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    user_cluster = kmeans.fit_predict(user_profiles)
    cluster_by_user_id = {
        uid: int(user_cluster[user_to_idx[uid]]) for uid in user_to_idx
    }

    cluster_user_counts = {
        int(c): int(np.sum(user_cluster == c)) for c in range(N_CLUSTERS)
    }
    print(f"  Users per cluster: {cluster_user_counts}")

    train_arrays = records_to_arrays(train_records, vocab, user_to_idx)
    test_arrays = records_to_arrays(test_records, vocab, user_to_idx)

    train_cluster_id = np.array(
        [cluster_by_user_id[uid] for uid in train_arrays["user_ids"]],
        dtype=np.int64,
    )
    test_cluster_id = np.array(
        [cluster_by_user_id[uid] for uid in test_arrays["user_ids"]],
        dtype=np.int64,
    )

    print("\n[Training global fallback model]")
    fallback_model = train_one_model(
        vocab, train_arrays, device, label="global", seed_offset=0,
    )

    cluster_models = {}
    cluster_train_counts = {}
    for c in range(N_CLUSTERS):
        mask = (train_cluster_id == c)
        cluster_train_counts[int(c)] = int(mask.sum())
        if mask.sum() < MIN_CLUSTER_SAMPLES:
            print(
                f"\n[Cluster {c}] only {mask.sum()} samples "
                f"(< {MIN_CLUSTER_SAMPLES}); using global fallback at test time"
            )
            continue
        print(f"\n[Training cluster {c} model]")
        sub = select_arrays(train_arrays, mask)
        cluster_models[int(c)] = train_one_model(
            vocab, sub, device, label=f"cluster {c}", seed_offset=c + 1,
        )

    print("\nPredicting on test set with cluster routing...")
    n_test = int(len(test_arrays["labels"]))
    probs = np.zeros(n_test, dtype=np.float32)
    routed_with_cluster_model = np.zeros(n_test, dtype=bool)

    for c in range(N_CLUSTERS):
        mask = (test_cluster_id == c)
        if not mask.any():
            continue
        sub_test = select_arrays(test_arrays, mask)
        if c in cluster_models:
            model = cluster_models[c]
            routed_with_cluster_model[mask] = True
        else:
            model = fallback_model
        sub_probs = predict_with_model(model, sub_test, device)
        probs[mask] = sub_probs

    predictions = []
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
            "cluster": int(test_cluster_id[i]),
            "used_cluster_model": bool(routed_with_cluster_model[i]),
        })

    metrics = compute_full_metrics(
        predictions,
        method_name="Clustering Personalization (RQ1)",
    )
    metrics["n_clusters"] = N_CLUSTERS
    metrics["min_cluster_train_samples"] = MIN_CLUSTER_SAMPLES
    metrics["cluster_user_counts"] = cluster_user_counts
    metrics["cluster_train_sample_count"] = cluster_train_counts
    metrics["trained_cluster_models"] = sorted(int(c) for c in cluster_models)
    metrics["hyperparameters"] = {
        "emb_dim": EMB_DIM,
        "hidden_dim": HIDDEN_DIM,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "seed": SEED,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_predictions(predictions, OUTPUT_PRED)
    save_metrics(metrics, OUTPUT_METRICS)

    print_metric_summary(metrics)
    print(f"Predictions -> {OUTPUT_PRED}")
    print(f"Metrics     -> {OUTPUT_METRICS}")


if __name__ == "__main__":
    main()
