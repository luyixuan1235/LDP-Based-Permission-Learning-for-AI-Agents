"""RQ1 Baseline 2: Single Model (No Personalization).

A single global MLP shared by all users. Predicts permission solely from the
context (query_id, receiver, generic_data_type, domain). No user identity,
demographics, or history are used.

This is the "lower bound" of personalization: how well can we do without
modeling user differences at all?

Inputs:
    ../data/processed_dataset.json
    ../queries.json
    ../data/data_types.csv

Outputs:
    ../results/no_personalization_predictions.json
    ../results/no_personalization_metrics.json
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


EMB_DIM = 16
HIDDEN_DIM = 128
EPOCHS = 40
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.2
SEED = 42

OUTPUT_PRED = os.path.join(RESULTS_DIR, "no_personalization_predictions.json")
OUTPUT_METRICS = os.path.join(RESULTS_DIR, "no_personalization_metrics.json")


class SingleModel(nn.Module):
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


def train_single_model(train_arrays, vocab, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    model = SingleModel(vocab).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    tensors = to_tensor_dict(train_arrays, device)
    n_samples = int(len(train_arrays["labels"]))

    print(f"  Training samples : {n_samples}")
    print(f"  Epochs           : {EPOCHS}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Learning rate    : {LR}")

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
            print(f"    epoch {epoch:3d}/{EPOCHS}  loss={avg:.4f}")
    return model


@torch.no_grad()
def predict(model, arrays, device):
    model.eval()
    tensors = to_tensor_dict(arrays, device)
    logits = model(
        tensors["query_idx"],
        tensors["receiver_idx"],
        tensors["datatype_idx"],
        tensors["domain_idx"],
    )
    return torch.sigmoid(logits).cpu().numpy()


def build_prediction_records(arrays, probs, threshold=0.5):
    records = []
    for i, p in enumerate(probs):
        p_allow = float(p)
        pred = int(p_allow >= threshold)
        conf = max(p_allow, 1.0 - p_allow)
        records.append({
            "participant_id": arrays["user_ids"][i],
            "query_id": int(arrays["query_ids"][i]),
            "permission_key": arrays["permission_keys"][i],
            "y_true": int(arrays["labels"][i]),
            "p_allow": p_allow,
            "pred": pred,
            "confidence": float(conf),
        })
    return records


def main():
    print("=" * 60)
    print("RQ1 Baseline: Single Model (No Personalization)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading data...")
    train_records, test_records, bio_features = build_records()
    print(f"  Train records : {len(train_records)}")
    print(f"  Test  records : {len(test_records)}")
    print(f"  Participants  : {len(bio_features)}")

    vocab = build_vocab(train_records)
    _, user_to_idx = build_user_profiles(train_records, bio_features, vocab)

    train_arrays = records_to_arrays(train_records, vocab, user_to_idx)
    test_arrays = records_to_arrays(test_records, vocab, user_to_idx)

    print("\nTraining single model...")
    model = train_single_model(train_arrays, vocab, device)

    print("\nPredicting on test set...")
    probs = predict(model, test_arrays, device)
    predictions = build_prediction_records(test_arrays, probs)

    metrics = compute_full_metrics(
        predictions,
        method_name="Single (No Personalization, RQ1)",
    )
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
