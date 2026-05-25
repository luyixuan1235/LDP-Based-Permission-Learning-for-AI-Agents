"""Shared data loading and feature engineering for RQ1 baselines.

Used by:
    - permission_no_personalization.py  (Single model, no personalization)
    - permission_clustering.py          (Clustering-based personalization)
    - permission_moe.py                 (MoE-only, centralized)
    - rq1_visualization.py              (line charts of all baselines)

Pipeline:
    processed_dataset.json + queries.json + data_types.csv
        -> normalized flat records (train / test)
        -> categorical vocabularies (query_id / receiver / generic_data_type / domain)
        -> per-user profile vectors (per-domain allow rate + bio features)
        -> numpy arrays ready to be wrapped as torch tensors

Only binary labels are kept: "Yes, always share" -> 1, "No, never share" -> 0.
Other labels (ask-first, etc.) are filtered, matching the paper's setup.
"""

import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROCESSED_DATASET_PATH = "../data/processed_dataset.json"
QUERIES_PATH = "../queries.json"
DATA_TYPES_PATH = "../data/data_types.csv"
RESULTS_DIR = "../results"


ALLOWED_LABELS = {
    "Yes, always share": 1,
    "No, never share": 0,
}

RECEIVER_ALIASES = {
    "mira": "LLM",
    "llm": "LLM",
}

UNKNOWN_TOKEN = "<UNK>"


def safe_int(value, default):
    """Coerce ``value`` to int; fall back to ``default`` on None / bad input."""
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)


def normalize_text(value):
    if value is None:
        return ""
    text = str(value).replace("\u2019", "'").strip()
    text = re.sub(r"\s*\(.*$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.;:")


def canonical_token(value):
    return normalize_text(value).lower()


def normalize_receiver(receiver):
    cleaned = normalize_text(receiver)
    return RECEIVER_ALIASES.get(cleaned.lower(), cleaned)


def load_generic_type_map(path=DATA_TYPES_PATH):
    mapping = {}
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        generic_type = str(row["Generic Data Type"]).strip()
        raw_types = str(row["Raw Data Type"]).split(",")
        for raw_type in raw_types:
            mapping[canonical_token(raw_type)] = generic_type
    return mapping


def parse_permission_key(permission_key):
    parts = permission_key.split(",", 1)
    if len(parts) == 2:
        receiver, data_type = parts
    else:
        receiver, data_type = "Unknown", permission_key
    return normalize_receiver(receiver), normalize_text(data_type)


def load_queries(path=QUERIES_PATH):
    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    return {int(q["id"]): q for q in queries}


def build_records(
    processed_dataset_path=PROCESSED_DATASET_PATH,
    queries_path=QUERIES_PATH,
    data_types_path=DATA_TYPES_PATH,
) -> Tuple[List[dict], List[dict], Dict[str, dict]]:
    """Load per-user train/test splits and turn them into flat records."""
    queries_by_id = load_queries(queries_path)
    generic_map = load_generic_type_map(data_types_path)

    with open(processed_dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    train_records: List[dict] = []
    test_records: List[dict] = []
    bio_features_by_user: Dict[str, dict] = {}

    for user_id, user_data in dataset.items():
        bio_features_by_user[user_id] = {
            "ai_familiarity": safe_int(user_data.get("ai_familiarity"), 3),
            "ai_frequency": safe_int(user_data.get("ai_frequency"), 3),
            "ai_trust": safe_int(user_data.get("ai_trust"), 3),
            "privacy_importance": safe_int(user_data.get("privacy_importance"), 3),
            "age": safe_int(user_data.get("age"), 3),
            "gender": safe_int(user_data.get("gender"), 0),
            "education": safe_int(user_data.get("education"), 3),
        }

        for split_key, target_list in [
            ("training", train_records),
            ("testing", test_records),
        ]:
            for example in user_data.get(split_key, []):
                query_id = int(example["id"])
                domain = queries_by_id.get(query_id, {}).get("domain", "Unknown")
                for permission_key, label in example.get("answer", {}).items():
                    binary = ALLOWED_LABELS.get(label)
                    if binary is None:
                        continue
                    receiver, data_type = parse_permission_key(permission_key)
                    generic_data_type = generic_map.get(
                        canonical_token(data_type), data_type
                    )
                    target_list.append({
                        "user_id": user_id,
                        "query_id": query_id,
                        "domain": domain,
                        "receiver": receiver,
                        "data_type": data_type,
                        "generic_data_type": generic_data_type,
                        "permission_key": permission_key,
                        "label": int(binary),
                    })

    return train_records, test_records, bio_features_by_user


def build_vocab(
    records: List[dict],
    fields=("query_id", "receiver", "generic_data_type", "domain"),
) -> Dict[str, Dict[object, int]]:
    """Build vocabularies from training records. Index 0 is reserved for unknowns."""
    vocab = {field: {UNKNOWN_TOKEN: 0} for field in fields}
    for rec in records:
        for field in fields:
            value = rec[field]
            if value not in vocab[field]:
                vocab[field][value] = len(vocab[field])
    return vocab


def encode_field(values, vocab_field):
    return np.array(
        [vocab_field.get(v, 0) for v in values],
        dtype=np.int64,
    )


def build_user_profiles(
    train_records: List[dict],
    bio_features: Dict[str, dict],
    vocab: Dict[str, Dict[object, int]],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build per-user profile vectors from training history + demographics.

    Profile = [allow_rate_per_domain (n_domains)] + [bio (7)] + [overall_allow_rate (1)]
    """
    domain_vocab = vocab["domain"]
    n_domains = len(domain_vocab)

    user_records: Dict[str, List[dict]] = {}
    for rec in train_records:
        user_records.setdefault(rec["user_id"], []).append(rec)

    user_ids = sorted(bio_features.keys())
    profiles = []
    for user_id in user_ids:
        recs = user_records.get(user_id, [])
        labels = [r["label"] for r in recs]
        overall_allow = float(np.mean(labels)) if labels else 0.5

        domain_allow = np.full(n_domains, overall_allow, dtype=np.float32)
        for d_name, d_idx in domain_vocab.items():
            if d_name == UNKNOWN_TOKEN:
                continue
            d_labels = [r["label"] for r in recs if r["domain"] == d_name]
            if d_labels:
                domain_allow[d_idx] = float(np.mean(d_labels))

        bio = bio_features[user_id]
        bio_vec = np.array([
            bio["ai_familiarity"] / 5.0,
            bio["ai_frequency"] / 5.0,
            bio["ai_trust"] / 5.0,
            bio["privacy_importance"] / 5.0,
            bio["age"] / 7.0,
            bio["gender"] / 3.0,
            bio["education"] / 6.0,
        ], dtype=np.float32)

        profile = np.concatenate([
            domain_allow,
            bio_vec,
            np.array([overall_allow], dtype=np.float32),
        ]).astype(np.float32)
        profiles.append(profile)

    profiles_arr = np.stack(profiles, axis=0).astype(np.float32)
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    return profiles_arr, user_to_idx


def records_to_arrays(
    records: List[dict],
    vocab: Dict[str, Dict[object, int]],
    user_to_idx: Dict[str, int],
) -> Dict[str, object]:
    """Convert records to numpy arrays + raw lists for downstream use."""
    return {
        "query_idx": encode_field([r["query_id"] for r in records], vocab["query_id"]),
        "receiver_idx": encode_field([r["receiver"] for r in records], vocab["receiver"]),
        "datatype_idx": encode_field(
            [r["generic_data_type"] for r in records], vocab["generic_data_type"]
        ),
        "domain_idx": encode_field([r["domain"] for r in records], vocab["domain"]),
        "user_profile_idx": np.array(
            [user_to_idx[r["user_id"]] for r in records], dtype=np.int64
        ),
        "labels": np.array([r["label"] for r in records], dtype=np.float32),
        "user_ids": [r["user_id"] for r in records],
        "permission_keys": [r["permission_key"] for r in records],
        "query_ids": [r["query_id"] for r in records],
    }


class ContextEncoder(nn.Module):
    """Embeds (query_id, receiver, generic_data_type, domain) into a context vector."""

    def __init__(self, vocab: Dict[str, Dict[object, int]], emb_dim: int = 16):
        super().__init__()
        self.query_emb = nn.Embedding(len(vocab["query_id"]), emb_dim)
        self.receiver_emb = nn.Embedding(len(vocab["receiver"]), emb_dim)
        self.datatype_emb = nn.Embedding(len(vocab["generic_data_type"]), emb_dim)
        self.domain_emb = nn.Embedding(len(vocab["domain"]), emb_dim)
        self.out_dim = emb_dim * 4

    def forward(self, query_idx, receiver_idx, datatype_idx, domain_idx):
        return torch.cat([
            self.query_emb(query_idx),
            self.receiver_emb(receiver_idx),
            self.datatype_emb(datatype_idx),
            self.domain_emb(domain_idx),
        ], dim=-1)


def make_batches(n_samples: int, batch_size: int, rng: np.random.Generator):
    """Yield shuffled batch index arrays."""
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        yield indices[start:start + batch_size]


def save_json(obj, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_predictions(predictions, output_path):
    save_json(predictions, output_path)


def save_metrics(metrics, output_path):
    save_json(metrics, output_path)


def compute_full_metrics(
    predictions: List[dict],
    method_name: str,
    high_conf_threshold: float = 0.8,
) -> dict:
    """Compute accuracy / precision / recall / F1 / FPR / FNR / AUC plus high-conf stats.

    Predictions must follow our standardized schema with fields:
        participant_id, y_true (0/1), p_allow (float), pred (0/1), confidence (float)
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score,
    )

    y_true = np.array([p["y_true"] for p in predictions])
    y_pred = np.array([p["pred"] for p in predictions])
    p_allow = np.array([p["p_allow"] for p in predictions])
    confidence = np.maximum(p_allow, 1.0 - p_allow)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    auc = float(roc_auc_score(y_true, p_allow)) if len(np.unique(y_true)) > 1 else float("nan")

    mask = confidence >= high_conf_threshold
    coverage = float(mask.mean())
    if mask.sum() > 0:
        high_conf_accuracy = float(accuracy_score(y_true[mask], y_pred[mask]))
    else:
        high_conf_accuracy = float("nan")

    return {
        "method": method_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "auc": auc,
        "high_conf_threshold": high_conf_threshold,
        "high_conf_accuracy": high_conf_accuracy,
        "coverage": coverage,
        "n_predictions": int(len(y_true)),
        "n_participants": int(len({p["participant_id"] for p in predictions})),
    }


def print_metric_summary(metrics: dict):
    print(f"\n{'='*60}")
    print(f"{metrics['method']}")
    print(f"{'='*60}")
    print(f"  Accuracy           : {metrics['accuracy']*100:6.2f}%")
    print(f"  Precision          : {metrics['precision']*100:6.2f}%")
    print(f"  Recall             : {metrics['recall']*100:6.2f}%")
    print(f"  F1                 : {metrics['f1']*100:6.2f}%")
    print(f"  FPR                : {metrics['fpr']*100:6.2f}%")
    print(f"  FNR                : {metrics['fnr']*100:6.2f}%")
    print(f"  AUC                : {metrics['auc']*100:6.2f}%")
    print(f"  High-conf Acc.     : {metrics['high_conf_accuracy']*100:6.2f}%  (tau={metrics['high_conf_threshold']})")
    print(f"  Coverage @ tau     : {metrics['coverage']*100:6.2f}%")
    print(f"  N predictions      : {metrics['n_predictions']}")
    print(f"  N participants     : {metrics['n_participants']}")
    print(f"{'='*60}\n")
