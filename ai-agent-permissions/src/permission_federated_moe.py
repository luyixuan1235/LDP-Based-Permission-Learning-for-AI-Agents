"""RQ1 Baseline 5: Personalized Federated MoE (no LDP).

Architecture:
    server shared model = ContextEncoder + Experts
    client private model = PersonalizedGate

Each federated round sends the shared encoder/experts to sampled users. Users
train a local shared-model copy together with their private gate on local data.
The server FedAvg-aggregates only the shared parameters; gates remain local and
are reused for that user's later local training and test-time routing.

Outputs:
    ../results/federated_moe_predictions.json
    ../results/federated_moe_metrics.json
"""

import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from permission_moe import Expert, load_balance_loss
from rq1_data_utils import (
    RESULTS_DIR,
    ContextEncoder,
    build_records,
    build_user_profiles,
    build_vocab,
    compute_full_metrics,
    load_queries,
    make_batches,
    print_metric_summary,
    records_to_arrays,
    save_metrics,
    save_predictions,
)


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


def env_str(name, default):
    value = os.environ.get(name, default)
    return str(value).strip()


N_EXPERTS = env_int("FED_MOE_N_EXPERTS", 4)
TOP_K = env_int("FED_MOE_TOP_K", 2)
EMB_DIM = env_int("FED_MOE_EMB_DIM", 16)
SEMANTIC_MAX_FEATURES = env_int("FED_MOE_SEMANTIC_MAX_FEATURES", 256)
SEMANTIC_DIM = env_int("FED_MOE_SEMANTIC_DIM", 32)
USE_LLM_SEMANTIC = env_int("FED_MOE_USE_LLM_SEMANTIC", 1)
LLM_SEMANTIC_BATCH_SIZE = env_int("FED_MOE_LLM_SEMANTIC_BATCH_SIZE", 12)
EXPERT_HIDDEN = env_int("FED_MOE_EXPERT_HIDDEN", 64)
GATE_HIDDEN = env_int("FED_MOE_GATE_HIDDEN", 64)
FED_ROUNDS = env_int("FED_MOE_ROUNDS", 30)
CLIENT_FRACTION = env_float("FED_MOE_CLIENT_FRACTION", 1.0)
LOCAL_EPOCHS = env_int("FED_MOE_LOCAL_EPOCHS", 2)
LOCAL_BATCH_SIZE = env_int("FED_MOE_LOCAL_BATCH_SIZE", 64)
LR = env_float("FED_MOE_LR", 1e-3)
WEIGHT_DECAY = env_float("FED_MOE_WEIGHT_DECAY", 1e-5)
DROPOUT = env_float("FED_MOE_DROPOUT", 0.2)
LOAD_BALANCE_COEF = env_float("FED_MOE_LOAD_BALANCE_COEF", 0.01)
LOCAL_PRIOR_ALPHA = env_float("FED_MOE_LOCAL_PRIOR_ALPHA", 0.1)
LOCAL_PRIOR_CONF_THRESHOLD = env_float("FED_MOE_LOCAL_PRIOR_CONF_THRESHOLD", 0.6)
LDP_MODE = env_str("FED_MOE_LDP_MODE", "none").lower()  # none | user_stats | gating_input
LDP_EPSILON = env_float("FED_MOE_LDP_EPSILON", 8.0)
OUTPUT_TAG = env_str("FED_MOE_OUTPUT_TAG", "")
SEED = env_int("FED_MOE_SEED", 42)


def tagged_filename(base_name):
    if not OUTPUT_TAG:
        return base_name
    stem, ext = os.path.splitext(base_name)
    return f"{stem}_{OUTPUT_TAG}{ext}"


OUTPUT_PRED = os.path.join(RESULTS_DIR, tagged_filename("federated_moe_predictions.json"))
OUTPUT_METRICS = os.path.join(RESULTS_DIR, tagged_filename("federated_moe_metrics.json"))
LLM_SEMANTIC_CACHE = os.path.join(RESULTS_DIR, "llm_semantic_context_features.json")


class SharedFederatedMoE(nn.Module):
    """Server-aggregated context encoder and experts."""

    def __init__(
        self,
        vocab,
        semantic_input_dim,
        n_experts=N_EXPERTS,
        emb_dim=EMB_DIM,
        semantic_dim=SEMANTIC_DIM,
        expert_hidden=EXPERT_HIDDEN,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.encoder = ContextEncoder(vocab, emb_dim=emb_dim)
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_input_dim, semantic_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.context_dim = self.encoder.out_dim + semantic_dim
        self.experts = nn.ModuleList([
            Expert(self.context_dim, expert_hidden, dropout)
            for _ in range(n_experts)
        ])

    def forward(
        self,
        query_idx,
        receiver_idx,
        datatype_idx,
        domain_idx,
        semantic_features,
    ):
        categorical_z = self.encoder(query_idx, receiver_idx, datatype_idx, domain_idx)
        semantic_z = self.semantic_proj(semantic_features)
        z = torch.cat([categorical_z, semantic_z], dim=-1)
        expert_probs = torch.stack(
            [torch.sigmoid(expert(z)) for expert in self.experts],
            dim=-1,
        )
        return z, expert_probs


class PersonalizedGate(nn.Module):
    """Client-private gate that maps local context/profile to expert weights."""

    def __init__(
        self,
        context_dim,
        user_profile_dim,
        n_experts=N_EXPERTS,
        top_k=TOP_K,
        gate_hidden=GATE_HIDDEN,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.net = nn.Sequential(
            nn.Linear(context_dim + user_profile_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, n_experts),
        )

    def forward(self, z, user_profile, ldp_mode="none", ldp_epsilon=8.0):
        gate_in = torch.cat([z, user_profile], dim=-1)
        if ldp_mode == "gating_input":
            scale = 1.0 / max(float(ldp_epsilon), 1e-6)
            gate_noise = torch.distributions.Laplace(
                loc=torch.zeros_like(gate_in),
                scale=torch.full_like(gate_in, scale),
            ).sample()
            gate_in = gate_in + gate_noise
        gate_weights = torch.softmax(self.net(gate_in), dim=-1)

        if self.top_k is not None and self.top_k < self.n_experts:
            top_vals, top_idx = torch.topk(gate_weights, self.top_k, dim=-1)
            sparse = torch.zeros_like(gate_weights)
            sparse.scatter_(1, top_idx, top_vals)
            denom = sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            gate_weights = sparse / denom

        return gate_weights


def to_tensor_dict(arrays, user_profiles, device):
    return {
        "query_idx": torch.from_numpy(arrays["query_idx"]).to(device),
        "receiver_idx": torch.from_numpy(arrays["receiver_idx"]).to(device),
        "datatype_idx": torch.from_numpy(arrays["datatype_idx"]).to(device),
        "domain_idx": torch.from_numpy(arrays["domain_idx"]).to(device),
        "semantic_features": torch.from_numpy(arrays["semantic_features"]).to(device),
        "user_profile_idx": torch.from_numpy(arrays["user_profile_idx"]).to(device),
        "labels": torch.from_numpy(arrays["labels"]).to(device),
        "user_profiles": torch.from_numpy(user_profiles).to(device),
    }


def build_user_index_map(arrays):
    """Map each user_profile_idx to local sample indices."""
    user_to_indices = {}
    for i, user_idx in enumerate(arrays["user_profile_idx"]):
        user_to_indices.setdefault(int(user_idx), []).append(i)
    return {
        user_idx: np.array(indices, dtype=np.int64)
        for user_idx, indices in user_to_indices.items()
    }


def clone_gate(gate):
    cloned = copy.deepcopy(gate)
    cloned.load_state_dict(copy.deepcopy(gate.state_dict()))
    return cloned


def combine_predictions(expert_probs, gate_weights):
    return (gate_weights * expert_probs).sum(dim=-1).clamp(1e-6, 1.0 - 1e-6)


def apply_user_stats_ldp(user_profiles, epsilon):
    """Add LDP-style Laplace noise to user profile statistics (range [0, 1])."""
    epsilon = max(float(epsilon), 1e-6)
    noisy = np.asarray(user_profiles, dtype=np.float32).copy()
    scale = 1.0 / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=noisy.shape).astype(np.float32)
    noisy = np.clip(noisy + noise, 0.0, 1.0)
    return noisy


def semantic_text_for_record(rec, queries_by_id):
    query = queries_by_id.get(int(rec["query_id"]), {})
    parts = [
        query.get("query", ""),
        rec.get("domain", ""),
        rec.get("receiver", ""),
        rec.get("generic_data_type", ""),
        rec.get("data_type", ""),
        rec.get("permission_key", ""),
        query.get("tool1", ""),
        query.get("tool2", ""),
        query.get("data_tool1", ""),
        query.get("data_tool2", ""),
        query.get("data_MIRA", ""),
    ]
    return " ".join(str(part) for part in parts if part)


def semantic_context_key(rec):
    return f"{int(rec['query_id'])}::{rec['permission_key']}"


def semantic_context_payload(rec, queries_by_id):
    query = queries_by_id.get(int(rec["query_id"]), {})
    return {
        "query_id": int(rec["query_id"]),
        "query": query.get("query", ""),
        "domain": rec.get("domain", ""),
        "receiver": rec.get("receiver", ""),
        "generic_data_type": rec.get("generic_data_type", ""),
        "data_type": rec.get("data_type", ""),
        "permission_key": rec.get("permission_key", ""),
        "tool1": query.get("tool1", ""),
        "tool2": query.get("tool2", ""),
    }


def load_llm_semantic_cache():
    if not os.path.exists(LLM_SEMANTIC_CACHE):
        return {}
    with open(LLM_SEMANTIC_CACHE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_llm_semantic_cache(cache):
    os.makedirs(os.path.dirname(LLM_SEMANTIC_CACHE), exist_ok=True)
    with open(LLM_SEMANTIC_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def parse_json_array(text):
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response did not contain a JSON array")
    return json.loads(text[start:end + 1])


def score_semantic_contexts_with_llm(contexts):
    if not USE_LLM_SEMANTIC:
        return {}

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  [semantic] OPENAI_API_KEY not found; using TF-IDF features only")
        return {}

    client_kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    model = os.getenv("OPENAI_MODEL", "o4-mini")

    cache = load_llm_semantic_cache()
    missing = [
        (key, payload)
        for key, payload in contexts.items()
        if key not in cache
    ]
    if not missing:
        print(f"  LLM semantic cache: {len(cache)} contexts")
        return cache

    print(
        f"  LLM semantic cache: {len(cache)} cached, "
        f"{len(missing)} to score with {model}"
    )
    system_prompt = (
        "You create non-user-specific semantic features for AI-agent permission "
        "prediction. Do not infer any particular user's preference. Score each "
        "query-permission context using only task semantics."
    )
    feature_names = [
        "data_necessity",
        "privacy_sensitivity",
        "receiver_relevance",
        "contextual_fit",
        "general_allow_likelihood",
    ]

    for start in range(0, len(missing), LLM_SEMANTIC_BATCH_SIZE):
        batch = missing[start:start + LLM_SEMANTIC_BATCH_SIZE]
        items = [
            {"id": key, **payload}
            for key, payload in batch
        ]
        user_prompt = (
            "Return ONLY a JSON array. For each item, output keys: id, "
            + ", ".join(feature_names)
            + ". Scores must be floats in [0,1].\n\n"
            + json.dumps(items, ensure_ascii=False)
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            rows = parse_json_array(content)
            for row in rows:
                key = str(row.get("id", ""))
                if key not in contexts:
                    continue
                cache[key] = [
                    float(np.clip(float(row.get(name, 0.5)), 0.0, 1.0))
                    for name in feature_names
                ]
            save_llm_semantic_cache(cache)
            print(f"    scored {min(start + len(batch), len(missing))}/{len(missing)}")
        except Exception as exc:
            print(f"  [semantic] LLM scoring failed: {exc}")
            print("  [semantic] Continuing with cached/TF-IDF features")
            break

    return cache


def build_llm_semantic_matrix(records, cache):
    features = []
    for rec in records:
        key = semantic_context_key(rec)
        features.append(cache.get(key, [0.5, 0.5, 0.5, 0.5, 0.5]))
    return np.array(features, dtype=np.float32)


def build_semantic_feature_matrices(train_records, test_records):
    """Vectorize non-private request semantics shared by all clients."""
    queries_by_id = load_queries()
    train_texts = [
        semantic_text_for_record(rec, queries_by_id)
        for rec in train_records
    ]
    test_texts = [
        semantic_text_for_record(rec, queries_by_id)
        for rec in test_records
    ]
    vectorizer = TfidfVectorizer(
        max_features=SEMANTIC_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=1,
        norm="l2",
    )
    train_features = vectorizer.fit_transform(train_texts).toarray().astype(np.float32)
    test_features = vectorizer.transform(test_texts).toarray().astype(np.float32)
    contexts = {}
    for rec in train_records + test_records:
        contexts[semantic_context_key(rec)] = semantic_context_payload(
            rec,
            queries_by_id,
        )
    llm_cache = score_semantic_contexts_with_llm(contexts)
    train_llm = build_llm_semantic_matrix(train_records, llm_cache)
    test_llm = build_llm_semantic_matrix(test_records, llm_cache)
    train_features = np.concatenate([train_features, train_llm], axis=1)
    test_features = np.concatenate([test_features, test_llm], axis=1)
    return train_features, test_features, vectorizer


def attach_semantic_features(arrays, semantic_features):
    arrays = dict(arrays)
    arrays["semantic_features"] = semantic_features
    return arrays


def local_train_user(
    server_shared,
    gate,
    tensors,
    sample_indices,
    device,
    rng,
):
    local_shared = copy.deepcopy(server_shared).to(device)
    local_gate = clone_gate(gate).to(device)
    local_shared.train()
    local_gate.train()

    params = list(local_shared.parameters()) + list(local_gate.parameters())
    optimizer = optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCELoss()
    sample_indices = np.asarray(sample_indices, dtype=np.int64)

    total_bce, total_lb, n_batches = 0.0, 0.0, 0
    for _ in range(LOCAL_EPOCHS):
        for batch_idx in make_batches(len(sample_indices), LOCAL_BATCH_SIZE, rng):
            idx_t = torch.from_numpy(sample_indices[batch_idx]).to(device)
            user_prof = tensors["user_profiles"][tensors["user_profile_idx"][idx_t]]

            z, expert_probs = local_shared(
                tensors["query_idx"][idx_t],
                tensors["receiver_idx"][idx_t],
                tensors["datatype_idx"][idx_t],
                tensors["domain_idx"][idx_t],
                tensors["semantic_features"][idx_t],
            )
            gate_w = local_gate(
                z,
                user_prof,
                ldp_mode=LDP_MODE,
                ldp_epsilon=LDP_EPSILON,
            )
            p_allow = combine_predictions(expert_probs, gate_w)
            bce = loss_fn(p_allow, tensors["labels"][idx_t])
            lb = load_balance_loss(gate_w)
            loss = bce + LOAD_BALANCE_COEF * lb

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_bce += float(bce.item())
            total_lb += float(lb.item())
            n_batches += 1

    return (
        copy.deepcopy(local_shared.state_dict()),
        copy.deepcopy(local_gate.state_dict()),
        int(len(sample_indices)),
        total_bce / max(n_batches, 1),
        total_lb / max(n_batches, 1),
    )


def fedavg_state_dicts(weighted_states):
    """Weighted average of model state dicts by local sample count."""
    total_weight = float(sum(weight for _, weight in weighted_states))
    aggregated = {}

    for key in weighted_states[0][0]:
        first_value = weighted_states[0][0][key]
        if not torch.is_floating_point(first_value):
            aggregated[key] = first_value.clone()
            continue

        avg = torch.zeros_like(first_value)
        for state, weight in weighted_states:
            avg += state[key] * (float(weight) / total_weight)
        aggregated[key] = avg

    return aggregated


def init_user_gates(user_profiles, context_dim, device):
    gates = {}
    user_profile_dim = int(user_profiles.shape[1])
    for user_idx in range(int(user_profiles.shape[0])):
        torch.manual_seed(SEED + user_idx)
        gates[user_idx] = PersonalizedGate(
            context_dim=context_dim,
            user_profile_dim=user_profile_dim,
        ).to(device)
    return gates


def build_local_prior_model(train_records):
    """Build local empirical priors from training history only."""
    global_rate = float(np.mean([r["label"] for r in train_records]))

    def build_counts(key_fn):
        counts = {}
        for rec in train_records:
            key = key_fn(rec)
            n, positives = counts.get(key, (0, 0))
            counts[key] = (n + 1, positives + int(rec["label"]))
        return counts

    return {
        "global_rate": global_rate,
        "user": build_counts(lambda r: r["user_id"]),
        "domain": build_counts(lambda r: (r["user_id"], r["domain"])),
        "datatype": build_counts(lambda r: (r["user_id"], r["generic_data_type"])),
        "permission": build_counts(lambda r: (r["user_id"], r["permission_key"])),
    }


def smoothed_rate(counts, key, fallback):
    if key not in counts:
        return fallback
    n, positives = counts[key]
    return float((positives + LOCAL_PRIOR_ALPHA * fallback) / (n + LOCAL_PRIOR_ALPHA))


def local_prior_probability(prior_model, rec):
    user_id = rec["user_id"]
    p_user = smoothed_rate(
        prior_model["user"],
        user_id,
        prior_model["global_rate"],
    )
    p_domain = smoothed_rate(
        prior_model["domain"],
        (user_id, rec["domain"]),
        p_user,
    )
    p_datatype = smoothed_rate(
        prior_model["datatype"],
        (user_id, rec["generic_data_type"]),
        p_domain,
    )
    return smoothed_rate(
        prior_model["permission"],
        (user_id, rec["permission_key"]),
        p_datatype,
    )


def apply_local_prior_calibration(probs, test_records, prior_model):
    calibrated = np.asarray(probs, dtype=np.float32).copy()
    used_prior = np.zeros(len(test_records), dtype=bool)

    for i, rec in enumerate(test_records):
        prior_p = local_prior_probability(prior_model, rec)
        if LDP_MODE == "user_stats":
            prior_p = float(
                np.clip(
                    prior_p + np.random.laplace(loc=0.0, scale=1.0 / max(LDP_EPSILON, 1e-6)),
                    0.0,
                    1.0,
                )
            )
        prior_conf = max(prior_p, 1.0 - prior_p)
        if prior_conf >= LOCAL_PRIOR_CONF_THRESHOLD:
            calibrated[i] = prior_p
            used_prior[i] = True

    return calibrated, used_prior


def train_federated_moe(train_arrays, user_profiles, vocab, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    server_shared = SharedFederatedMoE(
        vocab=vocab,
        semantic_input_dim=int(train_arrays["semantic_features"].shape[1]),
    ).to(device)
    user_gates = init_user_gates(
        user_profiles,
        server_shared.context_dim,
        device,
    )
    tensors = to_tensor_dict(train_arrays, user_profiles, device)
    user_to_indices = build_user_index_map(train_arrays)
    train_user_ids = sorted(user_to_indices)
    clients_per_round = max(1, int(np.ceil(len(train_user_ids) * CLIENT_FRACTION)))

    print(f"  Training samples : {len(train_arrays['labels'])}")
    print(f"  Users            : {len(train_user_ids)}")
    print(f"  Experts          : {N_EXPERTS}  (top_k = {TOP_K})")
    print(
        f"  Semantic features: {train_arrays['semantic_features'].shape[1]} "
        f"-> {SEMANTIC_DIM}"
    )
    print(f"  Fed rounds       : {FED_ROUNDS}")
    print(f"  Clients/round    : {clients_per_round}")
    print(f"  Local epochs     : {LOCAL_EPOCHS}")
    print(f"  Local batch size : {LOCAL_BATCH_SIZE}")
    print(f"  LB coef          : {LOAD_BALANCE_COEF}")

    for round_idx in range(1, FED_ROUNDS + 1):
        selected_users = rng.choice(
            train_user_ids,
            size=clients_per_round,
            replace=False,
        )
        weighted_states = []
        round_bce, round_lb = [], []

        for user_idx in selected_users:
            user_idx = int(user_idx)
            shared_state, gate_state, weight, avg_bce, avg_lb = local_train_user(
                server_shared=server_shared,
                gate=user_gates[user_idx],
                tensors=tensors,
                sample_indices=user_to_indices[user_idx],
                device=device,
                rng=rng,
            )
            weighted_states.append((shared_state, weight))
            user_gates[user_idx].load_state_dict(gate_state)
            round_bce.append(avg_bce)
            round_lb.append(avg_lb)

        server_shared.load_state_dict(fedavg_state_dicts(weighted_states))

        if round_idx == 1 or round_idx % 5 == 0 or round_idx == FED_ROUNDS:
            print(
                f"    round {round_idx:3d}/{FED_ROUNDS}  "
                f"bce={np.mean(round_bce):.4f}  lb={np.mean(round_lb):.4f}"
            )

    return server_shared, user_gates


@torch.no_grad()
def predict_federated_moe(server_shared, user_gates, arrays, user_profiles, device):
    server_shared.eval()
    for gate in user_gates.values():
        gate.eval()

    tensors = to_tensor_dict(arrays, user_profiles, device)
    n_samples = int(len(arrays["labels"]))
    probs = np.zeros(n_samples, dtype=np.float32)
    gate_weights = np.zeros((n_samples, N_EXPERTS), dtype=np.float32)

    user_to_indices = build_user_index_map(arrays)
    for user_idx, sample_indices in user_to_indices.items():
        idx_t = torch.from_numpy(sample_indices).to(device)
        user_prof = tensors["user_profiles"][tensors["user_profile_idx"][idx_t]]
        z, expert_probs = server_shared(
            tensors["query_idx"][idx_t],
            tensors["receiver_idx"][idx_t],
            tensors["datatype_idx"][idx_t],
            tensors["domain_idx"][idx_t],
            tensors["semantic_features"][idx_t],
        )
        gate_w = user_gates[int(user_idx)](
            z,
            user_prof,
            ldp_mode=LDP_MODE,
            ldp_epsilon=LDP_EPSILON,
        )
        p_allow = combine_predictions(expert_probs, gate_w)
        probs[sample_indices] = p_allow.cpu().numpy()
        gate_weights[sample_indices] = gate_w.cpu().numpy()

    return probs, gate_weights


def build_prediction_records(arrays, probs, gate_weights, used_local_prior=None):
    predictions = []
    expert_assignments = gate_weights.argmax(axis=-1)
    if used_local_prior is None:
        used_local_prior = np.zeros(len(probs), dtype=bool)

    for i, p in enumerate(probs):
        p_allow = float(p)
        pred = int(p_allow >= 0.5)
        conf = max(p_allow, 1.0 - p_allow)
        predictions.append({
            "participant_id": arrays["user_ids"][i],
            "query_id": int(arrays["query_ids"][i]),
            "permission_key": arrays["permission_keys"][i],
            "y_true": int(arrays["labels"][i]),
            "p_allow": p_allow,
            "pred": pred,
            "confidence": float(conf),
            "top_expert": int(expert_assignments[i]),
            "gate_weights": [float(w) for w in gate_weights[i]],
            "used_local_prior": bool(used_local_prior[i]),
        })

    return predictions


def main():
    print("=" * 60)
    print("RQ Baseline: Personalized Federated MoE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"LDP mode: {LDP_MODE}  (epsilon={LDP_EPSILON})")
    if OUTPUT_TAG:
        print(f"Output tag: {OUTPUT_TAG}")

    print("\nLoading data...")
    train_records, test_records, bio_features = build_records()
    print(f"  Train records : {len(train_records)}")
    print(f"  Test  records : {len(test_records)}")

    vocab = build_vocab(train_records)
    user_profiles, user_to_idx = build_user_profiles(
        train_records, bio_features, vocab
    )
    if LDP_MODE == "user_stats":
        print("Applying LDP noise to user profile statistics...")
        user_profiles = apply_user_stats_ldp(user_profiles, LDP_EPSILON)
    print(f"  Users         : {len(user_to_idx)}")
    print(f"  Profile dim   : {user_profiles.shape[1]}")

    train_arrays = records_to_arrays(train_records, vocab, user_to_idx)
    test_arrays = records_to_arrays(test_records, vocab, user_to_idx)

    print("\nBuilding non-private semantic context features...")
    train_semantic, test_semantic, vectorizer = build_semantic_feature_matrices(
        train_records,
        test_records,
    )
    train_arrays = attach_semantic_features(train_arrays, train_semantic)
    test_arrays = attach_semantic_features(test_arrays, test_semantic)
    print(f"  TF-IDF vocab    : {len(vectorizer.vocabulary_)}")

    print("\nTraining personalized federated MoE...")
    server_shared, user_gates = train_federated_moe(
        train_arrays, user_profiles, vocab, device
    )

    print("\nPredicting on test set with personalized gates...")
    probs, gate_weights = predict_federated_moe(
        server_shared, user_gates, test_arrays, user_profiles, device
    )

    print("\nApplying local-history prior calibration...")
    np.random.seed(SEED)
    prior_model = build_local_prior_model(train_records)
    probs, used_local_prior = apply_local_prior_calibration(
        probs,
        test_records,
        prior_model,
    )
    predictions = build_prediction_records(
        test_arrays,
        probs,
        gate_weights,
        used_local_prior=used_local_prior,
    )

    expert_assignments = gate_weights.argmax(axis=-1)
    expert_usage = {
        int(k): int((expert_assignments == k).sum()) for k in range(N_EXPERTS)
    }

    metrics = compute_full_metrics(
        predictions,
        method_name="Personalized Federated MoE (RQ1, No LDP)",
    )
    metrics["n_experts"] = N_EXPERTS
    metrics["top_k"] = TOP_K
    metrics["expert_usage"] = expert_usage
    metrics["federated"] = {
        "rounds": FED_ROUNDS,
        "client_fraction": CLIENT_FRACTION,
        "clients_per_round": max(
            1,
            int(np.ceil(len(build_user_index_map(train_arrays)) * CLIENT_FRACTION)),
        ),
        "local_epochs": LOCAL_EPOCHS,
        "aggregated_parameters": "ContextEncoder + Experts",
        "semantic_context": "TF-IDF(query/tool/domain/permission text)",
        "personalized_parameters": "Per-user gate",
        "ldp": LDP_MODE != "none",
    }
    metrics["ldp"] = {
        "mode": LDP_MODE,
        "epsilon": float(LDP_EPSILON),
        "noise_positions": {
            "user_statistics": bool(LDP_MODE == "user_stats"),
            "gating_input": bool(LDP_MODE == "gating_input"),
        },
    }
    metrics["local_prior_calibration"] = {
        "alpha": LOCAL_PRIOR_ALPHA,
        "confidence_threshold": LOCAL_PRIOR_CONF_THRESHOLD,
        "used_for_predictions": int(used_local_prior.sum()),
        "coverage": float(used_local_prior.mean()),
    }
    metrics["hyperparameters"] = {
        "emb_dim": EMB_DIM,
        "semantic_max_features": SEMANTIC_MAX_FEATURES,
        "semantic_dim": SEMANTIC_DIM,
        "expert_hidden": EXPERT_HIDDEN,
        "gate_hidden": GATE_HIDDEN,
        "local_batch_size": LOCAL_BATCH_SIZE,
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
