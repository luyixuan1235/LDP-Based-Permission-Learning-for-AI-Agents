"""
Offline LDP simulation for permission statistics.

This script simulates a setting where the server only observes locally
perturbed binary permission decisions, aggregates them into global priors,
and then uses those priors for simple global-only permission inference.

Data sources:
    - ../data/user_study.json
    - ../data/processed_dataset.json
    - ../data/data_types.csv
    - ../queries.json

Outputs:
    - ../results/ldp_offline_summary.json
    - ../results/ldp_offline_trials.csv
    - ../results/ldp_offline_item_priors.csv
"""

import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluation_utils import calculate_metrics, print_metrics


USER_STUDY_PATH = "../data/user_study.json"
PROCESSED_DATASET_PATH = "../data/processed_dataset.json"
DATA_TYPES_PATH = "../data/data_types.csv"
QUERIES_PATH = "../queries.json"
RESULTS_DIR = "../results"

DEFAULT_EPSILONS = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0]
DEFAULT_TRIALS = 20
DEFAULT_SEED = 42
DEFAULT_THRESHOLD = 0.5
TOP_K_VALUES = (5, 10)

ALLOWED_LABELS = {
    "Yes, always share": 1,
    "No, never share": 0,
}

RECEIVER_ALIASES = {
    "mira": "LLM",
    "llm": "LLM",
}

GRANULARITY_COLUMNS = {
    "fine": "key_fine",
    "medium": "key_medium",
    "coarse": "key_coarse",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run offline LDP simulation for permission statistics."
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=DEFAULT_EPSILONS,
        help="Privacy budgets to evaluate.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Number of Monte Carlo trials per epsilon.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Threshold used by GlobalOnly inference.",
    )
    return parser.parse_args()


def normalize_text(value: Optional[str]) -> str:
    """Normalize receivers and data types for consistent matching."""
    if value is None:
        return ""

    text = str(value).replace("\u2019", "'").strip()
    text = re.sub(r"\s*\(.*$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.;:")


def canonical_token(value: Optional[str]) -> str:
    """Convert a value to a canonical lookup token."""
    return normalize_text(value).lower()


def normalize_receiver(receiver: Optional[str]) -> str:
    """Normalize receiver aliases such as mira -> LLM."""
    cleaned = normalize_text(receiver)
    return RECEIVER_ALIASES.get(cleaned.lower(), cleaned)


def load_queries() -> Dict[int, dict]:
    """Load query metadata keyed by query id."""
    with open(QUERIES_PATH, "r", encoding="utf-8") as file:
        queries = json.load(file)
    return {int(query["id"]): query for query in queries}


def load_generic_type_map() -> Dict[str, str]:
    """Load raw-to-generic data type mappings."""
    mapping: Dict[str, str] = {}
    df = pd.read_csv(DATA_TYPES_PATH)

    for _, row in df.iterrows():
        generic_type = str(row["Generic Data Type"]).strip()
        raw_types = str(row["Raw Data Type"]).split(",")
        for raw_type in raw_types:
            mapping[canonical_token(raw_type)] = generic_type

    return mapping


def parse_permission_key(permission_key: str) -> Tuple[str, str]:
    """Split a permission key into normalized receiver and data type."""
    parts = permission_key.split(",", 1)
    if len(parts) == 2:
        receiver, data_type = parts
    else:
        receiver, data_type = "Unknown", permission_key

    return normalize_receiver(receiver), normalize_text(data_type)


def build_record(
    user_id: str,
    query_id: int,
    permission_key: str,
    label: str,
    queries_by_id: Dict[int, dict],
    generic_type_map: Dict[str, str],
) -> Optional[dict]:
    """Convert one permission response into a structured record."""
    binary_label = ALLOWED_LABELS.get(label)
    if binary_label is None:
        return None

    receiver, data_type = parse_permission_key(permission_key)
    query_info = queries_by_id.get(int(query_id), {})
    domain = query_info.get("domain", "Unknown")
    generic_data_type = generic_type_map.get(canonical_token(data_type), data_type)

    return {
        "user_id": user_id,
        "query_id": int(query_id),
        "receiver": receiver,
        "data_type": data_type,
        "generic_data_type": generic_data_type,
        "domain": domain,
        "label": int(binary_label),
        "key_fine": f"{int(query_id)}:::{receiver}:::{data_type}",
        "key_medium": f"{domain}:::{receiver}:::{generic_data_type}",
        "key_coarse": f"{domain}:::{generic_data_type}",
    }


def build_dataframe_from_examples(
    user_id: str,
    examples: Iterable[dict],
    queries_by_id: Dict[int, dict],
    generic_type_map: Dict[str, str],
) -> List[dict]:
    """Convert example lists into flat response records."""
    records: List[dict] = []

    for example in examples:
        query_id = int(example["id"])
        answers = example.get("answer", {})
        for permission_key, label in answers.items():
            record = build_record(
                user_id=user_id,
                query_id=query_id,
                permission_key=permission_key,
                label=label,
                queries_by_id=queries_by_id,
                generic_type_map=generic_type_map,
            )
            if record is not None:
                records.append(record)

    return records


def load_user_study_dataframe(
    processed_participant_ids: set,
    queries_by_id: Dict[int, dict],
    generic_type_map: Dict[str, str],
) -> pd.DataFrame:
    """Load filtered binary responses from the original user study file."""
    with open(USER_STUDY_PATH, "r", encoding="utf-8") as file:
        user_study = json.load(file)

    records: List[dict] = []
    for participant in user_study:
        user_id = participant["uuid"]
        if user_id not in processed_participant_ids:
            continue
        records.extend(
            build_dataframe_from_examples(
                user_id=user_id,
                examples=participant.get("task3", []),
                queries_by_id=queries_by_id,
                generic_type_map=generic_type_map,
            )
        )

    return pd.DataFrame(records)


def load_processed_split_dataframes(
    queries_by_id: Dict[int, dict],
    generic_type_map: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, set]:
    """Load processed training/testing splits."""
    with open(PROCESSED_DATASET_PATH, "r", encoding="utf-8") as file:
        processed_dataset = json.load(file)

    train_records: List[dict] = []
    test_records: List[dict] = []

    for user_id, participant_data in processed_dataset.items():
        train_records.extend(
            build_dataframe_from_examples(
                user_id=user_id,
                examples=participant_data.get("training", []),
                queries_by_id=queries_by_id,
                generic_type_map=generic_type_map,
            )
        )
        test_records.extend(
            build_dataframe_from_examples(
                user_id=user_id,
                examples=participant_data.get("testing", []),
                queries_by_id=queries_by_id,
                generic_type_map=generic_type_map,
            )
        )

    return (
        pd.DataFrame(train_records),
        pd.DataFrame(test_records),
        set(processed_dataset.keys()),
    )


def randomized_response(bits: np.ndarray, epsilon: float, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """Apply binary randomized response."""
    prob_truthful = np.exp(epsilon) / (np.exp(epsilon) + 1.0)
    keep_mask = rng.random(bits.shape[0]) < prob_truthful
    noisy_bits = np.where(keep_mask, bits, 1 - bits).astype(int)
    return noisy_bits, prob_truthful


def debias_probability(observed_rate: float, prob_truthful: float) -> float:
    """Debias a noisy Bernoulli rate estimate under randomized response."""
    denominator = (2.0 * prob_truthful) - 1.0
    if denominator == 0:
        return 0.5
    estimated = (observed_rate - (1.0 - prob_truthful)) / denominator
    return float(np.clip(estimated, 0.0, 1.0))


def estimate_group_rates(keys: pd.Series, labels: np.ndarray) -> Tuple[pd.Series, pd.Series]:
    """Estimate per-group mean rate and group counts."""
    frame = pd.DataFrame({"key": keys.to_numpy(), "label": labels})
    grouped = frame.groupby("key", sort=False)["label"]
    rates = grouped.mean()
    counts = grouped.size()
    return rates, counts


def apply_debiasing(noisy_rates: pd.Series, prob_truthful: float) -> pd.Series:
    """Debias a series of noisy rates."""
    debiased = (noisy_rates - (1.0 - prob_truthful)) / ((2.0 * prob_truthful) - 1.0)
    return debiased.clip(0.0, 1.0)


def top_k_jaccard(true_rates: pd.Series, estimated_rates: pd.Series, k: int) -> float:
    """Compute Jaccard similarity between top-k item sets."""
    if true_rates.empty:
        return 1.0

    k = min(k, len(true_rates))
    true_top = set(true_rates.nlargest(k).index)
    estimated_top = set(estimated_rates.nlargest(k).index)
    union = true_top | estimated_top
    if not union:
        return 1.0
    return float(len(true_top & estimated_top) / len(union))


def safe_spearman(true_rates: pd.Series, estimated_rates: pd.Series) -> float:
    """Compute Spearman correlation while handling degenerate cases."""
    if len(true_rates) <= 1:
        return 1.0

    correlation = true_rates.corr(estimated_rates, method="spearman")
    if pd.isna(correlation):
        if np.allclose(true_rates.to_numpy(), estimated_rates.to_numpy()):
            return 1.0
        return 0.0
    return float(correlation)


def exact_task_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    granularity: str,
    threshold: float,
) -> dict:
    """Evaluate a non-private exact global prior baseline."""
    key_column = GRANULARITY_COLUMNS[granularity]
    exact_rates, _ = estimate_group_rates(train_df[key_column], train_df["label"].to_numpy())
    fallback_rate = float(train_df["label"].mean())

    y_true = test_df["label"].to_numpy()
    y_score = test_df[key_column].map(exact_rates).fillna(fallback_rate).to_numpy()
    y_pred = (y_score >= threshold).astype(int)

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        method_name=f"GlobalOnly Exact ({granularity})",
        threshold=threshold,
    )
    metrics["granularity"] = granularity
    metrics["mechanism"] = "exact"
    metrics["epsilon"] = None
    metrics["train_groups"] = int(len(exact_rates))
    metrics["fallback_rate"] = fallback_rate
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def noisy_task_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    granularity: str,
    epsilon: float,
    rng: np.random.Generator,
    threshold: float,
) -> Tuple[dict, pd.Series]:
    """Evaluate GlobalOnly inference with noisy aggregated priors."""
    key_column = GRANULARITY_COLUMNS[granularity]
    train_labels = train_df["label"].to_numpy()
    noisy_bits, prob_truthful = randomized_response(train_labels, epsilon, rng)
    noisy_rates, _ = estimate_group_rates(train_df[key_column], noisy_bits)
    estimated_rates = apply_debiasing(noisy_rates, prob_truthful)

    overall_noisy_rate = float(noisy_bits.mean())
    fallback_rate = debias_probability(overall_noisy_rate, prob_truthful)

    y_true = test_df["label"].to_numpy()
    y_score = test_df[key_column].map(estimated_rates).fillna(fallback_rate).to_numpy()
    y_pred = (y_score >= threshold).astype(int)

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        method_name=f"GlobalOnly RR ({granularity}, eps={epsilon})",
        threshold=threshold,
    )
    metrics["granularity"] = granularity
    metrics["mechanism"] = "randomized_response"
    metrics["epsilon"] = float(epsilon)
    metrics["train_groups"] = int(len(estimated_rates))
    metrics["fallback_rate"] = fallback_rate
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    return metrics, estimated_rates


def summarize_numeric_columns(df: pd.DataFrame, group_columns: List[str], numeric_columns: List[str]) -> List[dict]:
    """Aggregate trial-level numeric results into mean/std summaries."""
    summaries: List[dict] = []
    grouped = df.groupby(group_columns, dropna=False, sort=False)

    for group_values, group_df in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        summary = {
            column: value
            for column, value in zip(group_columns, group_values)
        }
        summary["n_trials"] = int(len(group_df))

        for column in numeric_columns:
            if column not in group_df:
                continue
            values = pd.to_numeric(group_df[column], errors="coerce")
            summary[f"{column}_mean"] = float(values.mean())
            summary[f"{column}_std"] = float(values.std(ddof=0))

        summaries.append(summary)

    return summaries


def make_json_safe(value):
    """Recursively replace NaN values with JSON-friendly nulls."""
    if isinstance(value, dict):
        return {key: make_json_safe(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def main() -> None:
    """Run the offline simulation end-to-end."""
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("Offline LDP Simulation for AI-Agent Permission Statistics")
    print("=" * 72)
    print(f"Epsilons: {args.epsilons}")
    print(f"Trials:   {args.trials}")
    print(f"Seed:     {args.seed}")
    print(f"Threshold:{args.threshold}")

    queries_by_id = load_queries()
    generic_type_map = load_generic_type_map()
    train_df, test_df, processed_participant_ids = load_processed_split_dataframes(
        queries_by_id=queries_by_id,
        generic_type_map=generic_type_map,
    )
    full_df = load_user_study_dataframe(
        processed_participant_ids=processed_participant_ids,
        queries_by_id=queries_by_id,
        generic_type_map=generic_type_map,
    )

    print("\nLoaded data:")
    print(f"  Filtered participants: {len(processed_participant_ids)}")
    print(f"  Full binary responses (user_study): {len(full_df)}")
    print(f"  Processed train responses:          {len(train_df)}")
    print(f"  Processed test responses:           {len(test_df)}")

    trial_rows: List[dict] = []
    item_prior_rows: List[dict] = []

    full_true_rates: Dict[str, pd.Series] = {}
    full_counts: Dict[str, pd.Series] = {}

    for granularity, key_column in GRANULARITY_COLUMNS.items():
        true_rates, counts = estimate_group_rates(full_df[key_column], full_df["label"].to_numpy())
        full_true_rates[granularity] = true_rates.sort_index()
        full_counts[granularity] = counts.reindex(full_true_rates[granularity].index)

    for granularity in GRANULARITY_COLUMNS:
        exact_metrics = exact_task_metrics(
            train_df=train_df,
            test_df=test_df,
            granularity=granularity,
            threshold=args.threshold,
        )
        exact_stats_row = {
            "granularity": granularity,
            "mechanism": "exact",
            "epsilon": np.nan,
            "trial": 0,
            "stats_mae": 0.0,
            "stats_rmse": 0.0,
            "stats_weighted_mae": 0.0,
            "stats_spearman": 1.0,
            "stats_top5_jaccard": 1.0,
            "stats_top10_jaccard": 1.0,
            **exact_metrics,
        }
        trial_rows.append(exact_stats_row)

        for key, true_rate in full_true_rates[granularity].items():
            item_prior_rows.append(
                {
                    "granularity": granularity,
                    "mechanism": "exact",
                    "epsilon": np.nan,
                    "key": key,
                    "n_reports": int(full_counts[granularity].loc[key]),
                    "true_allow_rate": float(true_rate),
                    "estimated_allow_rate_mean": float(true_rate),
                }
            )

    for granularity, key_column in GRANULARITY_COLUMNS.items():
        print(f"\nSimulating granularity: {granularity}")
        true_rates = full_true_rates[granularity]
        counts = full_counts[granularity]

        item_estimate_sums = {
            epsilon: pd.Series(0.0, index=true_rates.index, dtype=float)
            for epsilon in args.epsilons
        }

        for epsilon in args.epsilons:
            print(f"  epsilon={epsilon}")

            for trial in range(1, args.trials + 1):
                rng = np.random.default_rng(args.seed + (trial * 10_000) + int(epsilon * 1_000))

                noisy_bits, prob_truthful = randomized_response(
                    full_df["label"].to_numpy(),
                    epsilon,
                    rng,
                )
                noisy_rates, _ = estimate_group_rates(full_df[key_column], noisy_bits)
                estimated_rates = apply_debiasing(
                    noisy_rates.reindex(true_rates.index),
                    prob_truthful,
                )
                item_estimate_sums[epsilon] = item_estimate_sums[epsilon].add(estimated_rates, fill_value=0.0)

                errors = estimated_rates - true_rates
                stats_mae = float(errors.abs().mean())
                stats_rmse = float(np.sqrt(np.mean(np.square(errors.to_numpy()))))
                stats_weighted_mae = float(
                    np.average(np.abs(errors.to_numpy()), weights=counts.to_numpy())
                )
                stats_spearman = safe_spearman(true_rates, estimated_rates)
                stats_top5 = top_k_jaccard(true_rates, estimated_rates, TOP_K_VALUES[0])
                stats_top10 = top_k_jaccard(true_rates, estimated_rates, TOP_K_VALUES[1])

                task_metrics, _ = noisy_task_metrics(
                    train_df=train_df,
                    test_df=test_df,
                    granularity=granularity,
                    epsilon=epsilon,
                    rng=np.random.default_rng(args.seed + (trial * 20_000) + int(epsilon * 2_000)),
                    threshold=args.threshold,
                )

                trial_rows.append(
                    {
                        "granularity": granularity,
                        "mechanism": "randomized_response",
                        "epsilon": float(epsilon),
                        "trial": trial,
                        "stats_mae": stats_mae,
                        "stats_rmse": stats_rmse,
                        "stats_weighted_mae": stats_weighted_mae,
                        "stats_spearman": stats_spearman,
                        "stats_top5_jaccard": stats_top5,
                        "stats_top10_jaccard": stats_top10,
                        **task_metrics,
                    }
                )

            mean_estimates = item_estimate_sums[epsilon] / float(args.trials)
            for key, true_rate in true_rates.items():
                item_prior_rows.append(
                    {
                        "granularity": granularity,
                        "mechanism": "randomized_response",
                        "epsilon": float(epsilon),
                        "key": key,
                        "n_reports": int(counts.loc[key]),
                        "true_allow_rate": float(true_rate),
                        "estimated_allow_rate_mean": float(mean_estimates.loc[key]),
                    }
                )

    trials_df = pd.DataFrame(trial_rows)
    item_priors_df = pd.DataFrame(item_prior_rows)

    summary_rows = summarize_numeric_columns(
        df=trials_df,
        group_columns=["granularity", "mechanism", "epsilon"],
        numeric_columns=[
            "stats_mae",
            "stats_rmse",
            "stats_weighted_mae",
            "stats_spearman",
            "stats_top5_jaccard",
            "stats_top10_jaccard",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "fpr",
            "fnr",
            "auc",
        ],
    )

    summary = {
        "config": {
            "epsilons": [float(epsilon) for epsilon in args.epsilons],
            "trials": int(args.trials),
            "seed": int(args.seed),
            "threshold": float(args.threshold),
            "top_k_values": list(TOP_K_VALUES),
        },
        "dataset": {
            "n_filtered_participants": len(processed_participant_ids),
            "n_full_binary_responses": int(len(full_df)),
            "n_train_binary_responses": int(len(train_df)),
            "n_test_binary_responses": int(len(test_df)),
            "n_full_users": int(full_df["user_id"].nunique()),
            "n_train_users": int(train_df["user_id"].nunique()),
            "n_test_users": int(test_df["user_id"].nunique()),
        },
        "summary": summary_rows,
    }
    summary = make_json_safe(summary)

    summary_path = os.path.join(RESULTS_DIR, "ldp_offline_summary.json")
    trials_path = os.path.join(RESULTS_DIR, "ldp_offline_trials.csv")
    item_priors_path = os.path.join(RESULTS_DIR, "ldp_offline_item_priors.csv")

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    trials_df.to_csv(trials_path, index=False)
    item_priors_df.to_csv(item_priors_path, index=False)

    print(f"\nSaved summary to {summary_path}")
    print(f"Saved trial details to {trials_path}")
    print(f"Saved item priors to {item_priors_path}")

    print("\nExact GlobalOnly baselines:")
    for granularity in GRANULARITY_COLUMNS:
        row = trials_df[
            (trials_df["mechanism"] == "exact")
            & (trials_df["granularity"] == granularity)
        ].iloc[0].to_dict()
        print_metrics(
            {
                "method": f"GlobalOnly Exact ({granularity})",
                "accuracy": float(row["accuracy"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "fpr": float(row["fpr"]),
                "fnr": float(row["fnr"]),
                "n_predictions": int(len(test_df)),
                "threshold": float(args.threshold),
            }
        )


if __name__ == "__main__":
    main()
