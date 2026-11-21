"""Compare the AI4I aggregation schemes at a common training false-alarm budget."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.model_selection import train_test_split

from experiment import MLP, SCHEMES, SEEDS, evaluate, load_data, make_clients, predict_logits, train


TRAIN_FPR_BUDGET = 0.02
LEARNING_RATE = 0.01  # Selected by the original FedAvg training-loss pilot.


def choose_threshold(probabilities, labels):
    fpr, _, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    # Thresholds are descending. The last eligible one detects the most failures within budget.
    threshold = float(thresholds[fpr <= TRAIN_FPR_BUDGET][-1])
    train_fp = int(np.count_nonzero(probabilities[labels == 0] >= threshold))
    return threshold, train_fp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to the original ai4i2020.csv")
    args = parser.parse_args()
    torch.set_num_threads(1)
    features, labels = load_data(args.csv)
    train_indices, test_indices = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels.numpy(), random_state=42,
    )
    clients = make_clients(features, labels, train_indices)
    totals = np.array([len(y) for _, y in clients])
    positives = np.array([int(y.sum()) for _, y in clients])
    fedavg = totals / totals.sum()
    minority = positives / positives.sum()
    train_features = features[train_indices]
    train_labels = labels[train_indices].numpy()
    train_negatives = int(np.count_nonzero(train_labels == 0))

    models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        initial_model = MLP(features.shape[1])
        for scheme, layers in SCHEMES.items():
            model = train(deepcopy(initial_model), clients, fedavg, minority, layers, seed, LEARNING_RATE)
            scores = predict_logits(model, train_features).double().sigmoid().numpy()
            threshold, train_fp = choose_threshold(scores, train_labels)
            if train_fp / train_negatives > TRAIN_FPR_BUDGET:
                raise RuntimeError("Train false-alarm budget exceeded.")
            models.append((seed, scheme, model, threshold, train_fp))
        print(f"Finished training and selecting thresholds for seed {seed}", flush=True)

    test_features = features[test_indices]
    test_labels = labels[test_indices]
    rows = []
    for seed, scheme, model, threshold, train_fp in models:
        metrics = evaluate(model, test_features, test_labels, threshold)
        test_scores = predict_logits(model, test_features).numpy()
        rows.append({
            "seed": seed,
            "scheme": scheme,
            "learning_rate": LEARNING_RATE,
            "threshold": threshold,
            "train_fp": train_fp,
            "train_fpr": train_fp / train_negatives,
            "test_fpr": metrics["fp"] / (metrics["fp"] + metrics["tn"]),
            "average_precision": average_precision_score(test_labels.numpy(), test_scores),
            **metrics,
        })

    results = pd.DataFrame(rows)
    destination = Path(__file__).with_name("false_alarm_results.csv")
    results.to_csv(destination, index=False)
    columns = ["train_fpr", "test_fpr", "recall", "precision", "f1", "average_precision"]
    print(results.groupby("scheme", sort=False)[columns].agg(["mean", "std"]).round(4).to_string())
    print(f"Saved {len(results)} runs to {destination}")


if __name__ == "__main__":
    main()
