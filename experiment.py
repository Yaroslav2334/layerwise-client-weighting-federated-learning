"""Run the four layer-wise aggregation schemes on the public AI4I 2020 CSV."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from torch import nn


ROUNDS = 60
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATES = (0.01, 0.005, 0.001)
PILOT_SEED = 0
SEEDS = (1, 2, 3, 4, 5)
CLIENT_SIZES = (800, 2400, 1200, 2000, 1600)
POSITIVE_COUNTS = (15, 30, 45, 71, 110)
METRICS = ["accuracy", "precision", "recall", "f1"]
SCHEMES = {
    "FedAvg": (),
    "All layers": ("hidden1", "hidden2", "output"),
    "Output only": ("output",),
    "Hidden 2 + output": ("hidden2", "output"),
}


class MLP(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.hidden1 = nn.Linear(feature_count, 16)
        self.hidden2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, inputs):
        hidden = torch.relu(self.hidden1(inputs))
        hidden = torch.relu(self.hidden2(hidden))
        return self.output(hidden).squeeze(-1)


def load_data(path):
    data = pd.read_csv(path)
    # Fixed scaling avoids sharing client feature statistics or fitting on test data.
    numeric_columns = [
        "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
        "Torque [Nm]", "Tool wear [min]",
    ]
    offsets = np.array([300, 310, 1500, 40, 0])
    scales = np.array([5, 5, 500, 20, 100])
    numeric = (data[numeric_columns].to_numpy() - offsets) / scales
    types = data["Type"].to_numpy()[:, None] == np.array(["L", "M", "H"])
    features = np.column_stack((numeric, types))
    labels = data["Machine failure"].to_numpy()
    if len(labels) != 10_000 or not np.isin(labels, [0, 1]).all() or labels.sum() != 339:
        raise ValueError("Expected the original AI4I data with 10,000 rows and 339 failures.")
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def make_clients(features, labels, train_indices):
    rng = np.random.default_rng(42)
    train_labels = labels[train_indices].numpy()
    pos = rng.permutation(train_indices[train_labels == 1])
    neg = rng.permutation(train_indices[train_labels == 0])
    if len(pos) != sum(POSITIVE_COUNTS) or len(train_indices) != sum(CLIENT_SIZES):
        raise ValueError("The predefined client partition requires the original 80% training split.")

    clients = []
    pos_start = neg_start = 0
    for client_size, pos_count in zip(CLIENT_SIZES, POSITIVE_COUNTS):
        neg_count = client_size - pos_count
        indices = np.concatenate((pos[pos_start:pos_start + pos_count], neg[neg_start:neg_start + neg_count]))
        clients.append((features[indices], labels[indices]))
        pos_start += pos_count
        neg_start += neg_count
    return clients


def train(global_model, clients, fedavg, minority, reweighted_layers, seed, learning_rate, pilot_losses=None):
    generator = torch.Generator().manual_seed(seed)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(ROUNDS):
        local_states = []
        for features, labels in clients:
            local_model = deepcopy(global_model)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
            for _ in range(LOCAL_EPOCHS):
                order = torch.randperm(len(labels), generator=generator)
                for batch in order.split(BATCH_SIZE):
                    optimizer.zero_grad()
                    loss = criterion(local_model(features[batch]), labels[batch])
                    loss.backward()
                    optimizer.step()
            local_states.append(local_model.state_dict())

        aggregate = {}
        for name in local_states[0]:
            layer = name.split(".")[0]
            weights = minority if layer in reweighted_layers else fedavg
            aggregate[name] = sum(weight * state[name] for weight, state in zip(weights, local_states))
        global_model.load_state_dict(aggregate)
        if pilot_losses is not None:
            with torch.no_grad():
                mean_loss = sum(
                    weight * criterion(global_model(features), labels).item()
                    for weight, (features, labels) in zip(fedavg, clients)
                )
            pilot_losses.append(mean_loss)
    return global_model


def select_learning_rate(clients, fedavg, minority, feature_count):
    torch.manual_seed(PILOT_SEED)
    initial_model = MLP(feature_count)
    candidates = {}
    for learning_rate in LEARNING_RATES:
        losses = []
        train(deepcopy(initial_model), clients, fedavg, minority, SCHEMES["FedAvg"],
              PILOT_SEED, learning_rate, pilot_losses=losses)
        first_mean = np.mean(losses[:5])
        last_mean = np.mean(losses[-5:])
        stable = np.isfinite(losses).all() and last_mean < first_mean
        print(f"Pilot LR={learning_rate:g}: first-five BCE={first_mean:.6f}, "
              f"last-five BCE={last_mean:.6f}, stable={stable}", flush=True)
        if stable:
            candidates[learning_rate] = last_mean
    if not candidates:
        raise RuntimeError("The pilot found no finite, decreasing training loss.")
    selected = min(candidates, key=candidates.get)
    print(f"Selected learning rate: {selected:g}", flush=True)
    return selected


@torch.inference_mode()
def predict_logits(model, features):
    logits = model(features)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite predictions.")
    return logits


def select_train_threshold(models, features, labels):
    scores = np.concatenate([
        predict_logits(model, features).double().sigmoid().numpy()
        for _, scheme, model in models if scheme == "FedAvg"
    ])
    targets = np.tile(labels.numpy(), len(scores) // len(labels))
    precision, recall, thresholds = precision_recall_curve(targets, scores)

    # The last precision-recall point has no corresponding threshold.
    precision = precision[:-1]
    recall = recall[:-1]
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    best = np.flatnonzero(f1 == f1.max())[-1]  # Prefer the higher threshold on a tie.
    threshold = float(thresholds[best])
    print(f"Selected Train threshold: {threshold:.12g} "
          f"(FedAvg pooled precision={precision[best]:.6f}, "
          f"recall={recall[best]:.6f}, F1={f1[best]:.6f})", flush=True)
    return threshold


def evaluate(model, features, labels, threshold=None):
    logits = predict_logits(model, features)
    if threshold is None:
        predicted = logits >= 0  # Probability threshold of 0.5.
    else:
        predicted = logits.double().sigmoid() >= threshold
    tn, fp, fn, tp = confusion_matrix(labels.numpy(), predicted.numpy(), labels=[0, 1]).ravel()
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to the original ai4i2020.csv")
    parser.add_argument("--train-threshold", action="store_true",
                        help="Select a shared threshold on FedAvg training predictions")
    parser.add_argument("--learning-rate", type=float, choices=LEARNING_RATES,
                        help="Reuse a learning rate already selected in the pilot")
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
    print(f"Client sizes: {totals}; minority counts: {positives}", flush=True)
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = select_learning_rate(clients, fedavg, minority, features.shape[1])
    else:
        print(f"Reusing pilot learning rate: {learning_rate:g}", flush=True)

    models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        initial_model = MLP(features.shape[1])
        for name, reweighted_layers in SCHEMES.items():
            model = train(deepcopy(initial_model), clients, fedavg, minority, reweighted_layers, seed,
                          learning_rate)
            models.append((seed, name, model))
        print(f"Finished training seed {seed}", flush=True)

    threshold = None
    if args.train_threshold:
        threshold = select_train_threshold(models, features[train_indices], labels[train_indices])

    test_features, test_labels = features[test_indices], labels[test_indices]
    print(f"Test examples: {len(test_labels)}; minority: {int(test_labels.sum())}", flush=True)
    rows = [
        {"seed": seed, "scheme": name, "learning_rate": learning_rate,
         **evaluate(model, test_features, test_labels, threshold)}
        for seed, name, model in models
    ]
    results = pd.DataFrame(rows)
    if args.train_threshold:
        results.insert(3, "threshold", threshold)
    filename = "threshold_results.csv" if args.train_threshold else "results.csv"
    output_path = Path(__file__).with_name(filename)
    results.to_csv(output_path, index=False)
    print(results.groupby("scheme", sort=False)[METRICS].agg(["mean", "std"]).round(4).to_string())
    print(f"Saved {len(results)} runs to {output_path}")


if __name__ == "__main__":
    main()
