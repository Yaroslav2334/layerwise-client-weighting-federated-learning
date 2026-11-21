# Layer-Wise Client Weighting for Minority-Class Recognition in Federated Learning

The simulation compares FedAvg and three schemes that apply minority-class-count weighting to different layers of one binary classifier.

## Setup

Install the Python dependencies and download `ai4i2020.csv` from the [UCI Machine Learning Repository](https://doi.org/10.24432/C5HS5C). Put the CSV beside the scripts or pass its path to the training commands.

```sh
python -m pip install numpy pandas scikit-learn torch matplotlib
```

## Reproduce the results

Run these commands from this folder.

```sh
python experiment.py ai4i2020.csv
python experiment.py ai4i2020.csv --train-threshold --learning-rate 0.01
python false_alarm_budget.py ai4i2020.csv
python plot_recall_vs_false_alarms.py
```

1. The first run performs the learning-rate pilot and writes `results.csv` for the default threshold of 0.5.
2. The second run uses the selected learning rate of 0.01 and writes `threshold_results.csv` for the shared threshold chosen from FedAvg training predictions.
3. The third run writes `false_alarm_results.csv` using per-model thresholds selected at a 2% training false-positive-rate limit.
4. The plot script reads the two threshold result files and writes the paper's figure in PDF and PNG formats.
