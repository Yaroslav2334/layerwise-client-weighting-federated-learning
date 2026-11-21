"""Reproduce the paper's recall versus false-positive-rate figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


SOURCE = Path(__file__).resolve().parent
SCHEMES = {
    "FedAvg": ("FedAvg", "#225B94"),
    "All layers": ("All layers", "#B9471C"),
    "Output only": ("Output only", "#1B765E"),
    "Hidden 2 + output": ("Last two layers", "#76469D"),
}


def mean_metrics(filename):
    runs = pd.read_csv(SOURCE / filename)
    expected = {(seed, scheme) for seed in range(1, 6) for scheme in SCHEMES}
    assert len(runs) == len(expected) and set(zip(runs.seed, runs.scheme)) == expected
    assert ((runs.fp + runs.tn == 1932) & (runs.tp + runs.fn == 68)).all()
    runs["fpr"] = runs.fp / (runs.fp + runs.tn)
    runs["recall"] = runs.tp / (runs.tp + runs.fn)
    return 100 * runs.groupby("scheme")[["fpr", "recall"]].mean()


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "text.color": "#26313C",
        "axes.labelcolor": "#26313C",
        "axes.edgecolor": "#72808F",
        "xtick.color": "#374452",
        "ytick.color": "#374452",
        "pdf.fonttype": 42,
    })
    shared = mean_metrics("threshold_results.csv")
    calibrated = mean_metrics("false_alarm_results.csv")
    figure, axes = plt.subplots(figsize=(6.15, 3.9))
    figure.subplots_adjust(left=0.120, right=0.965, bottom=0.147, top=0.790)
    axes.set(xlim=(0.86, 3.64), ylim=(16.0, 56.0),
             xlabel="False-positive rate (%)", ylabel="Failure recall (%)")
    axes.xaxis.set_major_locator(MultipleLocator(0.5))
    axes.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axes.yaxis.set_major_locator(MultipleLocator(10))
    axes.spines[["top", "right"]].set_visible(False)
    axes.tick_params(length=3)
    axes.grid(color="#E3E8ED", linewidth=0.7)
    axes.set_axisbelow(True)

    inset = axes.inset_axes([0.060, 0.619, 0.338, 0.330], zorder=5)
    inset.set(xlim=(1.875, 2.25), ylim=(33.85, 37.35))
    inset.xaxis.set_major_locator(MultipleLocator(0.1))
    inset.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    inset.yaxis.set_major_locator(MultipleLocator(1))
    inset.tick_params(length=2, labelsize=7, pad=1.5)
    inset.grid(color="#E3E8ED", linewidth=0.55)
    inset.set_axisbelow(True)
    inset.set_title("Detail near 2% false positives", fontsize=7.7, pad=3.0)
    for spine in inset.spines.values():
        spine.set_color("#A2ADBA")
        spine.set_linewidth(0.7)
    axes.add_patch(Rectangle((1.875, 33.85), 0.375, 3.5, fill=False,
                             edgecolor="#909DAB", linewidth=0.8, linestyle=(0, (3, 2)), zorder=2))

    scheme_handles = []
    for scheme, (label, color) in SCHEMES.items():
        x0, y0 = shared.loc[scheme]
        x1, y1 = calibrated.loc[scheme]
        axes.plot((x0, x1), (y0, y1), color=color, alpha=0.48, linewidth=1.25, zorder=2)
        axes.scatter(x0, y0, marker="o", s=45, facecolor="white", edgecolor=color, linewidth=1.6, zorder=4)
        axes.scatter(x1, y1, marker="s", s=40, facecolor=color, edgecolor="white", linewidth=0.65, zorder=4)
        inset.scatter(x1, y1, marker="s", s=31, facecolor=color, edgecolor="white", linewidth=0.55, zorder=4)
        scheme_handles.append(Line2D([], [], color=color, linewidth=2.4, label=label))
        print(f"{label}: shared ({x0:.2f}, {y0:.2f}), calibrated ({x1:.2f}, {y1:.2f})")

    marker_color = "#394652"
    threshold_handles = [
        Line2D([], [], color=marker_color, marker="o", linestyle="None", markerfacecolor="white",
               markeredgewidth=1.4, markersize=5.5, label="Shared threshold (0.1316)"),
        Line2D([], [], color=marker_color, marker="s", linestyle="None", markersize=5.1,
               label="Per-model threshold (2% train FPR limit)"),
    ]
    figure.legend(handles=scheme_handles, ncol=4, loc="center", bbox_to_anchor=(0.53, 0.949),
                  frameon=False, handlelength=1.35, handletextpad=0.52, columnspacing=1.55, fontsize=9.0)
    figure.legend(handles=threshold_handles, ncol=2, loc="center", bbox_to_anchor=(0.53, 0.877),
                  frameon=False, handlelength=1.1, handletextpad=0.55, columnspacing=1.35, fontsize=8.6)

    output = SOURCE / "recall_vs_false_alarms"
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    figure.savefig(output.with_suffix(".png"), dpi=450, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
