"""Plot the light positions selected by the RL policy from policy_results.csv."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="RL policy_results.csv")
    parser.add_argument("--out", type=Path, help="output PNG (default: best_light_map.png beside CSV)")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv.open(newline="")))
    if not rows:
        parser.error("policy CSV has no result rows")

    failed = len([r for r in rows if not r["best_policy_azimuth"]])
    rows = [r for r in rows if r["best_policy_azimuth"]]
    points = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis")
    az = np.array([float(row["best_policy_azimuth"]) for row in rows])
    el = np.array([float(row["best_policy_elevation"]) for row in rows])
    confidence = np.array([float(row["best_policy_confidence"]) for row in rows])
    colors = {"correct": "o", "wrong": "X", "none": "s"}

    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor="#f7f7f3")
    ax.set_facecolor("#f7f7f3")
    for verdict, marker in colors.items():
        indices = [i for i, row in enumerate(rows) if row["best_verdict"] == verdict]
        if indices:
            points = ax.scatter(az[indices], el[indices], c=confidence[indices], vmin=0, vmax=1,
                                cmap="viridis", marker=marker, s=64, edgecolors="#263238",
                                linewidths=0.45, label=verdict.title(), alpha=0.9)

    top = sorted(range(len(rows)), key=lambda i: confidence[i], reverse=True)[:10]
    for i in top:
        ax.annotate(rows[i]["object"], (az[i], el[i]), xytext=(4, 5),
                    textcoords="offset points", fontsize=7, color="#20262b")

    ax.set_xticks(range(0, 360, 30))
    ax.set_yticks((5, 25, 45, 65, 85))
    ax.set_xlim(-12, 372)
    ax.set_ylim(0, 90)
    ax.set_xlabel("Selected light azimuth (degrees)")
    ax.set_ylabel("Selected light elevation (degrees)")
    ax.set_title("RL-selected valid positions (%d objects without a valid choice)" % failed, loc="left", weight="bold")
    ax.grid(color="#c8ccca", alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="YOLOE verdict", frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))
    colorbar = fig.colorbar(points, ax=ax, pad=0.025)
    colorbar.set_label("Best visited YOLOE tag confidence")
    fig.tight_layout()
    output = args.out or args.csv.with_name("best_light_map.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
