"""Plot each RL-selected light on the table-centred hemisphere sweep map."""
import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP = REPO / "captures" / "sweep_tufaelz_yoloe"
DEFAULT_POSE = Path(os.environ.get("VRNAV_POSE", "/opt/Unrealprojects/tufaelz/Scripts/pose.json"))
CENTER = (1120.0, -250.0)
TABLE = (1080.0, -335.0, 80.0, 170.0)
RADIUS = 120.0
CONFIDENCE = LinearSegmentedColormap.from_list(
    "confidence", ["#eef4fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
VERDICT = {"correct": ("#0ca30c", "o"), "wrong": ("#d03b3b", "X"), "none": ("#9a9993", "s")}


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def draw_room(ax, camera, title):
    ax.set_facecolor("#fcfcfb")
    ax.set_aspect("equal")
    ax.axis("off")
    yaw = np.arctan2(CENTER[1] - camera[1], CENTER[0] - camera[0])
    for side in (-1, 1):
        angle = yaw + side * np.radians(30)
        ax.plot([camera[0], camera[0] + 420 * np.cos(angle)],
                [camera[1], camera[1] + 420 * np.sin(angle)], color="#86b6ef", lw=1.1, zorder=1)
    ax.add_patch(Rectangle(TABLE[:2], TABLE[2], TABLE[3], fc="#a0703c", ec="none", alpha=0.78, zorder=1))
    ax.add_patch(Circle(CENTER, RADIUS, fill=False, ec="#999b98", lw=1.3, zorder=2))
    ax.plot(*CENTER, "o", ms=6, color="#d03b3b", mec="white", mew=0.8, zorder=5)
    ax.plot(camera[0], camera[1], "o", ms=9, color="#2a78d6", mec="white", mew=1.0, zorder=5)
    ax.text(camera[0], camera[1] + 12, "camera", ha="center", va="top", fontsize=8, color="#2a78d6")
    ax.text(CENTER[0], CENTER[1] + 10, "object", ha="center", va="top", fontsize=8, color="white", weight="bold")
    ax.set_xlim(895, 1310)
    ax.set_ylim(-80, -420)
    ax.set_title(title, fontsize=12, weight="bold", pad=5)


def draw_object_map(name, sweep_rows, result, camera, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor="#fcfcfb")
    for ax, title in zip(axes, ("Sweep confidence", "Sweep verdict")):
        draw_room(ax, camera, title)

    for row in sweep_rows:
        x, y = float(row["light_x"]), float(row["light_y"])
        confidence = float(row["confidence"])
        axes[0].scatter(x, y, s=45, color=CONFIDENCE(confidence), edgecolors="#626762",
                        linewidths=0.45, zorder=4)
        color, marker = VERDICT.get(row["verdict"], VERDICT["none"])
        axes[1].scatter(x, y, s=48, color=color, marker=marker, edgecolors="white",
                        linewidths=0.55, zorder=4)

    has_choice = bool(result["best_policy_azimuth"])
    if has_choice:
        selected = (float(result["light_x_cm"]), float(result["light_y_cm"]))
        verdict_color, _ = VERDICT.get(result["best_verdict"], VERDICT["none"])
        for ax in axes:
            ax.scatter(*selected, marker="*", s=220, color="#f4c542", edgecolors="#171a1c",
                       linewidths=1.0, zorder=8)
            ax.annotate("RL choice", selected, xytext=(7, 7), textcoords="offset points",
                        fontsize=8, color="#15191b", weight="bold")
    
    norm = Normalize(0, 1)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CONFIDENCE), ax=axes[0], fraction=0.045, pad=0.02)
    bar.set_label("YOLOE tag confidence", fontsize=8)
    legend = [Line2D([], [], ls="none", marker=marker, ms=7, color=color, label=verdict.title())
              for verdict, (color, marker) in VERDICT.items()]
    axes[1].legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.01),
                   ncol=3, frameon=False, fontsize=8)
    if has_choice:
        fig.suptitle("%s (%s) | RL chose %d° azimuth, %d° elevation | confidence %.3f | %s" % (
            name, result["tag"], int(float(result["best_policy_azimuth"])),
            int(float(result["best_policy_elevation"])), float(result["best_policy_confidence"]),
            result["best_verdict"].upper()), fontsize=11, weight="bold")
    else:
        fig.suptitle(name + " | " + result.get("status", "No valid detection"), fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(output, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_csv", type=Path, help="RL policy_results.csv")
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP, help="matching YOLOE sweep folder")
    parser.add_argument("--pose", type=Path, default=DEFAULT_POSE, help="matching Unreal pose.json")
    parser.add_argument("--out", type=Path, help="output directory; default is beside policy_csv")
    args = parser.parse_args()

    policy_rows = read_rows(args.policy_csv)
    sweep_rows = read_rows(args.sweep / "sweep.csv")
    grouped = {}
    for row in sweep_rows:
        grouped.setdefault(row["object"], []).append(row)
    results = {row["object"]: row for row in policy_rows}
    if set(results) != set(grouped):
        parser.error("policy and sweep CSVs must contain the same objects")
    camera = json.loads(args.pose.read_text())["camera"]["location"]
    output_dir = args.out or args.policy_csv.parent / "hemisphere_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, result in results.items():
        image_path = output_dir / ("rl_light_map_%s.png" % name)
        draw_object_map(name, grouped[name], result, camera, image_path)
        paths.append((name, image_path))
    print("%d object maps -> %s" % (len(paths), output_dir))


if __name__ == "__main__":
    main()
