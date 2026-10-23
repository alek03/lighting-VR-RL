"""Train and evaluate PPO for bounded YOLOE-guided light search."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.light_env import ACTION_NAMES, DEFAULT_SWEEP, ELEVATIONS, LightSearchEnv, load_score_table, correct_confidence


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "deliverables" / "rl_light_search"


def write_results(model, env, score_table, output, sweep_dir, eval_episodes, seed):
    by_object = defaultdict(list)
    for name, azimuth, elevation in score_table:
        by_object[name].append((azimuth, elevation))
    rows = []
    examples = []
    rng = random.Random(seed + 1)
    for object_name in sorted(by_object):
        positions = sorted(set(by_object[object_name]))
        starts = rng.sample(positions, min(eval_episodes, len(positions)))
        best_seen, final_conf, starts_conf = [], [], []
        start_correct, final_correct, episode_success = [], [], []
        best_start = (*starts[0], 0.0)
        best_result = None
        for episode, (azimuth, elevation) in enumerate(starts):
            obs, info = env.reset(seed=seed + episode, options={
                "object_name": object_name,
                "azimuth": azimuth,
                "elevation": elevation,
            })
            start_conf = info["confidence"]
            episode_best = (start_conf, azimuth, elevation) if info["correct"] else None
            start_correct.append(info["correct"])
            starts_conf.append(start_conf)
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(int(action))
                candidate = (info["confidence"], info["azimuth"], info["elevation"])
                if info["correct"] and (episode_best is None or candidate[0] > episode_best[0]):
                    episode_best = candidate
            best_seen.append(episode_best[0] if episode_best else 0.0)
            episode_success.append(episode_best is not None)
            final_correct.append(info["correct"])
            final_conf.append(info["confidence"])
            if episode_best is not None and (best_result is None or episode_best[0] > best_result[0]):
                best_result = episode_best
                best_start = (azimuth, elevation, start_conf)

        valid_positions = [(az, el) for az, el in positions
                           if score_table[(object_name, az, el)]["verdict"] == "correct"]
        status = ("correct_detection_found" if best_result else
                  "policy_missed_valid_position" if valid_positions else
                  "no_correct_detection_in_sweep")
        best_confidence, best_azimuth, best_elevation = best_result or (0.0, "", "")
        source = score_table[(object_name, best_azimuth, best_elevation)] if best_result else {}
        oracle = max((correct_confidence(score_table[(object_name, az, el)])
                      for az, el in valid_positions), default=0.0)
        row = {
            "object": object_name,
            "tag": score_table[(object_name, *positions[0])]["tag"],
            "status": status,
            "sweep_correct_positions": len(valid_positions),
            "sweep_wrong_positions": sum(score_table[(object_name, az, el)]["verdict"] == "wrong" for az, el in positions),
            "sweep_no_detection_positions": sum(score_table[(object_name, az, el)]["verdict"] == "none" for az, el in positions),
            "start_detection_rate": float(np.mean(start_correct)),
            "final_detection_rate": float(np.mean(final_correct)),
            "episode_success_rate": float(np.mean(episode_success)),
            "evaluation_starts": len(starts),
            "mean_start_confidence": round(float(np.mean(starts_conf)), 4),
            "mean_final_confidence": round(float(np.mean(final_conf)), 4),
            "mean_best_seen_confidence": round(float(np.mean(best_seen)), 4),
            "best_policy_confidence": round(float(best_confidence), 4),
            "best_policy_azimuth": best_azimuth,
            "best_policy_elevation": best_elevation,
            "light_x_cm": source.get("light_x", ""),
            "light_y_cm": source.get("light_y", ""),
            "light_z_cm": source.get("light_z", ""),
            "best_frame": source.get("frame", ""),
            "best_verdict": source.get("verdict", ""),
            "sweep_oracle_confidence": round(float(oracle), 4),
            "policy_vs_start": round(float(best_confidence - np.mean(starts_conf)), 4),
            "best_evaluation_start_azimuth": best_start[0],
            "best_evaluation_start_elevation": best_start[1],
        }
        rows.append(row)
        examples.append((row, best_start))

    csv_path = output / "policy_results.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "objects": len(rows),
        "objective": "correct_detection_then_confidence",
        "unsuccessful_episode_penalty": 0.1,
        "objects_with_correct_detection": sum(r["status"] == "correct_detection_found" for r in rows),
        "objects_policy_missed": sum(r["status"] == "policy_missed_valid_position" for r in rows),
        "objects_without_correct_sweep_position": sum(r["status"] == "no_correct_detection_in_sweep" for r in rows),
        "start_detection_rate": float(np.mean([r["start_detection_rate"] for r in rows])),
        "final_detection_rate": float(np.mean([r["final_detection_rate"] for r in rows])),
        "episode_success_rate": float(np.mean([r["episode_success_rate"] for r in rows])),
        "evaluation_episodes_per_object": eval_episodes,
        "mean_start_confidence": round(float(np.mean([r["mean_start_confidence"] for r in rows])), 4),
        "mean_final_confidence": round(float(np.mean([r["mean_final_confidence"] for r in rows])), 4),
        "mean_best_seen_confidence": round(float(np.mean([r["mean_best_seen_confidence"] for r in rows])), 4),
        "mean_sweep_oracle_confidence": round(float(np.mean([r["sweep_oracle_confidence"] for r in rows])), 4),
        "mean_policy_improvement_over_start": round(float(np.mean([r["policy_vs_start"] for r in rows])), 4),
        "action_names": list(ACTION_NAMES),
        "max_steps": env.max_steps,
        "move_penalty": env.move_penalty,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot_results(rows, output / "confidence_summary.png")
    _example_contact_sheet(examples, sweep_dir, output / "example_frames.png")
    return summary


def _plot_results(rows, path):
    means = [
        np.mean([row["mean_start_confidence"] for row in rows]),
        np.mean([row["mean_final_confidence"] for row in rows]),
        np.mean([row["mean_best_seen_confidence"] for row in rows]),
        np.mean([row["sweep_oracle_confidence"] for row in rows]),
    ]
    labels = ["Start", "Final", "Best valid visited", "Valid sweep max"]
    colors = ["#77818a", "#3f80a5", "#3b986d", "#d0a13d"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="#f7f7f3")
    ax.set_facecolor("#f7f7f3")
    bars = ax.bar(labels, means, color=colors, width=0.62)
    ax.set_ylim(0, max(0.1, max(means) * 1.2))
    ax.set_ylabel("Correct-detection confidence (failures = 0)")
    ax.set_title("Bounded RL light search across %d objects" % len(rows), loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01,
                "%.3f" % value, ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _example_contact_sheet(examples, sweep_dir, path):
    preferred = {"Mug", "Apple", "Chair", "Phone", "Sunglasses", "Teapot"}
    selected = [item for item in examples if item[0]["object"] in preferred]
    if not selected:
        selected = examples[:6]
    tile_w, tile_h = 256, 286
    sheet = Image.new("RGB", (tile_w * 2, tile_h * len(selected)), "#f7f7f3")
    draw = ImageDraw.Draw(sheet)
    for index, (row, start) in enumerate(selected):
        object_name = row["object"]
        if row["status"] != "correct_detection_found":
            draw.text((8, index * tile_h + 6), object_name + ": " + row["status"], fill="#20262b")
            continue
        frames = (
            ("Start %.3f (%d°, %d°)" % (start[2], start[0], start[1]), start[0], start[1]),
            ("Best %.3f (%d°, %d°)" % (row["best_policy_confidence"],
             row["best_policy_azimuth"], row["best_policy_elevation"]),
             row["best_policy_azimuth"], row["best_policy_elevation"]),
        )
        y = index * tile_h
        draw.text((8, y + 6), object_name, fill="#20262b")
        for column, (caption, azimuth, elevation) in enumerate(frames):
            source = sweep_dir / "frames" / object_name / ("el%02d_az%03d.png" % (elevation, azimuth))
            if not source.is_file():
                continue
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((tile_w - 16, tile_h - 38))
                x = column * tile_w + (tile_w - image.width) // 2
                sheet.paste(image, (x, y + 26))
            draw.text((column * tile_w + 8, y + tile_h - 16), caption, fill="#343c41")
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP, help="YOLOE-scored training.scene sweep.csv")
    parser.add_argument("--out", type=Path, default=REPO / "deliverables" / "rl_correct_detection", help="model and result folder")
    parser.add_argument("--timesteps", type=int, default=80000)
    parser.add_argument("--eval-episodes", type=int, default=8, help="starting positions per object")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--move-penalty", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--live", action="store_true",
                        help="measure every training/evaluation step in Unreal with YOLOE")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    score_table = load_score_table(args.sweep)
    train_env = DummyVecEnv([lambda: Monitor(LightSearchEnv(
        score_table=score_table, sweep_csv=args.sweep, max_steps=args.max_steps,
        move_penalty=args.move_penalty, seed=args.seed, live=args.live))])
    model = PPO(
        "MlpPolicy", train_env, learning_rate=3e-4, n_steps=1024,
        batch_size=256, gamma=0.96, gae_lambda=0.92,
        ent_coef=0.025, verbose=1, seed=args.seed, device=args.device,
        policy_kwargs={"net_arch": [128, 128]},
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model_path = args.out / "ppo_light_search"
    model.save(model_path)
    eval_env = LightSearchEnv(score_table=score_table, sweep_csv=args.sweep, max_steps=args.max_steps,
                              move_penalty=args.move_penalty, seed=args.seed + 1, live=args.live)
    summary = write_results(model, eval_env, score_table, args.out, args.sweep.parent,
                            args.eval_episodes, args.seed)
    summary.update({"algorithm": "PPO", "timesteps": args.timesteps,
                    "seed": args.seed, "model": model_path.with_suffix(".zip").name,
                    "source_sweep": str(args.sweep)})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    eval_env.close()
    train_env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()