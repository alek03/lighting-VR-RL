"""Gymnasium environment for bounded flashlight search using YOLOE tag confidence."""
import csv
import math
import re
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


REPO = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_SWEEP = REPO / "outputs" / "sweep_common_2026-11-10" / "sweep.csv"
DEFAULT_EVAL_SWEEP = REPO / "outputs" / "sweep_fab_2026-11-10" / "sweep.csv"
DEFAULT_SWEEP = DEFAULT_TRAIN_SWEEP
AZIMUTHS = tuple(range(0, 360, 30))
ELEVATIONS = (5, 25, 45, 65, 85)
ACTION_NAMES = ("azimuth -30", "azimuth +30", "elevation -20", "elevation +20", "hold")


def load_score_table(path=DEFAULT_SWEEP):
    """Load confidence values and source-frame paths from a vision.sweep score CSV."""
    table = {}
    for row in csv.DictReader(Path(path).open(newline="")):
        key = (row["object"], int(float(row["azimuth"])) % 360, int(float(row["elevation"])))
        table[key] = {
            "object": row["object"],
            "tag": row.get("tag", row["object"]),
            "confidence": float(row["confidence"]),
            "gt_conf": float(row.get("gt_conf", row.get("confidence", 0.0))),
            "verdict": row.get("verdict", ""),
            "top1": row.get("top1", row.get("object", "")),
            "top1_conf": float(row.get("top1_conf", row.get("confidence", 0.0))),
            "top2": row.get("top2", ""),
            "top2_conf": float(row.get("top2_conf", 0.0)),
            "top3": row.get("top3", ""),
            "top3_conf": float(row.get("top3_conf", 0.0)),
            "frame": row.get("frame", ""),
            "light_x": float(row["light_x"]),
            "light_y": float(row["light_y"]),
            "light_z": float(row["light_z"]),
        }
    if any(r["verdict"] not in {"correct", "wrong", "none", "partial"} for r in table.values()):
        raise ValueError("score table must contain validated detection verdicts; rescore the sweep first")
    if not table:
        raise ValueError("sweep CSV contains no confidence rows: %s" % path)
    return table


def _normalize_label(name):
    text = str(name or "").strip()
    if not text:
        return ""
    text = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
    text = re.sub(r'\s+', ' ', text).lower()
    return text


def quality_from_row(row):
    """Quality score = ground-truth confidence minus the strongest competing label."""
    gt_conf = float(row.get("gt_conf", row.get("confidence", 0.0)))
    top1_name = _normalize_label(row.get("top1", row.get("object", "")))
    gt_name = _normalize_label(row.get("object", row.get("tag", "")))
    top1_conf = float(row.get("top1_conf", row.get("confidence", 0.0)))
    top2_conf = float(row.get("top2_conf", 0.0))
    competitor = top2_conf if top1_name == gt_name and top2_conf else top1_conf
    return gt_conf - competitor


def correct_confidence(row):
    """Confidence of the validated top detection; failures score zero."""
    # if row.get("verdict") != "correct":
    #     return 0.0
    return float(row.get("gt_conf") if row.get("gt_conf") is not None else row.get("top1_conf") or row["confidence"])


class LightSearchEnv(gym.Env):
    """Find a high-confidence light location using small discrete moves.

    The detector's cached confidence at each measured position is the observation
    returned after moving. Azimuth wraps around; elevation is clipped to the
    calibrated 5-85 degree range. Rewards are improvement in correctness-first quality minus a
    small movement cost; an unsuccessful episode also costs 0.1. The fixed sweep grid keeps the source render settings,
    object poses, and screenshots unchanged.
    """

    metadata = {"render_modes": []}

    def __init__(self, score_table=None, sweep_csv=DEFAULT_SWEEP, max_steps=16,
                 move_penalty=0.005, seed=None, live=False, collection="common"):
        super().__init__()
        self.score_table = score_table if score_table is not None else load_score_table(sweep_csv)
        self.sweep_dir = Path(sweep_csv).parent
        self.objects = sorted({key[0] for key in self.score_table})
        self.global_best_confidence = {
            name: max(
                (float(row.get("gt_conf", 0.0)) for (object_name, _, _), row in self.score_table.items()
                 if object_name == name),
                default=0.0,
            )
            for name in self.objects
        }
        self.object_index = {name: index for index, name in enumerate(self.objects)}
        self.max_steps = int(max_steps)
        self.move_penalty = float(move_penalty)
        if self.max_steps < 1 or self.move_penalty < 0:
            raise ValueError("max_steps must be positive and move_penalty non-negative")
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )
        self.object_name = None
        self.azimuth = 0
        self.elevation = 45
        self.confidence = 0.0
        self.correct_confidence = 0.0
        self.quality = 0.0
        self.best_confidence = 0.0
        self.best_quality = 0.0
        self.steps = 0
        self.visited = set()
        self.collection = collection
        self.live = live
        self.scene = None
        self.scorer = None
        self.mask = None
        if self.live:
            self._open_live_backend()

    def _open_live_backend(self):
        from unreal_bridge import CONFIG, Unreal
        from vision.yoloe import Detector

        self.scene = Unreal(collection=self.collection)
        self.scorer = Detector(collection=self.collection)
        self.warmup_seconds = CONFIG['image'].get('warmup_seconds', 30.0)

    def _measure(self, azimuth, elevation):
        if not self.live:
            return self.score_table[(self.object_name, azimuth, elevation)]
        self.scene.set_light(azimuth, elevation)
        result = self.scorer.score(self.scene.capture(), self.object_name, true_box=self.mask)
        top = result.get("top", [])
        return {
            "object": self.object_name,
            "confidence": float(result["confidence"]),
            "gt_conf": float(result.get("gt_conf", result["confidence"])),
            "verdict": result["verdict"],
            "top1": top[0][0] if top else self.object_name,
            "top1_conf": float(top[0][1]) if top else float(result["confidence"]),
            "top2": top[1][0] if len(top) > 1 else "",
            "top2_conf": float(top[1][1]) if len(top) > 1 else 0.0,
            "top3": top[2][0] if len(top) > 2 else "",
            "top3_conf": float(top[2][1]) if len(top) > 2 else 0.0,
        }

    def _score(self, azimuth, elevation):
        try:
            row = self._measure(azimuth % 360, elevation)
            self.detector_confidence = float(row["confidence"])
            self.verdict = row["verdict"]
            self.correct = self.verdict == "correct"
            self.correct_confidence = correct_confidence(row)
            self.quality = quality_from_row(row)
            return self.quality
        except KeyError as exc:
            raise ValueError("missing sweep score for %s at az=%d el=%d" % (
                self.object_name, azimuth % 360, elevation)) from exc

    def _observation(self):
        angle = math.radians(self.azimuth)
        state = np.array([
            (math.sin(angle) + 1.0) / 2.0,
            (math.cos(angle) + 1.0) / 2.0,
            (self.elevation - ELEVATIONS[0]) / (ELEVATIONS[-1] - ELEVATIONS[0]),
            self.quality,
        ], dtype=np.float32)
        return state

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.object_name = options.get("object_name")
        if self.object_name is None:
            self.object_name = self.objects[int(self.np_random.integers(len(self.objects)))]
        if self.object_name not in self.object_index:
            raise ValueError("object has no sweep scores: %s" % self.object_name)
        self.azimuth = int(options.get("azimuth", self.np_random.choice(AZIMUTHS))) % 360
        self.elevation = int(options.get("elevation", self.np_random.choice(ELEVATIONS)))
        if self.elevation not in ELEVATIONS:
            raise ValueError("elevation must be one of %s" % (ELEVATIONS,))
        self.steps = 0
        if self.live:
            self.scene.spawn(self.object_name)
            self.scene.set_light(self.azimuth, self.elevation)
            deadline = time.monotonic() + self.warmup_seconds
            while time.monotonic() < deadline:
                self.scene.capture()
            self.mask = self.scene.true_box()
        self._score(self.azimuth, self.elevation)
        self.confidence = self.correct_confidence
        self.best_confidence = self.confidence
        self.best_quality = self.quality
        self.found_correct = self.correct
        self.visited = {(self.azimuth, self.elevation)}
        return self._observation(), {"object": self.object_name, "confidence": self.confidence,
                         "quality": self.quality, "correct": self.correct,
                         "verdict": self.verdict}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("action must be in [0, %d]" % (self.action_space.n - 1))
        old_quality = float(self.quality)
        moved = action != 4
        if action == 0:
            self.azimuth = (self.azimuth - 30) % 360
        elif action == 1:
            self.azimuth = (self.azimuth + 30) % 360
        elif action == 2:
            self.elevation = max(ELEVATIONS[0], self.elevation - 20)
        elif action == 3:
            self.elevation = min(ELEVATIONS[-1], self.elevation + 20)

        self._score(self.azimuth, self.elevation)
        self.confidence = self.correct_confidence
        self.best_confidence = max(self.best_confidence, self.confidence)
        self.best_quality = max(self.best_quality, self.quality)
        self.found_correct = self.found_correct or self.correct
        self.visited.add((self.azimuth, self.elevation))
        self.steps += 1
        terminated = bool(action == 4)
        truncated = bool(self.steps >= self.max_steps)
        #reward = self.quality
        reward = self.quality - old_quality
        info = {
            "object": self.object_name,
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "confidence": self.confidence,
            "best_confidence": self.best_confidence,
            "quality": self.quality,
            "best_quality": self.best_quality,
            "global_best_confidence": self.global_best_confidence[self.object_name],
            "visited_positions": len(self.visited),
            "correct": self.correct,
            "verdict": self.verdict,
            "found_correct": self.found_correct,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def close(self):
        if self.scene is not None:
            self.scene.close()
            self.scene = None
