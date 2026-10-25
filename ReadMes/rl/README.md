# RL

Run the trainer from the repository root as a module so imports from the sibling
`unreal_bridge/` and `vision/` packages resolve:

```sh
.venv/bin/python -m RL.train_rl
```

Training uses the scored sweep CSV specified by `--sweep_csv` (default:
`outputs/sweep_2026-10-17/sweep.csv`). To evaluate the trained policy in Unreal as well:

```sh
.venv/bin/python -m RL.train_rl --train --evaluate_live
```

It will use the other two parts:

- `unreal_bridge`: `ue.spawn`, `ue.set_light`, `ue.capture` to act in the scene live
- `vision`: `Detector.score` for the reward, or a sweep's `sweep.csv` as a ready-made table of
  scores for every light position (fast, no Unreal)
