# rl/

Empty for now. The RL agent goes here; add a readme in this folder for each file as it is
written.

It will use the other two parts:

- `unreal_bridge`: `ue.spawn`, `ue.set_light`, `ue.capture` to act in the scene live
- `vision`: `Detector.score` for the reward, or a sweep's `sweep.csv` as a ready-made table of
  scores for every light position (fast, no Unreal)
