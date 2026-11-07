# RL light search

The Gymnasium environment uses YOLOE tag confidence as reward. The action set moves the light by
one azimuth/elevation step or holds position; camera pose, object placement, Unreal startup,
screenshots, and detector prompts stay on the existing project paths.

`training/object_classes.json` supplies 115 object names and matching YOLOE text tags. The sweep and
RL code uses all catalogue labels directly. Run a complete sweep to create YOLOE
scores and frames for all objects, then train PPO from that CSV:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
  .venv/bin/python -m training.scene sweep --out captures/sweep_yoloe
.venv/bin/python -m rl.train_light_search --sweep captures/sweep_yoloe/sweep.csv
```

The PPO run writes `deliverables/rl_light_search/ppo_light_search.zip`, `policy_results.csv`,
`summary.json`, and two figures. The CSV includes selected tag, confidence, verdict, light
coordinates, and source-frame path for each object. The 80,000-step run on all 115 objects
averaged 0.1237 start confidence, 0.2583 final confidence, and 0.2618 best-visited confidence;
the full-grid maximum averaged 0.2988. Optional `--live` scoring runs each action through the
existing Unreal `Scene` and YOLOE scorer; it requires the editor and is much slower.

To rescore existing screenshots without Unreal:

```bash
.venv/bin/python -m training.scene sweep --out <sweep-folder> --rescore
```

A sweep made before YOLOE must be rescored before training. The previous results are preserved as
`sweep_before_rescore.csv`. To cover all 115 catalog objects, the score table must contain frames
for all 115; rescoring a 107-object archive cannot create the missing eight object renders.


## Correct-detection objective

The current environment uses the existing vision verdict: the top prediction must meet the
vision threshold (0.25), match the target tag, and overlap the object mask according to the
vision scorer (IoU or object fraction at least 0.5). Correct confidence uses that top prediction's
confidence (`top1_conf`), not a potentially higher target-tag prediction elsewhere in the image.
Wrong/missing predictions have effective confidence zero.

Quality is `1 + correct confidence` for a correct detection and zero otherwise. The per-action
reward is the change in quality minus the movement cost (default 0.005). Episodes ending
without ever finding a correct detection receive an additional 0.1 penalty. STOP still ends
the episode, and the 16-action limit applies. No extra confidence reward is given for a wrong
or absent detection. Accuracy/success rates are reported separately from confidence.

Only correctly detected visited positions may be selected. Failure rows leave selected angles,
coordinates and frames blank. `status` distinguishes `correct_detection_found`,
`policy_missed_valid_position` (the cached sweep contains a valid spot), and
`no_correct_detection_in_sweep`. The latter describes the recorded grid only; it does not
prove that detection is impossible under other lighting, viewpoints, prompts or models.
Counts of wrong and absent sweep detections are also exported. Success is measured against
the known target and rendered mask, not independently annotated real-world accuracy.

The confidence chart now counts invalid detections as zero. Summary/CSV fields include start,
final, and episode detection success rates. Plot scripts support failure rows without drawing
a selected star. Old confidence-only policies are incompatible with the new observation;
retrain from scratch. Old deliverables are historical and are not rewritten automatically.
The new default output directory is `deliverables/rl_correct_detection`.

Example using the existing extra-large scores (no image rescoring required):

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m rl.train_light_search \
  --sweep captures/sweep_tufaelz_yoloe26x/sweep.csv \
  --out deliverables/tufaelz_yoloe26x/rl_correct_detection --device cpu
```
