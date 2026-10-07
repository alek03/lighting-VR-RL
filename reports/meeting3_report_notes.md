# Meeting 3 report notes

Running notes for the meeting 3 report. Started 2026-10-06; add to this as work happens.

## Goal for this stage

Repeat last week's lighting test (move the flashlight around one object at a time and see how the
vision model's confidence changes) across **all the new objects**, using **YOLO9000**, which can name
far more object types than the YOLO11m model used last week.

## What changed since last week

- **Unreal project updated by Bradley (Oct 5).** The object set grew from 10 to **115**: 105 new
  objects with their own materials, catalogued in `tufaelz/Config/objectlab_objects.json`. IDs 1–10
  kept their original objects.
- **Lighting changed.** Object materials no longer glow (emissive set to 0), and spawned objects now
  receive light only on lighting channel 1. The room's sun light (channel 0) no longer reaches them,
  so objects are visible only when the flashlight hits them.
  - Consequence for our code: the agent's spotlight copies BP_Torch's settings but not its lighting
    channels, so it most likely does not light the objects any more. **Needs fixing before any
    new runs.**
  - Consequence for last week's results: they were measured under the old lighting, so they are a
    baseline, not directly comparable.
- **Project access.** The project folder was briefly locked to Bradley during the update; it is
  open again (writable by everyone).
- **Vision model.** The lab set up YOLO9000 (Darknet) in `/opt/Unrealprojects/darknet` on Sep 30,
  plus one screenshot per object in `/opt/Unrealprojects/YOLO/data/`.

## Last week's baseline (YOLO11m)

Camera at 165 cm, light 120 cm from the object, 60 light positions (12 directions × 5 heights),
object filling 50% of a 640×640 frame. Only 8 of the 10 objects could be scored (COCO has no
shoe or lamp class).

| Object | Mean confidence | Best |
|---|---|---|
| Mug | 0.80 | 0.97 |
| Bottle | 0.71 | 0.90 |
| Phone | 0.68 | 0.92 |
| Bowl | 0.65 | 0.92 |
| Plant | 0.53 | 0.79 |
| Chair | 0.39 | 0.94 |
| Book | 0.22 | 0.73 |
| Apple | 0.02 | 0.16 |

Every object's worst light position scored close to 0, so light placement matters a lot.

## YOLO11m vs YOLO9000

| | YOLO11m (last week) | YOLO9000 (now) |
|---|---|---|
| Year | 2024 | 2016 (YOLOv2-based) |
| Classes | 80 (COCO) | ~9,000 (9,418 names in a WordNet tree) |
| Training | boxes for all 80 classes | boxes only for COCO's 80; the rest learned from ImageNet labels without boxes |
| Accuracy | high | much lower, especially on non-COCO classes |
| Software | `ultralytics` (Python) | Darknet (C; unmaintained) |

Reason for switching: coverage. YOLO11m can score only ~30 of the 115 objects.

## Ground-truth class dataset (`training/object_classes.json`)

**Purpose:** for each object, which YOLO9000 classes count as a correct identification.

**Method**
1. Object names come from the 3D model file names (`SM_AcousticGuitar` → "acoustic guitar").
2. Exact name lookup in YOLO9000's class list: 98/115 matched.
3. Words with several meanings resolved by their place in the class tree (shoe = footwear, not
   a case; ruler → "rule", because YOLO9000's "ruler" is a person).
4. The other 17 were matched by hand (synonyms, dropping extra words).
5. Cross-check: every object name tested against **all WordNet synonyms** of every YOLO9000
   class (YOLO9000 shows only one name per class, e.g. "hand blower" = hair dryer). This found
   two mapping errors (HockeyPuck → "puck"; Purse → "bag", the handbag class) and added two more
   mushroom classes.
6. Objects whose names have no synonym in YOLO9000 were dropped.

**Scoring rule:** a prediction is correct if it is an accepted class or a more specific class
under it ("coffee mug" counts for Mug). More general classes above it ("container") count only
as partial hits.

Phone also accepts the general "telephone" class (synonym "phone"), so any kind of phone counts.

**Result: 107 objects**

| Group | Count |
|---|---|
| YOLO9000 had box training for the object's class | 24 |
| Box training only for a more general parent ("ball", "timepiece") | 10 |
| Learned from photos only (least reliable) | 73 |
| Also scorable by YOLO11m (for comparison) | 31 |

**Dropped (8):** AudioCassette, BeltBuckle, DeskFan, HairClip, MortarAndPestle, ServingTray,
WoodenCrate (no synonym in YOLO9000), and TableTennisBall (YOLO9000 only knows "ping-pong ball").

**Decisions (2026-10-06)**
- Mug: "cup" counts as correct (the class YOLO9000 learned with boxes).
- Plant: "pot" counts as correct (YOLO9000 learned COCO's "potted plant" boxes under "pot").
- Keyboard: the generic "keyboard" class counts, so its sub-classes do too (typewriter, QWERTY,
  computer and piano keyboard).
- TableTennisBall: dropped.

## Concerns / risks

1. ~~The class mapping has not been tested against real YOLO9000 output.~~ Partly addressed: on
   live renders YOLO9000's specific guesses match the mapping ("Dixie cup" for Mug, "water
   bottle" for Bottle). The full review comes from the sweep's top-3 columns.
2. The 73 photo-only objects may hardly be detected at all, which would give them flat-zero
   sweeps that are useless for RL. (Open; the sweep will show.)
3. YOLO9000 often answers with a general class ("container", "ball"), so strict accuracy may
   look low; partial hits are reported separately.
4. The objects are 3D renders lit only by a flashlight in a dark room, which is harder than the
   photos YOLO9000 was trained on. (Open.)
5. ~~Per-class confidence from Darknet not verified.~~ Resolved: read from the output buffer
   (Step 1).
6. ~~Spotlight lighting channel and the 10-object limit.~~ Resolved (Step 2).
7. None of the new code or data is committed to GitHub yet. (Last step of the plan.)
8. **New:** 7 models are drawn at a different size from their stored bounds (Step 2); worked
   around by rescaling in our renders, but the VR scene still has them.

## Requirements set for the sweep (2026-10-06)

- **Overall goal:** deliverables (graphs and pictures) showing how YOLO9000's identification of
  the new objects changes with lighting angle.
- **Bounding-box check.** A confident prediction is meaningless if its box is around a shadow or
  something else. For every detection, record whether the box is really on the object, and to
  what degree. Approach: one object-only "mask" capture per object (exact silhouette, no shadows),
  then per box: centre-in-box, object fraction of the box (low = mostly shadow/background),
  coverage of the object, and IoU. Raw numbers are kept so the on-object threshold can be changed
  later.
- **Top-3 predictions per object × light angle,** with confidences, alongside the correct
  class(es), for manual checking of the class mapping. Because YOLO9000's classes are nested
  (P(container) ≥ P(mug)), "top 3" means its three best *specific* guesses, following the class
  tree and keeping the three best branches.
- Everything is recorded in one CSV row per object × light position, plus plain and annotated
  frames.

## Deliverables chosen (2026-10-06)

Confidence alone doesn't tell the whole story: every lighting map shows **accuracy** next to
confidence. Definitions: *confidence* = probability YOLO gives the correct class (even when it is
not its top answer); *correct* = YOLO's top answer is an accepted class **and** its box passes the
on-object check; *partial* = top answer is a more general class (e.g. "seat" for Chair).


**Folder layout** (`deliverables/`, kept out of git like `captures/`):

```
deliverables/
  README.md                          index: which file is which deliverable
  3_best_vs_worst.png                3. best vs worst lighting, all objects
  4_recognition_map.png              4. recognition map, all objects
  6_average_lighting_map.png         6. average lighting map, all objects
  objects/<Object>/
    1_lighting_map_<Object>.png      1. confidence + accuracy room maps for that object
    2_image_grid_<Object>.png        2. all 60 frames for that object
```

1. **Per-object lighting map: two top-view room maps side by side.** Same layout as the earlier
   room map (camera with its view lines, table, 120 cm light circle, one dot per flashlight
   position at its true top-down location, +Y down as in the Unreal editor). Left: dots coloured
   by confidence for the correct class. Right: dots marked by YOLO's top answer (correct /
   partial / wrong / no detection, by colour **and** marker shape). Panels titled just
   "Confidence" and "Accuracy"; no main title, overall-accuracy line or footnote. Chosen over
   grid and polar plots because it reads directly as "where in the room".
2. **Image grid per object:** all 60 frames (rows = light height, columns = direction), border
   coloured by verdict, caption = YOLO's answer and confidence. Each frame overlays the **true
   object box** (dashed magenta: the object's 3D bounds projected into the image) and **YOLO's
   box** (solid green/red), so a box drawn around a shadow or the lit table is visible at a
   glance. This replaces the separate box-quality map.
3. **Best vs worst lighting:** per object, the best and worst frame with confidence, YOLO's
   answer, both boxes, and a small top-view icon of where the light was.
4. **Recognition map of all objects:** x = accuracy (share of the 60 light positions where YOLO's
   top answer is correct), y = confidence (dot = average, line = worst to best light).
   Background bands: never recognised / lighting-dependent / always recognised.
6. **Average lighting map across all objects,** same room-map pair: mean confidence, and accuracy
   (% of objects correctly identified from each light position).

Dropped: box-quality map (7, folded into the image grid), ranking bar chart (5), verdict
breakdown (8), "what YOLO thinks it is" table (9; the CSV still supports it for manual review),
YOLO9000 vs YOLO11m scatter (10).

## Example deliverables (2026-10-06, for choosing formats)

`deliverables/` currently holds a full example set built from a fresh 60-position render of
Chair, Mug, Book and Bottle in the Sep 23 project (`tufaelz_old`) scored with YOLO11m, with real
accuracy. The true object box there is the projected 3D bounds; the real run will use an
object-only mask. The render reproduced last week's mean confidences (Chair 0.39, Mug 0.79,
Book 0.22, Bottle 0.71) and gave the first real **accuracies** (top answer correct and box on the
object): Chair 38%, Mug 93%, Book 33%, Bottle 93%.

**First evidence for the bounding-box concern:** with the flashlight high above the Chair (85°, and
65° from behind), YOLO11m's top answer is "bench", with the box drawn around the lit patch of
table rather than the chair. 18 of the Chair's 60 frames had YOLO's top box off the object (Book
4, Bottle 1, Mug 0). Confidence alone would have missed this; the box check catches it.

## Step 1 result: YOLO9000 scorer (2026-10-06)

**Built** in `training/vision.py` (`Yolo9000`). Darknet's standard `detect()` keeps only one label per
box, but its full per-class probabilities stay in the network's output buffer, so the scorer reads
all 9,418 of them for every box. Per frame it records:
- *confidence* = objectness × P(the object's accepted class), the best over all boxes;
- *top-3* = the three most specific classes with P ≥ 0.1, following the class tree (e.g. "Dixie
  cup", "teacup" for the Mug);
- *verdict* for YOLO's most confident box (objectness ≥ 0.25, else "none"): correct (accepted class
  or more specific, and box on the object), partial (only a more general class, e.g. "ball" for
  Basketball), wrong, or none.

**Lab screenshots not used.** A one-off test on the lab's existing screenshots (wide 90° VR
view, object ~1–2% of the frame) gave only 5 of 107 correct. Too small to judge the model or the
class mapping, and the sweep renders its own frames, so these screenshots are not used anywhere.

**On zoomed renders** (today's 60-position Chair/Mug/Book/Bottle sweep, Sep 23 project, object
filling 50% of the frame) YOLO9000 works and varies with the light:

| Object | YOLO9000 correct (of 60) | YOLO11m correct (same frames) |
|---|---|---|
| Mug | 53 (88%) | 93% |
| Bottle | 35 (58%) | 93% |
| Chair | 11 (18%) | 38% |
| Book | 6 (10%) | 33% |

So YOLO9000 is weaker than YOLO11m on the objects both know, as expected, but usable; its
specific guesses ("water bottle", "rocking chair") match the class mapping. The real mapping
review will come from the sweep's top-3 columns, since the screenshots are too small to judge.

## Step 2 result: Unreal fixes (2026-10-06)

- **Flashlight fixed.** Confirmed the diagnosis first: in the new project our spotlight lit the
  table but the Mug rendered as a pure black silhouette (and cast no shadow). After copying
  BP_Torch's lighting channels too (`command_server.py`), all objects render lit.
- **Object mask added** (`capture_mask` command; `Scene.capture_mask()`): the object rendered
  alone and unlit, giving its exact silhouette. Compared with the projected 3D box: IoU 0.74
  (Mug), 0.97 (Book); the 3D box is looser for rounded or irregular shapes, as expected.
- **`start_editor.sh`** now really selects the GPU (`-graphicsadapter`, mapped from the
  nvidia-smi index; verified on GPU 1), defaults to GPU 1, warns if placement or map load fail,
  and loads the map by file path (needed for the frozen copy).
- **Settle time** re-measured in the new lighting: still 0 frames (Mug, Book).
- **Every object rendered once** (`captures/object_check/contact_sheet.jpg`): 99 of 107 frame
  well (object spans 20–50% of the frame, silhouette matches its 3D box). Problems, all in the
  object models (reported size ≠ visible geometry):
  - *rendered larger than their reported size*, so the camera zooms in too far and the object
    overflows the frame: Baseball, SafetyPin, Whistle;
  - *rendered smaller than their reported size*, so the object is small in frame: Chair (its
    model changed in the Oct 5 update), AcousticGuitar, Apron;
  - *not visible at all*: Thimble (empty frame, empty mask).

  **Measured** (camera zoomed out, size taken from the silhouette; the Golf Ball, measured the same
  way, matches its reported 4.3 cm exactly, so the method is sound):

  | Object | Reported size | Drawn size | Ratio |
  |---|---|---|---|
  | Baseball | 7.4 cm | 15 cm | ×2 |
  | Whistle | 6.1 cm long | 15.5 cm | ×2.5 |
  | SafetyPin | 5.4 cm long | 22 cm | ×4 |
  | Chair | 91 cm tall | 50 cm | ×0.55 |
  | AcousticGuitar | 99 cm long | 50 cm | ×0.5 |
  | Apron | 87 cm | 44 cm | ×0.5 |
  | Thimble | 2.4 cm | not drawn | n/a |

  The drawn geometry is a uniformly scaled version of the reported box, scaled about the
  object's base (the Chair's silhouette sits 23 cm below the box centre, on the table). So for
  these models the drawn geometry and the stored size are out of sync. Which side changed varies:
  the Chair rendered correctly (~91 cm, matching its box) in the Sep 23 project, and its stored
  size is unchanged, so its *drawn* geometry shrank in the Oct 5 update (`SM_Chair` was one of the
  changed files). Why it happened is not verified; Bradley may know. It affects the VR scene too:
  people currently see e.g. a 15 cm baseball and a 50 cm chair and guitar.

  **Workaround (decided 2026-10-06): rescale them when placed.** Each object is drawn as a
  uniformly scaled copy of its stored size, scaled about its base (confirmed: the object's pivot
  sits on the table top). So our code scales the placed object by the inverse factor
  (`training/object_scale.json`; new `scale_object` command), which makes the drawn object match
  its stored, realistic size while staying on the table. Factors: Baseball ×0.49, Whistle ×0.40,
  SafetyPin ×0.25, Chair ×1.86, AcousticGuitar ×1.99, Apron ×1.98, Thimble ×0.2 (at scale 1 the
  Thimble is not drawn at all; at 0.2 it appears at its stored size). Verified by re-measuring:
  Baseball 7.3 cm (stored 7.4), Whistle 6.1 (6.1), SafetyPin 5.4 (5.4), Guitar 100 (99). After
  the fix all 107 objects frame correctly (no object overflowing the frame or missing; median
  silhouette-vs-box IoU 0.75). Only our renders are affected; Bradley's files are unchanged, and
  the VR scene still shows the mis-sized models until he fixes them.

## Step 3 result: the sweep (finished 2026-10-06, 59 min)

6,420 live renders (107 objects × 60 light positions), no errors; `captures/sweep_2026-10-06/`
(2.7 GB: `sweep.csv`, frames, masks). Verdicts with the current rule (top answer is an accepted
class or more specific, and YOLO's box overlaps the object's mask box with IoU ≥ 0.5):

- **11.6% of frames correct** (742), 17 partial, 3,865 wrong, 1,796 no detection.
- **26 objects lighting-dependent** (correct at 10–90% of light positions), **1 always recognised**
  (Vase), **80 never recognised** (<10%); 41 objects are correct at least once.
- **Box-trained classes do far better:** the 34 objects whose class YOLO9000 learned with boxes
  average 31% accuracy (20 of them lighting-dependent); the 73 photo-only objects average 2.4%
  (6 lighting-dependent). This confirms concern 2.
- Lighting-dependent, most to least accurate: Mug 88%, ComputerMouse 87%, TennisRacket 82%,
  Apple 73%, GolfBall 73%, Phone 72%, Baseball 63%, Bottle 58%, SaltShaker 53%, WallClock 48%,
  Bowl 45%, Scissors 40%, TeddyBear 40%, Umbrella 40%, Banana 28%, Shoe 25%, Microphone 25%,
  Hourglass 22%, Orange 22%, Keyboard 20%, Laptop 20%, Joystick 18%, Chair 17%, Book 13%,
  Padlock 13%, Thimble 10%.
- **Light position** (accuracy over all objects): best from the camera's side and front-right at
  25–45° high (e.g. 21% at 30° right of front, 25° high); worst low and from behind/left (2–7%).
  By height: 5° 7.6%, 25° 12.9%, 45° 13.6%, 65° 12.1%, 85° 11.6%.
- **Boxes:** 32% of YOLO's top boxes overlap the object's box by less than half (IoU < 0.5); 22%
  are mostly *not* object pixels (object fraction < 0.5: shadow, table or background).
- YOLO11m on its 31 objects: mean confidence 0.39 vs YOLO9000's 0.22 on the same frames.

**Mapping review (top answers of objects under 50% accuracy, computed from
`sweep.csv`).** Most wrong answers are genuinely wrong (Harmonica → "laptop", Dumbbell → "traffic
light", Lemon → "tennis ball", BeerGlass → "Dixie cup"); "book" and "traffic light" are YOLO9000's
most common wrong guesses for unfamiliar objects. Borderline cases: WallClock → "analog clock" /
"pendulum clock" (sibling kinds of clock, currently wrong); Plant → "vase" (its pot).

**Box-rule effect.** 763 top boxes lie mostly on the object (object fraction ≥ 0.5) but cover only
part of it (IoU < 0.5), e.g. Plant's pot, Joystick's stick, Laptop's keyboard, and currently
count as wrong; 315 boxes pass IoU ≥ 0.5 while being mostly not object. Object fraction alone
penalises thin or open objects (Scissors: correct-looking boxes are mostly background between the
blades). Accuracy under alternative rules:

| Rule | Correct frames | Lighting-dep. / always / never | Joystick | Laptop |
|---|---|---|---|---|
| IoU ≥ 0.5 (current) | 11.6% | 26 / 1 / 80 | 18% | 20% |
| object fraction ≥ 0.5 | 10.9% | 23 / 1 / 83 | 70% | 27% |
| either | 13.1% | 26 / 1 / 80 | 72% | 47% |
| no box check | 13.4% | 26 / 1 / 80 | 72% | 48% |

Adding "clock" for WallClock and "vase" for Plant raises WallClock to 100% and Plant to 80%
(with "either").

**Decisions (2026-10-06):** box rule **"either"**: YOLO's box counts as on the object if its IoU
with the object's mask box is ≥ 0.5 *or* at least half of the box is object pixels; a box that is
neither (around a shadow, the lit table or background) makes the answer wrong. **WallClock:** any
kind of clock counts. **Plant:** "vase" counts (YOLO9000's name for its pot). All 6,420 saved
frames were re-scored with these (`sweep --rescore`, no re-rendering); the earlier table is kept
as `sweep_before_rescore.csv`. The sweep now also records the top answer's WordNet ID
(`top1_wnid`).

## Final results (after re-scoring, 2026-10-06)

Rules: top answer is an accepted class or more specific, and YOLO's box is on the object (IoU ≥
0.5 with the mask box, or ≥ 50% object pixels); WallClock accepts any clock, Plant accepts vase.

- **14.3% of the 6,420 frames correct** (915), 23 partial, 3,686 wrong, 1,796 no detection.
- **26 objects lighting-dependent**, **2 always recognised** (Vase, WallClock), **79 never
  recognised**; 45 objects are correct at least once.
- **Box-trained classes:** 34 objects, mean accuracy 37.9%, 20 lighting-dependent.
  **Photo-only classes:** 73 objects, mean accuracy 3.2%, 6 lighting-dependent.
- **Lighting-dependent objects** (the usable set for RL with YOLO9000), most to least accurate:
  Mug 88%, ComputerMouse 88%, Umbrella 88%, TennisRacket 82%, Plant 80%, Apple 75%, GolfBall 73%,
  Phone 72%, Joystick 72%, Baseball 63%, Bottle 58%, SaltShaker 53%, Laptop 47%, Bowl 45%,
  Scissors 40%, TeddyBear 40%, Hourglass 33%, Banana 28%, Shoe 25%, Microphone 25%, Orange 22%,
  Keyboard 20%, Chair 17%, Book 13%, Padlock 13%, Thimble 10%.
- **Where the light works best** (accuracy over all objects): front-right of the camera's side at
  25–45° high (24% at 30°, 45° high); worst low and from behind/left (5–8%). By height: 5° 10.4%,
  25° 15.8%, 45° 16.5%, 65° 15.0%, 85° 13.6%.
- **Boxes off the object** (neither rule): 15% of YOLO's top boxes (696), e.g. the Chair's "bench"
  boxes on the lit table when the light is overhead.
- YOLO11m on its 31 objects: mean confidence 0.39 vs YOLO9000's 0.22 on the same frames.

**Why YOLO9000 does poorly here** (for the report): it learned box detection only for COCO's 80
classes (its other ~9,000 names come from ImageNet labels without boxes; its authors report ~16
mAP on such classes); our images are synthetic renders lit only by a spotlight in a dark room,
unlike its training photos (its common wrong guesses are shape/brightness matches: "traffic light"
for bright spots, "book" for flat rectangles, "Frisbee" for discs); many errors are near misses
among 9,000 fine-grained classes (BeerGlass → "Dixie cup", Lemon → "tennis ball"); and it is a
2016 model. Options worth discussing: an open-vocabulary detector (YOLO-World, OWLv2, Grounding
DINO), a fill light, or fine-tuning on renders (which changes what is measured).

**Deliverables built** (`python -m training.deliverables captures/sweep_2026-10-06`, 169 MB in
`deliverables/`): per-object lighting maps and image grids for all 107 objects, best vs worst for
the 6 objects with the largest confidence range (ComputerMouse, Vase, TennisRacket, Laptop,
Bottle, Apple), the recognition map (all lighting-dependent and always-recognised objects
labelled), and the average lighting map. Its colour scale runs from 0 to just above the best
position (25% accuracy, 0.15 confidence), shown on the colour bars, because averages over 107
objects span a narrow range.

## Setup for the sweep (2026-10-06)

- **Frozen project copy:** `/opt/Unrealprojects/tufaelz_frozen_2026-10-06` (Content, Config,
  Scripts, Docs and the project file from `tufaelz` as of Oct 6; 1.7 GB). The live project is
  being edited by others, so the sweep runs on this copy for reproducibility.
- **GPU 1 only** (announced to the lab). Unreal ignores `-gpuindex`; its `-graphicsadapter`
  numbering differs from `nvidia-smi`, and `-graphicsadapter=2` = nvidia-smi GPU 1 (verified).
  YOLO runs with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`.

**Why the camera zooms per object (and not one fixed view).** Object sizes range from a 2 cm
thimble to a 1 m guitar. With one fixed view wide enough for the guitar, a golf ball or safety pin
is ~12–15 px wide (~8–10 px in the 90° headset view), too small for YOLO at any light position, so
the test would measure object size, not lighting (the lab's wide screenshots: 5/107 correct). So,
as in the first runs, the zoom is set once per object (object fills 50% of the frame) and stays
fixed for all 60 light positions; only the light moves. A fixed headset-like view would be a
different study (what a person sees), in which small objects are unrecognisable regardless of
light.

**Terms used in the checks.** *Bounds* = the box Unreal stores with each model (its size; used for
placement, collision and our camera zoom); the "magenta box" is these bounds projected into the
image (`scene.projected_box`). *Drawn geometry* = what the GPU actually renders; the "green box" is
around the object's mask (the object rendered alone, unlit). Normally the two agree; Unreal has
no notion of an object's real-world size: that is set by whoever builds the model (1 Unreal unit
= 1 cm). The drawn size was measured from the silhouette's pixel width × the cm-per-pixel at the
object's distance (known camera position and field of view); validated on the Golf Ball (4.3 cm
measured = 4.3 cm stored ≈ 4.27 cm real).

## Step 3: the sweep (started 2026-10-06)

- `python -m training.scene sweep --out captures/sweep_2026-10-06`, launched as a detached
  process on the server (keeps running if the session disconnects; log `logs/sweep_run.log`;
  resumes where it stopped if re-run).
- 107 objects × 60 light positions = 6,420 live renders on the frozen copy, GPU 1 (Unreal and
  both YOLO models), ~35 s per object, ~70 min in total.
- Per object: its mask first, then each light position: frame saved, YOLO9000 scored (top-3,
  confidence, verdict), YOLO's box checked against the mask (`object_fraction` = share of the
  box that is object pixels; `coverage`; IoU; centre in box), YOLO11m confidence for the 31
  COCO-class objects. Output: `sweep.csv` (one row per object × position) and
  `frames/<Object>/`.
- Two-object test before launch: Mug 53/60 correct (same as on the Sep 23 project), Baseball
  (rescaled) 38/60 correct.
- **For the mapping review after the sweep:** Plant scored 0/60 correct. YOLO9000's top answer is
  "vase" in 55 of 60 frames (not an accepted class), and its box never overlaps the whole plant
  enough (IoU < 0.5 in all 60): it boxes the pot, not the foliage. Two questions to decide: should
  "vase" count for Plant, and should a box around a clear part of the object (high object
  fraction, low coverage) count as on the object? Frames are saved, so verdicts can be recomputed
  without re-rendering.

## For the model authors (Bradley and labmates)

**Message to send:** "For 7 objects (Baseball, Whistle, SafetyPin, Chair, AcousticGuitar,
Apron, Thimble) the drawn size doesn't match the bounds, e.g. the baseball's bounds say 7.4 cm but
it renders at 15 cm. Each object's drawn size should match its bounds, and both should be the
object's real-world size." Evidence: `captures/object_check/size_mismatch.png` (zoomed out;
magenta = bounds, green = drawn) and `captures/object_check/contact_sheet.jpg` (all 107).

**Guidelines for future objects**
1. In Blender: model at real-world size with metric units; apply scale (Ctrl+A → Scale) so the
   object's scale reads 1.0; origin at the bottom centre; export FBX with the same unit settings
   every time, one object per file.
2. In Unreal: import at uniform scale 1.0; if an object comes in the wrong size, fix it in Blender
   and re-import. If a mesh must be resized in Unreal, use the Static Mesh Editor's Build Scale
   and **Apply Changes** (rebuilds everything, including Nanite data); avoid tools that change
   only part of the mesh data.
3. Check after import: the Static Mesh Editor's approximate size should match real life; place it
   next to a reference (e.g. a 10 cm cube).
4. After any batch of new or changed objects, re-run a size check (render each object, compare its
   silhouette with its bounds), as done in Step 2.

## Plan (final, 2026-10-06)

Status: steps 1–4 done (sweep, re-score, deliverables); step 5 (GitHub) waits for approval.

1. **YOLO9000 scorer** in `training/vision.py`: confidence for the object's accepted classes,
   top-3 specific guesses, verdict (correct / partial / wrong / no detection).
   *Check:* score live renders and review the top-3 answers against `object_classes.json`
   (the class mapping is reviewed from the sweep's top-3 columns; lab screenshots are not used).
2. **Unreal fixes:** flashlight lights channel 1; `scene.py` uses the full object list;
   object-only mask capture for the true object box; `start_editor.sh` uses `-graphicsadapter`.
   *Check:* one picture of every object; mask agrees with the projected 3D box.
3. **Sweep:** 107 objects × 60 light positions (12 directions × 5 heights, light 120 cm from the
   object, camera at 165 cm, object filling 50% of a 640×640 frame), on the frozen copy and
   GPU 1. One CSV row per object × position (top-3, confidence, verdict, box check), plus saved
   frames. About 1–2 hours.
4. **Deliverables** regenerated into `deliverables/` (layout above).
5. **Commit to GitHub:** code and `object_classes.json` (images stay out).
