# Training data

## `object_classes.json`

This file maps every Unreal catalogue ID to its exact object name and YOLOE prompt tag. It contains
all 115 objects (IDs 1-115); no objects are filtered by whether their names occur in another
classification system. Tags use readable lowercase names, with explicit overrides for `cell phone`,
`potted plant`, `computer keyboard`, and `reflex camera`.

## `object_scale.json`

Scale corrections for Unreal models whose drawn size differs from their stored bounds. They are
applied by `Scene.load()` and do not affect camera pose or detector tags.

## `scene_config.json`

Camera framing, light radius, render settings, and YOLO11m comparison settings used by the existing
scene and capture scripts. YOLOE-26n is the primary sweep and RL detector.
