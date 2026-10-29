"""Light-position sweep: a picture of every object at every light position, then YOLOE scores.

    .venv/bin/python -m vision.sweep render --out outputs/sweep_2026-10-17   # needs Unreal running
    .venv/bin/python -m vision.sweep score  --out outputs/sweep_2026-10-17   # no Unreal needed

render  writes frames/<Object>/el45_az090.png (light 45 deg high, 90 deg to the camera's right),
        frames/<Object>/mask.png, positions.csv, and settings.json (camera, light, table).
        Objects already in positions.csv are skipped, so a stopped run continues where it left
        off. --objects limits the objects.
score   runs YOLOE on every picture and writes sweep.csv (one row per picture).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from unreal_bridge import CONFIG, mask_box, object_catalog

POSITION_FIELDS = ['object', 'elevation', 'azimuth', 'light_x', 'light_y', 'light_z', 'fov', 'frame']
SWEEP_FIELDS = POSITION_FIELDS + [
    'verdict', 'confidence', 'top1', 'top1_conf', 'top2', 'top2_conf', 'top3', 'top3_conf',
    'box_x0', 'box_y0', 'box_x1', 'box_y1', 'iou', 'object_fraction',
    'truth_x0', 'truth_y0', 'truth_x1', 'truth_y1']


def render(out, objects=None, collection='common', warmup_seconds=30.0):
    if not np.isfinite(warmup_seconds) or warmup_seconds < 0:
        raise ValueError('warmup_seconds must be finite and nonnegative')
    from unreal_bridge import Unreal
    settings_path = out / 'settings.json'
    if settings_path.exists():
        saved = json.loads(settings_path.read_text()).get('collection', 'common')
        if saved != collection:
            raise ValueError('Sweep belongs to collection %s; use a separate --out folder' % saved)
    catalog = object_catalog(collection)
    if objects is not None and set(objects) - set(catalog):
        raise ValueError('Objects outside selected collection: %s' % sorted(set(objects) - set(catalog)))
    table = out / 'positions.csv'
    done = {r['object'] for r in csv.DictReader(table.open())} if table.exists() else set()
    new_file = not table.exists()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with Unreal(collection=collection) as ue, table.open('a', newline='') as f:
        writer = csv.DictWriter(f, POSITION_FIELDS)
        if new_file:
            writer.writeheader()
            # The settings this sweep was made with, for the figures and for later reference.
            settings = {'camera': CONFIG['camera'], 'light': CONFIG['light'], 'image': CONFIG['image'],
                        'table': CONFIG['table'], 'project': CONFIG['unreal']['project'],
                        'collection': collection, 'objects': catalog,
                        'warmup_seconds': warmup_seconds}
            (out / 'settings.json').write_text(json.dumps(settings, indent=2))
        todo = [n for n in (objects or ue.objects()) if n not in done]
        for k, name in enumerate(todo):
            try:
                ue.spawn(name)
            except RuntimeError as error:
                print('skipping %s: %s' % (name, error))
                continue
            # A depth mask proves geometry exists, not that its materials/textures are ready.
            # Keep the final camera rendering while Unreal prepares the newly spawned asset.
            ue.set_light(CONFIG['light']['azimuths'][0], CONFIG['light']['elevations'][0])
            print('warming up %s for %.1f seconds' % (name, warmup_seconds), flush=True)
            deadline = time.monotonic() + warmup_seconds
            while time.monotonic() < deadline:
                ue.capture()  # discard: these frames may still have fallback materials
            folder = out / 'frames' / name
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(ue.capture_mask().astype(np.uint8) * 255).save(folder / 'mask.png')
            rows = []
            for el in CONFIG['light']['elevations']:
                for az in CONFIG['light']['azimuths']:
                    light = ue.set_light(az, el)
                    path = folder / ('el%02d_az%03d.png' % (el, az))
                    Image.fromarray(ue.capture()).save(path)
                    rows.append(dict(zip(POSITION_FIELDS, [name, el, az] + [round(v, 1) for v in light]
                                         + [round(ue.camera['fov'], 2), str(path.relative_to(out))])))
            writer.writerows(rows)              # only once all of the object's pictures exist
            f.flush()
            minutes = (time.time() - started) / 60
            print('[%d/%d] %s  (%.0f min so far, ~%.0f min left)'
                  % (k + 1, len(todo), name, minutes, minutes / (k + 1) * (len(todo) - k - 1)), flush=True)


def score(out, collection=None):
    from vision.yoloe import Detector
    settings = json.loads((out / 'settings.json').read_text())
    saved = settings.get('collection', 'common')
    if collection is not None and collection != saved:
        raise ValueError('Sweep belongs to collection %s' % saved)
    detector = Detector(objects=settings.get('objects'), collection=saved)
    positions = list(csv.DictReader((out / 'positions.csv').open()))
    truth = {}
    with (out / 'sweep.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, SWEEP_FIELDS)
        writer.writeheader()
        for k, row in enumerate(positions):
            name = row['object']
            if name not in truth:
                truth[name] = mask_box(np.asarray(Image.open(out / 'frames' / name / 'mask.png')) > 0)
                print('scoring', name, flush=True)
            r = detector.score(str(out / row['frame']), name, true_box=truth[name])
            row.update(verdict=r['verdict'], confidence=round(r['confidence'], 4))
            for i, (guess, conf) in enumerate(r['top'], 1):
                row['top%d' % i], row['top%d_conf' % i] = guess, conf
            if r['box']:
                row.update(zip(['box_x0', 'box_y0', 'box_x1', 'box_y1'], r['box']))
                row.update(iou=r.get('iou'), object_fraction=r.get('object_fraction'))
            if truth[name]:
                row.update(zip(['truth_x0', 'truth_y0', 'truth_x1', 'truth_y1'], truth[name]))
            writer.writerow(row)
    print('wrote', out / 'sweep.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['render', 'score'])
    ap.add_argument('--out', type=Path, required=True, help='sweep folder, e.g. outputs/sweep_2026-10-17')
    ap.add_argument('--objects', nargs='+', help='render only these objects (default: all)')
    ap.add_argument('--warmup-seconds', type=float, default=30.0,
                    help='render and discard frames after each spawn (default: 30 seconds)')
    ap.add_argument('--collection', choices=['common', 'new'], default=None,
                    help='spawner collection (render default: common; score: saved collection)')
    args = ap.parse_args()
    if args.step == 'render':
        render(args.out, args.objects, args.collection or 'common', args.warmup_seconds)
    else:
        score(args.out, args.collection)
