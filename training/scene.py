"""Scene geometry for the flashlight task, and calibration of camera framing and light radius.

Coordinates are Unreal world centimetres (left-handed, Z up). Light angles are relative to the
camera, so they mean the same thing whatever the table's orientation:

    azimuth   0 = light on the camera's side of the object, 90 = to the camera's right,
              180 = behind the object
    elevation 0 = level with the object's centre, 90 = directly above it

The camera stays at the horizontal position recorded in the project's Scripts/pose.json (its
height is configurable: that recording is at seated eye height), is aimed at the object, and zooms
so the object's bounding sphere fills a fixed fraction of the frame. Per-object zoom keeps the
reward about lighting rather than object size: one field of view for all ten objects either
crops the 91 cm chair or leaves the 1.6 cm-thick phone a few pixels tall.

    .venv/bin/python -m training.scene calibrate   # also rewrites calibration.md and the room map
    .venv/bin/python -m training.scene map         # redraw captures/room_map.png only
    .venv/bin/python -m training.scene sweep       # score a grid of light positions per object
    (all three need ./infra/start_editor.sh running)
"""
import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from bridge_client.unreal_client import UnrealClient, wait_for_file

REPO = Path(__file__).resolve().parents[1]
POSE = Path('/opt/Unrealprojects/tufaelz/Scripts/pose.json')
CONFIG = REPO / 'training' / 'scene_config.json'
MAP = REPO / 'captures' / 'room_map.png'
OBJECT_SCALE = json.loads((REPO / 'training' / 'object_scale.json').read_text())['scale']
OBJECTS = ('Mug', 'Shoe', 'Apple', 'Book', 'Bottle', 'Chair', 'Lamp', 'Bowl', 'Phone', 'Plant')


def _norm(v):
    return v / np.linalg.norm(v)


def look_at(src, dst):
    d = np.asarray(dst, float) - np.asarray(src, float)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return [pitch, yaw, 0.0]


def camera_pose(target, object_radius, fill, height=None):
    """Camera at the pose.json position (optionally at another height), aimed at the target and
    zoomed so a sphere of `object_radius` spans `fill` of the frame width."""
    loc = np.array(json.loads(POSE.read_text())['camera']['location'], float)
    if height is not None:
        loc[2] = height
    distance = np.linalg.norm(loc - np.asarray(target, float))
    fov = math.degrees(2 * math.atan(object_radius / (fill * distance)))
    return {'location': loc.tolist(), 'rotation': look_at(loc, target), 'fov': fov}


def light_position(centre, camera_location, azimuth, elevation, radius):
    """Point on the sphere around `centre`, with angles relative to the camera (see module doc)."""
    centre = np.asarray(centre, float)
    towards_camera = np.asarray(camera_location, float) - centre
    towards_camera[2] = 0
    towards_camera = _norm(towards_camera)
    right = np.array([towards_camera[1], -towards_camera[0], 0.0])   # camera's right, Z-up LH
    az, el = math.radians(azimuth), math.radians(elevation)
    horizontal = math.cos(az) * towards_camera + math.sin(az) * right
    return (centre + radius * (math.cos(el) * horizontal + math.sin(el) * np.array([0, 0, 1.0]))).tolist()


def projected_box(camera, centre, size_cm, image_size):
    """The object's 3D bounding box (world-aligned, centred on `centre`) projected into the
    camera image: [x0, y0, x1, y1] in pixels. Unreal cameras look along +X with Y right, Z up."""
    pitch, yaw = math.radians(camera['rotation'][0]), math.radians(camera['rotation'][1])
    fwd = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    up = np.array([-math.sin(pitch) * math.cos(yaw), -math.sin(pitch) * math.sin(yaw), math.cos(pitch)])
    focal = (image_size / 2) / math.tan(math.radians(camera['fov']) / 2)
    xs, ys = [], []
    for corner in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
        d = np.asarray(centre, float) + corner * np.asarray(size_cm, float) / 2 - np.asarray(camera['location'], float)
        xs.append(image_size / 2 + focal * d.dot(right) / d.dot(fwd))
        ys.append(image_size / 2 - focal * d.dot(up) / d.dot(fwd))
    return [max(0.0, min(xs)), max(0.0, min(ys)), min(image_size, max(xs)), min(image_size, max(ys))]


def mask_box(mask):
    """Tight box [x0, y0, x1, y1] around the True pixels of a mask, or None if it is empty."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


class Scene:
    """One object on the table, the fixed camera, and the flashlight."""

    def __init__(self, ue, width=640, height=640, world='editor', camera_height=None, fill=0.6,
                 supersample=2):
        self.ue, self.width, self.height, self.world = ue, width, height, world
        # Scene captures get no anti-aliasing here, so render larger and downsample instead.
        self.supersample = supersample
        self.camera_height, self.fill = camera_height, fill
        self.info = ue.call('describe', world=world, nearby_cm=900)
        self.sizes = {n: v['radius_cm'] for n, v in self.info['objects'].items()}
        self.torch = self.info['torch_light']
        self.tmp = Path(tempfile.mkdtemp(prefix='vrnav_', dir='/dev/shm'))
        self.name = self.centre = self.camera = None
        ue.call('light_setup', world=world)

    def close(self):
        for call in ('light_cleanup', 'capture_cleanup'):
            self.ue.call(call)
        self.ue.call('remove', world=self.world)
        for f in self.tmp.glob('*'):
            f.unlink()
        self.tmp.rmdir()

    def load(self, name):
        """Put `name` on the table (scaled if its model needs it, see object_scale.json) and frame it.
        The centre is taken before scaling: it is where the realistic-size object sits."""
        self.name = name
        self.centre = self.ue.call('spawn', object=name, world=self.world)['object_bounds_center_cm']
        if name in OBJECT_SCALE:
            self.ue.call('scale_object', scale=OBJECT_SCALE[name], world=self.world)
        self.aim_camera()
        return self.centre

    def aim_camera(self, fill=None):
        self.fill = fill or self.fill
        self.camera = camera_pose(self.centre, self.sizes[self.name], self.fill, self.camera_height)
        self.ue.call('capture_setup', world=self.world, camera=self.camera,
                     width=self.width * self.supersample, height=self.height * self.supersample)

    def light(self, azimuth, elevation, radius):
        loc = light_position(self.centre, self.camera['location'], azimuth, elevation, radius)
        self.ue.call('set_light', location=loc, target=self.centre)

    def capture(self, settle_frames=0):
        path = self.tmp / 'frame.png'
        path.unlink(missing_ok=True)
        self.ue.call('capture', path=str(path), settle_frames=settle_frames)
        wait_for_file(str(path))
        img = Image.open(path).convert('RGB')
        if self.supersample > 1:
            img = img.resize((self.width, self.height), Image.LANCZOS)
        return np.asarray(img)

    def capture_mask(self):
        """Boolean image (at output size) of the pixels the object covers, from an unlit
        object-only render, so shadows and lighting do not change it."""
        path = self.tmp / 'mask.png'
        path.unlink(missing_ok=True)
        self.ue.call('capture_mask', path=str(path), world=self.world)
        wait_for_file(str(path))
        grey = Image.open(path).convert('L').resize((self.width, self.height), Image.BOX)
        return np.asarray(grey) > 0

    def object_box(self):
        """The current object's 3D bounds projected into the frame, as a cross-check on the mask."""
        size = self.info['objects'][self.name]['size_cm']
        return projected_box(self.camera, self.centre, size, self.width)


# -- calibration ------------------------------------------------------------------------------

# Light directions averaged over, since confidence depends strongly on direction.
CAL_DIRECTIONS = [(0, 30), (60, 45), (-60, 45), (135, 35)]


def _sheet(rows, path, thumb=160):
    """rows: list of (label, [(caption, image), ...]) -> one contact sheet."""
    from PIL import ImageDraw
    ncol = max(len(r[1]) for r in rows)
    sheet = Image.new('RGB', (90 + ncol * thumb, len(rows) * (thumb + 16) + 4), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, (label, cells) in enumerate(rows):
        y = i * (thumb + 16) + 2
        draw.text((4, y + thumb // 2), label, fill=(230, 230, 230))
        for j, (caption, img) in enumerate(cells):
            sheet.paste(Image.fromarray(img).resize((thumb, thumb)), (90 + j * thumb, y))
            draw.text((92 + j * thumb, y + thumb + 1), caption, fill=(200, 200, 120))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)


def calibrate_settle(scene, radius):
    """Frames needed after a light move before the image stops changing (Lumen, temporal AA)."""
    a, b = (-70, 20), (40, 45)

    def after_move(settle):
        scene.light(*a, radius)
        scene.capture(40)                       # converge at A so every trial starts alike
        scene.light(*b, radius)
        return scene.capture(settle).astype(float)

    ref, ref2 = after_move(80), after_move(80)
    floor = float(np.abs(ref - ref2).mean())   # frame-to-frame noise when fully converged
    rows = [(s, float(np.abs(after_move(s) - ref).mean())) for s in (0, 1, 2, 4, 8, 16, 32)]
    tolerance = max(1.5 * floor, 0.05)
    chosen = next((s for s, d in rows if d <= tolerance), rows[-1][0])
    return {'settle_frames': chosen, 'noise_floor': round(floor, 4),
            'diff_by_frames': {s: round(d, 4) for s, d in rows}, 'tolerance': round(tolerance, 4)}


def sweep(scene, scorer, settings, configure, settle, sheet_name, caption):
    """For every object and setting: `configure(setting)` returns the light radius to use; score
    the mean YOLO confidence over CAL_DIRECTIONS, plus brightness and clipping on the object."""
    results, rows = {}, []
    for name in OBJECTS:
        scene.load(name)
        results[name], cells = {}, []
        for setting in settings:
            radius = configure(setting)
            half = max(8, int(scene.fill * scene.width / 2))
            cy, cx = scene.height // 2, scene.width // 2
            confs, bright, clipped = [], [], []
            for az, el in CAL_DIRECTIONS:
                scene.light(az, el, radius)
                img = scene.capture(settle)
                confs.append(scorer.score(img, name))
                patch = img[cy - half:cy + half, cx - half:cx + half].mean(axis=2)
                bright.append(patch.mean())
                clipped.append((patch >= 250).mean())
                if (az, el) == CAL_DIRECTIONS[1]:
                    cells.append((caption % setting, img))
            results[name][setting] = {
                'conf': None if confs[0] is None else round(float(np.mean(confs)), 3),
                'conf_spread': None if confs[0] is None else round(float(np.ptp(confs)), 3),
                'brightness': round(float(np.mean(bright)), 1),
                'clipped_frac': round(float(np.mean(clipped)), 3)}
        rows.append((name, cells))
        print('  %-7s' % name, {s: results[name][s]['conf'] for s in settings}, flush=True)
    _sheet(rows, REPO / 'captures' / 'calibration' / sheet_name)
    return results


def _best(results, prefer, allowed=None, tolerance=0.02):
    """Mean confidence per setting across the COCO-scorable objects. Settings within `tolerance`
    of the best are a tie -- each mean covers only a few light directions, so smaller gaps are
    noise -- broken by `prefer` (min or max of the setting value)."""
    settings = [s for s in next(iter(results.values())) if allowed is None or allowed(s)]
    means = {}
    for s in settings:
        vals = [results[o][s]['conf'] for o in results if results[o][s]['conf'] is not None]
        means[s] = float(np.mean(vals)) if vals else 0.0
    top = max(means.values())
    chosen = prefer(s for s in settings if means[s] >= top - tolerance)
    return chosen, {s: round(m, 3) for s, m in means.items()}


# Light positions scored by `sweep` (and drawn on the room map): 12 directions x 5 heights.
SWEEP_AZ_STEP = 30
SWEEP_ELEVATIONS = [5, 25, 45, 65, 85]


def draw_map(info, config, path):
    """Top and side view of the room with the configured camera, zoom range and light sphere."""
    from PIL import ImageDraw, ImageFont
    fonts = '/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf'
    font, bold = ImageFont.truetype(fonts % '', 13), ImageFont.truetype(fonts % '-Bold', 15)
    near = {n['actor']: n for n in info['nearby']}

    def bounds(n):
        return ([n['centre'][i] - n['half_size'][i] for i in range(3)],
                [n['centre'][i] + n['half_size'][i] for i in range(3)])

    floor_lo, floor_hi = bounds(next(v for k, v in near.items() if 'Floor' in k))
    table_lo, table_hi = bounds(next(v for k, v in near.items() if 'Table' in k))
    table = info['table_centre_cm']
    cam_cfg, light_cfg = config['camera'], config['light']
    radius = light_cfg['radius_cm']
    el_lo, el_hi = light_cfg['elevation_range_deg']
    cam = camera_pose(table, 1, 1, cam_cfg['height_cm'])['location']
    obj_z = table_hi[2] + info['objects']['Mug']['size_cm'][2] / 2
    obj = [table[0], table[1], obj_z]

    # Frame the camera, table and light sphere rather than the whole (mostly empty) room.
    S = 1.0
    X0, X1 = cam[0] - 120, min(floor_hi[0], table[0] + radius + 250) + 60
    Y0, Y1 = max(floor_lo[1] - 40, table[1] - radius - 250), min(floor_hi[1] + 40, table[1] + radius + 200)
    Z0, Z1 = -60, 470
    W = int((X1 - X0) * S)
    top = Image.new('RGB', (W, int((Y1 - Y0) * S)), (250, 250, 247))
    side = Image.new('RGB', (W, int((Z1 - Z0) * S)), (250, 250, 247))
    dt, ds = ImageDraw.Draw(top), ImageDraw.Draw(side)

    # +Y downward, as in Unreal's own top view: looking along +X, the camera's right is down.
    def tp(x, y):
        return ((x - X0) * S, (y - Y0) * S)

    def sp(x, z):
        return ((x - X0) * S, (Z1 - z) * S)

    def rect(d, f, a, b, **kw):
        p, q = f(*a), f(*b)
        d.rectangle([min(p[0], q[0]), min(p[1], q[1]), max(p[0], q[0]), max(p[1], q[1])], **kw)

    def dot(d, f, xy, r, colour):
        x, y = f(*xy)
        d.ellipse([x - r, y - r, x + r, y + r], fill=colour)

    grey, dark, wood = (205, 205, 200), (90, 90, 90), (160, 110, 60)
    blue, pale_blue, amber = (40, 90, 200), (150, 180, 235), (220, 140, 10)

    rect(dt, tp, floor_lo[:2], floor_hi[:2], outline=dark, width=2)
    rect(ds, sp, (floor_lo[0], floor_lo[2]), (floor_hi[0], floor_hi[2]), fill=grey)
    for k, n in near.items():
        lo, hi = bounds(n)
        if 'Wall' in k:
            rect(dt, tp, lo[:2], hi[:2], fill=grey)
        # From the side only walls thin along X (and the roof) are edges; a wall running
        # along X would cover the whole view.
        if 'Roof' in k or ('Wall' in k and n['half_size'][0] < 60):
            rect(ds, sp, (lo[0], lo[2]), (hi[0], hi[2]), fill=grey)

    # Zoom range: the narrowest and widest per-object fields of view.
    fovs = {}
    for name, o in info['objects'].items():
        centre = [table[0], table[1], table_hi[2] + o['size_cm'][2] / 2]
        fovs[name] = camera_pose(centre, o['radius_cm'], cam_cfg['fill'], cam_cfg['height_cm'])
    narrow, wide = min(fovs, key=lambda n: fovs[n]['fov']), max(fovs, key=lambda n: fovs[n]['fov'])
    for name, shade in ((wide, pale_blue), (narrow, blue)):
        pose = fovs[name]
        pitch, yaw = math.radians(pose['rotation'][0]), math.radians(pose['rotation'][1])
        half = math.radians(pose['fov'] / 2)
        for sign in (-1, 1):
            a, b = yaw + sign * half, pitch + sign * half
            dt.line([tp(*cam[:2]), tp(cam[0] + 400 * math.cos(a), cam[1] + 400 * math.sin(a))], fill=shade, width=2)
            ds.line([sp(cam[0], cam[2]), sp(cam[0] + 300 * math.cos(b), cam[2] + 300 * math.sin(b))], fill=shade, width=2)

    rect(dt, tp, table_lo[:2], table_hi[:2], fill=wood)
    rect(ds, sp, (table_lo[0], table_hi[2] - 6), (table_hi[0], table_hi[2]), fill=wood)
    for lx in (table_lo[0] + 6, table_hi[0] - 6):
        rect(ds, sp, (lx - 3, 0), (lx + 3, table_hi[2]), fill=wood)

    # Light sphere: its equator from above; the usable elevation band from the side.
    r = radius * S
    cx, cy = tp(*table[:2])
    dt.ellipse([cx - r, cy - r, cx + r, cy + r], outline=amber, width=3)
    ox, oz = sp(obj[0], obj[2])
    ds.arc([ox - r, oz - r, ox + r, oz + r], 180 + el_lo, 360 - el_lo, fill=amber, width=3)
    ds.arc([ox - r, oz - r, ox + r, oz + r], 360 - el_lo, 180 + el_lo, fill=(235, 215, 180), width=1)
    # The sweep grid: from above, one ring of dots per height (higher = smaller ring).
    for el in SWEEP_ELEVATIONS:
        for az in range(0, 360, SWEEP_AZ_STEP):
            p = light_position(obj, cam, az, el, radius)
            dot(dt, tp, p[:2], 3, amber)
            dot(ds, sp, (p[0], p[2]), 3, amber)
    # Direction labels just outside the sphere, clear of the dots and view lines.
    for label, az in (('front', 0), ('right', 90), ('behind', 180), ('left', 270)):
        p = light_position(obj, cam, az, 0, radius + 30)
        x, y = tp(*p[:2])
        w = dt.textlength(label, font=font)
        dt.text((x - w / 2, y - 7), label, font=bold, fill=(150, 90, 0))
    dot(dt, tp, obj[:2], 4, (200, 30, 30))
    dot(ds, sp, (obj[0], obj[2]), 4, (200, 30, 30))
    dot(dt, tp, cam[:2], 7, blue)
    dot(ds, sp, (cam[0], cam[2]), 7, blue)

    legend = [
        ('camera', blue, 'height %.0f cm, %.0f cm from the table (pose.json position)'
         % (cam[2], math.dist(cam[:2], table[:2]))),
        ('zoom', blue, 'object fills %.0f%% of the frame: %.0f° (%s) to %.0f° (%s)'
         % (100 * cam_cfg['fill'], fovs[narrow]['fov'], narrow, fovs[wide]['fov'], wide)),
        ('light', amber, 'sphere radius %.0f cm, elevation %d-%d° above the object centre'
         % (radius, el_lo, el_hi)),
        ('dots', amber, 'light positions tested: %d directions x %d heights (%s°) = %d'
         % (360 // SWEEP_AZ_STEP, len(SWEEP_ELEVATIONS), ', '.join('%d' % e for e in SWEEP_ELEVATIONS),
            360 // SWEEP_AZ_STEP * len(SWEEP_ELEVATIONS))),
    ]
    pad = 60
    img = Image.new('RGB', (W + 2 * pad, top.height + side.height + 110 + 20 * len(legend)), (250, 250, 247))
    d = ImageDraw.Draw(img)
    d.text((pad, 10), 'TOP VIEW (cm; +X right, +Y down, as in the Unreal editor)', font=bold, fill=(20, 20, 20))
    img.paste(top, (pad, 34))
    y = 34 + top.height + 16
    d.text((pad, y), 'SIDE VIEW (camera on the left, looking toward the table; +Z up)', font=bold, fill=(20, 20, 20))
    img.paste(side, (pad, y + 24))
    y += 24 + side.height + 12
    for i, (name, colour, text) in enumerate(legend):
        d.text((pad, y + i * 20), name, font=bold, fill=colour)
        d.text((pad + 70, y + i * 20), text, font=font, fill=(40, 40, 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def write_report(report, config, path):
    """Markdown tables of every calibration measurement, with the chosen settings marked."""
    fill, radius = config['camera']['fill'], config['light']['radius_cm']
    dirs = ', '.join('(%d°, %d°)' % d for d in CAL_DIRECTIONS)

    def table(results, chosen, fmt, extra=None):
        settings = list(next(iter(results.values())))
        head = ['Object'] + [('**%s**' % fmt % float(s)) if float(s) == chosen else fmt % float(s)
                             for s in settings]
        lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
        for obj, row in results.items():
            cells = []
            for s in settings:
                v = row[s]
                if v['conf'] is None:
                    cells.append('–')
                else:
                    cell = '%.2f' % v['conf']
                    if extra:
                        cell += extra(v)
                    cells.append(('**%s**' % cell) if float(s) == chosen else cell)
            lines.append('| %s | %s |' % (obj, ' | '.join(cells)))
        return '\n'.join(lines)

    settle = report['settle']
    md = [
        '# Scene calibration',
        '',
        'Generated by `python -m training.scene calibrate`; do not edit by hand. Chosen values are '
        'in bold and saved to `scene_config.json`.',
        '',
        '**Method.** Each cell is the mean YOLO11m confidence for the object\'s own COCO class, '
        'over the light directions (azimuth, elevation) %s. Frames are %dx%d, rendered at %dx '
        'and downsampled. Camera at %s cm height, pose from `%s`. "–" = no COCO class (Shoe, Lamp).'
        % (dirs, config['camera']['width'], config['camera']['height'],
           config['render'].get('supersample', 2), config['camera']['height_cm'] or 'pose.json',
           Path(config['camera']['pose_file']).name),
        '',
        '## Settle time',
        '',
        'Mean absolute pixel difference from a fully converged frame, N frames after moving the '
        'light. Noise floor (two converged frames) %.3f; chosen: **%d frames**.'
        % (settle['noise_floor'], settle['settle_frames']),
        '',
        '| Frames | ' + ' | '.join(str(k) for k in settle['diff_by_frames']) + ' |',
        '|' + '---|' * (len(settle['diff_by_frames']) + 1),
        '| Difference | ' + ' | '.join('%.3f' % v for v in settle['diff_by_frames'].values()) + ' |',
        '',
        '## Framing: fraction of the frame the object fills',
        '',
        'Light at %s cm. Mean over objects: %s. Rule: best mean; within 0.02 counts as a tie, '
        'broken toward the larger fill.' % (report.get('framing_radius', '120'), report['fill_mean_conf']),
        '',
        table(report['framing'], fill, '%.2f'),
        '',
        '## Light radius',
        '',
        'At fill %.2f. Cells: confidence (object brightness 0-255). Mean over objects: %s. Rule: only '
        'radii where the torch\'s full-intensity cone covers every scorable object (>= %.0f cm, the '
        'chair); best mean, ties toward the closer, brighter radius.'
        % (fill, report['radius_mean_conf'], report.get('radius_needed_cm', 0)),
        '',
        table(report['radius'], radius, '%.0f cm', extra=lambda v: ' (%.0f)' % v['brightness']),
        '',
        '## Sensitivity to light direction at the chosen setting',
        '',
        'Spread = best minus worst confidence across the light directions. A small spread means the '
        'flashlight barely changes the score, so that object gives the agent little to learn.',
        '',
        '| Object | Confidence | Spread |',
        '|---|---|---|',
    ]
    for obj, row in report['radius'].items():
        v = row[str(radius)] if str(radius) in row else row[radius]
        if v['conf'] is not None:
            md.append('| %s | %.2f | %.2f |' % (obj, v['conf'], v['conf_spread']))
    path.write_text('\n'.join(md) + '\n')


def calibrate(args):
    from training.vision import COCO_CLASS, Scorer
    scorer = Scorer(device=args.device)
    started = time.time()
    with UnrealClient(timeout=300) as ue:
        scene = Scene(ue, args.size, args.size, args.world, args.camera_height)
        inner = float(scene.torch['inner_cone_angle'])
        # The torch's full-intensity cone covers an object only beyond this distance.
        min_radius = {n: round(scene.sizes[n] / math.tan(math.radians(inner)), 1) for n in scene.sizes}
        try:
            print('settle...', flush=True)
            scene.load('Mug')
            settle = calibrate_settle(scene, radius=100)
            print('  ', settle, flush=True)
            s = settle['settle_frames']

            print('framing (fraction of frame the object fills)...', flush=True)
            framing = sweep(scene, scorer, args.fills,
                            lambda f: (scene.aim_camera(f), args.framing_radius)[1],
                            s, 'framing.jpg', 'fill %.2f')
            # On a tie, frame larger: more pixels on the object shows more lighting detail.
            fill, fill_means = _best(framing, prefer=max)

            print('light radius at fill %.2f...' % fill, flush=True)
            scene.fill = fill
            radii = sweep(scene, scorer, args.radii, lambda r: r, s, 'radius.jpg', '%d cm')
            # Only radii at which the torch's full-intensity cone covers every scorable object;
            # on a tie, the closest (brightest) one.
            needed = max(min_radius[n] for n in OBJECTS if COCO_CLASS[n] is not None)
            radius, radius_means = _best(radii, prefer=min, allowed=lambda r: r >= needed)
        finally:
            scene.close()
    config = {
        'camera': {'pose_file': str(POSE), 'height_cm': args.camera_height, 'fill': fill,
                   'width': args.size, 'height': args.size},
        'light': {'radius_cm': radius, 'min_radius_cm_by_object': min_radius,
                  'elevation_range_deg': [5, 85], 'settings_from': 'BP_Torch'},
        'render': {'settle_frames': s, 'world': args.world, 'supersample': scene.supersample},
        'vision': {'weights': 'yolo11m.pt'},
    }
    CONFIG.write_text(json.dumps(config, indent=2) + '\n')
    report = {'settle': settle, 'fill_mean_conf': fill_means, 'radius_mean_conf': radius_means,
              'framing_radius': args.framing_radius, 'radius_needed_cm': needed,
              'framing': framing, 'radius': radii, 'seconds': round(time.time() - started)}
    (REPO / 'logs').mkdir(exist_ok=True)
    (REPO / 'logs' / 'calibration.json').write_text(json.dumps(report, indent=2) + '\n')
    write_report(report, config, REPO / 'training' / 'calibration.md')
    draw_map(scene.info, config, MAP)
    print(json.dumps({'chosen': config, 'fill_mean_conf': fill_means,
                      'radius_mean_conf': radius_means}, indent=2))


SWEEP_FIELDS = ['object', 'id', 'elevation', 'azimuth', 'light_x', 'light_y', 'light_z', 'frame',
                'accepted', 'verdict', 'confidence',
                'top1', 'top1_wnid', 'top1_conf', 'top2', 'top2_conf', 'top3', 'top3_conf',
                'box_x0', 'box_y0', 'box_x1', 'box_y1', 'objectness',
                'iou', 'object_fraction', 'coverage', 'centre_in_box',
                'truth_x0', 'truth_y0', 'truth_x1', 'truth_y1', 'yolo11m_confidence']


def frame_row(yolo9000, path, name, mask, base):
    """One sweep row: `base` (object, position, light, YOLO11m) plus YOLO9000's result for the frame."""
    r = yolo9000.score(path, name, mask=mask)
    row = dict(base, accepted='; '.join(r['accepted']), verdict=r['verdict'], confidence=round(r['confidence'], 4))
    top = r['top3'] + [{}] * (3 - len(r['top3']))
    for i, t in enumerate(top, 1):
        row['top%d' % i], row['top%d_conf' % i] = t.get('class', ''), t.get('confidence', '')
    row['top1_wnid'] = top[0].get('wnid', '')
    if r.get('truth_box'):
        row.update(zip(('truth_x0', 'truth_y0', 'truth_x1', 'truth_y1'), [round(v) for v in r['truth_box']]))
    b = r.get('top_box')
    if b:
        row.update(zip(('box_x0', 'box_y0', 'box_x1', 'box_y1'), b['box']))
        row.update({k: b[k] for k in ('objectness', 'iou', 'object_fraction', 'coverage', 'centre_in_box') if k in b})
    return row


def _progress(k, total, name, rows, started):
    counts = {v: sum(r['verdict'] == v for r in rows) for v in ('correct', 'partial', 'wrong', 'none')}
    elapsed = time.time() - started
    print('  [%d/%d] %-16s %s  mean conf %.2f  (%.0f min elapsed, ~%.0f min left)' % (
        k + 1, total, name, counts, np.mean([float(r['confidence']) for r in rows]),
        elapsed / 60, elapsed / (k + 1) * (total - k - 1) / 60), flush=True)


def sweep_sphere(args):
    """Score every light position on an azimuth x elevation grid for every object, with YOLO9000
    (and YOLO11m where the object has a COCO class), using the calibrated camera and radius.

    Writes <out>/sweep.csv (one row per object x light position), <out>/frames/<Object>/*.png
    and each object's mask. Objects already complete in the CSV are skipped, so an interrupted
    run continues where it stopped. The true object box is the mask's box; `object_fraction` is
    the share of YOLO's box covered by object pixels (low = the box is mostly shadow or table).

    With --rescore, no rendering: the saved frames are scored again (e.g. after changing
    object_classes.json or the box rule), keeping each row's light position and YOLO11m score.
    The previous table is kept as sweep_before_rescore.csv."""
    import csv
    from training.vision import COCO_CLASS, Scorer, Yolo9000
    out = Path(args.out)
    table = out / 'sweep.csv'
    if args.rescore:
        old = list(csv.DictReader(table.open()))
        backup = out / 'sweep_before_rescore.csv'
        table.replace(backup)
        yolo9000, started = Yolo9000(), time.time()
        names = list(dict.fromkeys(r['object'] for r in old))
        with table.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
            writer.writeheader()
            for k, name in enumerate(names):
                mask = np.asarray(Image.open(out / 'frames' / name / 'mask.png')) > 0
                rows = []
                for r in (r for r in old if r['object'] == name):
                    base = {key: r[key] for key in ('object', 'id', 'elevation', 'azimuth', 'light_x', 'light_y',
                                                    'light_z', 'frame', 'yolo11m_confidence')}
                    rows.append(frame_row(yolo9000, out / r['frame'], name, mask, base))
                writer.writerows(rows); f.flush()
                _progress(k, len(names), name, rows, started)
        print('wrote', table, '(previous table: %s)' % backup)
        return
    cfg = json.loads(CONFIG.read_text())
    height = args.camera_height if args.camera_height is not None else cfg['camera']['height_cm']
    radius, settle = cfg['light']['radius_cm'], cfg['render']['settle_frames']
    azimuths = list(range(0, 360, args.az_step))
    catalogue = {o['object']: o for o in json.loads((REPO / 'training' / 'object_classes.json').read_text())['objects']}
    names = args.objects or list(catalogue)
    (out / 'frames').mkdir(parents=True, exist_ok=True)
    per_object = len(azimuths) * len(args.elevations)
    done = {}
    if table.exists():
        for row in csv.DictReader(table.open()):
            done[row['object']] = done.get(row['object'], 0) + 1
    # Drop rows of an object that was interrupted part-way; it is redone from the start.
    if any(0 < n < per_object for n in done.values()):
        rows = [r for r in csv.DictReader(table.open()) if done[r['object']] == per_object]
        with table.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=SWEEP_FIELDS); w.writeheader(); w.writerows(rows)
        done = {k: n for k, n in done.items() if n == per_object}
    todo = [n for n in names if n not in done]
    print('%d objects, %d already done, %d to go' % (len(names), len(names) - len(todo), len(todo)), flush=True)
    yolo9000, yolo11m = Yolo9000(), Scorer(device=args.device)
    new_file = not table.exists()
    started = time.time()
    with UnrealClient(timeout=300) as ue, table.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
        if new_file:
            writer.writeheader()
        scene = Scene(ue, cfg['camera']['width'], cfg['camera']['height'], cfg['render']['world'],
                      height, cfg['camera']['fill'], cfg['render'].get('supersample', 2))
        try:
            for k, name in enumerate(todo):
                scene.load(name)
                folder = out / 'frames' / name
                folder.mkdir(exist_ok=True)
                mask = scene.capture_mask()
                Image.fromarray((mask * 255).astype(np.uint8)).save(folder / 'mask.png')
                rows = []
                for el in args.elevations:
                    for az in azimuths:
                        light = light_position(scene.centre, scene.camera['location'], az, el, radius)
                        scene.light(az, el, radius)
                        img = scene.capture(settle)
                        path = folder / ('el%02d_az%03d.png' % (el, az))
                        Image.fromarray(img).save(path)
                        base = {'object': name, 'id': catalogue[name]['id'], 'elevation': el, 'azimuth': az,
                                'light_x': round(light[0], 1), 'light_y': round(light[1], 1), 'light_z': round(light[2], 1),
                                'frame': str(path.relative_to(out)),
                                'yolo11m_confidence': '' if COCO_CLASS.get(name) is None else round(yolo11m.score(img, name), 4)}
                        rows.append(frame_row(yolo9000, path, name, mask, base))
                writer.writerows(rows); f.flush()
                _progress(k, len(todo), name, rows, started)
        finally:
            scene.close()
    print('wrote', table)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    cal = sub.add_parser('calibrate', help='measure settle time, pick framing and light radius')
    cal.add_argument('--world', default='editor', choices=['editor', 'pie'])
    cal.add_argument('--size', type=int, default=640, help='square render size (YOLO native)')
    cal.add_argument('--camera-height', type=float, default=None,
                     help='camera height in cm (default: pose.json, recorded at seated eye height)')
    cal.add_argument('--fills', type=float, nargs='+', default=[0.35, 0.5, 0.65, 0.8])
    cal.add_argument('--framing-radius', type=float, default=120,
                     help='light radius while choosing the framing (covers the largest object)')
    cal.add_argument('--radii', type=float, nargs='+', default=[50, 70, 90, 120, 160, 220])
    cal.add_argument('--device', default='cuda:0', help='GPU for YOLO; Unreal renders on GPU 3')
    sub.add_parser('map', help='draw captures/room_map.png from scene_config.json and the level')
    sw = sub.add_parser('sweep', help='score a grid of light positions for every object (YOLO9000)')
    sw.add_argument('--camera-height', type=float, default=None, help='default: scene_config.json')
    sw.add_argument('--az-step', type=int, default=SWEEP_AZ_STEP)
    sw.add_argument('--elevations', type=int, nargs='+', default=SWEEP_ELEVATIONS)
    sw.add_argument('--objects', nargs='+', default=None, help='default: every object in object_classes.json')
    sw.add_argument('--out', required=True, help='output folder, e.g. captures/sweep_2026-10-06')
    sw.add_argument('--device', default='cuda:0', help='YOLO11m device; pin the GPU with CUDA_VISIBLE_DEVICES')
    sw.add_argument('--rescore', action='store_true', help='re-score the saved frames (no Unreal needed)')
    args = ap.parse_args()
    if args.cmd == 'calibrate':
        calibrate(args)
    elif args.cmd == 'sweep':
        sweep_sphere(args)
    elif args.cmd == 'map':
        with UnrealClient() as ue:
            info = ue.call('describe', nearby_cm=900)
        draw_map(info, json.loads(CONFIG.read_text()), MAP)
        print(MAP)


if __name__ == '__main__':
    main()
