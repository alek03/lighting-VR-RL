"""Figures for the lighting sweep (training.scene sweep), written to deliverables/.

    .venv/bin/python -m training.deliverables captures/sweep_2026-10-06

deliverables/
  0_overview.png                    thumbnail of every deliverable next to its file name
  3_best_vs_worst.png               objects where the light matters most: best vs worst frame
  4_recognition_map.png             every object by accuracy and confidence
  6_average_lighting_map.png        top view: mean confidence and accuracy per light position
  objects/<Object>/1_lighting_map_<Object>.png   top view: confidence and result per position
  objects/<Object>/2_image_grid_<Object>.png     all 60 frames with YOLO's box and the true box

Numbers match the deliverable list in reports/meeting3_report_notes.md. The true object box is
the box around the object's mask (the object rendered alone).
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import PercentFormatter
from PIL import Image

from training.scene import POSE, light_position

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'deliverables'
TABLE = (1080, -335, 80, 170)       # table top x, y, width, depth (cm), from captures/room_map.png
OBJ = [1120.0, -250.0, 0.0]         # table centre (the spawner)
RADIUS = 120.0
INK, INK2, SURF, WOOD = '#0b0b0b', '#52514e', '#fcfcfb', '#a0703c'
BLUE = LinearSegmentedColormap.from_list('b', ['#eef4fc', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
GREEN = LinearSegmentedColormap.from_list('g', ['#eef7ee', '#bfe3bf', '#7cc67c', '#34a234', '#0c7a0c', '#075207'])
TRUE_COL = '#d63bd0'
# verdict -> (label, colour, marker, face)
VERDICTS = {'correct': ('Correct', '#0ca30c', 'o', '#0ca30c'), 'partial': ('Partial (too general)', '#fab219', '^', '#fab219'),
            'wrong': ('Wrong', '#d03b3b', 'X', '#d03b3b'), 'none': ('No detection', '#9a9993', 'o', 'none')}
SYMBOL = {'correct': '✓', 'partial': '~', 'wrong': '✗', 'none': '–'}
DIRS = {0: 'front', 90: 'right', 180: 'behind', 270: 'left'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK})


def load(sweep):
    """Rows grouped by object, plus each object's light grid."""
    rows = defaultdict(list)
    for r in csv.DictReader((Path(sweep) / 'sweep.csv').open()):
        for k in ('elevation', 'azimuth'):
            r[k] = int(float(r[k]))
        r['confidence'] = float(r['confidence'])
        rows[r['object']].append(r)
    first = next(iter(rows.values()))
    azimuths = sorted({r['azimuth'] for r in first})
    elevations = sorted({r['elevation'] for r in first})
    return rows, azimuths, elevations


def grids(frames, azimuths, elevations):
    conf = np.zeros((len(elevations), len(azimuths)))
    verdict = np.empty(conf.shape, object)
    for r in frames:
        i, j = elevations.index(r['elevation']), azimuths.index(r['azimuth'])
        conf[i, j], verdict[i, j] = r['confidence'], r['verdict']
    return conf, verdict


def label(r):
    if r['verdict'] == 'none' or not r['top1']:
        return '– nothing'
    return '%s %s %.2f' % (SYMBOL[r['verdict']], r['top1'], float(r['top1_conf']))


def boxes(r):
    true = [float(r['truth_%s' % k]) for k in ('x0', 'y0', 'x1', 'y1')] if r['truth_x0'] else None
    yolo = [float(r['box_%s' % k]) for k in ('x0', 'y0', 'x1', 'y1')] if r['box_x0'] and r['verdict'] != 'none' else None
    return true, yolo


# ---- the top-view room map --------------------------------------------------------------------

def room(ax, title, camera):
    ax.set_facecolor(SURF); ax.set_aspect('equal'); ax.axis('off')
    yaw = math.atan2(OBJ[1] - camera[1], OBJ[0] - camera[0])
    for s in (-1, 1):
        a = yaw + s * math.radians(30)
        ax.plot([camera[0], camera[0] + 420 * math.cos(a)], [camera[1], camera[1] + 420 * math.sin(a)], color='#86b6ef', lw=1.2, zorder=1)
    ax.add_patch(Rectangle(TABLE[:2], TABLE[2], TABLE[3], fc=WOOD, ec='none', alpha=0.85, zorder=1))
    ax.add_patch(Circle(OBJ[:2], RADIUS, fill=False, ec='#b9b8b3', lw=1.5, zorder=2))
    ax.plot(*OBJ[:2], 'o', ms=6, color='#d03b3b', mec='white', mew=1, zorder=6)
    ax.plot(*camera[:2], 'o', ms=12, color='#2a78d6', mec='white', mew=1.5, zorder=6)
    for text, a, r in (('front', 13, RADIUS + 16), ('right', 90, RADIUS + 24), ('behind', 180, RADIUS + 30), ('left', 270, RADIUS + 22)):
        x, y = light_position(OBJ, camera, a, 0, r)[:2]
        ax.text(x, y, text, ha='center', va='center', fontsize=11, color=INK2, fontweight='bold')
    ax.text(camera[0], camera[1] + 12, 'camera', ha='center', va='top', fontsize=10, color='#2a78d6', fontweight='bold')
    ax.set_xlim(905, 1300); ax.set_ylim(-92, -408)            # +Y down, as in the Unreal editor
    ax.set_title(title, fontsize=13, fontweight='bold', pad=6)


def dots(azimuths, elevations, camera):
    """(row, col, x, y, size) per light position; lowest lights first so the overhead ring is on top."""
    for i, e in enumerate(elevations):
        for j, a in enumerate(azimuths):
            x, y = light_position(OBJ, camera, a, e, RADIUS)[:2]
            yield i, j, x, y, (11 if e < 80 else 8)


def colorbar(fig, rect, cmap, text, fmt=None, top=1.0):
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(0, top), cmap), cax=fig.add_axes(rect), format=fmt)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=9, length=0)
    cb.set_label(text, fontsize=10, color=INK2)


def verdict_legend(fig, anchor):
    fig.legend(handles=[Line2D([], [], ls='none', marker=mk, ms=9, color=c, mfc=f, mec=('white' if f != 'none' else c), label=n)
                        for n, c, mk, f in VERDICTS.values()],
               loc='upper center', bbox_to_anchor=anchor, ncol=4, frameon=False, fontsize=10)


# ---- 1: lighting map per object -------------------------------------------------------------

def lighting_map(frames, azimuths, elevations, camera, path):
    conf, verdict = grids(frames, azimuths, elevations)
    fig = plt.figure(figsize=(15, 7.0), facecolor=SURF)
    a1 = fig.add_axes([0.0, 0.1, 0.44, 0.84]); room(a1, 'Confidence', camera)
    for i, j, x, y, ms in dots(azimuths, elevations, camera):
        a1.plot(x, y, 'o', ms=ms, color=BLUE(conf[i, j]), mec='#6f6e69', mew=0.7, zorder=4)
    colorbar(fig, [0.45, 0.3, 0.01, 0.45], BLUE, 'Confidence')
    a2 = fig.add_axes([0.53, 0.1, 0.44, 0.84]); room(a2, 'Accuracy', camera)
    for i, j, x, y, ms in dots(azimuths, elevations, camera):
        _, col, mk, face = VERDICTS[verdict[i, j]]
        a2.plot(x, y, mk, ms=ms + (1 if mk == 'X' else 0), color=col, mfc=face,
                mec='white' if face != 'none' else col, mew=0.8 if face != 'none' else 1.5, zorder=4)
    verdict_legend(fig, (0.75, 0.09))
    fig.savefig(path, dpi=150, facecolor=SURF); plt.close(fig)


# ---- 2: image grid per object ---------------------------------------------------------------

def frame(ax, sweep, r, lw=3.5):
    ax.imshow(Image.open(Path(sweep) / r['frame'])); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(VERDICTS[r['verdict']][1]); s.set_linewidth(lw)
    true, yolo = boxes(r)
    if true:
        ax.add_patch(Rectangle(true[:2], true[2] - true[0], true[3] - true[1], fill=False, ec=TRUE_COL, lw=1.4, ls=(0, (3, 2))))
    if yolo:
        ax.add_patch(Rectangle(yolo[:2], yolo[2] - yolo[0], yolo[3] - yolo[1], fill=False, ec=VERDICTS[r['verdict']][1], lw=1.8))


def box_legend(n=4):
    return [Line2D([], [], color=TRUE_COL, lw=2, ls=(0, (3, 2)), label='True object box'),
            Line2D([], [], color=VERDICTS['correct'][1], lw=2.2, label='YOLO box: correct'),
            Line2D([], [], color=VERDICTS['partial'][1], lw=2.2, label='YOLO box: partial'),
            Line2D([], [], color=VERDICTS['wrong'][1], lw=2.2, label='YOLO box: wrong'),
            Line2D([], [], ls='none', marker='s', ms=10, mfc='none', mec=VERDICTS['none'][1], mew=2.5, label='No detection')][:n]


def image_grid(sweep, frames, azimuths, elevations, path):
    by_pos = {(r['elevation'], r['azimuth']): r for r in frames}
    fig, axes = plt.subplots(len(elevations), len(azimuths), figsize=(19, 9.4), facecolor=SURF, squeeze=False)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.885, bottom=0.03, wspace=0.06, hspace=0.32)
    for row, e in enumerate(reversed(elevations)):                 # highest light on top
        for col, a in enumerate(azimuths):
            ax = axes[row, col]; r = by_pos[(e, a)]
            frame(ax, sweep, r)
            ax.set_xlabel(label(r), fontsize=7.5, labelpad=2, color=INK)
            if row == 0:
                ax.set_title('%d°%s' % (a, ('\n' + DIRS[a]) if a in DIRS else '\n '), fontsize=9.5, color=INK2)
            if col == 0:
                ax.set_ylabel('%d° high' % e, fontsize=10, color=INK2, fontweight='bold')
    fig.legend(handles=box_legend(5), loc='upper center', ncol=5, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.995))
    fig.savefig(path, dpi=110, facecolor=SURF); plt.close(fig)


# ---- 3: best vs worst light -----------------------------------------------------------------

def light_icon(ax, a, e):
    ax.set_xlim(-1.6, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ec='#b9b8b3', lw=1))
    ax.plot(-1.45, 0, 'o', ms=5, color='#2a78d6'); ax.plot(0, 0, 'o', ms=3, color='#d03b3b')
    r, t = math.cos(math.radians(e)), math.radians(a)
    ax.plot(-r * math.cos(t), -r * math.sin(t), 'o', ms=7, color='#fab219', mec=INK, mew=0.6)


def best_worst(sweep, rows, names, path):
    n = len(names)
    fig = plt.figure(figsize=(12.5, 3.35 * n + 0.9), facecolor=SURF)
    top = 1 - 0.6 / (3.35 * n + 0.9); h = top / n
    for i, name in enumerate(names):
        fs = rows[name]
        best = max(fs, key=lambda r: (r['verdict'] == 'correct', r['confidence']))
        worst = min(fs, key=lambda r: (r['confidence'], r['verdict'] == 'correct'))
        y = top - (i + 1) * h
        fig.text(0.02, y + 0.5 * h, name, fontsize=14, fontweight='bold', va='center')
        for k, (title, r) in enumerate((('Best light', best), ('Worst light', worst))):
            x = 0.16 + k * 0.42
            frame(fig.add_axes([x, y + 0.04 * h, 0.26, 0.86 * h]), sweep, r, lw=4)
            where = DIRS.get(r['azimuth'], '%d°' % r['azimuth'])
            fig.text(x + 0.27, y + 0.78 * h, title, fontsize=12, fontweight='bold', color=VERDICTS[r['verdict']][1])
            fig.text(x + 0.27, y + 0.62 * h, 'confidence %.2f' % r['confidence'], fontsize=11)
            fig.text(x + 0.27, y + 0.49 * h, 'YOLO: %s' % label(r), fontsize=10, color=INK2)
            fig.text(x + 0.27, y + 0.36 * h, 'light: %s, %d° high' % (where, r['elevation']), fontsize=10, color=INK2)
            light_icon(fig.add_axes([x + 0.27, y + 0.06 * h, 0.1, 0.27 * h]), r['azimuth'], r['elevation'])
    fig.legend(handles=box_legend(4), loc='upper center', ncol=4, frameon=False, fontsize=10.5, bbox_to_anchor=(0.55, 1.0))
    fig.savefig(path, dpi=130, facecolor=SURF); plt.close(fig)


# ---- 4: recognition map ---------------------------------------------------------------------

def recognition_map(rows, box_trained, path):
    stats = []
    for name, fs in rows.items():
        c = np.array([r['confidence'] for r in fs])
        stats.append((name, np.mean([r['verdict'] == 'correct' for r in fs]), c.mean(), c.min(), c.max()))
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=SURF)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.1); ax.set_facecolor(SURF)
    for x0, x1, text in [(0, 0.1, 'Never recognised'), (0.1, 0.9, 'Lighting-dependent'), (0.9, 1.0, 'Always recognised')]:
        dep = text == 'Lighting-dependent'
        ax.axvspan(x0, x1, color='#e6effb' if dep else '#f1f0ec', zorder=0, lw=0)
        ax.text((x0 + x1) / 2, 1.03, text, ha='center', va='bottom', fontsize=10.5, fontweight='bold',
                color='#1c5cab' if dep else INK2, transform=ax.get_xaxis_transform())
    colours = {True: '#2a78d6', False: '#eb6834'}
    for name, acc, mean, lo, hi in stats:
        c = colours[name in box_trained]
        ax.plot([acc, acc], [lo, hi], color=c, alpha=0.35, lw=1.5, solid_capstyle='round', zorder=2)
        ax.plot(acc, mean, 'o', ms=8, color=c, mec=SURF, mew=1.5, zorder=3)
    # Label every object that is recognised at least sometimes; never-recognised ones pile up at 0%
    # and stay unlabelled. Labels go left of dots near the right edge and step down to avoid overlaps.
    placed = []
    for name, acc, mean, lo, hi in sorted(stats, key=lambda s: (-s[2])):
        if acc < 0.1:
            continue
        right = acc < 0.85
        x_text, y_text = acc + (0.012 if right else -0.012), mean
        clash = lambda yt: any(abs(yt - y) < 0.022 and abs(x_text - x) < 0.13 and side == right for x, y, side in placed)
        while clash(y_text):
            y_text -= 0.022
        if y_text < 0.02:                       # would fall below the axis: step upwards instead
            y_text = mean
            while clash(y_text):
                y_text += 0.022
        placed.append((x_text, y_text, right))
        ax.annotate(name, (acc, mean), xytext=(x_text, y_text), textcoords='data', va='center',
                    ha='left' if right else 'right', fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle='-', color='#b9b8b3', lw=0.6) if abs(y_text - mean) > 0.005 else None)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel('Accuracy: share of the light positions where YOLO9000\'s top answer is correct', fontsize=11, color=INK2)
    ax.set_ylabel('Confidence (dot = average, line = worst to best light)', fontsize=11, color=INK2)
    ax.grid(color='#e4e3df', lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    ax.legend(handles=[Line2D([], [], ls='none', marker='o', ms=8, color=colours[True], label='YOLO9000 learned this class with boxes'),
                       Line2D([], [], ls='none', marker='o', ms=8, color=colours[False], label='Learned from photo labels only')],
              loc='upper left', frameon=False, fontsize=10)
    fig.savefig(path, dpi=150, facecolor=SURF); plt.close(fig)
    return stats


# ---- 6: average lighting map ----------------------------------------------------------------

def average_map(rows, azimuths, elevations, camera, path):
    gs = [grids(fs, azimuths, elevations) for fs in rows.values()]
    mean_conf = np.mean([g[0] for g in gs], 0)
    acc = np.mean([g[1] == 'correct' for g in gs], 0)
    # Averages over many objects span a narrow range; the colour scale runs from 0 to just above
    # the best position (shown on the colour bar) so the pattern is visible.
    top_c, top_a = math.ceil(mean_conf.max() * 20) / 20, math.ceil(acc.max() * 20) / 20
    fig = plt.figure(figsize=(15, 7.0), facecolor=SURF)
    a1 = fig.add_axes([0.0, 0.06, 0.43, 0.86]); room(a1, 'Confidence (average over objects)', camera)
    for i, j, x, y, ms in dots(azimuths, elevations, camera):
        a1.plot(x, y, 'o', ms=ms, color=BLUE(mean_conf[i, j] / top_c), mec='#6f6e69', mew=0.7, zorder=4)
    colorbar(fig, [0.44, 0.25, 0.01, 0.5], BLUE, 'Mean confidence', top=top_c)
    a2 = fig.add_axes([0.505, 0.06, 0.43, 0.86]); room(a2, 'Accuracy (share of objects identified)', camera)
    for i, j, x, y, ms in dots(azimuths, elevations, camera):
        a2.plot(x, y, 'o', ms=ms, color=GREEN(acc[i, j] / top_a), mec='#6f6e69', mew=0.7, zorder=4)
    colorbar(fig, [0.945, 0.25, 0.01, 0.5], GREEN, 'Objects identified correctly', fmt=PercentFormatter(1.0), top=top_a)
    fig.savefig(path, dpi=150, facecolor=SURF); plt.close(fig)


# ---- 0: overview ----------------------------------------------------------------------------

def overview(example, path):
    items = [('objects/%s/1_lighting_map_%s.png' % (example, example), '1_lighting_map_<Object>.png', '1. Lighting map', 'one per object, in objects/<Object>/'),
             ('objects/%s/2_image_grid_%s.png' % (example, example), '2_image_grid_<Object>.png', '2. Image grid', 'one per object, in objects/<Object>/'),
             ('3_best_vs_worst.png', '3_best_vs_worst.png', '3. Best vs worst lighting', 'top level'),
             ('4_recognition_map.png', '4_recognition_map.png', '4. Recognition map', 'top level'),
             ('6_average_lighting_map.png', '6_average_lighting_map.png', '6. Average lighting map', 'top level')]
    fig = plt.figure(figsize=(18, 13), facecolor=SURF)
    rects = [(0.02, 0.53, 0.40, 0.40), (0.45, 0.53, 0.53, 0.40), (0.03, 0.04, 0.22, 0.42), (0.30, 0.04, 0.32, 0.40), (0.66, 0.04, 0.32, 0.40)]
    for (src, fname, title, where), r in zip(items, rects):
        ax = fig.add_axes(r); ax.imshow(Image.open(OUT / src)); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color('#cdcdc8')
        ax.set_title('%s\n' % title, fontsize=15, fontweight='bold', loc='left', pad=4)
        ax.text(0, 1.012, '%s   ·   %s' % (fname, where), transform=ax.transAxes, fontsize=10.5, color=INK2, family='DejaVu Sans Mono', va='bottom')
    fig.savefig(path, dpi=110, facecolor=SURF); plt.close(fig)


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sweep', help='sweep output folder (contains sweep.csv and frames/)')
    ap.add_argument('--best-worst', type=int, default=6, help='objects shown in 3_best_vs_worst.png')
    ap.add_argument('--out', default=str(OUT), help='output folder (default: deliverables/)')
    args = ap.parse_args()
    OUT = Path(args.out)
    rows, azimuths, elevations = load(args.sweep)
    camera = json.loads(POSE.read_text())['camera']['location']
    classes = json.loads((REPO / 'training' / 'object_classes.json').read_text())['objects']
    box_trained = {o['object'] for o in classes if o['yolo9000_box_trained']}
    (OUT / 'objects').mkdir(parents=True, exist_ok=True)
    for name, frames in rows.items():
        d = OUT / 'objects' / name
        d.mkdir(exist_ok=True)
        lighting_map(frames, azimuths, elevations, camera, d / ('1_lighting_map_%s.png' % name))
        image_grid(args.sweep, frames, azimuths, elevations, d / ('2_image_grid_%s.png' % name))
    stats = recognition_map(rows, box_trained, OUT / '4_recognition_map.png')
    # Best vs worst: the objects where the light matters most (largest confidence range).
    striking = [s[0] for s in sorted(stats, key=lambda s: -(s[4] - s[3]))[:args.best_worst]]
    best_worst(args.sweep, rows, striking, OUT / '3_best_vs_worst.png')
    average_map(rows, azimuths, elevations, camera, OUT / '6_average_lighting_map.png')
    overview(striking[0], OUT / '0_overview.png')
    print('%d objects -> %s' % (len(rows), OUT))


if __name__ == '__main__':
    main()
