"""Figures for the RL agent's results ONLY, written to <out>/rl_deliverables/.

Independent of vision.deliverables (the sweep-only figures) -- no imports from or calls into
that script. Run this AFTER vision.sweep (render + score) and the RL training have produced
their files. It only ever shows what the RL agent found, compared against the true best
position (read from sweep.csv, for reference only -- never computed by the RL agent itself).

    .venv/bin/python -m vision.rl_deliverables outputs/sweep_2026-10-17 \
        --rl-table outputs/optimal_light_positions_table.csv \
        --rl outputs/optimal_light_positions.csv

<out>/rl_deliverables/
  0_overview.png               thumbnail of every deliverable next to its file name
  objects/<Object>.png         RL's chosen frame (YOLO + true box), room map, true-best marker
  2_ranking.png                every object ranked by RL confidence vs the true best available
  3_optimality_gap.png         gap between RL's result and the true best, per object
  4_restart_sensitivity.png    how much the table (Phase 1.5) result depends on start position
  summary.csv                  one row per object: RL result(s), true best, gaps, restart spread

Inputs:
  sweep            folder from vision.sweep (needs sweep.csv, settings.json, frames/). Used
                    only as reference context: the true best achievable position and the real
                    rendered frame for whatever position the RL agent chose.
  --rl-table       outputs/optimal_light_positions_table.csv  (Phase 1.5, table-only rollouts)
  --rl             outputs/optimal_light_positions.csv        (Phase 2, verified against Unreal)
At least one of --rl-table / --rl is required. Figures compare both when both are given.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from PIL import Image

from unreal_bridge import light_position

INK, INK2, SURF, SLAB = '#0b0b0b', '#52514e', '#fcfcfb', '#b9b6ae'
TABLE_COL, LIVE_COL, TRUTH_COL = '#2a78d6', '#d4a017', '#9a9993'
TRUE_BOX_COL = '#d63bd0'
VERDICT_COL = {'correct': '#0ca30c', 'wrong': '#d03b3b', 'none': '#9a9993'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK})

OUT = None


# ---- loading ---------------------------------------------------------------------------------

def load_sweep(sweep):
    """All rows, grouped by object and indexed by (object, elevation, azimuth) for exact lookup
    of whatever grid cell the RL agent reports -- its actions only ever land on real grid cells,
    so this should always hit exactly."""
    rows, by_pos = {}, {}
    for r in csv.DictReader((Path(sweep) / 'sweep.csv').open()):
        r['elevation'], r['azimuth'] = int(float(r['elevation'])), int(float(r['azimuth']))
        r['confidence'] = float(r['confidence'])
        rows.setdefault(r['object'], []).append(r)
        by_pos[(r['object'], r['elevation'], r['azimuth'])] = r
    return rows, by_pos


def true_best(rows):
    """{object: best row by verdict then confidence} -- the real achievable ceiling, for
    reference only. The RL agent never sees or uses this during training."""
    return {name: max(fs, key=lambda r: (r['verdict'] == 'correct', r['confidence'])) for name, fs in rows.items()}


def load_rl_csv(path):
    if not path or not Path(path).exists():
        return {}
    out = {}
    for row in csv.DictReader(Path(path).open()):
        row['az'] = int(round(float(row['az'])))
        row['el'] = int(round(float(row['el'])))
        row['conf'] = float(row['conf'])
        row['acc'] = float(row.get('acc', 0) or 0)
        if row.get('conf_std_across_restarts') not in (None, ''):
            row['conf_std_across_restarts'] = float(row['conf_std_across_restarts'])
        out[row['object']] = row
    return out


def resolve_row(by_pos, azimuths, elevations, obj, az, el):
    """Exact (object, elevation, azimuth) lookup; nearest-grid-cell fallback only as a safety
    net (e.g. rounding drift), since the agent's discrete action space should always land exactly."""
    if (obj, el, az) in by_pos:
        return by_pos[(obj, el, az)]
    best_az = min(azimuths, key=lambda a: min(abs(a - az), 360 - abs(a - az)))
    best_el = min(elevations, key=lambda e: abs(e - el))
    return by_pos.get((obj, best_el, best_az))


def boxes(r):
    true = [float(r['truth_%s' % k]) for k in ('x0', 'y0', 'x1', 'y1')] if r.get('truth_x0') else None
    yolo = [float(r['box_%s' % k]) for k in ('x0', 'y0', 'x1', 'y1')] if r.get('box_x0') and r['verdict'] != 'none' else None
    return true, yolo


def label(r):
    if r['verdict'] == 'none' or not r.get('top1'):
        return 'no detection'
    return '%s  %.2f confidence' % (r['top1'], float(r['top1_conf']))


# ---- room map (self-contained; no dependency on vision.deliverables) -------------------------

def room(ax, title, camera, obj, radius, table):
    ax.set_facecolor(SURF); ax.set_aspect('equal'); ax.axis('off')
    if table:
        ax.add_patch(Rectangle(table[:2], table[2], table[3], fc=SLAB, ec='none', alpha=0.85, zorder=1))
    ax.add_patch(Circle(obj[:2], radius, fill=False, ec='#b9b8b3', lw=1.5, zorder=2))
    ax.plot(*obj[:2], 'o', ms=6, color='#d03b3b', mec='white', mew=1, zorder=6)
    ax.plot(*camera[:2], 'o', ms=12, color='#2a78d6', mec='white', mew=1.5, zorder=6)
    ax.text(camera[0], camera[1] + 12, 'camera', ha='center', va='top', fontsize=9.5, color='#2a78d6', fontweight='bold')
    pad = radius + 45
    ax.set_xlim(min(camera[0], obj[0] - pad) - 25, max(camera[0], obj[0] + pad) + 25)
    ax.set_ylim(max(camera[1], obj[1] + pad) + 25, min(camera[1], obj[1] - pad) - 25)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=6)


def mark(ax, camera, obj, radius, az, el, color, text, marker='*', ms=16):
    x, y = light_position(obj, camera, az, el, radius)[:2]
    ax.plot(x, y, marker, ms=ms, color=color, mec=INK, mew=0.8, zorder=8)
    ax.annotate(text, (x, y), xytext=(0, -14), textcoords='offset points', ha='center', va='top',
                fontsize=8.5, color=color, fontweight='bold', zorder=8)


def legend_handles():
    return [Line2D([], [], ls='none', marker='*', ms=13, color=TABLE_COL, mec=INK, label='RL (table, Phase 1.5)'),
            Line2D([], [], ls='none', marker='*', ms=13, color=LIVE_COL, mec=INK, label='RL (live-verified, Phase 2)'),
            Line2D([], [], ls='none', marker='o', ms=9, color=TRUTH_COL, mec=INK, label='True best (full grid)')]


# ---- objects/<Object>.png : one result card per object ---------------------------------------

def object_card(sweep, obj, table_r, live_r, truth_r, by_pos, azimuths, elevations, camera, obj_centre, radius, table, path):
    chosen = live_r or table_r
    chosen_source = 'live-verified (Phase 2)' if live_r else 'table-only (Phase 1.5)'
    chosen_row = resolve_row(by_pos, azimuths, elevations, obj, chosen['az'], chosen['el'])

    fig = plt.figure(figsize=(13.5, 6.0), facecolor=SURF)
    a1 = fig.add_axes([0.02, 0.1, 0.36, 0.8])
    a1.imshow(Image.open(Path(sweep) / chosen_row['frame']))
    a1.set_xticks([]); a1.set_yticks([])
    for s in a1.spines.values():
        s.set_color(VERDICT_COL[chosen_row['verdict']]); s.set_linewidth(4)
    true, yolo = boxes(chosen_row)
    if true:
        a1.add_patch(Rectangle(true[:2], true[2] - true[0], true[3] - true[1], fill=False, ec=TRUE_BOX_COL, lw=1.6, ls=(0, (3, 2))))
    if yolo:
        a1.add_patch(Rectangle(yolo[:2], yolo[2] - yolo[0], yolo[3] - yolo[1], fill=False, ec=VERDICT_COL[chosen_row['verdict']], lw=2.2))
    a1.set_title('RL-chosen frame  (%s)' % chosen_source, fontsize=11.5, fontweight='bold')
    a1.set_xlabel(label(chosen_row), fontsize=10, color=INK2)

    a2 = fig.add_axes([0.46, 0.08, 0.5, 0.84])
    room(a2, 'Light position', camera, obj_centre, radius, table)
    mark(a2, camera, obj_centre, radius, truth_r['azimuth'], truth_r['elevation'], TRUTH_COL, 'true best', marker='o', ms=11)
    if table_r:
        mark(a2, camera, obj_centre, radius, table_r['az'], table_r['el'], TABLE_COL, 'RL table')
    if live_r:
        mark(a2, camera, obj_centre, radius, live_r['az'], live_r['el'], LIVE_COL, 'RL live')
    fig.legend(handles=legend_handles(), loc='upper center', ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))

    gap_txt = []
    if table_r:
        gap_txt.append('table: conf %.2f (true best %.2f, gap %.2f)   acc %.2f   verdict %s'
                        % (table_r['conf'], truth_r['confidence'], truth_r['confidence'] - table_r['conf'], table_r['acc'], table_r.get('verdict', '?')))
    if live_r:
        gap_txt.append('live:  conf %.2f (true best %.2f, gap %.2f)   acc %.2f   verdict %s'
                        % (live_r['conf'], truth_r['confidence'], truth_r['confidence'] - live_r['conf'], live_r['acc'], live_r.get('verdict', '?')))
    fig.text(0.02, 0.015, '\n'.join(gap_txt), fontsize=10, color=INK2, family='DejaVu Sans Mono')
    fig.suptitle(obj, fontsize=15, fontweight='bold', x=0.02, ha='left', y=0.985)
    fig.savefig(path, dpi=130, facecolor=SURF); plt.close(fig)


# ---- 2: ranking --------------------------------------------------------------------------------

def ranking(objects, truth, table_rl, live_rl, path):
    rank_key = lambda o: (live_rl.get(o) or table_rl.get(o) or {'conf': 0})['conf']
    objects = sorted(objects, key=rank_key, reverse=True)
    y = np.arange(len(objects))
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(objects) + 1.5), facecolor=SURF)
    ax.set_facecolor(SURF)
    for i, obj in enumerate(objects):
        tb = truth[obj]['confidence']
        ax.plot([0, tb], [i, i], color='#d9d8d3', lw=5, solid_capstyle='round', zorder=1)
        ax.plot(tb, i, 'o', ms=7, color=TRUTH_COL, mec=INK, mew=0.5, zorder=3)
        if obj in table_rl:
            ax.plot(table_rl[obj]['conf'], i, '*', ms=13, color=TABLE_COL, mec=INK, mew=0.6, zorder=4)
        if obj in live_rl:
            ax.plot(live_rl[obj]['conf'], i, '*', ms=13, color=LIVE_COL, mec=INK, mew=0.6, zorder=5)
    ax.set_yticks(y); ax.set_yticklabels(objects, fontsize=9.5)
    ax.set_xlim(0, 1.02); ax.set_xlabel('Confidence', fontsize=11, color=INK2)
    ax.set_title('RL result vs. true best available position, by object', fontsize=13, fontweight='bold')
    ax.grid(axis='x', color='#e4e3df', lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    fig.legend(handles=legend_handles(), loc='upper center', ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140, facecolor=SURF); plt.close(fig)


# ---- 3: optimality gap -------------------------------------------------------------------------

def optimality_gap(objects, truth, table_rl, live_rl, path):
    objects = sorted(objects, key=lambda o: (truth[o]['confidence'] - (live_rl.get(o) or table_rl.get(o) or {'conf': 0})['conf']))
    x = np.arange(len(objects))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(objects)), 5), facecolor=SURF)
    ax.set_facecolor(SURF)
    has_table, has_live = bool(table_rl), bool(live_rl)
    if has_table:
        gaps = [truth[o]['confidence'] - table_rl[o]['conf'] if o in table_rl else 0 for o in objects]
        ax.bar(x - w / 2 if has_live else x, gaps, width=w if has_live else w * 1.6, color=TABLE_COL, label='table (Phase 1.5)')
    if has_live:
        gaps = [truth[o]['confidence'] - live_rl[o]['conf'] if o in live_rl else 0 for o in objects]
        ax.bar(x + w / 2 if has_table else x, gaps, width=w if has_table else w * 1.6, color=LIVE_COL, label='live (Phase 2)')
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(objects, rotation=60, ha='right', fontsize=8.5)
    ax.set_ylabel('True best confidence  −  RL confidence', fontsize=10.5, color=INK2)
    ax.set_title('Optimality gap: how far RL landed from the true best position', fontsize=13, fontweight='bold')
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis='y', color='#e4e3df', lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURF); plt.close(fig)


# ---- 4: restart sensitivity (table / Phase 1.5 only) --------------------------------------------

def restart_sensitivity(objects, table_rl, path):
    objects = [o for o in objects if o in table_rl and 'conf_std_across_restarts' in table_rl[o]]
    if not objects:
        return False
    objects = sorted(objects, key=lambda o: -table_rl[o]['conf_std_across_restarts'])
    x = np.arange(len(objects))
    stds = [table_rl[o]['conf_std_across_restarts'] for o in objects]
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(objects)), 4.5), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.bar(x, stds, color=TABLE_COL)
    ax.set_xticks(x); ax.set_xticklabels(objects, rotation=60, ha='right', fontsize=8.5)
    ax.set_ylabel('Std. dev. of confidence across restarts', fontsize=10.5, color=INK2)
    n = next(iter(table_rl.values())).get('n_restarts', '?')
    ax.set_title('Sensitivity to starting position  (%s random restarts per object, Phase 1.5)' % n,
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', color='#e4e3df', lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURF); plt.close(fig)
    return True


# ---- 0: overview -------------------------------------------------------------------------------

def overview(example_obj, has_restart, path):
    items = [('objects/%s.png' % example_obj, 'objects/<Object>.png', 'Per-object result', 'one per object, in objects/'),
             ('2_ranking.png', '2_ranking.png', 'Ranking', 'top level'),
             ('3_optimality_gap.png', '3_optimality_gap.png', 'Optimality gap', 'top level')]
    if has_restart:
        items.append(('4_restart_sensitivity.png', '4_restart_sensitivity.png', 'Restart sensitivity', 'top level'))
    n = len(items)
    fig = plt.figure(figsize=(18, 6.5 * ((n + 1) // 2)), facecolor=SURF)
    cols = 2
    for i, (src, fname, title, where) in enumerate(items):
        r, c = divmod(i, cols)
        ax = fig.add_axes([0.03 + c * 0.49, 0.97 - (r + 1) / ((n + 1) // 2) + 0.02, 0.45, 1 / ((n + 1) // 2) - 0.08])
        ax.imshow(Image.open(OUT / src)); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color('#cdcdc8')
        ax.set_title('%s\n' % title, fontsize=14, fontweight='bold', loc='left', pad=4)
        ax.text(0, 1.012, '%s   ·   %s' % (fname, where), transform=ax.transAxes, fontsize=9.5,
                color=INK2, family='DejaVu Sans Mono', va='bottom')
    fig.savefig(path, dpi=110, facecolor=SURF); plt.close(fig)


# ---- summary.csv ---------------------------------------------------------------------------------

def write_summary(objects, truth, table_rl, live_rl, path):
    fields = ['object', 'true_best_conf', 'true_best_acc', 'true_best_az', 'true_best_el',
              'table_conf', 'table_acc', 'table_az', 'table_el', 'table_gap', 'table_conf_std_restarts',
              'live_conf', 'live_acc', 'live_az', 'live_el', 'live_gap']
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for obj in objects:
            t = truth[obj]
            row = {'object': obj, 'true_best_conf': round(t['confidence'], 4), 'true_best_acc': t['verdict'] == 'correct',
                   'true_best_az': t['azimuth'], 'true_best_el': t['elevation']}
            if obj in table_rl:
                tr = table_rl[obj]
                row.update(table_conf=round(tr['conf'], 4), table_acc=round(tr['acc'], 4), table_az=tr['az'], table_el=tr['el'],
                           table_gap=round(t['confidence'] - tr['conf'], 4),
                           table_conf_std_restarts=round(tr.get('conf_std_across_restarts', 0), 4) if 'conf_std_across_restarts' in tr else '')
            if obj in live_rl:
                lr = live_rl[obj]
                row.update(live_conf=round(lr['conf'], 4), live_acc=round(lr['acc'], 4), live_az=lr['az'], live_el=lr['el'],
                           live_gap=round(t['confidence'] - lr['conf'], 4))
            w.writerow(row)


# ---- main -----------------------------------------------------------------------------------

def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep', help='sweep folder (contains sweep.csv, settings.json and frames/)')
    ap.add_argument('--rl-table', help='outputs/optimal_light_positions_table.csv (Phase 1.5)')
    ap.add_argument('--rl', help='outputs/optimal_light_positions.csv (Phase 2, live-verified)')
    ap.add_argument('--out', help='output folder (default: <sweep>/rl_deliverables)')
    args = ap.parse_args()
    if not args.rl_table and not args.rl:
        ap.error('pass at least one of --rl-table or --rl')

    OUT = Path(args.out or Path(args.sweep) / 'rl_deliverables')
    (OUT / 'objects').mkdir(parents=True, exist_ok=True)

    rows, by_pos = load_sweep(args.sweep)
    truth = true_best(rows)
    settings = json.loads((Path(args.sweep) / 'settings.json').read_text())
    camera, radius = settings['camera']['location'], settings['light']['radius_cm']
    azimuths = sorted({r['azimuth'] for r in next(iter(rows.values()))})
    elevations = sorted({r['elevation'] for r in next(iter(rows.values()))})

    lights = np.array([[float(r['light_x']), float(r['light_y'])] for fs in rows.values() for r in fs])
    obj_centre = list(lights.mean(axis=0)) + [0.0]
    size = settings['table']['size_cm']
    table = (obj_centre[0] - size[0] / 2, obj_centre[1] - size[1] / 2, size[0], size[1]) if size else None

    table_rl = load_rl_csv(args.rl_table)
    live_rl = load_rl_csv(args.rl)
    objects = sorted(set(table_rl) | set(live_rl))
    if not objects:
        raise SystemExit('no objects found in --rl-table / --rl CSVs')

    for obj in objects:
        object_card(args.sweep, obj, table_rl.get(obj), live_rl.get(obj), truth[obj], by_pos,
                    azimuths, elevations, camera, obj_centre, radius, table, OUT / 'objects' / ('%s.png' % obj))

    ranking(objects, truth, table_rl, live_rl, OUT / '2_ranking.png')
    optimality_gap(objects, truth, table_rl, live_rl, OUT / '3_optimality_gap.png')
    has_restart = restart_sensitivity(objects, table_rl, OUT / '4_restart_sensitivity.png')
    write_summary(objects, truth, table_rl, live_rl, OUT / 'summary.csv')
    overview(objects[0], has_restart, OUT / '0_overview.png')
    print('%d objects -> %s' % (len(objects), OUT))


if __name__ == '__main__':
    main()
