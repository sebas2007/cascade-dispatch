#!/usr/bin/env python3
"""Cut OpenStreetMap data (.pbf or .osm) into small JSON tiles that Cascade Dispatch streams while you fly.

usage: python3 tile_osm.py OUT_DIR input1.osm.pbf [input2 ...] [--tile 4000] [--bbox S W N E]

World metres use the same projection as the game (x east, z = -north) around the NYC reference point,
so a tile never needs re-projecting in the browser.  Each tile is  t_<ix>_<iz>.json  where
ix = floor(x / tile), iz = floor(z / tile).  An index.json lists every tile that has content.
"""
import sys, os, json, math, argparse, collections
import osmium
import random
import numpy as np
import mapbox_earcut as earcut

LAT0, LON0 = 40.73, -73.41
KX, KZ = 111320 * math.cos(math.radians(LAT0)), 110540.0

def xz(lat, lon): return ((lon - LON0) * KX, -(lat - LAT0) * KZ)

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('inputs', nargs='+')
ap.add_argument('--tile', type=float, default=4000.0)
ap.add_argument('--bbox', nargs=4, type=float, metavar=('S', 'W', 'N', 'E'))
ap.add_argument('--min-building', type=float, default=14.0, help='drop buildings smaller than this many m2')
ap.add_argument('--max-per-tile', type=int, default=9000, help='safety cap on buildings per tile')
args = ap.parse_args()
T = args.tile

RW = {'motorway': 16, 'motorway_link': 8, 'trunk': 14, 'trunk_link': 7, 'primary': 14, 'primary_link': 7, 'secondary': 12, 'secondary_link': 7, 'tertiary': 10, 'tertiary_link': 6, 'residential': 8, 'unclassified': 8, 'living_street': 6, 'service': 5, 'track': 4, 'cycleway': 2.2}
RC = {'motorway': 0, 'motorway_link': 0, 'trunk': 0, 'trunk_link': 0, 'primary': 1, 'primary_link': 1, 'secondary': 1, 'secondary_link': 1, 'tertiary': 2, 'tertiary_link': 2, 'residential': 3, 'unclassified': 3, 'living_street': 3, 'service': 4, 'track': 5, 'cycleway': 6}
LU = {'grass': 0, 'meadow': 0, 'village_green': 0, 'recreation_ground': 0, 'farmland': 0, 'orchard': 0, 'residential': 1, 'commercial': 2, 'retail': 2, 'industrial': 2, 'forest': 3, 'cemetery': 7, 'construction': 9, 'brownfield': 9, 'allotments': 0}
HOUSE = {'house', 'detached', 'residential', 'semidetached_house', 'terrace'}
SMALL = {'garage', 'garages', 'shed', 'carport', 'roof', 'service'}
BIG = {'commercial', 'retail', 'industrial', 'warehouse', 'apartments', 'school', 'church', 'office'}
HEIGHT = {'house': 7.5, 'detached': 7.5, 'residential': 8, 'garage': 3.2, 'garages': 3.2, 'shed': 3, 'service': 4.5, 'commercial': 9, 'retail': 8, 'industrial': 9, 'hangar': 10, 'apartments': 12, 'school': 9, 'church': 10, 'roof': 3.5, 'carport': 3, 'warehouse': 9}
ORDER = [1, 2, 0, 3, 7, 8, 9, 6, 10, 4, 5]

def dp(p, eps):
    if len(p) < 3: return p
    a, b = p[0], p[-1]; dx, dz = b[0] - a[0], b[1] - a[1]; L = math.hypot(dx, dz); best = -1; bi = 0
    for i in range(1, len(p) - 1):
        q = p[i]; d = abs((q[0] - a[0]) * dz - (q[1] - a[1]) * dx) / L if L > 1e-9 else math.hypot(q[0] - a[0], q[1] - a[1])
        if d > best: best, bi = d, i
    if best > eps: return dp(p[:bi + 1], eps)[:-1] + dp(p[bi:], eps)
    return [a, b]

def flat(p): return [round(v, 1) for q in p for v in q]
def area(p): return 0.5 * sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))
def key(x, z): return (math.floor(x / T), math.floor(z / T))

def tri(p):
    # triangulate a polygon here so the phone never has to
    v = np.array(p, dtype=np.float32).reshape(-1, 2)
    try: return [int(i) for i in earcut.triangulate_float32(v, np.array([len(v)], dtype=np.uint32))]
    except Exception: return []

def inside(px, pz, poly):
    # vectorised point in polygon (ray casting); poly is a flat [x,z,...] list
    xs, zs = poly[0::2], poly[1::2]; n = len(xs); c = np.zeros(px.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, zi, xj, zj = xs[i], zs[i], xs[j], zs[j]
        if zi != zj:
            m = ((zi > pz) != (zj > pz)) & (px < (xj - xi) * (pz - zi) / (zj - zi) + xi)
            c ^= m
        j = i
    return c

def make_trees(tl, rng):
    # forests, tree rows and yard trees, baked as x, z, scale*100
    out = []; forests = [e[1] for e in tl['land'] if e[0] == 3]
    A = 0.0
    for p in forests:
        xs, zs = p[0::2], p[1::2]; A += abs(sum(xs[i] * zs[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * zs[i] for i in range(len(xs)))) / 2
    sp = max(13.0, math.sqrt(A / 3200.0)) if A else 13.0
    for p in forests:
        xs, zs = p[0::2], p[1::2]
        gx, gz = np.meshgrid(np.arange(min(xs), max(xs), sp), np.arange(min(zs), max(zs), sp))
        if gx.size == 0: continue
        px = gx.ravel() + (np.random.RandomState(1).rand(gx.size) - 0.5) * sp * 0.7
        pz = gz.ravel() + (np.random.RandomState(2).rand(gx.size) - 0.5) * sp * 0.7
        ok = inside(px, pz, p)
        for x, z in zip(px[ok], pz[ok]): out += [round(float(x), 1), round(float(z), 1), int(55 + rng.random() * 50)]
    for r in tl['rows']:
        for i in range(0, len(r) - 3, 2):
            dx, dz = r[i + 2] - r[i], r[i + 3] - r[i + 1]; n = int(math.hypot(dx, dz) / 8)
            for k in range(n + 1): out += [round(r[i] + dx * k / max(1, n), 1), round(r[i + 1] + dz * k / max(1, n), 1), int(50 + rng.random() * 40)]
    for b in tl['bld']:
        if b[0] != 0 or rng.random() > 0.55: continue
        p = b[2]; cx, cz = sum(p[0::2]) / (len(p) / 2), sum(p[1::2]) / (len(p) / 2); a = rng.random() * 6.283; r = 9 + rng.random() * 8
        out += [round(cx + math.cos(a) * r, 1), round(cz + math.sin(a) * r, 1), int(40 + rng.random() * 40)]
    while len(out) > 3 * 3500:
        i = rng.randrange(len(out) // 3) * 3; del out[i:i + 3]
    return out

def inbox(lat, lon):
    if not args.bbox: return True
    s, w, n, e = args.bbox; return s <= lat <= n and w <= lon <= e

def bheight(t):
    if 'height' in t:
        try: return float(t['height'].split()[0])
        except Exception: pass
    if 'building:levels' in t:
        try: return float(t['building:levels']) * 3.3 + 1.2
        except Exception: pass
    return HEIGHT.get(t.get('building'), 6)

tiles = collections.defaultdict(lambda: {'roads': [], 'bld': [], 'land': [], 'rails': [], 'rows': [], 'trees': []})

class H(osmium.SimpleHandler):
    def __init__(self):
        super().__init__(); self.n = collections.Counter()

    def way(self, w):
        t = w.tags
        hw = t.get('highway'); rail = t.get('railway'); tr = t.get('natural')
        if hw in RW or rail == 'rail' or tr == 'tree_row':
            try: pts = [(n.lat, n.lon) for n in w.nodes if n.location.valid()]
            except Exception: return
            if len(pts) < 2 or not inbox(*pts[0]): return
            p = [xz(a, b) for a, b in pts]
            if hw in RW: p = dp(p, 0.8); kind = 'roads'
            elif rail == 'rail': p = dp(p, 1.0); kind = 'rails'
            else: p = dp(p, 1.5); kind = 'rows'
            # split the line where it crosses tile borders so each tile owns its own pieces
            run = [p[0]]; k0 = key(*p[0])
            for a, b in zip(p, p[1:]):
                kb = key((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                if kb != k0:
                    if len(run) > 1: self.put(kind, k0, run, hw, t)
                    run = [a]; k0 = kb
                run.append(b)
            if len(run) > 1: self.put(kind, k0, run, hw, t)

    def put(self, kind, k, run, hw, t):
        tl = tiles[k]; self.n[kind] += 1
        if kind == 'roads': tl['roads'].append([RC[hw], RW[hw], flat(run), 1 if t.get('bridge') in ('yes', 'viaduct') else 0])
        else: tl[kind].append(flat(run))

    def area(self, a):
        t = a.tags
        is_b = 'building' in t or t.get('aeroway') in ('terminal', 'hangar')
        c = None
        if not is_b:
            if t.get('natural') == 'water' or t.get('leisure') == 'swimming_pool' or t.get('waterway') == 'riverbank': c = 4
            elif t.get('natural') in ('wood', 'scrub'): c = 3
            elif t.get('natural') in ('sand', 'beach'): c = 6
            elif t.get('natural') in ('grassland', 'heath'): c = 0
            elif t.get('landuse') in LU: c = LU[t['landuse']]
            elif t.get('leisure') in ('park', 'garden', 'golf_course'): c = 0
            elif t.get('leisure') in ('pitch', 'playground', 'track'): c = 8
            elif t.get('amenity') == 'parking': c = 5
            if c is None: return
        try: rings = list(a.outer_rings())
        except Exception: return
        for r in rings:
            try: pts = [(n.lat, n.lon) for n in r]
            except Exception: continue
            if len(pts) < 4 or not inbox(*pts[0]): continue
            p = dp([xz(a_, b_) for a_, b_ in pts], 0.35 if is_b else 1.5)[:-1]
            if len(p) < 3: continue
            if area(p) < 0: p = p[::-1]
            A = abs(area(p))
            cx, cz = sum(q[0] for q in p) / len(p), sum(q[1] for q in p) / len(p); tl = tiles[key(cx, cz)]
            if is_b:
                if A < args.min_building or len(tl['bld']) >= args.max_per_tile: continue
                b = t.get('building', 'yes')
                cat = 2 if (t.get('aeroway') == 'hangar' or b == 'hangar') else 3 if t.get('aeroway') == 'terminal' else 0 if b in HOUSE else 4 if b in SMALL else 1 if b in BIG else 5
                tl['bld'].append([cat, round(bheight(t), 1), flat(p), [] if (cat == 0 and len(p) <= 8) else tri(flat(p))]); self.n['bld'] += 1
            elif A > 60:
                tl['land'].append([c, flat(p), tri(flat(p)) if c != 3 else []]); self.n['land'] += 1

h = H()
for f in args.inputs:
    print('reading', f, flush=True)
    h.apply_file(f, locations=True, idx='flex_mem')
os.makedirs(args.out, exist_ok=True)
index = {'tile': T, 'lat0': LAT0, 'lon0': LON0, 'tiles': {}}
for (ix, iz), tl in tiles.items():
    tl['land'].sort(key=lambda e: ORDER.index(e[0]))
    tl['roads'].sort(key=lambda r: (r[3], -r[0]))
    tl['trees'] = make_trees(tl, random.Random(ix * 7919 + iz * 104729))
    name = 't_%d_%d.json' % (ix, iz)
    with open(os.path.join(args.out, name), 'w') as f: json.dump({'v': 1, 'ix': ix, 'iz': iz, **tl}, f, separators=(',', ':'))
    index['tiles']['%d_%d' % (ix, iz)] = [len(tl['roads']), len(tl['bld']), len(tl['land'])]
with open(os.path.join(args.out, 'index.json'), 'w') as f: json.dump(index, f, separators=(',', ':'))
print('wrote', len(tiles), 'tiles', dict(h.n))
