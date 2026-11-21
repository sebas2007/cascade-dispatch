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


import struct, time, gzip
def pack_tile(d):
    """compact binary tile: JSON skeleton + delta/zigzag/varint number arrays (no server compression needed)"""
    data = bytearray()
    def varint(n):
        while n >= 0x80: data.append((n & 0x7f) | 0x80); n >>= 7
        data.append(n)
    def enc(v, stride):
        if isinstance(v, list):
            if len(v) >= 6 and all(isinstance(x, (int, float)) for x in v):
                isf = any(isinstance(x, float) for x in v)
                vals = [int(round(x * 10)) if isf else int(x) for x in v]; st = stride if isf else 1; off = len(data)
                for i, x in enumerate(vals):
                    dv = x - (vals[i - st] if i >= st else 0); varint((dv << 1) if dv >= 0 else ((-dv) << 1) - 1)
                return '~%s%d:%d:%d' % ('f' if isf else 'i', st, off, len(v))
            return [enc(x, stride) for x in v]
        return v
    meta = {k: (enc(v, 3 if k == 'trees' else 2) if isinstance(v, list) else v) for k, v in d.items()}
    j = json.dumps(meta, separators=(',', ':')).encode()
    return struct.pack('<I', len(j)) + j + bytes(data)

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

# ---- airports the game knows (KFRG uses the hand-built pack, so it is not listed) ----
AIRPORTS = {  # icao: (lat, lon, search radius m)
    'KJFK': (40.6398, -73.7789, 5200), 'KLGA': (40.7769, -73.8740, 3000), 'KISP': (40.7952, -73.1002, 3500),
    'KTEB': (40.8501, -74.0608, 2500), 'KHPN': (41.0670, -73.7076, 2800), 'KFOK': (40.8437, -72.6318, 3500),
    'KHWV': (40.8218, -72.8696, 2500), 'KBDR': (41.1635, -73.1262, 2500)}
AP_XY = {k: (xz(v[0], v[1]), v[2]) for k, v in AIRPORTS.items()}
ap_data = {k: {'rw': [], 'tw': [], 'apr': []} for k in AIRPORTS}
def ap_of(x, z):
    best, bd = None, 1e18
    for k, (c, r) in AP_XY.items():
        d = math.hypot(x - c[0], z - c[1])
        if d < r and d < bd: best, bd = k, d
    return best
def fnum(v, d):
    try: return float(str(v).split()[0])
    except Exception: return d

tiles = collections.defaultdict(lambda: {'roads': [], 'bld': [], 'land': [], 'rails': [], 'rows': [], 'trees': []})

class H(osmium.SimpleHandler):
    def __init__(self):
        super().__init__(); self.n = collections.Counter()

    def way(self, w):
        t = w.tags
        aw = t.get('aeroway')
        if aw in ('runway', 'taxiway'):
            try: pts = [xz(n.lat, n.lon) for n in w.nodes if n.location.valid()]
            except Exception: pts = []
            if len(pts) >= 2:
                k = ap_of(*pts[len(pts) // 2])
                if k:
                    if aw == 'runway': ap_data[k]['rw'].append({'ref': t.get('ref', ''), 'w': fnum(t.get('width'), 30), 'p': [pts[0], pts[-1]]})
                    else: ap_data[k]['tw'].append([fnum(t.get('width'), 18), flat(dp(pts, 1.0))])
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
        if t.get('aeroway') == 'apron':
            try: rings0 = list(a.outer_rings())
            except Exception: rings0 = []
            for r in rings0:
                try: pp = [xz(n.lat, n.lon) for n in r]
                except Exception: continue
                if len(pp) < 4: continue
                pp = dp(pp, 1.5)[:-1]
                if len(pp) < 3: continue
                k = ap_of(sum(q[0] for q in pp) / len(pp), sum(q[1] for q in pp) / len(pp))
                if k and abs(area(pp)) > 400:
                    if area(pp) < 0: pp = pp[::-1]
                    ap_data[k]['apr'].append([flat(pp), tri(flat(pp))])
            return
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
    name = 't_%d_%d.bin' % (ix, iz)
    with open(os.path.join(args.out, name), 'wb') as f: f.write(gzip.compress(pack_tile({'v': 1, 'ix': ix, 'iz': iz, **tl}), 9, mtime=0))
    index['tiles']['%d_%d' % (ix, iz)] = [len(tl['roads']), len(tl['bld']), len(tl['land'])]
index['fmt'] = 'bin1gz'; index['ver'] = int(time.time())
with open(os.path.join(args.out, 'index.json'), 'w') as f: json.dump(index, f, separators=(',', ':'))
MAGVAR = -13.0
def angd(a, b): return (a - b + 540) % 360 - 180
def build_airport(k, d):
    rws = []
    for r in d['rw']:
        p0, p1 = r['p']; L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if L < 300: continue
        hdg = (math.degrees(math.atan2(p1[0] - p0[0], -(p1[1] - p0[1]))) + 360) % 360
        parts = [q.strip() for q in r['ref'].split('/')] if r['ref'] else []
        def num(q):
            m = ''.join(c for c in q if c.isdigit()); return int(m) if m else None
        n1 = num(parts[0]) if parts else None
        if n1 is None:
            n1 = round(((hdg - MAGVAR) % 360) / 10) % 36 or 36; a_name = str(n1); parts = [a_name]
        if abs(angd(hdg, n1 * 10 + MAGVAR)) > 90: p0, p1 = p1, p0; hdg = (hdg + 180) % 360
        a = parts[0]
        if len(parts) > 1: b = parts[1]
        else:
            n2 = (n1 + 18 - 1) % 36 + 1; suf = {'L': 'R', 'R': 'L', 'C': 'C'}.get(a[-1:], ''); b = str(n2) + suf
        rws.append({'a': a, 'b': b, 'w': round(r['w'] * 3.281), 'len': round(L * 3.281), 'p0': [round(p0[0], 1), round(p0[1], 1)], 'p1': [round(p1[0], 1), round(p1[1], 1)], 'hdg': round(hdg, 1)})
    # lateral offset of the parallel taxiway for each runway (positive = to the right of the 'a' heading)
    for r in rws:
        h = math.radians(r['hdg']); ux, uz = math.sin(h), -math.cos(h); rx, rz = math.cos(h), math.sin(h)
        cx, cz = (r['p0'][0] + r['p1'][0]) / 2, (r['p0'][1] + r['p1'][1]) / 2; half = r['len'] / 3.281 / 2
        score = collections.defaultdict(float)
        for wd, fl in d['tw']:
            for i in range(0, len(fl) - 3, 2):
                ax, az, bx, bz = fl[i], fl[i + 1], fl[i + 2], fl[i + 3]; sl = math.hypot(bx - ax, bz - az)
                if sl < 20: continue
                th = (math.degrees(math.atan2(bx - ax, -(bz - az))) + 360) % 360
                if min(abs(angd(th, r['hdg'])), abs(angd(th, r['hdg'] + 180))) > 6: continue
                mx, mz = (ax + bx) / 2 - cx, (az + bz) / 2 - cz; s_, t_ = mx * ux + mz * uz, mx * rx + mz * rz
                if abs(s_) < half and 60 < abs(t_) < 500: score[round(t_ / 12) * 12] += sl
        r['twl'] = max(score, key=score.get) if score else None
    return {'rw': rws, 'tw': d['tw'], 'apr': d['apr']}
out = {k: build_airport(k, d) for k, d in ap_data.items() if d['rw']}
with open(os.path.join(args.out, 'airports.js'), 'w') as f:
    f.write('window.REAL_APS=' + json.dumps(out, separators=(',', ':')) + ';')
print('airports:', {k: (len(v['rw']), len(v['tw']), len(v['apr'])) for k, v in out.items()})
print('wrote', len(tiles), 'tiles', dict(h.n))

# ---- real terrain elevation (AWS open terrain tiles, "terrarium" format) -> world/terrain.js ----
def make_terrain(outdir, fetch=None):
    import base64, io, subprocess, urllib.request
    try:
        from PIL import Image
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pillow'])
        from PIL import Image
    Z = 10; X0, X1, Z0, Z1, D = -66000.0, 64000.0, -56000.0, 44000.0, 300.0
    nx, nz = int((X1 - X0) / D) + 1, int((Z1 - Z0) / D) + 1
    gx, gz = np.meshgrid(X0 + np.arange(nx) * D, Z0 + np.arange(nz) * D)
    lat = LAT0 - gz / KZ; lon = LON0 + gx / KX
    n = 2 ** Z * 256
    px = (lon + 180.0) / 360.0 * n
    py = (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / math.pi) / 2 * n
    tx0, tx1, ty0, ty1 = int(px.min() // 256), int(px.max() // 256), int(py.min() // 256), int(py.max() // 256)
    mos = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), dtype=np.float32)
    def get(x, y):
        url = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/%d/%d/%d.png' % (Z, x, y)
        req = urllib.request.Request(url, headers={'User-Agent': 'CascadeDispatch-tiler'})
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as r: return r.read()
            except Exception as e: err = e
        raise err
    fetch = fetch or get
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            im = np.asarray(Image.open(io.BytesIO(fetch(tx, ty))).convert('RGB'), dtype=np.float32)
            mos[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = im[:, :, 0] * 256 + im[:, :, 1] + im[:, :, 2] / 256 - 32768
    fx, fy = px - tx0 * 256 - 0.5, py - ty0 * 256 - 0.5
    ix, iy = np.clip(np.floor(fx).astype(int), 0, mos.shape[1] - 2), np.clip(np.floor(fy).astype(int), 0, mos.shape[0] - 2)
    ax, ay = np.clip(fx - ix, 0, 1), np.clip(fy - iy, 0, 1)
    h = mos[iy, ix] * (1 - ax) * (1 - ay) + mos[iy, ix + 1] * ax * (1 - ay) + mos[iy + 1, ix] * (1 - ax) * ay + mos[iy + 1, ix + 1] * ax * ay
    u8 = np.clip(np.round(np.maximum(h, 0) / 2.0), 0, 255).astype(np.uint8)
    with open(os.path.join(outdir, 'terrain.js'), 'w') as f:
        f.write('window.REAL_TERR=' + json.dumps({'x0': X0, 'z0': Z0, 'd': D, 'nx': nx, 'nz': nz, 's': 2, 'b': base64.b64encode(u8.tobytes()).decode()}, separators=(',', ':')) + ';')
    print('terrain:', nx, 'x', nz, 'max', float(h.max()), 'm')

if __name__ == '__main__' or True:
    try: make_terrain(args.out)
    except Exception as e: print('terrain skipped:', e)
