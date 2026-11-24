#!/usr/bin/env python3
"""Download FAA airport diagrams (d-TPP chart code APD) and turn them into game data.

usage: python3 tools/fetch_faa.py [--airports KJFK,KLGA,...] [--pdf-dir DIR]

Every FAA diagram has a latitude/longitude grid printed on it. The tool finds that grid, works out which
whole minutes of latitude/longitude the lines are, and so pins the drawing onto the game map exactly.
From the drawing it then reads the tower, the taxiway letters and the buildings, and it checks the OSM taxiways
against the diagram's grey pavement. Output: world/faa.js (window.FAA_APS) and world/faa_report.json.
"""
import sys, os, re, json, math, argparse, urllib.request, datetime, collections
import xml.etree.ElementTree as ET
import numpy as np
import pymupdf as fitz

ap = argparse.ArgumentParser()
ap.add_argument('--airports', default='')
ap.add_argument('--pdf-dir', default='')
args = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {'User-Agent': 'cascade-dispatch/1.0'}
LAT0, LON0 = 40.73, -73.41                     # the game's reference point
KX, KZ = 111320 * math.cos(math.radians(LAT0)), 110540.0

def get(url, t=90): return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=t).read()
def load_real():
    s = open(os.path.join(ROOT, 'world', 'airports.js')).read()
    return json.loads(s[s.index('=') + 1:].rstrip().rstrip(';'))
def find_cycle():
    d = datetime.date.today()
    for back in range(0, 120, 7):
        cyc = (d - datetime.timedelta(days=back)).strftime('%y%m')
        try: return cyc, get(f'https://aeronav.faa.gov/d-tpp/{cyc}/xml_data/d-TPP_Metafile.xml', 120)
        except Exception: pass
    raise SystemExit('could not find a d-TPP cycle')
def apd_files(xml):
    out = {}
    for ar in ET.fromstring(xml).iter('airport_name'):
        ids = {(ar.get('icao_ident') or '').upper(), ('K' + (ar.get('apt_ident') or '').upper()) if ar.get('apt_ident') else ''}
        for rec in ar.iter('record'):
            if (rec.findtext('chart_code') or '') == 'APD':
                for i in ids:
                    if i: out.setdefault(i, rec.findtext('pdf_name'))
                break
    return out

def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0); s0, d0 = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d0.T @ s0 / len(src)); D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: D[1, 1] = -1
    R = U @ D @ Vt; sc = (S * np.diag(D)).sum() / (s0 ** 2).sum() * len(src)
    return sc, R, mu_d - sc * R @ mu_s

def rough_fit(pg, real):
    """Coarse alignment from the runway numbers printed at the runway ends (only used to choose the right grid minute)."""
    anchors = {}
    for r in real['rw']:
        for nm, p in zip((r['a'].lstrip('0'), r['b'].lstrip('0')), (r['p0'], r['p1'])): anchors.setdefault(nm.upper(), []).append(p)
    src, dst = [], []
    for w in pg.get_text('words'):
        t = w[4].upper().lstrip('0')
        if t in anchors and re.fullmatch(r'\d{1,2}[LRC]?', t): src.append(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)); dst.append(anchors[t][0])
    if len(src) < 3: return None
    src = np.array(src); dst = np.array(dst)
    for _ in range(5):
        sc, R, t = umeyama(src, dst); res = np.linalg.norm((sc * (R @ src.T)).T + t - dst, axis=1)
        if len(src) <= 4 or np.sqrt((res ** 2).mean()) < 40: break
        k = res.argmax(); src = np.delete(src, k, 0); dst = np.delete(dst, k, 0)
    return sc, R, t

def grid_lines(pg):
    """x positions of vertical thin lines and y positions of horizontal ones (graticule lines and their edge ticks), frame excluded"""
    V = collections.defaultdict(float); H = collections.defaultdict(float)
    fr = None
    for d in pg.get_drawings():
        if d['fill'] is None and d['rect'].width > 250 and d['rect'].height > 400: fr = d['rect']
    for d in pg.get_drawings():
        if d['color'] and d['fill'] is None and d['width'] is not None and d['width'] < 1.0:
            for it in d['items']:
                if it[0] != 'l': continue
                a, b = it[1], it[2]
                if abs(a.x - b.x) < 0.05 and abs(a.y - b.y) >= 6: V[round(a.x * 2) / 2] += abs(a.y - b.y)
                elif abs(a.y - b.y) < 0.05 and abs(a.x - b.x) >= 6: H[round(a.y * 2) / 2] += abs(a.x - b.x)
    def pick(D, lo, hi):
        ks = sorted(k for k, v in D.items() if v >= 60 and not (fr and (abs(k - lo) < 1.5 or abs(k - hi) < 1.5)))
        out = []
        for k in ks:
            if out and k - out[-1] < 1.5: continue
            out.append(k)
        return out
    return (pick(V, fr.x0, fr.x1) if fr else pick(V, -9, -9)), (pick(H, fr.y0, fr.y1) if fr else pick(H, -9, -9))

def georef(pg, real):
    """page point -> game world metres:  w = A @ p + t
    The drawing carries a lat/lon grid. Its spacing (1, 1/2 or 1/4 minute), which way is north and which grid lines they are
    are all found by trying every option and keeping the one where the OSM runways and taxiways land on the drawing's pavement."""
    V, H = grid_lines(pg)
    if len(V) < 2 or len(H) < 2: return None, 'no lat/lon grid found'
    dV = sorted({round(abs(a - b), 1) for i, a in enumerate(V) for b in V[i + 1:] if abs(a - b) > 25})
    dH = sorted({round(abs(a - b), 1) for i, a in enumerate(H) for b in H[i + 1:] if abs(a - b) > 25})
    pairs = []
    for x in dV:
        for y in dH:
            hi, lo = max(x, y), min(x, y)
            if abs(lo / hi - 0.757) < 0.05: pairs.append((x, y))
    if not pairs: return None, 'grid spacing does not look like a lat/lon grid'
    pts = []
    for r in real['rw']:
        for f in np.linspace(0, 1, 25): pts.append((r['p0'][0] + (r['p1'][0] - r['p0'][0]) * f, r['p0'][1] + (r['p1'][1] - r['p0'][1]) * f))
    for tws in real.get('tw', []):
        flat = tws[1] if isinstance(tws[1], list) else tws[1:]
        pts += [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 6)]
    pts = np.array(pts[:900])
    pm = pg.get_pixmap(dpi=100, colorspace=fitz.csGRAY); k = 100 / 72
    img = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width)
    cx_, cy_ = pg.rect.width / 2, pg.rect.height / 2
    tgt = np.array([np.mean([(r['p0'][0] + r['p1'][0]) / 2 for r in real['rw']]), np.mean([(r['p0'][1] + r['p1'][1]) / 2 for r in real['rw']])])
    best = None
    for sv, sh in pairs:
        latV = sv > sh                                   # the wider spacing is latitude (1' of latitude is longer than 1' of longitude)
        spl, spo = (sv, sh) if latV else (sh, sv)
        ixn = 0 if latV else 1; ixe = 1 - ixn
        lat_lines, lon_lines = (V, H) if latV else (H, V)
        for I in (1.0, 0.5, 0.25, 2.0):                  # grid interval in minutes
            mx, mz = KX / 60 * I / spo, KZ / 60 * I / spl
            for se in (1, -1):
                for sn in (1, -1):
                    A = np.zeros((2, 2)); A[0, ixe] = se * mx; A[1, ixn] = sn * mz
                    Ai = np.linalg.inv(A); t_c = tgt - A @ np.array([cx_, cy_])
                    cand = []
                    for ax, lines, wi, per_deg, origin, sgn in ((ixn, lat_lines, 1, KZ, LAT0, -1), (ixe, lon_lines, 0, KX, LON0, 1)):
                        sp_ = spl if ax == ixn else spo                      # a real grid line has a neighbour exactly one spacing away
                        refs = [l for l in lines if any(abs(abs(l - l2) - sp_) < 0.8 for l2 in lines)] or lines
                        c0 = refs[0]; w_est = A[wi, ax] * c0 + t_c[wi]
                        m0 = round((w_est / (sgn * per_deg) + origin) * 60 / I) * I
                        cand.append([sgn * ((m0 + d * I) / 60 - origin) * per_deg - A[wi, ax] * c0 for d in (-3, -2, -1, 0, 1, 2, 3)])
                    for tz in cand[0]:
                        for tx in cand[1]:
                            tw = np.array([tx, tz]); q = ((pts - tw) @ Ai.T) * k
                            xi, yi = q[:, 0].astype(int), q[:, 1].astype(int)
                            ok = (xi > 3) & (xi < pm.width - 3) & (yi > 3) & (yi < pm.height - 3)
                            if ok.sum() < 20: continue
                            v = img[yi[ok], xi[ok]]
                            score = float((((v > 190) & (v < 225)) | (v < 70)).mean()) * float(ok.mean())
                            if best is None or score > best[0]: best = (score, A.copy(), tw.copy())
    if best is None or best[0] < 0.5: return None, 'no orientation made the OSM layout fit the diagram'
    return (best[1], best[2]), 'ok'

def subpaths(d):
    """split a drawing into closed polygons (curves are reduced to their end points)"""
    out, cur = [], []
    for it in d['items']:
        if it[0] == 'l': a, b = it[1], it[2]
        elif it[0] == 'c': a, b = it[1], it[4]
        elif it[0] == 're': out.append([it[1].tl, it[1].tr, it[1].br, it[1].bl]); continue
        else: continue
        if cur and (abs(cur[-1].x - a.x) > 0.05 or abs(cur[-1].y - a.y) > 0.05):
            if len(cur) > 2: out.append(cur)
            cur = []
        if not cur: cur.append(a)
        cur.append(b)
    if len(cur) > 2: out.append(cur)
    return out
def area(p): return 0.5 * abs(sum(p[i].x * p[(i + 1) % len(p)].y - p[(i + 1) % len(p)].x * p[i].y for i in range(len(p))))

def _seg_x(p1, p2, p3, p4):
    d = lambda a, b, c: (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
    return d(p1, p2, p3) * d(p1, p2, p4) < 0 and d(p3, p4, p1) * d(p3, p4, p2) < 0
def is_simple(P):
    n = len(P)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1: continue
            if _seg_x(P[i], P[(i + 1) % n], P[j], P[(j + 1) % n]): return False
    return True
def obb(P):
    best = None; pts = np.array(P)
    for k in range(0, 90, 3):
        a = math.radians(k); c, s_ = math.cos(a), math.sin(a); u = pts @ np.array([[c, -s_], [s_, c]])
        w, h = np.ptp(u[:, 0]), np.ptp(u[:, 1])
        if best is None or w * h < best[0]: best = (w * h, a, u.min(0), u.max(0))
    _, a, lo, hi = best; c, s_ = math.cos(a), math.sin(a); R = np.array([[c, -s_], [s_, c]])
    return [list((np.array(q) @ R.T).round(1)) for q in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))]

def building_height(a_m2):                           # guesses from footprint size: hangars/small, large halls, terminals
    return 6 if a_m2 < 400 else 9 if a_m2 < 2500 else 14 if a_m2 < 12000 else 20

def process(icao, pdf_bytes, real):
    doc = fitz.open(stream=pdf_bytes, filetype='pdf'); pg = doc[0]
    g, msg = georef(pg, real)
    if g is None: return None, msg
    A, tw = g
    T = lambda p: [round(float(v), 1) for v in A @ np.array([p[0], p[1]], float) + tw]
    Ai = np.linalg.inv(A); words = pg.get_text('words')
    wbox = [fitz.Rect(w[:4]) for w in words]
    # --- pavement agreement: how many OSM taxiway points land on the diagram's grey pavement
    pm = pg.get_pixmap(dpi=144, colorspace=fitz.csGRAY); k = 144 / 72
    img = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width)
    def on_pave(w):
        q = Ai @ (np.array(w) - tw); x, y = int(q[0] * k), int(q[1] * k)
        if not (3 <= x < pm.width - 3 and 3 <= y < pm.height - 3): return None
        patch = img[y - 2:y + 3, x - 2:x + 3]
        return bool(((patch > 190) & (patch < 225)).mean() > 0.4 or (patch < 60).mean() > 0.4)  # grey pavement, or runway paint
    tot = hit = 0; off = []
    for tws in real.get('tw', []):
        flat = tws[1] if isinstance(tws[1], list) else tws[1:]
        for i in range(0, len(flat) - 1, 2):
            r = on_pave((flat[i], flat[i + 1]))
            if r is None: continue
            tot += 1; hit += r
            if not r and len(off) < 30: off.append([round(flat[i]), round(flat[i + 1])])
    pave = round(100 * hit / tot, 1) if tot else 0.0
    out = {'pavement_pct': pave, 'off_pavement': off}
    # --- tower: arrow head next to the TWR label
    twr = None
    for w in words:
        if w[4].upper() in ('TWR', 'ATCT'):
            cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2; best = None
            for d in pg.get_drawings():
                if d.get('fill') != (0.0, 0.0, 0.0): continue
                r = d['rect']
                if r.width > 8 or r.height > 8 or len(d['items']) != 3: continue
                dist = math.hypot((r.x0 + r.x1) / 2 - cx, (r.y0 + r.y1) / 2 - cy)
                if dist < 32 and (best is None or dist < best[0]): best = (dist, d)
            if best:
                pts = subpaths(best[1])[0]; tip = max(pts, key=lambda q: math.hypot(q.x - cx, q.y - cy)); twr = T((tip.x, tip.y))
            else: twr = T((cx, cy))
            break
    out['twr'] = twr
    # --- taxiway letters
    cenx = np.mean([(r['p0'][0] + r['p1'][0]) / 2 for r in real['rw']]); cenz = np.mean([(r['p0'][1] + r['p1'][1]) / 2 for r in real['rw']])
    taxi = []
    for w in words:
        if re.fullmatch(r'[A-Z]{1,2}[0-9]?', w[4]) and w[4] not in ('TWR', 'N', 'S', 'E', 'W', 'D', 'P', 'H1', 'HS'):
            p = T(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))
            if math.hypot(p[0] - cenx, p[1] - cenz) < 5000: taxi.append({'n': w[4], 'p': p})
    out['taxi'] = taxi
    pave_pts = [(flat[i], flat[i + 1]) for tws in real.get('tw', []) for flat in [tws[1] if isinstance(tws[1], list) else tws[1:]] for i in range(0, len(flat) - 1, 2)] or [(cenx, cenz)]
    # --- buildings: black filled shapes that are not text, arrows or symbols
    blds = []
    fx0, fy0, fx1, fy1 = 1e9, 1e9, -1e9, -1e9                      # the map frame = extent of the lat/lon grid lines
    fr = [d['rect'] for d in pg.get_drawings() if d['color'] == (0.0, 0.0, 0.0) and d['fill'] is None and d['rect'].width > 250 and d['rect'].height > 400]
    if fr: fx0, fy0, fx1, fy1 = min(r.x0 for r in fr) + 1, min(r.y0 for r in fr) + 1, max(r.x1 for r in fr) - 1, max(r.y1 for r in fr) - 1
    else: fx0, fy0, fx1, fy1 = 18, 44, 369, 549
    for d in pg.get_drawings():
        if d.get('fill') != (0.0, 0.0, 0.0): continue
        for poly in subpaths(d):
            a = area(poly)
            if a < 14: continue
            xs = [q.x for q in poly]; ys = [q.y for q in poly]; bw, bh = max(xs) - min(xs), max(ys) - min(ys)
            if bw > 120 or bh > 120: continue
            r = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
            cx_, cy_ = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            if not (fx0 < cx_ < fx1 and fy0 < cy_ < fy1): continue                                       # outside the map frame (titles, notes)
            if any(b.contains(fitz.Point(cx_, cy_)) for b in wbox): continue                              # letters and their holes
            if bw / max(bh, 0.1) > 12 or bh / max(bw, 0.1) > 12: continue                                # lines and arrows
            wp = [T((q.x, q.y)) for q in poly]
            if len(wp) > 60: step = len(wp) // 40 + 1; wp = wp[::step]
            if any(math.hypot(q[0] - cenx, q[1] - cenz) > 6000 for q in wp): continue
            if not is_simple(wp): wp = obb(wp)                       # a path with crossing lines would not extrude cleanly
            cxw = sum(q[0] for q in wp) / len(wp); czw = sum(q[1] for q in wp) / len(wp)
            if min(math.hypot(cxw - q[0], czw - q[1]) for q in pave_pts) > 350: continue   # not near any taxiway/apron (compass rose, legend symbols)
            a_m2 = a * abs(np.linalg.det(A)); blds.append({'p': wp, 'h': building_height(a_m2)})
    out['bld'] = blds[:400]
    return out, 'ok'

def main():
    real = load_real()
    want = [a.strip().upper() for a in args.airports.split(',') if a.strip()] or sorted(real)
    out = {}; report = {}
    if args.pdf_dir:
        files = {f[:-4].upper(): os.path.join(args.pdf_dir, f) for f in os.listdir(args.pdf_dir) if f.lower().endswith('.pdf')}
        fetch = lambda k: open(files[k], 'rb').read(); names = {k: k for k in files}
    else:
        cyc, xml = find_cycle(); names = apd_files(xml); print('cycle', cyc, 'airports with diagrams', len(names))
        def fetch(k):
            b = get(f'https://aeronav.faa.gov/d-tpp/{cyc}/{names[k]}')
            os.makedirs(os.path.join(ROOT, 'data', 'faa'), exist_ok=True); open(os.path.join(ROOT, 'data', 'faa', k + '.pdf'), 'wb').write(b)
            return b
    for k in want:
        if k not in real or k not in names: report[k] = 'no data'; continue
        try: res, msg = process(k, fetch(k), real[k])
        except Exception as e: res, msg = None, 'error: ' + str(e)
        if not res: report[k] = {'ok': False, 'reason': msg}; print(k, msg); continue
        ok = res['pavement_pct'] >= 75
        report[k] = {'ok': ok, 'osm_taxiways_on_faa_pavement_pct': res['pavement_pct'], 'tower': bool(res['twr']), 'taxi_labels': len(res['taxi']), 'buildings': len(res['bld']), 'taxiway_points_off_pavement': res['off_pavement'][:10]}
        print(k, {a: b for a, b in report[k].items() if a != 'taxiway_points_off_pavement'})
        if ok: res.pop('off_pavement'); out[k] = res
    open(os.path.join(ROOT, 'world', 'faa.js'), 'w').write('window.FAA_APS=' + json.dumps(out, separators=(',', ':')) + ';\n')
    json.dump(report, open(os.path.join(ROOT, 'world', 'faa_report.json'), 'w'), indent=1)
main()
