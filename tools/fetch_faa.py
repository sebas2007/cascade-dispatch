#!/usr/bin/env python3
"""Download FAA airport diagrams (d-TPP chart code APD) and turn them into game data.

usage: python3 tools/fetch_faa.py [--airports KJFK,KLGA,...] [--pdf-dir DIR]   (run inside GitHub Actions)

For each airport it
  1. downloads the current airport diagram PDF from aeronav.faa.gov,
  2. aligns the drawing to the game map using the runway numbers printed at the runway ends
     (matched against world/airports.js, which comes from OpenStreetMap),
  3. reads the tower position, taxiway letters and building shapes from the drawing,
  4. writes world/faa.js  (window.FAA_APS)  and world/faa_report.json (alignment quality per airport).
An airport only gets FAA data in the game when its alignment error is small; otherwise it is left alone.
"""
import sys, os, re, json, math, argparse, urllib.request, datetime
import xml.etree.ElementTree as ET
import numpy as np
import pymupdf as fitz

ap = argparse.ArgumentParser()
ap.add_argument('--airports', default='')
ap.add_argument('--pdf-dir', default='')
ap.add_argument('--max-rms', type=float, default=60.0, help='metres; above this the airport is rejected')
args = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {'User-Agent': 'cascade-dispatch/1.0'}

def get(url, t=90):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=t).read()

def load_real():
    s = open(os.path.join(ROOT, 'world', 'airports.js')).read()
    return json.loads(s[s.index('=') + 1:].rstrip().rstrip(';'))

def find_cycle():
    d = datetime.date.today()
    for back in range(0, 90, 7):  # d-TPP cycles are named YYMM, e.g. 2611; try recent ones
        for dd in (d - datetime.timedelta(days=back),):
            for cyc in (dd.strftime('%y%m'),):
                try:
                    x = get(f'https://aeronav.faa.gov/d-tpp/{cyc}/xml_data/d-TPP_Metafile.xml', 120)
                    return cyc, x
                except Exception:
                    pass
    raise SystemExit('could not find a d-TPP cycle')

def apd_files(xml):
    root = ET.fromstring(xml); out = {}
    for ar in root.iter('airport_name'):
        icao = (ar.get('icao_ident') or '').upper()
        for rec in ar.iter('record'):
            if (rec.findtext('chart_code') or '') == 'APD':
                out[icao] = rec.findtext('pdf_name'); break
    return out

def umeyama(src, dst):  # similarity transform (optionally mirrored not needed: PDF y is flipped, handled by caller)
    mu_s, mu_d = src.mean(0), dst.mean(0); s0, d0 = src - mu_s, dst - mu_d
    C = d0.T @ s0 / len(src); U, S, Vt = np.linalg.svd(C)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: D[1, 1] = -1
    R = U @ D @ Vt; sc = (S * np.diag(D)).sum() / (s0 ** 2).sum() * len(src)
    t = mu_d - sc * R @ mu_s
    return sc, R, t

def runway_end_names(r):
    return r['a'].upper().lstrip('0') or r['a'], r['b'].upper().lstrip('0') or r['b']

def process(icao, pdf_bytes, real):
    doc = fitz.open(stream=pdf_bytes, filetype='pdf'); pg = doc[0]
    words = pg.get_text('words')  # x0,y0,x1,y1,text
    rw = real['rw']
    anchors = {}
    for r in rw:
        for nm, p in zip(runway_end_names(r), (r['p0'], r['p1'])): anchors.setdefault(nm, []).append(p)
    src, dst = [], []
    for w in words:
        t = w[4].upper().lstrip('0')
        if t in anchors and re.fullmatch(r'\d{1,2}[LRC]?', t):
            src.append(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)); dst.append(anchors[t][0])
    if len(src) < 3: return None, f'only {len(src)} runway numbers matched'
    src = np.array(src, float); dst = np.array(dst, float)
    # PDF y grows downward; game z grows south (downward) too, so no mirror. Robust fit: drop worst while rms is high.
    for _ in range(4):
        sc, R, t = umeyama(src, dst); res = np.linalg.norm((sc * (R @ src.T)).T + t - dst, axis=1); rms = float(np.sqrt((res ** 2).mean()))
        if rms <= args.max_rms or len(src) <= 4: break
        k = int(res.argmax()); src = np.delete(src, k, 0); dst = np.delete(dst, k, 0)
    T = lambda p: [round(float(v), 1) for v in sc * (R @ np.array(p, float)) + t]
    # runway numbers sit a little beyond the runway end, so the fit has a few metres of built-in bias; fine for tower/taxi labels.
    out = {'n': int(len(src)), 'rms': round(rms, 1), 'scale': round(float(sc), 4)}
    twr = None
    for w in words:
        if w[4].upper() in ('TWR', 'ATCT', 'TOWER'): twr = T(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)); break
    out['twr'] = twr
    out['taxi'] = [{'n': w[4], 'p': T(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))} for w in words if re.fullmatch(r'[A-Z]{1,2}[0-9]?', w[4]) and w[4] not in ('TWR', 'N', 'S', 'E', 'W')]
    blds = []
    for d in pg.get_drawings():
        if d.get('fill') is None or not d.get('closePath', True) and d['type'] != 'f': continue
        r = d['rect']
        if r.width < 4 or r.height < 4 or r.width > 400 or r.height > 400: continue
        pts = []
        for it in d['items']:
            if it[0] == 'l': pts += [it[1], it[2]]
            elif it[0] == 're': pts += [it[1].tl, it[1].tr, it[1].br, it[1].bl]
        if len(pts) < 3: continue
        poly = []
        for q in pts:
            if not poly or (q.x, q.y) != tuple(poly[-1]): poly.append((q.x, q.y))
        if len(poly) < 3 or len(poly) > 40: continue
        f = d['fill']
        if max(f) - min(f) > 0.12 or sum(f) / 3 > 0.9 or sum(f) / 3 < 0.35: continue  # buildings are mid-grey; skip colour, white and black
        blds.append([T(q) for q in poly])
    out['bld'] = blds[:250]
    return out, 'ok'

def main():
    real = load_real()
    want = [a.strip().upper() for a in args.airports.split(',') if a.strip()] or sorted(real)
    out = {}; report = {}
    if args.pdf_dir:
        files = {f[:-4].upper(): os.path.join(args.pdf_dir, f) for f in os.listdir(args.pdf_dir) if f.lower().endswith('.pdf')}
        fetch = lambda k: open(files[k], 'rb').read()
        names = {k: k for k in files}
    else:
        cyc, xml = find_cycle(); names = apd_files(xml); print('cycle', cyc, 'airports with diagrams', len(names))
        fetch = lambda k: get(f'https://aeronav.faa.gov/d-tpp/{cyc}/{names[k]}')
    for k in want:
        if k not in real or k not in names: report[k] = 'no data'; continue
        try:
            res, msg = process(k, fetch(k), real[k])
        except Exception as e:
            res, msg = None, 'error: ' + str(e)
        if not res: report[k] = msg; print(k, msg); continue
        ok = res['rms'] <= args.max_rms
        report[k] = {'ok': ok, 'rms_m': res['rms'], 'runway_anchors': res['n'], 'tower': bool(res['twr']), 'taxi_labels': len(res['taxi']), 'buildings': len(res['bld'])}
        print(k, report[k])
        if ok: out[k] = res
    open(os.path.join(ROOT, 'world', 'faa.js'), 'w').write('window.FAA_APS=' + json.dumps(out, separators=(',', ':')) + ';\n')
    json.dump(report, open(os.path.join(ROOT, 'world', 'faa_report.json'), 'w'), indent=1)

main()
