#!/usr/bin/env python3
"""Find the airports inside the play area.  usage: python3 find_airports.py filtered.osm.pbf OUT.json [--radius-km 185.2] [--max 40]

Input is an extract that only holds aeroway=aerodrome and aeroway=runway (made with osmium tags-filter).
Keeps airports with a four-letter ICAO code (or a FAA code, which gets a K in front) and a runway of at least 600 m,
and keeps the --max longest-runway ones. The airports the game already knows are always kept.
"""
import sys, json, math, argparse
import osmium
LAT0, LON0 = 40.73, -73.41
KX, KZ = 111320 * math.cos(math.radians(LAT0)), 110540.0
ap = argparse.ArgumentParser(); ap.add_argument('pbf'); ap.add_argument('out'); ap.add_argument('--radius-km', type=float, default=185.2); ap.add_argument('--max', type=int, default=40)
args = ap.parse_args()
KEEP = {'KJFK', 'KLGA', 'KISP', 'KTEB', 'KHPN', 'KFOK', 'KHWV', 'KBDR'}
xz = lambda lat, lon: ((lon - LON0) * KX, -(lat - LAT0) * KZ)
fields, runways = [], []
class H(osmium.SimpleHandler):
    def node(self, n):
        t = n.tags
        if t.get('aeroway') == 'aerodrome' and n.location.valid(): fields.append(({x.k: x.v for x in t}, n.location.lat, n.location.lon, 0))
    def way(self, w):
        t = w.tags
        try: pts = [(n.lat, n.lon) for n in w.nodes if n.location.valid()]
        except Exception: return
        if len(pts) < 2: return
        if t.get('aeroway') == 'runway':
            a, b = xz(*pts[0]), xz(*pts[-1]); runways.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, math.hypot(a[0] - b[0], a[1] - b[1]), a, b))
        elif t.get('aeroway') == 'aerodrome':
            fields.append(({x.k: x.v for x in t}, sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts), 1))
H().apply_file(args.pbf, locations=True)
out = {}
for t, lat, lon, _ in fields:
    icao = (t.get('icao') or '').upper(); faa = (t.get('faa') or t.get('ref') or '').upper()
    code = icao if len(icao) == 4 else ('K' + faa if len(faa) == 3 and faa.isalnum() else '')
    if not code or code == 'KFRG': continue
    x, z = xz(lat, lon)
    if math.hypot(x, z) > args.radius_km * 1000: continue
    near = [r for r in runways if math.hypot(r[0] - x, r[1] - z) < 4000]
    if not near: continue
    longest = max(r[2] for r in near)
    if longest < 600 and code not in KEEP: continue
    rad = min(5200, max(1800, max(max(math.hypot(r[3][0] - x, r[3][1] - z), math.hypot(r[4][0] - x, r[4][1] - z)) for r in near) + 900))
    try: elev = round(float(str(t.get('ele', '0')).split()[0]) * 3.281)
    except Exception: elev = 0
    cur = out.get(code)
    if cur and cur['longest'] >= longest: continue
    out[code] = {'id': code, 'name': t.get('name', code), 'city': t.get('addr:city', ''), 'lat': round(lat, 5), 'lon': round(lon, 5), 'elev': elev, 'radius': round(rad), 'longest': round(longest)}
BUILTIN = {'KJFK': (40.6398, -73.7789, 5200), 'KLGA': (40.7769, -73.8740, 3000), 'KISP': (40.7952, -73.1002, 3500), 'KTEB': (40.8501, -74.0608, 2500), 'KHPN': (41.0670, -73.7076, 2800), 'KFOK': (40.8437, -72.6318, 3500), 'KHWV': (40.8218, -72.8696, 2500), 'KBDR': (41.1635, -73.1262, 2500)}
for code, (la, lo, rad) in BUILTIN.items():   # the airports the game already has are always kept, even if OSM lacks a code tag
    if code not in out: out[code] = {'id': code, 'name': code, 'city': '', 'lat': la, 'lon': lo, 'elev': 0, 'radius': rad, 'longest': 99999}
L = sorted(out.values(), key=lambda a: (a['id'] not in KEEP, -a['longest']))[:max(args.max, len(KEEP))]
json.dump(L, open(args.out, 'w'), indent=1); print(len(L), 'airports:', ' '.join(a['id'] for a in L))
