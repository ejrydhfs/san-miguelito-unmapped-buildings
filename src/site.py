"""Build the static GitHub Pages site in docs/ (Pages: main branch, /docs).

Data files are split by verdict so the map can lazy-load what the user toggles
on; nothing is inlined in the HTML (the page stays a few KB).
"""
from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from shapely.geometry import shape

from common import OUT, ROOT
from diff2 import MIN_REM, MIN_FRAC, BUFFER, CV_MIN_REM

SITE = ROOT / "docs"
DATA = SITE / "data"
ND = 6          # ~0.11 m
REPO = "ejrydhfs/san-miguelito-unmapped-buildings"


def rnd(coords):
    if isinstance(coords[0], (int, float)):
        return [round(coords[0], ND), round(coords[1], ND)]
    return [rnd(c) for c in coords]


def geom_out(g):
    t = g["type"]
    if t == "Polygon":
        return {"type": "Polygon", "coordinates": [rnd(r) for r in g["coordinates"]]}
    return {"type": "MultiPolygon",
            "coordinates": [[rnd(r) for r in p] for p in g["coordinates"]]}


def simplify_geom(g, tol=0.5):
    """Drop points closer than ~tol metres (shapely simplify on projected geom)."""
    from common import to_ll, to_utm
    s = to_utm(shape(g)).simplify(tol, preserve_topology=True)
    if s.is_empty:
        return geom_out(g)
    return geom_out(json.loads(json.dumps(to_ll(s).__geo_interface__)))


def write_json(path, obj):
    path.write_text(json.dumps(obj, separators=(",", ":")))


def build_data():
    DATA.mkdir(parents=True, exist_ok=True)
    cands = json.loads((OUT / "candidates.geojson").read_text())["features"]
    stats = json.loads((OUT / "diff2_stats.json").read_text())
    detail = json.loads((OUT / "stats_detail.json").read_text()) \
        if (OUT / "stats_detail.json").exists() else {}

    by_verdict = defaultdict(list)
    counts = Counter()
    for f in cands:
        p = f["properties"]
        v = p.get("verdict", "unconfirmed")
        counts[v] += 1
        by_verdict[v].append({
            "type": "Feature",
            "properties": {
                "a": p.get("area_m2"),
                "c": p.get("cv_cover"),
                "s": p.get("source"),
                "d": p.get("nearest_osm_m"),
                "m": p.get("ms_area_m2"),
                "u": p.get("uncovered_frac"),
                "lat": round(shape(f["geometry"]).centroid.y, 6),
                "lon": round(shape(f["geometry"]).centroid.x, 6),
            },
            "geometry": simplify_geom(f["geometry"]),
        })

    for v, feats in by_verdict.items():
        write_json(DATA / f"candidates_{v}.json",
                   {"type": "FeatureCollection", "features": feats})
        print(v, len(feats), f"{(DATA / f'candidates_{v}.json').stat().st_size/1e6:.1f} MB")

    grid = json.loads((OUT / "priority_grid.geojson").read_text())["features"]
    write_json(DATA / "grid.json",
               {"type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": f["properties"],
                              "geometry": geom_out(f["geometry"])} for f in grid]})

    osm = json.loads((OUT / "osm_buildings.geojson").read_text())["features"]
    keep = []
    for f in osm:
        c = shape(f["geometry"]).centroid
        if 9.02 <= c.y <= 9.10 and -79.56 <= c.x <= -79.43:
            keep.append({"type": "Feature", "properties": {},
                         "geometry": simplify_geom(f["geometry"])})
    write_json(DATA / "osm.json", {"type": "FeatureCollection", "features": keep})
    print("osm context", len(keep))

    corr = json.loads((OUT / "corregimientos.geojson").read_text())["features"]
    write_json(DATA / "corregimientos.json",
               {"type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": f["properties"],
                              "geometry": geom_out(f["geometry"])} for f in corr]})

    write_json(DATA / "stats.json", {
        "district": stats["district"],
        "district_short": stats.get("district_short", "San Miguelito"),
        "osm_buildings": stats["osm_buildings"],
        "ml_buildings": stats["ml_buildings"],
        "candidates": stats["candidates"],
        "area_m2": stats["area_m2"],
        "median_area_m2": stats["median_area_m2"],
        "by_source": stats["by_source"],
        "verdicts": dict(counts),
        "recall_pct": detail.get("ml_layer_recall_on_osm_pct"),
        "median_iou": detail.get("ml_vs_osm_median_iou"),
        "corregimientos": detail.get("corregimientos", []),
        "imagery": "Esri World Imagery · 30 cm · 2025-08-08/11",
        "detector": "Microsoft Global Buildings 2026-02 + HOTOSM DINOv3/UperNet",
    })


CSS = """
:root{--bg:#fff;--fg:#111;--muted:#5b6470;--line:#e3e6ea;--accent:#c92a2a;
      --warn:#e8590c;--grey:#868e96;--osm:#1c7ed6;--violet:#6741d9}
*{box-sizing:border-box}
html,body{margin:0;height:100%;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--fg)}
#map{position:fixed;top:48px;left:0;right:0;bottom:0}
header{position:fixed;top:0;left:0;right:0;height:48px;z-index:1100;background:#fff;
  border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px;padding:0 16px}
header h1{font-size:15px;margin:0;font-weight:650}
header .meta{color:var(--muted);font-size:12px}
header nav{margin-left:auto;display:flex;gap:14px;font-size:13px}
header a{color:var(--muted);text-decoration:none}
header a:hover,header a.on{color:var(--fg)}
#panel{position:fixed;top:64px;left:16px;width:320px;max-height:calc(100vh - 96px);
  overflow:auto;background:#fff;border:1px solid var(--line);border-radius:10px;
  z-index:1000;padding:14px;box-shadow:0 2px 14px #00000014}
#panel h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 8px}
.kpis{display:grid;grid-template-columns:1fr auto;gap:2px 8px;margin-bottom:14px;font-size:13px}
.kpis b{font-variant-numeric:tabular-nums;font-weight:600}
.kpis .big{font-size:20px}
label.row{display:flex;align-items:center;gap:7px;margin:5px 0;cursor:pointer;font-size:13px}
.sw{width:11px;height:11px;border-radius:2px;flex:0 0 auto}
.cell{display:flex;justify-content:space-between;padding:3px 5px;border-radius:4px;cursor:pointer;font-variant-numeric:tabular-nums}
.cell:hover{background:#f2f5ff}
.note{color:var(--muted);font-size:11.5px;border-top:1px solid var(--line);padding-top:8px;margin-top:10px}
.note a{color:#0b62c4}
.leaflet-popup-content{font-size:12.5px;line-height:1.45}
.pill{display:inline-block;padding:0 6px;border-radius:9px;color:#fff;font-size:11px;font-weight:600}
main{margin:64px auto 40px;max-width:860px;padding:0 20px}
main h2{margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:6px;font-size:18px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
code{background:#f4f6f8;padding:1px 4px;border-radius:3px;font-size:12.5px}
@media(max-width:720px){#panel{width:auto;right:16px;max-height:52vh}main{margin-top:56px}}
"""

MAP_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Unmapped buildings · San Miguelito, Panama City</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%23c92a2a'/%3E%3Cpath d='M9 23V11l7-4 7 4v12h-5v-6h-4v6z' fill='white'/%3E%3C/svg%3E">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<link rel="stylesheet" href="assets/style.css">
</head><body>
<header>
  <h1>Unmapped buildings — San Miguelito</h1>
  <span class="meta" id="sub">loading…</span>
  <nav><a class="on" href="index.html">Map</a><a href="about.html">Method &amp; data</a></nav>
</header>
<div id="map"></div>
<aside id="panel">
  <div class="kpis" id="kpis"></div>
  <h2>Layers</h2>
  <label class="row"><input type="checkbox" data-layer="verified" checked>
    <span class="sw" style="background:var(--accent)"></span>verified <span class="cnt" data-c="verified"></span></label>
  <label class="row"><input type="checkbox" data-layer="probable" checked>
    <span class="sw" style="background:var(--warn)"></span>probable <span class="cnt" data-c="probable"></span></label>
  <label class="row"><input type="checkbox" data-layer="unconfirmed">
    <span class="sw" style="background:var(--grey)"></span>unconfirmed <span class="cnt" data-c="unconfirmed"></span></label>
  <label class="row"><input type="checkbox" data-layer="osm">
    <span class="sw" style="background:var(--osm)"></span>mapped OSM buildings</label>
  <label class="row"><input type="checkbox" data-layer="grid">
    <span class="sw" style="background:var(--violet);opacity:.5"></span>priority grid (400 m)</label>
  <h2 style="margin-top:14px">Top cells</h2>
  <div id="cells"></div>
  <div class="note">Imagery © Esri, Maxar, Earthstar Geographics. OSM ©
    <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a> (ODbL).
    Detections: Microsoft Global Buildings (CDLA-Permissive-2.0) + HOTOSM DINOv3 model.
    <a href="about.html">Method, licences, caveats →</a></div>
</aside>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="assets/app.js"></script>
</body></html>"""

JS = """
const COLORS = {verified:'#c92a2a', probable:'#e8590c', unconfirmed:'#868e96'};
const map = L.map('map', {preferCanvas:true, zoomControl:true});
window.__map = map;
const hash = new URLSearchParams(location.hash.slice(1) || '');
map.setView([Number(hash.get('lat')) || 9.048, Number(hash.get('lon')) || -79.494],
            Number(hash.get('z')) || 13);
map.on('moveend', () => {
  const c = map.getCenter();
  history.replaceState(null, '', `#z=${map.getZoom()}&lat=${c.lat.toFixed(5)}&lon=${c.lng.toFixed(5)}`);
});
const esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {maxZoom:19, attribution:'Imagery © Esri, Maxar, Earthstar Geographics'}).addTo(map);
const osmBase = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:19, attribution:'© OpenStreetMap contributors'});
L.control.layers({'Satellite (Esri)':esri,'OSM map':osmBase},null,{position:'topright'}).addTo(map);
L.control.scale({imperial:false}).addTo(map);

const groups = {};
const loaded = {};
const loading = html => `<span style="color:#868e96">${html}…</span>`;

function popup(p){
  const cls = {verified:'#c92a2a', probable:'#e8590c', unconfirmed:'#868e96'}[p.v] || '#555';
  const cv = p.c == null ? '–' : Number(p.c).toFixed(2);
  const origin = p.s === 'cv'
    ? '<div>found by the model only</div>'
    : `<div>uncovered part of a ${Number(p.m).toFixed(0)} m² detection
         (${Math.round(100*Number(p.u))} % of it is unmapped)</div>`;
  const near = p.d == null ? '' :
    (Number(p.d) <= 1.5 ? '<div>adjacent to a mapped building</div>'
                        : `<div>${Number(p.d).toFixed(0)} m from the nearest mapped building</div>`);
  return `<div><span class="pill" style="background:${cls}">${p.v}</span>
    <b>${Number(p.a).toFixed(0)} m²</b></div>${origin}
    <div>imagery coverage: ${cv}</div>${near}
    <div><a target="_blank" rel="noopener"
      href="https://www.openstreetmap.org/edit?editor=id#map=19/${p.lat}/${p.lon}">open in iD editor ↗</a></div>`;
}

async function loadLayer(name){
  if (loaded[name]) return groups[name];
  const file = (name === 'grid' || name === 'osm') ? name : `candidates_${name}`;
  const res = await fetch(`data/${file}.json`);
  if (!res.ok) throw new Error(`${file}.json: HTTP ${res.status}`);
  const gj = await res.json();
  if (name === 'grid'){
    const maxC = Math.max(...gj.features.map(f => f.properties.count));
    groups.grid = L.geoJSON(gj, {style: f => ({color:'#6741d9', weight:1,
      fillColor:'#6741d9', fillOpacity:0.05 + 0.28*f.properties.count/maxC}),
      onEachFeature: (f,l) => l.bindTooltip(`${f.properties.count} unmapped · ${f.properties.area_m2} m²`)});
  } else if (name === 'osm'){
    groups.osm = L.geoJSON(gj, {style: () => ({color:'#1c7ed6', weight:1, fill:false})});
  } else {
    groups[name] = L.geoJSON(gj, {
      style: () => styleFor(name),
      onEachFeature: (f,l) => { f.properties.v = name; l.bindPopup(popup(f.properties)); }});
  }
  loaded[name] = true;
  return groups[name];
}

function styleFor(name){
  const z = map.getZoom();
  return {color: COLORS[name], weight:1,
          fillOpacity: name === 'unconfirmed' ? 0.08 : (z >= 16 ? 0.22 : 0.10)};
}
map.on('zoomend', () => {
  for (const k of ['verified','probable','unconfirmed'])
    if (loaded[k]) groups[k].setStyle(() => styleFor(k));
});

const checkboxes = [...document.querySelectorAll('input[data-layer]')];
async function sync(){
  for (const cb of checkboxes){
    const name = cb.dataset.layer;
    const want = cb.checked;
    if (want){
      let g;
      try { g = await loadLayer(name); } catch (e) { console.warn(e); continue; }
      if (!map.hasLayer(g)) g.addTo(map);
    } else if (loaded[name] && map.hasLayer(groups[name])){
      map.removeLayer(groups[name]);
    }
  }
}
checkboxes.forEach(cb => cb.onchange = sync);

(async () => {
  const st = await (await fetch('data/stats.json')).json();
  document.getElementById('sub').textContent =
    `${st.district_short || st.district} · ${st.imagery}`;
  document.getElementById('kpis').innerHTML = `
    <span class="big"><b>${st.candidates.toLocaleString()}</b></span><span class="big">unmapped buildings</span>
    <span>OSM has</span><b>${st.osm_buildings.toLocaleString()}</b>
    <span>detected in total</span><b>${st.ml_buildings.toLocaleString()}</b>
    <span>OSM coverage</span><b>${(100*st.osm_buildings/st.ml_buildings).toFixed(0)}%</b>
    <span>unmapped area</span><b>${(st.area_m2/1e6).toFixed(1)} km²</b>`;
  for (const [k,v] of Object.entries(st.verdicts)){
    const el = document.querySelector(`[data-c="${k}"]`);
    if (el) el.textContent = v.toLocaleString();
  }
  const grid = await (await fetch('data/grid.json')).json();
  const key = {}; grid.features.forEach(f => key[f.properties.grid] = f);
  document.getElementById('cells').innerHTML = grid.features.slice(0,12).map(f =>
    `<div class="cell" data-g="${f.properties.grid}"><span>cell ${f.properties.grid}</span>
      <span>${f.properties.count}</span></div>`).join('');
  document.querySelectorAll('#cells .cell').forEach(el => el.onclick = () => {
    const g = key[el.dataset.g];
    if (g) map.fitBounds(L.geoJSON(g).getBounds());
  });
  await sync();
})();
"""

ABOUT = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Method &amp; data · San Miguelito building footprints</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%23c92a2a'/%3E%3Cpath d='M9 23V11l7-4 7 4v12h-5v-6h-4v6z' fill='white'/%3E%3C/svg%3E">
<link rel="stylesheet" href="assets/style.css"></head><body>
<header><h1>Unmapped buildings — San Miguelito</h1>
<span class="meta">method, data, licences</span>
<nav><a href="index.html">Map</a><a class="on" href="about.html">Method &amp; data</a></nav></header>
<main>__BODY__</main></body></html>"""


def build_pages():
    stats = json.loads((DATA / "stats.json").read_text())
    rows = "\n".join(
        f"<tr><td>{c['corregimiento']}</td><td class=n>{c['osm_buildings']:,}</td>"
        f"<td class=n>{c['unmapped']:,}</td><td class=n>{c['unmapped_verified']:,}</td></tr>"
        for c in stats["corregimientos"])
    body = f"""
<h2>What this is</h2>
<p>Building footprints detected by machine learning on imagery from
<b>2025-08-08/11</b> (Esri World Imagery, 30 cm) that have no counterpart in
OpenStreetMap, for the district of <b>{stats.get('district_short','San Miguelito')}</b> (49.9 km², Panama City).
{stats['candidates']:,} candidates over ~{stats['area_m2']/1e6:.1f} km²; OSM currently
maps {stats['osm_buildings']:,} of the {stats['ml_buildings']:,} buildings detected in the
district ({100*stats['osm_buildings']/stats['ml_buildings']:.0f}%).</p>

<h2>Pipeline</h2>
<ol>
<li><b>Boundary &amp; OSM</b> — district polygon from OSM relation 8389079 (Nominatim),
building ways and multipolygon relations from the Overpass API.</li>
<li><b>Detection layer</b> — Microsoft Global Buildings (2026-02-03 release) for the two
Panama quadkeys. Recall against already-mapped OSM buildings:
{stats.get('recall_pct','–')}%, median shape IoU {stats.get('median_iou','–')}.</li>
<li><b>Difference</b> — UTM 17N, each footprint minus the union of OSM (and, for the
second source, Microsoft) polygons buffered by {BUFFER:g} m; uncovered parts
≥ {MIN_REM:g} m² and ≥ {MIN_FRAC:.0%} of the footprint become candidates
(model-only blobs need ≥ {CV_MIN_REM:g} m²). This keeps half-mapped rows of houses
instead of dropping them.</li>
<li><b>Scene check</b> — a DINOv3 ViT-S/16 + UperNet building segmenter
(<code>hotosm/dinov3s-buildings</code>, ONNX) is run over all
4,576 z18 tiles of the district; blobs that neither dataset covers are added as
<em>source: model</em> candidates, and the same masks score every candidate
(<code>cv_cover</code> = share of the footprint within 1.8 m of a predicted building).</li>
<li><b>Ranking</b> — candidates binned into 400 m cells and ordered by count.</li>
</ol>

<p>{stats['candidates']:,} candidates come from the Microsoft layer
({stats['by_source'].get('ms',0):,}) and {stats['by_source'].get('cv',0):,} from the
segmentation model alone (parts of a built-up block that neither OSM nor Microsoft
covers).</p>

<h2>Confidence</h2>
<p><span class="pill" style="background:#c92a2a">verified</span> cv_cover ≥ 0.60 —
a building is visible at that location. &nbsp;
<span class="pill" style="background:#e8590c">probable</span> ≥ 0.30. &nbsp;
<span class="pill" style="background:#868e96">unconfirmed</span> &lt; 0.30 — sheds,
walls, patios, or the roof is hidden by trees. Calibration: 91 % of already-mapped
OSM buildings score <em>verified</em> with the same metric.</p>

<h2>By corregimiento</h2>
<table><thead><tr><th>corregimiento</th><th class=n>OSM buildings</th>
<th class=n>candidates</th><th class=n>verified</th></tr></thead>
<tbody>{rows}</tbody></table>

<h2>Data</h2>
<ul>
<li><a href="data/candidates_verified.json">candidates_verified.json</a>,
<a href="data/candidates_probable.json">candidates_probable.json</a>,
<a href="data/candidates_unconfirmed.json">candidates_unconfirmed.json</a> — GeoJSON, WGS84</li>
<li><a href="data/grid.json">grid.json</a> — 400 m priority cells</li>
<li><a href="data/osm.json">osm.json</a> — mapped OSM buildings (context)</li>
<li><a href="data/corregimientos.json">corregimientos.json</a> — sub-district boundaries</li>
<li><a href="data/stats.json">stats.json</a> — headline numbers</li>
</ul>
<p>Pipeline source and the JOSM reference layer are in the
<a href="https://github.com/__REPO__">repository</a>.</p>

<h2>Do not bulk-import</h2>
<p>Microsoft's footprints are licensed <b>CDLA-Permissive-2.0</b>, not ODbL, and contain
false positives. Use this map as a <em>reference</em>: open the same location in iD and
trace the building from the Esri imagery. That is also why <em>verified</em> means
"a building is here", not "this polygon is the building".</p>

<h2>Caveats</h2>
<ul>
<li>The segmenter is trained on HOT/OpenAerialMap very-high-resolution tiles; at
0.59 m/px it merges row-houses, so coverage is a screen, not a survey.</li>
<li>Footprints mapped with a 1–2 m offset are treated as mapped (buffer), by design.</li>
<li>Imagery is from August 2025; anything built later is absent.</li>
<li>Small structures below 35 m² uncovered area are not reported.</li>
</ul>

<h2>Licences</h2>
<ul>
<li>Imagery: Esri World Imagery (Maxar/Vantor, Earthstar Geographics) — attribution required, not redistributed here.</li>
<li>OSM data: © OpenStreetMap contributors, ODbL.</li>
<li>Microsoft Global Buildings: CDLA-Permissive-2.0.</li>
<li>Model <code>hotosm/dinov3s-buildings</code>: CC-BY-4.0 (DINOv3 backbone under its own licence).</li>
</ul>
"""
    (SITE / "index.html").write_text(MAP_PAGE)
    (SITE / "assets").mkdir(exist_ok=True)
    (SITE / "assets" / "style.css").write_text(CSS)
    (SITE / "assets" / "app.js").write_text(JS)
    (SITE / "about.html").write_text(ABOUT.replace("__BODY__", body).replace("__REPO__", REPO))
    (SITE / ".nojekyll").write_text("")
    (SITE / "404.html").write_text(
        '<!doctype html><meta http-equiv="refresh" content="0;url=./index.html">')


def main():
    SITE.mkdir(exist_ok=True)
    build_data()
    build_pages()
    print("site built:", sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file()) / 1e6, "MB")


if __name__ == "__main__":
    main()
