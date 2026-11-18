"""Cross-checks and breakdowns.

1. Reverse diff: how many OSM buildings have a corresponding ML footprint
   (measures recall of the detection layer).
2. Per-corregimiento counts (district sub-areas) for the report.

Usage: python stats.py
"""
from __future__ import annotations

import json
from collections import defaultdict

from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from common import DATA, OUT, polys_of, to_ll, to_utm, write_geojson


def stitch(members):
    """Stitch relation member ways (with geometry) into polygons."""
    segs = []
    for m in members:
        if m.get("role") != "outer" or not m.get("geometry"):
            continue
        pts = [(p["lon"], p["lat"]) for p in m["geometry"]]
        if len(pts) >= 2:
            segs.append(pts)
    rings, used = [], set()
    for i, s in enumerate(segs):
        if i in used:
            continue
        ring = list(s)
        used.add(i)
        changed = True
        while changed and ring[0] != ring[-1]:
            changed = False
            for j, t in enumerate(segs):
                if j in used:
                    continue
                if t[0] == ring[-1]:
                    ring += t[1:]
                elif t[-1] == ring[-1]:
                    ring += t[::-1][1:]
                elif t[-1] == ring[0]:
                    ring = t[:-1] + ring
                elif t[0] == ring[0]:
                    ring = t[::-1][:-1] + ring
                else:
                    continue
                used.add(j)
                changed = True
                break
        if ring[0] == ring[-1] and len(ring) >= 4:
            rings.append(ring)
    if not rings:
        return None
    polys = [Polygon(r) for r in rings]
    polys = [p if p.is_valid else Polygon(p.exterior) for p in polys]
    return unary_union(polys)


def main():
    osm = [{"g": to_utm(shape(f["geometry"])), "p": f["properties"]}
           for f in json.loads((OUT / "osm_buildings.geojson").read_text())["features"]]
    ms = [{"g": to_utm(shape(f["geometry"])), "p": f["properties"]}
          for f in json.loads((OUT / "ms_buildings.geojson").read_text())["features"]]
    tree = STRtree([m["g"] for m in ms])

    import numpy as np
    hit, ious = 0, []
    for o in osm:
        g = o["g"]
        idxs = tree.query(g)
        cov = 0.0
        best = 0.0
        if len(idxs):
            cov = unary_union([g.intersection(ms[i]["g"]) for i in idxs]).area / g.area
            for i in idxs:
                mg = ms[i]["g"]
                inter = g.intersection(mg).area
                best = max(best, inter / (g.area + mg.area - inter))
        if cov >= 0.5:
            hit += 1
            ious.append(best)
    recall = round(100 * hit / max(1, len(osm)), 1)
    geo_iou = round(float(np.median(ious)), 3) if ious else None

    # corregimientos
    geo = json.loads((DATA / "corregimientos_geom.json").read_text())
    corr = []
    for el in geo["elements"]:
        poly = stitch(el.get("members", []))
        if poly is None:
            continue
        corr.append({"name": el["tags"]["name"], "g": to_utm(poly)})

    cands = [{"g": to_utm(shape(f["geometry"])), "p": f["properties"]}
             for f in json.loads((OUT / "candidates.geojson").read_text())["features"]]
    bbox = box(-79.56, 9.02, -79.43, 9.10)
    ms_in = [m for m in ms if bbox.contains(to_ll(m["g"]).centroid)]

    rows = []
    for c in corr:
        poly = c["g"]
        n_osm = sum(1 for o in osm if poly.contains(o["g"].centroid))
        n_ms = sum(1 for m in ms_in if poly.contains(m["g"].centroid))
        mine = [x for x in cands if poly.contains(x["g"].centroid)]
        rows.append({
            "corregimiento": c["name"],
            "km2": round(poly.area / 1e6, 2),
            "osm_buildings": n_osm,
            "ml_buildings": n_ms,
            "unmapped": len(mine),
            "unmapped_verified": sum(1 for x in mine if x["p"].get("verdict") == "verified"),
        })
    rows.sort(key=lambda r: -r["unmapped"])

    out = {"ml_layer_recall_on_osm_pct": recall,
           "ml_vs_osm_median_iou": geo_iou,
           "ml_vs_osm_iou_p25_p75": [round(float(np.percentile(ious, 25)), 3),
                                     round(float(np.percentile(ious, 75)), 3)] if ious else None,
           "osm_buildings_tested": len(osm),
           "ml_in_bbox": len(ms_in),
           "corregimientos": rows}
    print(json.dumps(out, indent=1))
    (OUT / "stats_detail.json").write_text(json.dumps(out, indent=1))
    write_geojson(OUT / "corregimientos.geojson",
                  [{"geometry": to_ll(c["g"]), "properties": {"name": c["name"]}} for c in corr])


if __name__ == "__main__":
    main()
