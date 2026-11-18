"""Gap-closing diff (v2).

Improvements over diff.py
  * candidates are the *uncovered remainder* of an ML footprint, not the whole
    footprint: a row of houses where only one is mapped now yields the other
    houses instead of being dropped as "partial".
  * OSM/MS are buffered by 1.5 m before subtraction, so a 1-2 m digitising
    offset does not leave slivers.
  * a second source: model blobs (cv_all.geojson) minus both datasets, which
    surfaces buildings that neither OSM nor Microsoft has.

Output: out/candidates.geojson (district only) with source=ms|cv.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

from shapely.geometry import box, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from common import DATA, OUT, polys_of, to_ll, to_utm, write_geojson

BUFFER = 1.5            # m, absorbs digitising offsets
MIN_REM = 30.0          # m2 of uncovered area to become a candidate
MIN_FRAC = 0.30         # share of the ML footprint that must be uncovered
CV_MIN_REM = 60.0       # model-only blobs are noisier: demand a bigger remainder
CV_MIN_BLOB = 100.0
MS_MAX_AREA = 20000.0
DEDUPE_IOU = 0.55
GRID_M = 400.0


def load(path, name):
    feats = json.loads((OUT / path).read_text())["features"]
    out = []
    for f in feats:
        for p in polys_of(shape(f["geometry"])):
            p = to_utm(p)
            if p.area >= 10:
                out.append({"g": p, "p": f["properties"]})
    print(f"{name}: {len(out)}")
    return out


def main():
    bnd = json.loads((DATA / "san_miguelito_boundary.json").read_text())[0]
    district = to_utm(shape(bnd["geojson"]))

    osm = load("osm_buildings.geojson", "osm")
    ms = load("ms_buildings.geojson", "ms")
    ogeom = [o["g"] for o in osm]
    obuf = [g.buffer(BUFFER) for g in ogeom]
    oTree = STRtree(ogeom)
    obTree = STRtree(obuf)

    ms_keep = [m for m in ms if m["g"].area <= MS_MAX_AREA]
    candidates = []

    # ---- source 1: Microsoft footprints minus OSM -------------------------
    for m in ms_keep:
        g = m["g"]
        idx = obTree.query(g)
        rem = g.difference(unary_union([obuf[i] for i in idx])) if len(idx) else g
        if rem.is_empty:
            continue
        parts = [p for p in polys_of(rem)
                 if p.area >= MIN_REM and p.area / g.area >= MIN_FRAC]
        if not parts:
            continue
        rem = unary_union(parts)
        if not district.intersects(rem):
            continue
        near = oTree.nearest(g)
        candidates.append({
            "g": rem,
            "p": {
                "source": "ms",
                "area_m2": round(rem.area, 1),
                "ms_area_m2": round(g.area, 1),
                "uncovered_frac": round(rem.area / g.area, 3),
                "nearest_osm_m": round(g.distance(ogeom[int(near)]), 1),
                "ms_conf": m["p"].get("conf"),
            },
        })
    print("ms remainder candidates:", len(candidates))

    # ---- source 2: model blobs minus OSM and MS ---------------------------
    if not (OUT / "cv_all.geojson").exists():
        print("cv_all.geojson missing - skipping source 2")
        cv = []
    else:
        cv = load("cv_all.geojson", "cv")
    mg = [m["g"] for m in ms_keep]
    mbuf = [g.buffer(BUFFER) for g in mg]
    allg = ogeom + mg
    allbuf = obuf + mbuf
    ref = STRtree(allg)
    refbuf = STRtree(allbuf)
    n_before = len(candidates)
    for b in cv:
        g = b["g"]
        if g.area < CV_MIN_BLOB:
            continue
        idx = refbuf.query(g)
        rem = g.difference(unary_union([allbuf[i] for i in idx])) if len(idx) else g
        if rem.is_empty:
            continue
        parts = [p for p in polys_of(rem) if p.area >= CV_MIN_REM]
        if not parts:
            continue
        rem = unary_union(parts)
        if not district.intersects(rem):
            continue
        near = ref.nearest(g)
        candidates.append({
            "g": rem,
            "p": {
                "source": "cv",
                "area_m2": round(rem.area, 1),
                "ms_area_m2": None,
                "uncovered_frac": round(rem.area / max(g.area, 1e-6), 3),
                "nearest_osm_m": round(g.distance(allg[int(near)]), 1),
                "ms_conf": None,
            },
        })
    print("cv-only remainder candidates:", len(candidates) - n_before)

    # ---- dedupe: MS geometry wins over CV blobs ---------------------------
    candidates.sort(key=lambda c: (c["p"]["source"] != "ms", -c["g"].area))
    keep = []
    grid = {}
    CELL = 60.0
    for c in candidates:
        g = c["g"]
        minx, miny, maxx, maxy = g.bounds
        keys = {(int(x // CELL), int(y // CELL))
                for x in range(int(minx), int(maxx) + 1, int(CELL))
                for y in range(int(miny), int(maxy) + 1, int(CELL))}
        dup = False
        for k in keys:
            for j in grid.get(k, ()):
                kg = keep[j]["g"]
                if not kg.intersects(g):
                    continue
                inter = kg.intersection(g).area
                if inter / (kg.area + g.area - inter) > DEDUPE_IOU:
                    dup = True
                    break
            if dup:
                break
        if not dup:
            keep.append(c)
            for k in keys:
                grid.setdefault(k, []).append(len(keep) - 1)
    print("after dedupe:", len(keep))

    # ---- district clip + outputs ------------------------------------------
    inside = []
    for c in keep:
        p = c["g"].intersection(district)
        if p.is_empty or p.area < 15:
            continue
        c["g"] = p.simplify(0.3, preserve_topology=True)
        c["p"]["area_m2"] = round(c["g"].area, 1)
        inside.append(c)

    write_geojson(OUT / "candidates.geojson",
                  [{"geometry": to_ll(c["g"]), "properties": c["p"]} for c in inside])

    osm_d = sum(1 for o in osm if district.contains(o["g"].centroid))
    ms_d = sum(1 for m in ms if district.contains(m["g"].centroid))
    # 400 m priority grid ---------------------------------------------------
    xmin, _, _, ymax = district.bounds
    cells = {}
    for c in inside:
        cx, cy = c["g"].centroid.coords[0]
        key = (int((cx - xmin) // GRID_M), int((ymax - cy) // GRID_M))
        cells.setdefault(key, []).append(c["p"]["area_m2"])
    grid = []
    for (ix, iy), areas in cells.items():
        x0 = xmin + ix * GRID_M
        y1 = ymax - iy * GRID_M
        grid.append({"geometry": to_ll(box(x0, y1 - GRID_M, x0 + GRID_M, y1)),
                     "properties": {"count": len(areas),
                                    "area_m2": round(sum(areas)),
                                    "grid": f"{ix}_{iy}"}})
    grid.sort(key=lambda f: -f["properties"]["count"])
    write_geojson(OUT / "priority_grid.geojson", grid)
    print("grid cells:", len(grid), "top:", grid[0]["properties"])

    stats = {
        "district": bnd["display_name"],
        "district_short": "San Miguelito, Panama City",
        "osm_buildings": osm_d,
        "ml_buildings": ms_d,
        "osm_buildings_bbox": len(osm),
        "ml_buildings_bbox": len(ms),
        "candidates": len(inside),
        "by_source": dict(Counter(c["p"]["source"] for c in inside)),
        "area_m2": round(sum(c["p"]["area_m2"] for c in inside)),
        "median_area_m2": sorted(c["p"]["area_m2"] for c in inside)[len(inside) // 2],
    }
    print(json.dumps(stats, indent=2))
    (OUT / "diff2_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    sys.exit(main())
