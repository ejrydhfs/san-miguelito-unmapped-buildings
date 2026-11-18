"""Diff the ML-derived footprint layer against OSM buildings.

Everything is projected to UTM 17N. For each ML polygon the covered fraction
of its area is measured against the OSM building polygons; the uncovered
remainder is the candidate set of unmapped buildings.

Outputs
    out/missing_buildings.geojson          candidates in the whole bbox
    out/missing_buildings_district.geojson candidates inside San Miguelito
    out/priority_grid.geojson              400 m cells ranked by candidates
    out/diff_stats.json
"""
from __future__ import annotations

import json
import math
import sys

from shapely.geometry import box, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from common import DATA, OUT, polys_of, to_ll, to_utm, write_geojson

MATCH_RATIO = 0.55      # OSM covers >= 55% of the ML footprint -> mapped
PARTIAL_RATIO = 0.20    # some overlap -> mapped, but offset/shape differs
MIN_AREA, MAX_AREA = 12.0, 4000.0
DEDUPE_IOU = 0.60
GRID_M = 400.0
CELL = 50.0             # dedupe bucket size (m)


def load_layer(path):
    feats = json.loads((OUT / path).read_text())["features"]
    out = []
    for f in feats:
        for p in polys_of(shape(f["geometry"])):
            p = to_utm(p)
            if p.area >= MIN_AREA:
                out.append({"geom": p, "props": f["properties"]})
    return out


def cover_ratio(g, tree, geoms):
    idxs = tree.query(g)
    if not len(idxs):
        return 0.0
    inter = unary_union([g.intersection(geoms[i]) for i in idxs])
    return 0.0 if inter.is_empty else min(1.0, inter.area / g.area)


def dedupe(cands):
    """Drop ML polygons that duplicate another candidate (same roof twice)."""
    grid: dict[tuple[int, int], list[int]] = {}
    keep = []
    for c in sorted(cands, key=lambda c: -c["geom"].area):
        g = c["geom"]
        minx, miny, maxx, maxy = g.bounds
        keys = {(int(x // CELL), int(y // CELL))
                for x in range(int(minx), int(maxx) + 1, int(CELL))
                for y in range(int(miny), int(maxy) + 1, int(CELL))}
        dup = False
        for k in keys:
            for j in grid.get(k, ()):
                kg = keep[j]["geom"]
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
    return keep


def main():
    osm = load_layer("osm_buildings.geojson")
    ms = load_layer("ms_buildings.geojson")
    geoms = [o["geom"] for o in osm]
    tree = STRtree(geoms)
    print(f"osm={len(osm)} ms={len(ms)}")

    missing, matched, partial = [], 0, 0
    for m in ms:
        g = m["geom"]
        if g.area < MIN_AREA or g.area > MAX_AREA:
            continue
        cov = cover_ratio(g, tree, geoms)
        if cov >= MATCH_RATIO:
            matched += 1
            continue
        if cov >= PARTIAL_RATIO:
            partial += 1
            continue
        near = tree.nearest(g)
        missing.append({
            "geom": g,
            "props": {
                "area_m2": round(g.area, 1),
                "ms_conf": m["props"].get("conf"),
                "osm_cover": round(cov, 3),
                "nearest_osm_m": round(g.distance(geoms[int(near)]), 1),
            },
        })
    print(f"matched={matched} partial={partial} missing_raw={len(missing)}")
    keep = dedupe(missing)
    print(f"after dedupe={len(keep)}")

    write_geojson(OUT / "missing_buildings.geojson",
                  [{"geometry": to_ll(c["geom"]), "properties": c["props"]} for c in keep])

    bnd = json.loads((DATA / "san_miguelito_boundary.json").read_text())[0]
    district = shape(bnd["geojson"])
    district_utm = to_utm(district)
    inside = [c for c in keep if district_utm.contains(c["geom"].centroid)]
    for c in inside:
        c["props"]["district"] = "San Miguelito"
    write_geojson(OUT / "missing_buildings_district.geojson",
                  [{"geometry": to_ll(c["geom"]), "properties": c["props"]} for c in inside])

    osm_in = sum(1 for o in osm if district_utm.contains(o["geom"].centroid))
    ms_in = sum(1 for m in ms if district_utm.contains(m["geom"].centroid))
    areas = sorted(c["props"]["area_m2"] for c in inside)
    stats = {
        "district": bnd["display_name"],
        "bbox_wgs84": [-79.56, 9.02, -79.43, 9.10],
        "osm_buildings": osm_in,
        "ml_buildings": ms_in,
        "ml_buildings_bbox": len(ms),
        "district_km2": round(district_utm.area / 1e6, 2),
        "osm_vs_ml_coverage_pct": round(100 * osm_in / max(1, ms_in), 1),
        "matched": matched,
        "partial": partial,
        "missing_in_district": len(inside),
        "missing_area_m2": round(sum(areas)),
        "median_missing_area_m2": areas[len(areas) // 2] if areas else 0,
        "matched_ratio": round(matched / max(1, matched + partial + len(keep)), 4),
    }
    print(json.dumps(stats, indent=2))
    (OUT / "diff_stats.json").write_text(json.dumps(stats, indent=2))

    # ---- priority grid: 400 m cells ranked by unmapped buildings -----------
    xmin, _, _, ymax = district_utm.bounds
    cells = {}
    for c in inside:
        cx, cy = c["geom"].centroid.coords[0]
        key = (int((cx - xmin) // GRID_M), int((ymax - cy) // GRID_M))
        cells.setdefault(key, []).append(c["props"]["area_m2"])
    grid_feats = []
    for (ix, iy), cell_areas in cells.items():
        x0 = xmin + ix * GRID_M
        y1 = ymax - iy * GRID_M
        grid_feats.append({
            "geometry": to_ll(box(x0, y1 - GRID_M, x0 + GRID_M, y1)),
            "properties": {"count": len(cell_areas),
                           "area_m2": round(sum(cell_areas)),
                           "grid": f"{ix}_{iy}"},
        })
    grid_feats.sort(key=lambda f: -f["properties"]["count"])
    write_geojson(OUT / "priority_grid.geojson", grid_feats)
    print(f"grid cells={len(grid_feats)} top={grid_feats[0]['properties']}")


if __name__ == "__main__":
    sys.exit(main())
