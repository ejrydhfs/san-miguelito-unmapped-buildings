"""Turn Overpass `out geom` output into clean building polygons.

- ways -> LinearRing -> Polygon
- multipolygon relations (building) -> outer rings only
- drops building=no, degenerate shapes
- does NOT clip to the district (matching needs the ring just outside it)
"""
from __future__ import annotations

import json
import sys

from shapely.geometry import LinearRing, MultiPolygon, Polygon
from shapely.validation import make_valid

from common import DATA, OUT, polys_of, to_utm, write_geojson

RAW = DATA / "osm_buildings_raw.json"


def ring(coords):
    pts = [(n["lon"], n["lat"]) for n in coords]
    if len(pts) < 3:
        return None
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    r = LinearRing(pts)
    return r if r.is_valid and not r.is_empty else None


def main():
    d = json.loads(RAW.read_text())
    feats, bad = [], 0
    for el in d["elements"]:
        tags = el.get("tags", {})
        if tags.get("building") in ("no", "roof"):
            continue
        if el["type"] == "way":
            r = ring(el.get("geometry") or [])
            if r is None:
                bad += 1
                continue
            geom = Polygon(r)
        elif el["type"] == "relation":
            parts = []
            for m in el.get("members", []):
                if m.get("role") != "outer":
                    continue
                r = ring(m.get("geometry") or [])
                if r is not None:
                    parts.append(Polygon(r))
            if not parts:
                bad += 1
                continue
            geom = Polygon(parts[0].exterior)
        else:
            continue
        if not geom.is_valid:
            geom = make_valid(geom)
        ps = [p for p in polys_of(geom) if p.area > 1e-10]
        if not ps:
            bad += 1
            continue
        geom = ps[0] if len(ps) == 1 else MultiPolygon(ps)
        feats.append({
            "geometry": geom,
            "properties": {
                "osm": ("w" if el["type"] == "way" else "r") + str(el["id"]),
                "osm_id": el["id"],
                "building": tags.get("building", "yes"),
                "area_m2": round(to_utm(geom).area, 1),
            },
        })
    feats.sort(key=lambda f: f["properties"]["osm_id"])
    write_geojson(OUT / "osm_buildings.geojson", feats)
    print(f"OSM buildings kept={len(feats)} dropped={bad}")


if __name__ == "__main__":
    sys.exit(main())
