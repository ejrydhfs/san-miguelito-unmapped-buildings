"""Extract Microsoft GlobalMLBuildingFootprints (2026-02 vintage) for the
quadkeys covering the San Miguelito bbox.

The download is GeoJSONL (one feature per line) despite the .csv.gz filename.
"""
from __future__ import annotations

import gzip
import json
import sys

from shapely.geometry import box, shape

from common import CACHE, DATA, OUT, polys_of, to_utm, write_geojson

TARGET = box(-79.56, 9.02, -79.43, 9.10)


def main(quadkeys):
    feats, seen = [], set()
    for qk in quadkeys:
        path = CACHE / f"ms_{qk}.csv.gz"
        n = kept = 0
        # file is GeoJSONL despite the .csv.gz extension
        with gzip.open(path, "rt") as fh:
            for line in fh:
                n += 1
                try:
                    feat = json.loads(line)
                    g = shape(feat["geometry"])
                except Exception:
                    continue
                if g.is_empty or not g.intersects(TARGET):
                    continue
                conf = feat.get("properties", {}).get("confidence", 0.0)
                for p in polys_of(g):
                    # fingerprint to survive duplicated rows across files
                    key = p.wkb
                    if key in seen:
                        continue
                    seen.add(key)
                    feats.append({
                        "geometry": p,
                        "properties": {
                            "src": "ms_global_buildings_2026-02",
                            "conf": round(float(conf or 0), 3),
                            "area_m2": round(to_utm(p).area, 1),
                        },
                    })
                    kept += 1
        print(f"{qk}: rows={n} kept={kept}")
    write_geojson(OUT / "ms_buildings.geojson", feats)
    print("total MS candidates in bbox:", len(feats))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
