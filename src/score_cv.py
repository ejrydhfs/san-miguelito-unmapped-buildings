"""Score polygons against the cached segmentation masks (no re-inference).

cv_cover      share of the footprint within ~1.8 m of a predicted building pixel
cv_cover_raw  share of the footprint that is predicted building pixel

verdict thresholds are calibrated on existing OSM buildings: 91 % of them reach
cv_cover >= 0.6 on Panama imagery.

Usage: python score_cv.py [--control]
       --control also scores all OSM buildings and prints the calibration
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import cv2
import numpy as np
from shapely.geometry import shape

from common import OUT, lonlat_to_tile
from scan_cv import Z, mask_path
from verify_cv import poly_mask

KERNEL = np.ones((7, 7), np.uint8)


def score(feats, label):
    by_tile = {}
    for i, f in enumerate(feats):
        minx, miny, maxx, maxy = shape(f["geometry"]).bounds
        x0, y0 = lonlat_to_tile(minx, maxy, Z)
        x1, y1 = lonlat_to_tile(maxx, miny, Z)
        for x in range(int(x0), int(x1) + 1):
            for y in range(int(y0), int(y1) + 1):
                by_tile.setdefault((x, y), []).append(i)
    acc = {i: [0, 0, 0] for i in range(len(feats))}
    missing_tiles = 0
    for n, (t, ids) in enumerate(by_tile.items(), 1):
        mp = mask_path(*t)
        if not mp.exists():
            missing_tiles += 1
            continue
        m = cv2.imread(str(mp), 0)
        if m is None:
            missing_tiles += 1
            continue
        raw = m > 0
        dil = cv2.dilate(raw.astype(np.uint8), KERNEL) > 0
        x, y = t
        for i in ids:
            pm = poly_mask(shape(feats[i]["geometry"]), x, y)
            k = int(pm.sum())
            if not k:
                continue
            a = acc[i]
            a[0] += int((raw & pm).sum())
            a[1] += int((dil & pm).sum())
            a[2] += k
        if n % 500 == 0:
            print(f"  {label}: {n}/{len(by_tile)} tiles")
    out = []
    for i, f in enumerate(feats):
        raw, dil, n = acc[i]
        f["properties"]["cv_cover_raw"] = round(raw / n, 3) if n else None
        f["properties"]["cv_cover"] = round(dil / n, 3) if n else None
        out.append(f)
    print(f"  {label}: {missing_tiles} tiles without mask")
    return out


def main(control=False):
    cands = json.loads((OUT / "candidates.geojson").read_text())["features"]
    print("candidates", len(cands))
    score(cands, "candidates")
    for f in cands:
        v = f["properties"]["cv_cover"]
        f["properties"]["verdict"] = ("verified" if v is not None and v >= 0.60 else
                                      "probable" if v is not None and v >= 0.30 else
                                      "unconfirmed")
    OUT.joinpath("candidates.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": cands}))
    print(Counter(f["properties"]["verdict"] for f in cands))

    if control:
        osm = json.loads((OUT / "osm_buildings.geojson").read_text())["features"]
        osm = [f for f in osm if 9.02 <= shape(f["geometry"]).centroid.y <= 9.10]
        score(osm, "osm-control")
        v = [f["properties"]["cv_cover"] for f in osm if f["properties"]["cv_cover"] is not None]
        ok = sum(1 for x in v if x >= 0.60)
        rep = {"osm_control_verified_pct": round(100 * ok / max(1, len(v)), 1),
               "osm_control_median_cover": round(float(np.median(v)), 3),
               "n": len(v)}
        print(rep)
        OUT.joinpath("cv_control.json").write_text(json.dumps(rep, indent=1))
        OUT.joinpath("osm_cv_control.geojson").write_text(json.dumps(
            {"type": "FeatureCollection", "features": osm}))


if __name__ == "__main__":
    main("--control" in sys.argv)
