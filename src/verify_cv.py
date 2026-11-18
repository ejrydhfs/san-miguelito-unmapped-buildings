"""Second-opinion pass: run the HOTOSM DINOv3 building segmenter over the
candidate footprints and keep the pixels covered by the segmentation.

Every candidate is rasterised in the imagery tile(s) it touches; the mean
building probability inside the footprint and the fraction of the footprint
above the model threshold tell us whether a building is actually visible in
the current imagery.

Model: hotosm/dinov3s-buildings (DINOv3 ViT-S/16 + UperNet, ONNX).
Threshold from the model card: 0.4371 on the building class (channel 0).
"""
from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np
from shapely.geometry import shape

from common import OUT, tile_bounds, lonlat_to_tile
from eval_model import building_prob, load_model
from imagery import fetch, tile_path

Z = 18
THR = 0.4371
SIZE = 256
KERNEL = np.ones((7, 7), np.uint8)


def tiles_for(geoms):
    out = {}
    for i, g in enumerate(geoms):
        minx, miny, maxx, maxy = g.bounds
        x0, y0 = lonlat_to_tile(minx, maxy, Z)
        x1, y1 = lonlat_to_tile(maxx, miny, Z)
        for x in range(int(x0), int(x1) + 1):
            for y in range(int(y0), int(y1) + 1):
                out.setdefault((x, y), []).append(i)
    return out


def poly_mask(geom, tx, ty):
    """Rasterise a lon/lat polygon into a tile-local uint8 mask."""
    polys = getattr(geom, "geoms", [geom])
    pts = []
    for p in polys:
        ext = np.asarray(p.exterior.coords, dtype=np.float64)
        gx = (ext[:, 0] + 180.0) / 360.0 * 2 ** Z
        gy = (1.0 - np.arcsinh(np.tan(np.radians(ext[:, 1]))) / np.pi) / 2.0 * 2 ** Z
        px = (gx - tx) * SIZE
        py = (gy - ty) * SIZE
        pts.append(np.round(np.stack([px, py], 1)).astype(np.int32))
    m = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillPoly(m, pts, 1)
    return m.astype(bool)


def main(limit=None):
    cands = json.loads((OUT / "missing_buildings_district.geojson").read_text())["features"]
    osm = json.loads((OUT / "osm_buildings.geojson").read_text())["features"]
    cgeom = [shape(f["geometry"]) for f in cands]
    ogeom = [shape(f["geometry"]) for f in osm]
    ctile = tiles_for(cgeom)
    otile = tiles_for(ogeom)
    tiles = sorted(set(ctile) | set(otile))
    if limit:
        tiles = tiles[:limit]
    print(f"candidates={len(cands)} tiles={len(tiles)}")
    fetch(Z, tiles, workers=32)

    sess = load_model()
    # accumulated per polygon: sum prob, px count, cover count, max prob
    # per polygon: sum prob, px count, px near a building blob, max prob
    acc = {i: [0.0, 0, 0, 0.0] for i in range(len(cgeom) + len(ogeom))}
    t0 = time.time()
    missing_img = 0
    for n, (x, y) in enumerate(tiles, 1):
        p = tile_path(Z, x, y)
        img = cv2.imread(str(p))
        if img is None:
            missing_img += 1
            continue
        prob = building_prob(sess, img, norm=True)
        # a footprint pixel counts as covered when a predicted building blob is
        # within ~3 px (1.8 m): absorbs the model's 1-2 px boundary slop
        dil = cv2.dilate((prob >= THR).astype(np.uint8), KERNEL)
        for i in ctile.get((x, y), ()):
            m = poly_mask(cgeom[i], x, y)
            npx = int(m.sum())
            if npx == 0:
                continue
            a = acc[i]
            a[0] += float(prob[m].sum())
            a[1] += npx
            a[2] += int(dil[m].sum())
            a[3] = max(a[3], float(prob[m].max()))
        off = len(cgeom)
        for i in otile.get((x, y), ()):
            m = poly_mask(ogeom[i], x, y)
            npx = int(m.sum())
            if npx == 0:
                continue
            a = acc[off + i]
            a[0] += float(prob[m].sum())
            a[1] += npx
            a[2] += int(dil[m].sum())
            a[3] = max(a[3], float(prob[m].max()))
        if n % 200 == 0:
            print(f"  {n}/{len(tiles)} tiles  {time.time()-t0:.0f}s")

    def summarize(feats, base):
        out = []
        for i, f in enumerate(feats):
            s, npx, cov, mx = acc[base + i]
            if npx == 0:
                continue
            f["properties"].update({
                "cv_prob": round(s / npx, 3),
                "cv_cover": round(cov / npx, 3),
                "cv_max": round(mx, 3),
            })
            out.append(f)
        return out

    cands = summarize(cands, 0)
    osmv = summarize(osm, len(cgeom))

    def cls(p):
        if p["cv_cover"] >= 0.60 and p["cv_prob"] >= 0.40:
            return "verified"
        if p["cv_cover"] >= 0.30:
            return "probable"
        return "unconfirmed"

    for f in cands:
        f["properties"]["verdict"] = cls(f["properties"])
    for f in osmv:
        f["properties"]["verdict"] = cls(f["properties"])

    OUT.joinpath("missing_verified.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": cands}))
    OUT.joinpath("osm_cv_control.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": osmv}))

    def dist(feats):
        from collections import Counter
        c = Counter(f["properties"]["verdict"] for f in feats)
        return {k: round(v / max(1, len(feats)), 3) for k, v in c.items()}

    report = {
        "tiles_scanned": len(tiles),
        "missing_tile_reads": missing_img,
        "candidates": dist(cands),
        "osm_control": dist(osmv),
        "threshold": THR,
        "model": "hotosm/dinov3s-buildings",
    }
    print(json.dumps(report, indent=2))
    OUT.joinpath("cv_stats.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
