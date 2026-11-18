"""Full-district segmentation scan (z18), two passes.

Pass 1  inference on every tile that covers the district -> binary mask PNGs
Pass 2  stitch masks into 2-row bands, vectorise, keep each blob once

Blobs are what the model sees as built-up; the diff in diff2.py subtracts the
existing OSM + Microsoft footprints to expose blind spots of both datasets.
"""
from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np
from shapely.geometry import Polygon, shape

from common import DATA, OUT, CACHE, lonlat_to_tile, tile_bounds, write_geojson
from eval_model import building_prob, load_model
from imagery import fetch, tile_path

Z = 18
THR = 0.4371
SIZE = 256
MASKS = CACHE / "masks18"
MIN_AREA_PX = 30          # ~10 m2


def district_tiles(margin=1):
    bnd = json.loads((DATA / "san_miguelito_boundary.json").read_text())[0]
    g = shape(bnd["geojson"])
    w, s, e, n = g.bounds
    x0, y0 = lonlat_to_tile(w, n, Z)
    x1, y1 = lonlat_to_tile(e, s, Z)
    xs = range(int(x0) - margin, int(x1) + 1 + margin)
    ys = range(int(y0) - margin, int(y1) + 1 + margin)
    return [(x, y) for x in xs for y in ys]


def mask_path(x, y):
    return MASKS / f"{x}_{y}.png"


def pass1(tiles):
    MASKS.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tiles if not mask_path(*t).exists()]
    print(f"pass1: {len(tiles)} tiles, {len(todo)} to compute")
    fetch(Z, todo, workers=24)
    sess = load_model()
    t0 = time.time()
    n_none = 0
    for n, (x, y) in enumerate(todo, 1):
        img = cv2.imread(str(tile_path(Z, x, y)))
        if img is None:
            n_none += 1
            cv2.imwrite(str(mask_path(x, y)), np.zeros((SIZE, SIZE), np.uint8))
            continue
        prob = building_prob(sess, img, norm=True)
        cv2.imwrite(str(mask_path(x, y)), ((prob >= THR) * 255).astype(np.uint8))
        if n % 100 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el:.0f}s  {n/max(el,1e-9):.2f} tile/s  none={n_none}")
    print(f"pass1 done in {time.time()-t0:.0f}s, tiles without imagery: {n_none}")


def band_to_polys(y, xs):
    """Vectorise rows y..y+1 into polygons belonging to row y."""
    band = np.zeros((2 * SIZE, len(xs) * SIZE), np.uint8)
    for i, x in enumerate(xs):
        p = mask_path(x, y)
        if p.exists():
            m = cv2.imread(str(p), 0)
            band[0:SIZE, i * SIZE:(i + 1) * SIZE] = m
        p2 = mask_path(x, y + 1)
        if p2.exists():
            m2 = cv2.imread(str(p2), 0)
            band[SIZE:, i * SIZE:(i + 1) * SIZE] = m2
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA_PX:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cy = m["m01"] / m["m00"]
        if cy >= SIZE:          # belongs to the next band
            continue
        poly = c.reshape(-1, 2).astype(np.int32)
        geo = _px_to_lonlat(poly, xs, y)
        if geo is not None:
            out.append({"geometry": geo, "area_px": area})
    return out


def _px_to_lonlat(pts, xs, y):
    # pixel (global band) -> global tile coords -> lon/lat
    out = []
    for px, py in pts:
        tx = xs[0] + px / SIZE
        ty = y + py / SIZE
        lon, lat = _tile_to_lonlat(tx, ty)
        out.append((lon, lat))
    if len(out) < 4:
        return None
    p = Polygon(out)
    if not p.is_valid:
        p = p.buffer(0)
        if p.is_empty:
            return None
    return p


def _tile_to_lonlat(x, y):
    n = 2 ** Z
    return x / n * 360.0 - 180.0, np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))


def main(limit_rows=None, smoke=False):
    tiles = district_tiles()
    if smoke:
        xs0 = sorted({x for x, _ in tiles})[:12]
        ys0 = sorted({y for _, y in tiles})[:2]
        tiles = [(x, y) for x in xs0 for y in ys0]
        limit_rows = 1
    pass1(tiles)
    xs = sorted({x for x, _ in tiles})
    ys = sorted({y for _, y in tiles})
    print(f"pass2: {len(xs)}x{len(ys)} tiles")
    feats = []
    for k, y in enumerate(ys):
        if limit_rows and k >= limit_rows:
            break
        polys = band_to_polys(y, xs)
        for r in polys:
            feats.append({"geometry": r["geometry"],
                          "properties": {"area_px": round(r["area_px"], 1),
                                         "row": y}})
        print(f"  row {y} ({k+1}/{len(ys)}): +{len(polys)} blobs, total {len(feats)}")
    write_geojson(OUT / "cv_all.geojson", feats)
    print("blobs:", len(feats))


if __name__ == "__main__":
    if sys.argv[1:2] == ["smoke"]:
        main(smoke=True)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
