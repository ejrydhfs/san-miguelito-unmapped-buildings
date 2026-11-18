"""Exports for mappers: CSV, JOSM reference layer, QA contact sheet."""
from __future__ import annotations

import csv

import cv2
import numpy as np
from shapely.geometry import shape

from common import OUT, lonlat_to_tile
from imagery import fetch, tile_path

Z = 18


def r6(coords):
    if isinstance(coords[0], (int, float)):
        return [round(coords[0], 6), round(coords[1], 6)]
    return [r6(c) for c in coords]


def trim(feats, keys):
    out = []
    for f in feats:
        g = f["geometry"]
        if g["type"] == "Polygon":
            geom = {"type": "Polygon", "coordinates": [r6(r) for r in g["coordinates"]]}
        else:
            geom = {"type": "MultiPolygon",
                    "coordinates": [[r6(r) for r in part] for part in g["coordinates"]]}
        out.append({"type": "Feature", "properties": {k: f["properties"].get(k) for k in keys},
                    "geometry": geom})
    return out


def make_csv(cands, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "lat", "lon", "area_m2", "verdict", "source", "cv_cover",
                    "nearest_osm_m", "ms_conf"])
        for f in cands:
            c = shape(f["geometry"]).centroid
            p = f["properties"]
            w.writerow([p.get("id", ""), round(c.y, 7), round(c.x, 7), p.get("area_m2"),
                        p.get("verdict"), p.get("source"), p.get("cv_cover"),
                        p.get("nearest_osm_m"), p.get("ms_conf")])


def write_osm(cands, path):
    """Untagged ways for JOSM, so mappers can trace over the candidates.

    Negative ids mark them as new objects; the file has no tags on purpose so an
    accidental upload cannot add untagged building outlines to OSM.
    """
    lines = ["<?xml version='1.0' encoding='UTF-8'?>",
             "<osm version='0.6' generator='osm-missing-buildings'>"]
    nid = -1
    for f in cands:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            for ring in poly:
                ids = []
                for lon, lat in ring[:-1]:
                    lines.append(f"  <node id='{nid}' lat='{lat:.7f}' lon='{lon:.7f}' version='1'/>")
                    ids.append(nid)
                    nid -= 1
                lines.append(f"  <way id='{nid}' version='1'>")
                nid -= 1
                lines += [f"    <nd ref='{i}'/>" for i in ids]
                lines.append("  </way>")
    lines.append("</osm>")
    __import__("pathlib").Path(path).write_text("\n".join(lines))
    print("osm reference layer:", len(cands), "ways")


def qa_sheet(cands, path, n=12, seed=7):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cands), size=n, replace=False)
    tiles_needed = []
    for i in pick:
        g = shape(cands[i]["geometry"])
        c = g.centroid
        x, y = lonlat_to_tile(c.x, c.y, Z)
        tiles_needed.append((int(x), int(y)))
    fetch(Z, sorted(set(tiles_needed)))
    tiles = []
    for i, (x, y) in zip(pick, tiles_needed):
        img = cv2.imread(str(tile_path(Z, x, y)))
        g = shape(cands[i]["geometry"])
        polys = getattr(g, "geoms", [g])
        for p in polys:
            ext = np.asarray(p.exterior.coords)
            gx = (ext[:, 0] + 180) / 360 * 2 ** Z
            gy = (1 - np.arcsinh(np.tan(np.radians(ext[:, 1]))) / np.pi) / 2 * 2 ** Z
            px = (gx - x) * 256
            py = (gy - y) * 256
            cv2.polylines(img, [np.round(np.stack([px, py], 1)).astype(np.int32)],
                          True, (0, 255, 0), 2)
        prop = cands[i]["properties"]
        label = (f"{prop.get('verdict', '?')} a={prop['area_m2']:.0f} "
                 f"c={prop.get('cv_cover') or 0:.2f}")
        cv2.rectangle(img, (0, 0), (256, 18), (0, 0, 0), -1)
        cv2.putText(img, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles.append(cv2.resize(img, (420, 420), interpolation=cv2.INTER_CUBIC))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(path), np.vstack(rows))


