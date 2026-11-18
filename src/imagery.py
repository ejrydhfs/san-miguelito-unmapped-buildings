"""Fetch and cache web-mercator imagery tiles (Esri World Imagery).

Esri World Imagery is used instead of Google's satellite tiles: Google's ToS
forbid scraping/downloading tiles, Esri publishes a documented tile endpoint
that the OSM iD editor itself uses (attribution: Esri, Maxar, Earthstar
Geographics, and the GIS User Community).
"""
from __future__ import annotations

import concurrent.futures as cf
import sys
import time
from pathlib import Path

import requests

from common import CACHE, tile_range, tile_bounds

URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
       "World_Imagery/MapServer/tile/{z}/{y}/{x}")
UA = {"User-Agent": "osm-missing-buildings/0.1 (building footprint QA)"}


def tile_path(z, x, y):
    p = CACHE / "tiles" / f"{z}" / f"{x}_{y}.jpg"
    return p


def fetch_one(z, x, y, session):
    p = tile_path(z, x, y)
    if p.exists() and p.stat().st_size > 1000:
        return p, True
    p.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            r = session.get(URL.format(z=z, x=x, y=y), timeout=30)
            if r.status_code == 200 and len(r.content) > 500:
                p.write_bytes(r.content)
                return p, False
        except requests.RequestException:
            pass
        time.sleep(1 + attempt)
    return p, False


def fetch(z, tiles, workers=24):
    got = 0
    t0 = time.time()
    with requests.Session() as s:
        s.headers.update(UA)
        with cf.ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(fetch_one, z, x, y, s): (x, y) for x, y in tiles}
            for i, f in enumerate(cf.as_completed(futs), 1):
                p, cached = f.result()
                got += 1
                if i % 500 == 0:
                    print(f"  {i}/{len(tiles)} tiles ({time.time()-t0:.0f}s)")
    return got


def main():
    z = int(sys.argv[1])
    w, s, e, n = map(float, sys.argv[2:6])
    xs, ys = tile_range((w, s, e, n), z)
    tiles = [(x, y) for x in xs for y in ys]
    print(f"z={z} tiles={len(tiles)} bbox={w},{s},{e},{n}")
    fetch(z, tiles)
    print("done")


if __name__ == "__main__":
    main()
