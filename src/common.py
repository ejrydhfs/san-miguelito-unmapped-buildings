"""Shared geometry / web-mercator / GeoJSON helpers."""
from __future__ import annotations

import json
import math
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import (GeometryCollection, MultiPolygon, Polygon, box,
                              shape)
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
DATA, CACHE, OUT = ROOT / "data", ROOT / "cache", ROOT / "out"
for d in (DATA, CACHE, OUT):
    d.mkdir(exist_ok=True)

# UTM 17N — metres, area-preserving enough for Panama
TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
TO_LL = Transformer.from_crs("EPSG:32617", "EPSG:4326", always_xy=True)

R = 6378137.0


def to_utm(geom):
    return shp_transform(TO_UTM.transform, geom)


def to_ll(geom):
    return shp_transform(TO_LL.transform, geom)


# ---------------------------------------------------------------- web mercator
def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x: float, y: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def tile_bounds(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    """west, south, east, north"""
    w, n_ = tile_to_lonlat(x, y, z)
    e, s = tile_to_lonlat(x + 1, y + 1, z)
    return w, s, e, n_


def tile_range(bounds, z: int):
    w, s, e, n = bounds
    x0, y0 = lonlat_to_tile(w, n, z)
    x1, y1 = lonlat_to_tile(e, s, z)
    return range(int(x0), int(x1) + 1), range(int(y0), int(y1) + 1)


# ---------------------------------------------------------------- geojson I/O
def load_geojson(path):
    return json.loads(Path(path).read_text())


def iter_features(gj):
    t = gj.get("type")
    if t == "FeatureCollection":
        yield from gj["features"]
    elif t == "Feature":
        yield gj
    else:
        yield {"type": "Feature", "properties": {}, "geometry": gj}


def polys_of(geom):
    """Explode any geometry into a list of Polygons (dropping empties)."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        out = []
        for g in geom.geoms:
            out.extend(polys_of(g))
        return out
    if hasattr(geom, "geoms"):
        out = []
        for g in geom.geoms:
            out.extend(polys_of(g))
        return out
    return []


def write_geojson(path, features, props_round=7):
    from shapely.geometry import mapping

    feats = []
    for f in features:
        geom = f["geometry"]
        geoj = geom.__geo_interface__ if hasattr(geom, "__geo_interface__") else geom
        feats.append({"type": "Feature",
                      "properties": f.get("properties", {}),
                      "geometry": geoj})
    Path(path).write_text(json.dumps({"type": "FeatureCollection",
                                      "features": feats}))
