"""Exports for mappers: CSV, JOSM reference layer, QA contact sheet."""
from __future__ import annotations

import json
import sys

from shapely.geometry import shape

from common import OUT
from report import make_csv, qa_sheet, write_osm


def main():
    cands = json.loads((OUT / "candidates.geojson").read_text())["features"]
    for i, f in enumerate(cands):
        c = shape(f["geometry"]).centroid
        f["properties"]["id"] = f"c{i}"
        f["properties"]["lat"] = round(c.y, 7)
        f["properties"]["lon"] = round(c.x, 7)
    make_csv(cands, OUT / "candidates.csv")
    # JOSM layer: only the compact, dataset-derived footprints. Model blobs have
    # hundreds of vertices each and would make a 100 MB file.
    keep = [f for f in cands if f["properties"].get("verdict") == "verified"
            and f["properties"].get("source") == "ms"]
    write_osm(keep, OUT / "candidates.osm")
    qa_sheet(cands, OUT / "qa_sample.png")
    cvonly = [f for f in cands if f["properties"].get("source") == "cv"]
    if len(cvonly) >= 12:
        qa_sheet(cvonly, OUT / "qa_cv_only.png", n=12, seed=3)
    print("exports written")


if __name__ == "__main__":
    sys.exit(main())
