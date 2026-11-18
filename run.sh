#!/usr/bin/env bash
# Full pipeline: San Miguelito unmapped-building survey.
# Needs: uv, ~2 GB disk for caches, ~1 h on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")"

uv venv -p 3.12 .venv 2>/dev/null || true
uv pip install -p .venv/bin/python numpy pillow requests shapely pyproj \
  onnxruntime opencv-python-headless scipy pyarrow

# ---- raw inputs -------------------------------------------------------------
[ -f data/san_miguelito_boundary.json ] || curl -s -A "osm-missing-buildings/0.1" \
  "https://nominatim.openstreetmap.org/search?q=San+Miguelito,+Panama&format=json&polygon_geojson=1&limit=1" \
  -o data/san_miguelito_boundary.json
[ -f data/osm_buildings_raw.json ] || curl -s -A "osm-missing-buildings/0.1" \
  --data-urlencode 'data=[out:json][timeout:250];(way["building"](9.02,-79.56,9.10,-79.43);relation["building"](9.02,-79.56,9.10,-79.43););out geom;' \
  -o data/osm_buildings_raw.json https://overpass-api.de/api/interpreter
[ -f data/corregimientos_geom.json ] || curl -s -A "osm-missing-buildings/0.1" \
  --data-urlencode 'data=[out:json][timeout:180];rel(id:8389112,8389284,10583752,10583798,11482516,11482888,11482967,11483078,11483091);out geom;' \
  -o data/corregimientos_geom.json https://overpass.kumi.systems/api/interpreter

# model (220 MB, ONNX)
[ -f cache/hot_dinov3s.onnx ] || curl -sL \
  "https://huggingface.co/hotosm/dinov3s-buildings/resolve/main/model.onnx" \
  -o cache/hot_dinov3s.onnx

# ---- pipeline ---------------------------------------------------------------
.venv/bin/python src/prepare_osm.py
.venv/bin/python src/fetch_ms.py 032221132 032221133
.venv/bin/python src/scan_cv.py          # ~50 min: imagery + masks + blobs
.venv/bin/python src/diff2.py            # remainder diff (MS + model)
.venv/bin/python src/score_cv.py         # verdicts from cached masks
.venv/bin/python src/stats.py            # recall, IoU, corregimiento table
.venv/bin/python src/export.py           # CSV, JOSM layer, QA sheet
.venv/bin/python src/site.py             # static site in docs/
echo "open docs/index.html (or: python -m http.server -d docs)"
