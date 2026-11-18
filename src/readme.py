"""Generate the repository README.md from the pipeline outputs."""
from __future__ import annotations

import json

from common import OUT, ROOT

REPO = "ejrydhfs/san-miguelito-unmapped-buildings"
SITE_URL = "https://ejrydhfs.github.io/san-miguelito-unmapped-buildings/"


def main():
    st = json.loads((OUT / "diff2_stats.json").read_text())
    detail = json.loads((OUT / "stats_detail.json").read_text())
    cands = json.loads((OUT / "candidates.geojson").read_text())["features"]
    counts = {}
    for f in cands:
        v = f["properties"].get("verdict", "?")
        counts[v] = counts.get(v, 0) + 1
    total = len(cands)
    pct = lambda k: round(100 * counts.get(k, 0) / max(1, total))

    rows = "\n".join(
        f"| {c['corregimiento']} | {c['osm_buildings']:,} | {c['unmapped']:,} | "
        f"{c['unmapped_verified']:,} |"
        for c in detail["corregimientos"] if c["unmapped"])

    md = f"""# Unmapped building footprints — San Miguelito, Panama City

**Live map: <{SITE_URL}>**

Machine-learning building detection over the district of San Miguelito (49.9 km²,
Panama City), compared against OpenStreetMap, and screened on 30 cm imagery from
August 2025.

| metric | value |
|---|--:|
| OSM buildings in the district | {st['osm_buildings']:,} |
| buildings detected in the district | {st['ml_buildings']:,} |
| OSM coverage of what exists | **{100*st['osm_buildings']/max(1,st['ml_buildings']):.0f} %** |
| unmapped candidates | **{total:,}** ({st['area_m2']/1e6:.1f} km², median {st['median_area_m2']} m²) |
| … building visible on imagery | {counts.get('verified',0):,} ({pct('verified')} %) |
| … probable | {counts.get('probable',0):,} ({pct('probable')} %) |
| … unconfirmed (sheds, patios, trees) | {counts.get('unconfirmed',0):,} ({pct('unconfirmed')} %) |
| candidates from the Microsoft layer | {st['by_source'].get('ms',0):,} |
| candidates found by the model only | {st['by_source'].get('cv',0):,} |

## Pipeline

1. **OSM** — district polygon (relation 8389079) + building ways/relations from Overpass.
2. **Detection layer** — Microsoft Global Buildings, 2026-02-03 release, Panama
   quadkeys `032221132`/`032221133`. Recall on already-mapped OSM buildings
   **{detail['ml_layer_recall_on_osm_pct']} %**, median shape IoU
   {detail['ml_vs_osm_median_iou']}.
3. **Remainder diff** — UTM 17N; each footprint minus OSM (and Microsoft, for the
   model-only source) buffered by 1.5 m. Uncovered parts ≥ 30 m² and ≥ 30 % of the
   footprint become candidates, so half-mapped rows of houses still surface.
4. **Scene check** — `hotosm/dinov3s-buildings` (DINOv3 ViT-S/16 + UperNet, ONNX)
   over all z18 tiles of the district ({4576} tiles): blobs missing from both
   datasets are added, and every candidate is scored (`cv_cover` = share of the
   footprint within 1.8 m of a predicted building pixel; 91 % of existing OSM
   buildings score ≥ 0.6).
5. **Ranking** — 400 m cells ordered by candidate count.

## Layout

```
src/            pipeline (Python, uv venv)
  prepare_osm.py   Overpass JSON -> clean polygons
  fetch_ms.py      Microsoft footprints for the Panama quadkeys
  scan_cv.py       imagery download, segmentation, mask vectorisation
  diff2.py         remainder diff (Microsoft + model)
  score_cv.py      verdicts from cached masks
  stats.py         recall / IoU / per-corregimiento table
  export.py        CSV, JOSM layer, QA sheet
  site.py          static site -> docs/
docs/           GitHub Pages site (index.html + data/*.json)
out/            GeoJSON, CSV, .osm, QA sheet, stats
run.sh          full pipeline
```

Reproduce: `./run.sh` (Python 3.12, ~1 h, ~2 GB of caches).

## Per corregimiento

| corregimiento | OSM buildings | candidates | verified |
|---|--:|--:|--:|
{rows}

## Caveats

- **Do not bulk-import.** Microsoft's layer is CDLA-Permissive-2.0, not ODbL, and
  contains false positives (sheds, walls, patios). Use the map as a reference and
  trace buildings from the imagery — `verified` means "a building is here", not
  "this polygon is the building".
- The segmenter merges row-houses at 0.59 m/px and misses dark roofs; `unconfirmed`
  means "needs a human look".
- Footprints mapped with a 1–2 m offset are treated as mapped by design.
- Imagery is from August 2025; later construction is absent.

## Licences

Code: MIT. Derived layers: CC-BY-4.0. Data sources: OSM (ODbL), Microsoft Global
Buildings (CDLA-Permissive-2.0), model `hotosm/dinov3s-buildings` (CC-BY-4.0),
imagery © Esri / Maxar / Earthstar Geographics (attribution only, not redistributed).
"""
    (ROOT / "README.md").write_text(md)
    print(md[:800])


if __name__ == "__main__":
    main()
