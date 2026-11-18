# Unmapped building footprints — San Miguelito, Panama City

**Live map: <https://ejrydhfs.github.io/san-miguelito-unmapped-buildings/>**

Machine-learning building detection over the district of San Miguelito (49.9 km²,
Panama City), compared against OpenStreetMap, and screened on 30 cm imagery from
August 2025.

| metric | value |
|---|--:|
| OSM buildings in the district | 12,088 |
| buildings detected in the district | 71,060 |
| OSM coverage of what exists | **17 %** |
| unmapped candidates | **62,709** (11.3 km², median 127.6 m²) |
| … building visible on imagery | 57,525 (92 %) |
| … probable | 2,583 (4 %) |
| … unconfirmed (sheds, patios, trees) | 2,601 (4 %) |
| candidates from the Microsoft layer | 57,106 |
| candidates found by the model only | 5,603 |

## Pipeline

1. **OSM** — district polygon (relation 8389079) + building ways/relations from Overpass.
2. **Detection layer** — Microsoft Global Buildings, 2026-02-03 release, Panama
   quadkeys `032221132`/`032221133`. Recall on already-mapped OSM buildings
   **86.2 %**, median shape IoU
   0.512.
3. **Remainder diff** — UTM 17N; each footprint minus OSM (and Microsoft, for the
   model-only source) buffered by 1.5 m. Uncovered parts ≥ 30 m² and ≥ 30 % of the
   footprint become candidates, so half-mapped rows of houses still surface.
4. **Scene check** — `hotosm/dinov3s-buildings` (DINOv3 ViT-S/16 + UperNet, ONNX)
   over all z18 tiles of the district (4576 tiles): blobs missing from both
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
| Rufina Alfaro | 2,749 | 10,963 | 10,299 |
| José Domingo Espinar | 1,073 | 9,824 | 9,255 |
| Belisario Porras | 338 | 9,781 | 8,665 |
| Belisario Frías | 121 | 9,299 | 8,572 |
| Omar Torrijos | 587 | 8,081 | 7,601 |
| Amelia Denis de Icaza | 2,539 | 6,122 | 5,462 |
| Arnulfo Arias | 2,706 | 4,807 | 4,315 |
| Mateo Iturralde | 48 | 2,383 | 2,165 |
| Victoriano Lorenzo | 1,930 | 1,449 | 1,191 |

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
