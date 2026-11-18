"""Model evaluation harness: score the HOTOSM DINOv3 UperNet against OSM
labels for a block of imagery tiles.

Usage:
    python eval_model.py zoom tx ty nx ny      # tile block, e.g. 18 73165 124465 2 2
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np
import onnxruntime as ort
from shapely.geometry import shape
from shapely.ops import unary_union

from common import DATA, OUT, tile_bounds, lonlat_to_tile
from imagery import fetch, tile_path

MODEL = "cache/hot_dinov3s.onnx"
NORM_MEAN = np.array([0.4296737853453577, 0.4001659668453235, 0.34333372802741474],
                     dtype=np.float32)
NORM_STD = np.array([0.2056069389373208, 0.16738555558380538, 0.1598986422586595],
                    dtype=np.float32)


def load_model(path=MODEL):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 12
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def prep(img_bgr, norm=True):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if norm:
        rgb = (rgb - NORM_MEAN) / NORM_STD
    return rgb.transpose(2, 0, 1)[None]


def building_prob(sess, img_bgr, norm=True):
    """Building probability for a 256x256 RGB window.

    The exported graph has a 3-channel softmax head; channel 0 is the
    building class (verified on hotosm/vhr-building-segmentation: mean IoU
    0.68 with channel 0, 0.0 with channel 1).
    """
    logits = sess.run(None, {"image": prep(img_bgr, norm)})[0][0]  # (3,H,W)
    e = np.exp(logits - logits.max(0, keepdims=True))
    return (e / e.sum(0, keepdims=True))[0]


def mosaic(z, x0, y0, nx, ny):
    img = np.zeros((ny * 256, nx * 256, 3), np.uint8)
    for j in range(ny):
        for i in range(nx):
            p = tile_path(z, x0 + i, y0 + j)
            t = cv2.imread(str(p))
            if t is None:
                t = np.zeros((256, 256, 3), np.uint8)
            img[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = t
    return img


def sliding(sess, img, win=256, stride=96, norm=True):
    h, w = img.shape[:2]
    prob = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for y in range(0, h - win + 1, stride):
        for x in range(0, w - win + 1, stride):
            p = building_prob(sess, img[y:y + win, x:x + win], norm)
            prob[y:y + win, x:x + win] += p
            cnt[y:y + win, x:x + win] += 1
    return prob / np.maximum(cnt, 1)


def osm_mask(z, x0, y0, nx, ny, shape_img):
    """Rasterise OSM building polygons that intersect the tile block."""
    w, s, e, n = tile_bounds(x0, y0, z)[0], tile_bounds(x0, y0 + ny - 1, z)[1], \
        tile_bounds(x0 + nx - 1, y0, z)[2], tile_bounds(x0, y0, z)[3]
    feats = json.load(open(OUT / "osm_buildings.geojson"))["features"]
    mask = np.zeros(shape_img, np.uint8)
    for f in feats:
        g = shape(f["geometry"])
        if not g.intersects(__import__("shapely").box(w, s, e, n)):
            continue
        polys = getattr(g, "geoms", [g])
        for p in polys:
            ext = np.array(p.exterior.coords)
            tx = (ext[:, 0] + 180) / 360 * 2 ** z
            ty = (1 - np.arcsinh(np.tan(np.radians(ext[:, 1]))) / np.pi) / 2 * 2 ** z
            px = (tx - x0) * 256
            py = (ty - y0) * 256
            cv2.fillPoly(mask, [np.round(np.stack([px, py], 1)).astype(np.int32)], 1)
    return mask


def metrics(pred, gt):
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    return {
        "iou": round(tp / max(1, tp + fp + fn), 3),
        "precision": round(tp / max(1, tp + fp), 3),
        "recall": round(tp / max(1, tp + fn), 3),
        "pred_px": int(pred.sum()), "gt_px": int(gt.sum()),
    }


def main():
    z, x0, y0, nx, ny = map(int, sys.argv[1:6])
    tiles = [(x0 + i, y0 + j) for i in range(nx) for j in range(ny)]
    fetch(z, tiles)
    img = mosaic(z, x0, y0, nx, ny)
    sess = load_model()
    gt = osm_mask(z, x0, y0, nx, ny, img.shape[:2])
    results = {}
    for norm in (True, False):
        prob = sliding(sess, img, norm=norm)
        for thr in (0.4371, 0.5, 0.6):
            results[f"norm={norm},thr={thr}"] = metrics((prob > thr).astype(np.uint8), gt)
        cv2.imwrite(f"cache/prob_norm{int(norm)}.png",
                    (np.clip(prob, 0, 1) * 255).astype(np.uint8))
    print(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    main()
