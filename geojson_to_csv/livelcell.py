import os
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import label as cc_label
from pycocotools.coco import COCO
import polars as pl


def relabel_consecutive(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.size == 0:
        return mask.copy()

    fg = mask > 0
    if not np.any(fg):
        return mask.copy()

    uniq = np.unique(mask[fg])
    out = mask.copy()
    out_fg = out[fg]
    out[fg] = 1 + np.searchsorted(uniq, out_fg)

    n = int(uniq.size)
    safe_dtype = np.min_scalar_type(n)
    if np.issubdtype(out.dtype, np.integer) and np.iinfo(out.dtype).max >= n:
        return out
    else:
        return out.astype(safe_dtype, copy=False)


def split_disconnected(mask: np.ndarray, connectivity: int = 2) -> np.ndarray:
    out = mask.copy()
    next_id = int(out.max()) + 1
    for lab in np.unique(out):
        if lab == 0:
            continue
        cc = cc_label(out == lab, connectivity=connectivity)
        n = int(cc.max())
        if n <= 1:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        keep = int(sizes.argmax() + 1)
        for c in range(1, n + 1):
            if c != keep:
                out[cc == c] = next_id
                next_id += 1
    return out


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0
    union = np.count_nonzero(a | b)
    return inter / union


def geometry_rings(geometry):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if gtype == "Polygon":
        for ring in coords:
            yield [(x, y) for x, y in ring]
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                yield [(x, y) for x, y in ring]


def rasterize_geometry_to_mask(geom, height, width) -> np.ndarray:
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    rings = list(geometry_rings(geom))
    if not rings:
        return np.zeros((height, width), dtype=np.uint8)

    outer = rings[0]
    draw.polygon(outer, outline=0, fill=1)

    for hole in rings[1:]:
        draw.polygon(hole, outline=0, fill=0)

    return np.array(mask_img, dtype=np.uint8)


def load_geojson_masks(geojson_path: str, height: int, width: int) -> dict[str, np.ndarray]:
    geojson_path = Path(geojson_path)
    with geojson_path.open("r") as f:
        data = json.load(f)

    features = data.get("features", [])
    roi_masks: dict[str, np.ndarray] = {}

    for idx, feat in enumerate(features):
        geom = feat.get("geometry")
        if not geom:
            continue
        props = feat.get("properties", {})
        roi_name = props.get("name", f"roi_{idx}")
        mask = rasterize_geometry_to_mask(geom, height, width)
        roi_masks[roi_name] = mask.astype(bool)  # store as boolean directly

    return roi_masks


# --- Paths & config ---

LIVECELL_DIR = "/home/carlos/Pictures/samj_rebuttal/livecell"
ANN_FILE = os.path.join(LIVECELL_DIR, "livecell_coco_test.json")
LIVELCELL_JSON_DIR = "/home/carlos/Desktop/RESULTS_QUPATH/livecell_save"
RES_FILE = "/home/carlos/eclipse-workspace-test/scripts/res_qupath_second/livecell.csv"
os.makedirs(os.path.dirname(RES_FILE), exist_ok=True)

models = [
    #"sam2_tiny",
    #"sam2_small",
    #"sam2_base",
    "vit_h_em",
    #"vit_h_lm",
    #"vit_t",
]

promtp_types = ["points", "rect"]

coco = COCO(ANN_FILE)

img_ids = coco.getImgIds()
imgs = coco.loadImgs(img_ids)

scores_mat = np.zeros((len(imgs), len(models) * len(promtp_types)), dtype="float64")
f_names = []

for row_idx, coco_info in enumerate(imgs):
    print(row_idx)
    gf_name = coco_info["file_name"]
    f_names.append(gf_name)

    H, W = coco_info["height"], coco_info["width"]
    ann_ids = coco.getAnnIds(imgIds=[coco_info["id"]])
    anns = coco.loadAnns(ann_ids)

    # --- precompute all GT masks ONCE per image ---
    gt_masks = [coco.annToMask(ann).astype(bool) for ann in anns]
    n_inst = len(gt_masks)

    for i_model, model in enumerate(models):
        gm_name = f"{gf_name}_{model}.geojson"
        gm_path = os.path.join(LIVELCELL_JSON_DIR, gm_name)

        # --- load all ROI masks for this image+model ONCE ---
        try:
            roi_masks = load_geojson_masks(gm_path, H, W)
        except FileNotFoundError:
            print("Missing GeoJSON:", gm_path)
            continue

        for k, prompt_name in enumerate(promtp_types):
            ious = np.zeros(n_inst, dtype=float)

            # iterate over instances with index so it matches naming
            for inst_idx in range(n_inst):
                roi_name = f"{prompt_name}_{inst_idx}_{model}"
                pred = roi_masks.get(roi_name, None)
                if pred is None:
                    # stays 0.0
                    continue

                gt = gt_masks[inst_idx]
                iou_val = binary_iou(gt, pred)
                ious[inst_idx] = iou_val

            nonzero = ious[ious > 0]
            mean_nonzero = nonzero.mean() if nonzero.size > 0 else 0.0
            scores_mat[row_idx, i_model * len(promtp_types) + k] = mean_nonzero


# --- Save results ---

cols = [f"{model_type}_{prompt_type}"
        for model_type in models
        for prompt_type in promtp_types]

df = pl.DataFrame(scores_mat, schema=cols)
df = df.with_columns(pl.Series("file_names", f_names))
df.write_csv(RES_FILE)
