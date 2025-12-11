#!/usr/bin/env python3
"""
Create one mask image per ROI in a GeoJSON file.

Usage:
    python geojson_to_instance_masks.py pos7_fr_60.tif_sam2_small.geojson output_masks/
"""

import os
import math
import json
from pathlib import Path

from PIL import Image, ImageDraw

import tifffile

import numpy as np
from skimage.measure import label as cc_label

import polars as pl




def relabel_consecutive(mask: np.ndarray) -> np.ndarray:
    """
    Relabel an instance segmentation mask so that foreground labels are consecutive (1..N).
    Background (0) is preserved as 0. Order is by ascending original label, so any
    skipped numbers cause subsequent labels to shift down.

    Examples:
      [0,1,1,2,0,8] -> [0,1,1,2,0,3]

    Parameters
    ----------
    mask : np.ndarray
        Integer array of any shape. 0 is background; positive integers are object labels.

    Returns
    -------
    np.ndarray
        New mask with labels in 1..N (0 remains 0). Dtype chosen to safely hold N.
    """
    mask = np.asarray(mask)
    if mask.size == 0:
        return mask.copy()

    # Foreground positions
    fg = mask > 0
    if not np.any(fg):
        return mask.copy()

    # Unique positive labels, sorted
    uniq = np.unique(mask[fg])  # strictly > 0 and sorted

    # Map each positive label to its new consecutive id:
    # For any value v in uniq, its position in uniq is searchsorted(v),
    # so new_id = index + 1  (1-based)
    # This avoids building a huge lookup table up to max(label).
    out = mask.copy()
    out_fg = out[fg]
    out[fg] = 1 + np.searchsorted(uniq, out_fg)

    # Choose a safe integer dtype that can hold N labels
    n = int(uniq.size)
    safe_dtype = np.min_scalar_type(n)
    if np.issubdtype(out.dtype, np.integer) and np.iinfo(out.dtype).max >= n:
        return out  # original dtype can hold the result
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
        sizes = np.bincount(cc.ravel())[1:]     # sizes of components 1..n
        keep = int(sizes.argmax() + 1)          # largest keeps original label
        for c in range(1, n + 1):
            if c != keep:
                out[cc == c] = next_id
                next_id += 1
    return out

def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    # Assume a, b are binary (0/1) masks with same shape.
    inter = np.count_nonzero((a != 0) & (b != 0))
    if inter == 0:
        return 0.0
    union = np.count_nonzero((a != 0) | (b != 0))
    return inter / union


def geometry_rings(geometry):
    """
    Yield rings for a geometry as lists of (x, y).
    Returns an iterator of lists: [ring0, ring1, ...]
    ring0 is the outer boundary; others are holes.
    """
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
    # Create blank mask (8-bit, 0 background)
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    rings = list(geometry_rings(geom))
    if not rings:
        return np.zeros((height, width), dtype=np.uint8)

    # (x, y) are assumed already in pixel coords.
    outer = rings[0]
    draw.polygon(outer, outline=0, fill=1)

    for hole in rings[1:]:
        draw.polygon(hole, outline=0, fill=0)

    return np.array(mask_img, dtype=np.uint8)


def load_geojson_masks(geojson_path: str, height: int, width: int) -> dict[str, np.ndarray]:
    """
    Load a GeoJSON file once and return {roi_name: binary_mask}.
    """
    geojson_path = Path(geojson_path)
    with geojson_path.open("r") as f:
        data = json.load(f)

    features = data.get("features", [])
    roi_masks = {}

    for idx, feat in enumerate(features):
        geom = feat.get("geometry")
        if not geom:
            continue

        props = feat.get("properties", {})
        roi_name = props.get("name", f"roi_{idx}")

        mask = rasterize_geometry_to_mask(geom, height, width)
        roi_masks[roi_name] = mask

    return roi_masks



#NEURIPS_JSON_DIR = "/home/carlos/Videos/QuPath-v0.6.0-Linux/neurips/results/neurips_save"
NEURIPS_JSON_DIR = "/home/carlos/Desktop/RESULTS_QUPATH/neurips_save"
NEURIPS_IM_DIR = "/home/carlos/Pictures/samj_rebuttal/neurips/Testing/Public/images"
NEURIPS_MASK_DIR = "/home/carlos/Pictures/samj_rebuttal/neurips/Testing/Public/labels"
RES_FILE = "/home/carlos/eclipse-workspace-test/scripts/res_qupath_second/neurips.csv"
os.makedirs(os.path.dirname(RES_FILE), exist_ok=True)

all_files = os.listdir(NEURIPS_IM_DIR)
all_files.sort()

models = [
    #"sam2_tiny",
    #"sam2_small",
    #"sam2_base",
    "vit_h_em",
    "vit_h_lm",
    #"vit_t",
]

promtp_types = ["points", "rect"]


scores_mat = np.zeros((len(all_files), len(models) * len(promtp_types)), dtype="float64")
f_names = []
for row_idx, im_name in enumerate(all_files):
    print(row_idx)
    gf_name = im_name
    last_point_ind = len(im_name) - 1 - im_name[::-1].index(".")
    mask_name = im_name[:last_point_ind] + "_label.tiff"
    f_names.append(im_name)
    mask = tifffile.imread(os.path.join(NEURIPS_MASK_DIR, mask_name))
    mask = relabel_consecutive(split_disconnected(mask, connectivity=2))

    H, W = mask.shape[:2]

    for i_model, model in enumerate(models):
        gm_name = f"{gf_name}_{model}.geojson"
        gm_path = os.path.join(NEURIPS_JSON_DIR, gm_name)

        # --- load all ROI masks for this image+model ONCE ---
        try:
            roi_masks = load_geojson_masks(gm_path, H, W)
        except FileNotFoundError:
            print("Missing GeoJSON:", gm_path)
            # leave zeros in scores_mat for this model
            continue

        for k, prompt_name in enumerate(promtp_types):
            ious = []
            # labels are 1..mask.max()
            for i_mask in range(1, mask.max() + 1):
                roi_name = f"{prompt_name}_{i_mask - 1}_{model}"
                single_gj_mask = roi_masks.get(roi_name, None)
                if single_gj_mask is None:
                    # not found → IoU = 0
                    # print(gm_name, roi_name)
                    ious.append(0.0)
                    continue

                gt = (mask == i_mask)
                pred = (single_gj_mask != 0)
                iou_val = binary_iou(gt, pred)
                ious.append(iou_val)

            ious = np.array(ious, dtype=float)
            nonzero = ious[ious > 0]
            mean_nonzero = nonzero.mean() if nonzero.size > 0 else 0.0
            scores_mat[row_idx, i_model * len(promtp_types) + k] = mean_nonzero
            #print(row_idx, model, prompt_name, mean_nonzero)


import polars as pl
cols = []
for model_type in (models):
    for prompt_type in (promtp_types):
        cols.append(f"{model_type}_{prompt_type}")

df = pl.DataFrame(scores_mat, schema=cols)
df = df.with_columns(pl.Series("file_names", f_names))
df.write_csv(RES_FILE)




    
