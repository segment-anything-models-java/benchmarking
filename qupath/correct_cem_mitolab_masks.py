import json
import os
import platform
import random
import subprocess
import sys
import tempfile

import numpy as np
import tifffile
from skimage.measure import label as cc_label
from PIL import Image

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



CEM_MITOLAB_DIR = r'C:\Users\carlos\datasets\cem_mitolab\cem_mitolab\11037\data\cem_mitolab\cem_mitolab'
REAL_FOLDER = "images"
MASK_FOLDER = "masks"

RESULTS_PATH = os.path.join(os.getcwd(), "tmp_cem_mitolab")
if not os.path.isdir(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
POINT_PROMPTS = os.path.join(os.path.abspath(os.path.join(os.path.dirname(CEM_MITOLAB_DIR), '..')), "point_prompts")
if not os.path.isdir(POINT_PROMPTS):
    os.makedirs(POINT_PROMPTS)

QUPATH_PATH = os.path.join(r'C:\Users\carlos\datasets\cem_mitolab', "qupath")
QUPATH_PATH_IMS = os.path.join(r'C:\Users\carlos\datasets\cem_mitolab', "qupath_ims")

if not os.path.isdir(QUPATH_PATH):
    os.makedirs(QUPATH_PATH)
if not os.path.isdir(QUPATH_PATH_IMS):
    os.makedirs(QUPATH_PATH_IMS)



all_files = os.listdir(CEM_MITOLAB_DIR)
all_files.sort()
cc = 0
for _, ff in enumerate(all_files):
    all_file_2 = os.listdir(os.path.join(CEM_MITOLAB_DIR, ff, REAL_FOLDER))
    all_file_2.sort()
    for __, ff2 in enumerate(all_file_2):
        cc += 1

cc = -1
for ff in (all_files):
    all_file_2 = os.listdir(os.path.join(CEM_MITOLAB_DIR, ff, REAL_FOLDER))
    all_file_2.sort()
    for ff2 in all_file_2:
        cc += 1
        im = tifffile.imread(os.path.join(CEM_MITOLAB_DIR, ff, REAL_FOLDER, ff2))
        tifffile.imwrite(os.path.join(QUPATH_PATH_IMS, ff + '_' + ff2), im)
        mask = tifffile.imread(os.path.join(CEM_MITOLAB_DIR, ff, MASK_FOLDER, ff2))
        mask = relabel_consecutive(split_disconnected(mask, connectivity=2))

        bboxes = []
        points = []
        for i in range(1, mask.max() + 1):
            m = mask == i
            inds = np.where(m)
            bottom, top = int(inds[0].min()), int(inds[0].max())
            left, right = int(inds[1].min()), int(inds[1].max())
            # bboxes.append([[left, bottom, right - left, top - bottom]])
            bboxes.extend([[left, bottom, right - left + 1, top - bottom + 1]])

        if not os.path.exists(os.path.join(POINT_PROMPTS, ff + '_' + ff2 + ".npy")):
            print(cc, os.path.join(POINT_PROMPTS, ff + '_' + ff2 + ".npy"))
            continue
        points = np.load(os.path.join(POINT_PROMPTS, ff + '_' + ff2 + ".npy"))
        with open(os.path.join(QUPATH_PATH, f"point_prompts_{ff + '_' + ff2}.json"), "w") as f:
            json.dump(points.tolist(), f)
        with open(os.path.join(QUPATH_PATH, f"bbox_prompts_{ff + '_' + ff2}.json"), "w") as f:
            json.dump(bboxes, f)
print("Done!")