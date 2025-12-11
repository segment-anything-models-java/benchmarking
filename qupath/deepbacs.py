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



DEEPBACS_DIR = "/home/carlos/Pictures/samj_rebuttal/deepbacs/ims/"
REAL_FOLDER = "src"
MASK_FOLDER = "mask"

POINT_PROMPTS = os.path.join(DEEPBACS_DIR, "../point_prompts")
if not os.path.isdir(POINT_PROMPTS):
    os.makedirs(POINT_PROMPTS)

QUPATH_PATH = os.path.join(DEEPBACS_DIR, "qupath")

if not os.path.isdir(QUPATH_PATH):
    os.makedirs(QUPATH_PATH)

f_names = []
all_files = os.listdir(os.path.join(DEEPBACS_DIR, MASK_FOLDER))
all_files.sort()

for cc, ff in enumerate(all_files):
    print(cc)
    f_names.append(ff)
    mask = tifffile.imread(os.path.join(DEEPBACS_DIR, MASK_FOLDER, ff))
    mask = relabel_consecutive(split_disconnected(mask, connectivity=2))

    bboxes = []
    # for i in range(33, 34):
    for i in range(1, mask.max() + 1):
        m = mask == i
        inds = np.where(m)
        bottom, top = int(inds[0].min()), int(inds[0].max())
        left, right = int(inds[1].min()), int(inds[1].max())
        # bboxes.append([[left, bottom, right - left, top - bottom]])
        bboxes.extend([[left, bottom, right - left + 1, top - bottom + 1]])

    points = np.load(os.path.join(POINT_PROMPTS, ff + ".npy"))
    with open(os.path.join(QUPATH_PATH, f"point_prompts_{ff}.json"), "w") as f:
        json.dump(points.tolist(), f)
    with open(os.path.join(QUPATH_PATH, f"bbox_prompts_{ff}.json"), "w") as f:
        json.dump(bboxes, f)
print("Done!")