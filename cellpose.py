import numpy as np
import os
import subprocess
import random
import tifffile
import json
import platform
import tempfile
from skimage.measure import label as cc_label

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

def iou_diagonal_fast(gt, pred):
    n = gt.max()
    ious = np.empty(n)
    for i in range(1, n+1):
        mask = (gt == i) | (pred == i)
        inter = np.count_nonzero((gt == i) & (pred == i))
        ious[i-1] = inter / np.count_nonzero(mask)
    return ious



N_POINT_PROMPTS = 3

SCRIPT_PATH = "scripts/default.py"

CELLPOSE_DIR = "/home/carlos/Pictures/samj_rebuttal/cellpose/"
REAL_FOLDER = "test"
MASK_FOLDER = "test"
RESULTS_PATH = os.path.join(os.getcwd(), "tmp")
if not os.path.isdir(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
POINT_PROMPTS = os.path.join(CELLPOSE_DIR, "point_prompts")
if not os.path.isdir(POINT_PROMPTS):
    os.makedirs(POINT_PROMPTS)

FIJI_PATH = "/home/carlos/Desktop/Fiji.app"

if platform.system() == "Linux":
    FIJI_EXEC = "ImageJ-linux64"
elif platform.system() == "Windows":
    FIJI_EXEC = "ImageJ-windows-x64.exe"
elif platform.system() == "Darwin":  # macOS
    FIJI_EXEC = " Contents/MacOS/ImageJ-macos-x64"
elif platform.system() == "Darwin" and ("arm64" in platform.machine() or "aarch64" in platform.machine()):  # macOS
    FIJI_EXEC = " Contents/MacOS/fiji-macos-arm64"
else:
    raise RuntimeError(f"Unsupported OS: {platform.system()}")



f_names = []
model_types = ["tiny", "small", "large", "eff", "effvit"]
promtp_types = ["points", "bboxes"]
scores_mat = np.zeros((len(os.listdir(os.path.join(CELLPOSE_DIR, REAL_FOLDER))), len(model_types) * len(promtp_types)), dtype="float64")

all_files = os.listdir(os.path.join(CELLPOSE_DIR, REAL_FOLDER))
all_files.sort()
for ii, ff in enumerate(all_files):
    last_point_ind = len(ff) - 1 - ff[::-1].index("_")
    mask_name = ff[:last_point_ind] + "_masks.png"
    f_names.append(ff)
    mask_pre = tifffile.imread(os.path.join(CELLPOSE_DIR, MASK_FOLDER, mask_name))
    mask = split_disconnected(mask_pre, connectivity=2)
    bboxes = []
    points = []
    for i in range(1, mask.max() + 1):
        inds = np.where(mask == i)
        bottom, top = int(inds[0].min()), int(inds[0].max())
        left, right = int(inds[1].min()), int(inds[1].max())
        #bboxes.append([[left, bottom, right - left, top - bottom]])
        bboxes.extend([[left, bottom, right - left + 1, top - bottom + 1]])

        point_inds = random.sample(range(inds[0].shape[0]), np.min([inds[0].shape[0], N_POINT_PROMPTS]))
        xs = inds[1][point_inds]
        ys = inds[0][point_inds]
        pps = []
        for j in range(N_POINT_PROMPTS):
            #pps.append([[int(xs[j]), int(ys[j])]])
            if j >= inds[0].shape[0]:
                pps.append([int(-1), int(-1)])
                continue
            pps.append([int(xs[j]), int(ys[j])])

        #points.append([pps])
        points.append(pps)
    np.save(os.path.join(POINT_PROMPTS, ff + ".npy"), np.array(points))

    with open(os.path.join(os.getcwd(), SCRIPT_PATH), "r") as f:
        modified_lines = f.read()
        
    modified_script = \
            f"im_path=r'{os.path.join(CELLPOSE_DIR, REAL_FOLDER, ff)}'\n" \
            + f"bboxes='{json.dumps(bboxes)}'\n" \
            + f"points='{json.dumps(points)}'\n" \
            + f"tmp_path=r'{RESULTS_PATH}'\n" \
            + "".join(modified_lines)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp_file:
        tmp_script_path = tmp_file.name
        tmp_file.write(modified_script.encode())

    command = [
        os.path.join(FIJI_PATH, FIJI_EXEC),
        "--ij2",
        "--headless",
        "--console",
        "--run",
        tmp_script_path
    ]


    # Run the command
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    finally:
        os.remove(tmp_script_path)
    if "[ERROR]" in result.stderr:
        import shlex
        print(shlex.join(command))
        print("something happened")
        raise Exception()

    for j, model_type in enumerate(model_types):
        for k, prompt_type in enumerate(promtp_types):
            ious = []
            for pn in range(1, mask.max() + 1):
                path_to_tmp = os.path.join(RESULTS_PATH, f"pred_{model_type}_{prompt_type}_{pn - 1}.npy")
                tmp_file = np.load(path_to_tmp).T
                iou = iou_diagonal_fast((mask == pn) * 1, tmp_file)
                ious.append(iou[0])
            ious = np.array(ious)
            scores_mat[ii, j * len(promtp_types) + k] = ious.mean()


import polars as pl
cols = []
for model_type in (model_types):
    for prompt_type in (promtp_types):
        cols.append(f"{model_type}_{prompt_type}")

df = pl.DataFrame(scores_mat, schema=cols)
df = df.with_columns(pl.Series("file_names", f_names))
df.write_csv("cellpose.csv")