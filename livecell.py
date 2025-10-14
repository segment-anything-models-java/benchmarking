import numpy as np
import os
import subprocess
import random
import tifffile
import json
import platform
from pycocotools.coco import COCO

def iou_diagonal_fast(gt, pred):
    n = gt.max()
    ious = np.empty(n)
    for i in range(1, n+1):
        mask = (gt == i) | (pred == i)
        inter = np.count_nonzero((gt == i) & (pred == i))
        ious[i-1] = inter / np.count_nonzero(mask)
    return ious



N_POINT_PROMPTS = 3

SCRIPT_PATH = "scripts/livecell.py"

LIVECELL_DIR = "/home/carlos/Pictures/samj_rebuttal/livecell/"
REAL_FOLDER = "livecell_test_images"
ANN_FILE = os.path.join(LIVECELL_DIR, "livecell_coco_test.json")
RESULTS_PATH = os.path.join(os.getcwd(), "tmp")
if not os.path.isdir(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
POINT_PROMPTS = os.path.join(LIVECELL_DIR, "point_prompts")
if not os.path.isdir(POINT_PROMPTS):
    os.makedirs(POINT_PROMPTS)



FIJI_PATH = "/home/carlos/Desktop/Fiji.app"

if platform.system() == "Linux":
    FIJI_EXEC = "ImageJ-linux64"
elif platform.system() == "Windows":
    FIJI_EXEC = "fiji-windows-x64.exe"
elif platform.system() == "Darwin":  # macOS
    FIJI_EXEC = " Contents/MacOS/fiji-macos-x64"
elif platform.system() == "Darwin" and ("arm64" in platform.machine() or "aarch64" in platform.machine()):  # macOS
    FIJI_EXEC = " Contents/MacOS/fiji-macos-arm64"
else:
    raise RuntimeError(f"Unsupported OS: {platform.system()}")


coco = COCO(ANN_FILE)


f_names = []
model_types = ["tiny", "small", "large", "eff", "effvit"]
promtp_types = ["points", "bboxes"]

n_ims = len(coco.loadImgs(coco.getImgIds()))
scores_mat = np.zeros((n_ims, len(model_types) * len(promtp_types)), dtype="float64")



for ii, coco_info in enumerate(coco.loadImgs(coco.getImgIds())):
    f_names.append(coco_info["file_name"])
    im = tifffile.imread(os.path.join(LIVECELL_DIR, REAL_FOLDER, coco_info["file_name"]))

    H, W = coco_info["height"], coco_info["width"]
    ann_ids = coco.getAnnIds(imgIds=[coco_info["id"]])
    anns = coco.loadAnns(ann_ids)

    bboxes = []
    points = []
    #for i in range(33, 34):
    for i, ann in enumerate(anns, start=1):
        m = coco.annToMask(ann).astype(bool)
        inds = np.where(m)
        bottom, top = int(inds[0].min()), int(inds[0].max())
        left, right = int(inds[1].min()), int(inds[1].max())
        #bboxes.append([[left, bottom, right - left, top - bottom]])
        bboxes.extend([[left, bottom, right - left + 1, top - bottom + 1]])

        point_inds = random.sample(range(inds[0].shape[0]), N_POINT_PROMPTS)
        xs = inds[1][point_inds]
        ys = inds[0][point_inds]
        pps = []
        for j in range(N_POINT_PROMPTS):
            #pps.append([[int(xs[j]), int(ys[j])]])
            pps.append([int(xs[j]), int(ys[j])])

        #points.append([pps])
        points.append(pps)
    np.save(os.path.join(POINT_PROMPTS, coco_info["file_name"] + ".npy"), np.array(points))

    command = [
        os.path.join(FIJI_PATH, FIJI_EXEC),
        "--ij2",
        "--headless",
        "--console",
        "--run",
        os.path.join(os.getcwd(), SCRIPT_PATH),
        f"im_path=\"{os.path.join(LIVECELL_DIR, REAL_FOLDER, coco_info['file_name'])}\", bboxes=\"{json.dumps(bboxes)}\", points=\"{json.dumps(points)}\", " \
        + f"tmp_path=\"{RESULTS_PATH}\""
    ]


    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
    if "[ERROR]" in result.stderr:
        import shlex
        print(shlex.join(command))
        print("something happened")
        raise Exception()

    for j, model_type in enumerate(model_types):
        for k, prompt_type in enumerate(promtp_types):
            ious = []
            for pn, ann in enumerate(anns, start=1):
                m = coco.annToMask(ann).astype(bool)
                path_to_tmp = os.path.join(RESULTS_PATH, f"pred_{model_type}_{prompt_type}_{pn - 1}.npy")
                tmp_file = np.load(path_to_tmp).T
                iou = iou_diagonal_fast(m * 1, tmp_file)
                ious.append(iou[0])
            ious = np.array(ious)
            scores_mat[ii, j * len(promtp_types) + k] = ious.mean()

np.save("livecell.npy", scores_mat)

import polars as pl
cols = []
for model_type in (model_types):
    for prompt_type in (promtp_types):
        cols.append(f"{model_type}_{prompt_type}")

df = pl.DataFrame(scores_mat, schema=cols)
df = df.with_columns(pl.Series("file_names", f_names))
df.write_csv("livecell.csv")