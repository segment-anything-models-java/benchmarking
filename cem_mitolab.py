import json
import os
import platform
import random
import subprocess
import sys
import tempfile

import numpy as np
import tifffile
from pycocotools.coco import COCO

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
    for i in range(1, n + 1):
        mask = (gt == i) | (pred == i)
        inter = np.count_nonzero((gt == i) & (pred == i))
        ious[i - 1] = inter / np.count_nonzero(mask)
    return ious


N_POINT_PROMPTS = 3

SCRIPT_PATH = "scripts/default.py"

CEM_MITOLAB_DIR = "/home/carlos/Pictures/samj_rebuttal/mito_em/11037/data/cem_mitolab"
REAL_FOLDER = "images"
MASK_FOLDER = "masks"

RESULTS_PATH = os.path.join(os.getcwd(), "tmp_cem_mitolab")
if not os.path.isdir(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
POINT_PROMPTS = os.path.join(os.path.abspath(os.path.join(os.path.dirname(CEM_MITOLAB_DIR), '..')), "point_prompts")
if not os.path.isdir(POINT_PROMPTS):
    os.makedirs(POINT_PROMPTS)

MAX_STR_LEN = 20_000

FIJI_PATH = "/home/carlos/Desktop/Fiji.app"

if platform.system() == "Linux":
    FIJI_EXEC = "ImageJ-linux64"
elif platform.system() == "Windows":
    FIJI_EXEC = "ImageJ-win64.exe"
elif platform.system() == "Darwin":  # macOS
    FIJI_EXEC = " Contents/MacOS/ImageJ-macos-x64"
elif platform.system() == "Darwin" and ("arm64" in platform.machine() or "aarch64" in platform.machine()):  # macOS
    FIJI_EXEC = " Contents/MacOS/fiji-macos-arm64"
else:
    raise RuntimeError(f"Unsupported OS: {platform.system()}")


f_names = []
model_types = ["tiny", "small", "large", "eff", "effvit"]
promtp_types = ["points", "bboxes"]

all_files = os.listdir(CEM_MITOLAB_DIR)
all_files.sort()
cc = 0
for ff in (all_files):
    all_file_2 = os.listdir(os.path.join(CEM_MITOLAB_DIR, ff, REAL_FOLDER))
    all_file_2.sort()
    for ff2 in all_file_2:
        cc += 1
scores_mat = np.zeros((cc, len(model_types) * len(promtp_types)), dtype="float64")

cc = -1
for ff in (all_files):
    all_file_2 = os.listdir(os.path.join(CEM_MITOLAB_DIR, ff, REAL_FOLDER))
    all_file_2.sort()
    for ff2 in all_file_2:
        cc += 1
        print(cc)
        f_names.append(ff2)
        im = tifffile.imread(os.path.join(CEM_MITOLAB_DIR, ff, REAL_FOLDER, ff2))
        mask = tifffile.imread(os.path.join(CEM_MITOLAB_DIR, ff, MASK_FOLDER, ff2))
        mask = split_disconnected(mask, connectivity=2)

        bboxes = []
        points = []
        # for i in range(33, 34):
        for i in range(1, mask.max() + 1):
            m = mask == i
            inds = np.where(m)
            bottom, top = int(inds[0].min()), int(inds[0].max())
            left, right = int(inds[1].min()), int(inds[1].max())
            # bboxes.append([[left, bottom, right - left, top - bottom]])
            bboxes.extend([[left, bottom, right - left + 1, top - bottom + 1]])

            point_inds = random.sample(
                range(inds[0].shape[0]),
                np.min([N_POINT_PROMPTS, inds[0].shape[0]]),
            )
            xs = inds[1][point_inds]
            ys = inds[0][point_inds]
            pps = []
            for j in range(N_POINT_PROMPTS):
                # pps.append([[int(xs[j]), int(ys[j])]])
                if j >= inds[0].shape[0]:
                    pps.append([int(-1), int(-1)])
                    continue
                pps.append([int(xs[j]), int(ys[j])])

            # points.append([pps])
            points.append(pps)
        np.save(os.path.join(POINT_PROMPTS, ff + "_" + ff2 + ".npy"), np.array(points))

        with open(os.path.join(os.getcwd(), SCRIPT_PATH)) as og_script:
            script_content = og_script.read()
        bboxes_str = "\"" + json.dumps(bboxes) + "\""
        points_str = "\"" + json.dumps(points) + "\""
        if len(bboxes_str) > MAX_STR_LEN:
            with open(os.path.join(RESULTS_PATH, "bboxes.json"), "w") as f:
                json.dump(bboxes, f, indent=2)
                bboxes_str = f"json.dumps(json.load(open(r'{os.path.join(RESULTS_PATH, 'bboxes.json')}')))"
        if len(points_str) > MAX_STR_LEN:
            with open(os.path.join(RESULTS_PATH, "points.json"), "w") as f:
                json.dump(points, f, indent=2)
                points_str = f"json.dumps(json.load(open(r'{os.path.join(RESULTS_PATH, 'points.json')}')))"
        script_content = (
            f"import json\n"
            f"im_path=r'{os.path.join(CEM_MITOLAB_DIR, ff, REAL_FOLDER, ff2)}'\n"
            f"bboxes={bboxes_str}\npoints={points_str}\n"
            f"tmp_path=r'{RESULTS_PATH}'\n"
            + script_content
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as temp_script:
            temp_script.write(script_content)
            temp_script_path = temp_script.name

        command = [
            os.path.join(FIJI_PATH, FIJI_EXEC),
            "--ij2",
            "--headless",
            "--console",
            "--run",
            temp_script_path,
        ]

        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)
        if "[ERROR]" in result.stderr:
            import shlex

            print(shlex.join(command))
            print(result.stderr, file=sys.stderr)
            raise Exception()

        os.remove(temp_script_path)
        for j, model_type in enumerate(model_types):
            for k, prompt_type in enumerate(promtp_types):
                ious = []
                for pn in range(1, mask.max() + 1):
                    path_to_tmp = os.path.join(
                        RESULTS_PATH, f"pred_{model_type}_{prompt_type}_{pn - 1}.npy"
                    )
                    tmp_file = np.load(path_to_tmp).T
                    iou = iou_diagonal_fast((mask == pn) * 1, tmp_file)
                    ious.append(iou[0])
                ious = np.array(ious)
                scores_mat[cc, j * len(promtp_types) + k] = ious.mean()

import polars as pl

cols = []
for model_type in (model_types):
    for prompt_type in (promtp_types):
        cols.append(f"{model_type}_{prompt_type}")

df = pl.DataFrame(scores_mat, schema=cols)
df = df.with_columns(pl.Series("file_names", f_names))
df.write_csv("cem_mitolab.csv")
