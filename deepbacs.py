import numpy as np
import os
import subprocess
import random
import tifffile
import json
import platform

def iou_diagonal_fast(gt, pred):
    n = gt.max()
    ious = np.empty(n)
    for i in range(1, n+1):
        mask = (gt == i) | (pred == i)
        inter = np.count_nonzero((gt == i) & (pred == i))
        ious[i-1] = inter / np.count_nonzero(mask)
    return ious



N_POINT_PROMPTS = 1

SCRIPT_PATH = "scripts/deepbacs.py"

DEEPBACS_DIR = "/home/carlos/Pictures/samj_rebuttal/deepbacs/test/"
REAL_FOLDER = "brightfield"
MASK_FOLDER = "masks_RoiMap"
RESULTS_PATH = os.path.join(os.getcwd(), "tmp")
if not os.path.isdir(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

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



f_names = []
model_types = ["tiny", "small", "large", "eff", "effvit"]
promtp_types = ["points", "bboxes"]
scores_mat = np.zeros((len(os.listdir(os.path.join(DEEPBACS_DIR, REAL_FOLDER))), len(model_types) * len(promtp_types)), dtype="float64")

for ii, ff in enumerate(os.listdir(os.path.join(DEEPBACS_DIR, REAL_FOLDER))):
    mask = tifffile.imread(os.path.join(DEEPBACS_DIR, MASK_FOLDER, ff))
    im = tifffile.imread(os.path.join(DEEPBACS_DIR, REAL_FOLDER, ff))
    bboxes = []
    points = []
    #for i in range(33, 34):
    for i in range(1, mask.max() + 1):
        inds = np.where(mask == i)
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
        points.extend(pps)

    command = [
        os.path.join(FIJI_PATH, FIJI_EXEC),
        "--ij2",
        "--headless",
        "--console",
        "--run",
        os.path.join(os.getcwd(), SCRIPT_PATH),
        f"im_path=\"{os.path.join(DEEPBACS_DIR, REAL_FOLDER, ff)}\", bboxes=\"{json.dumps(bboxes)}\", points=\"{json.dumps(points)}\", " \
        + f"tmp_path=\"{RESULTS_PATH}\""
    ]


    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
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
np.save("deepbacs.npy", scores_mat)