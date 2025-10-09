import numpy as np
import os
import subprocess
import random

def iou_diagonal_fast(gt, pred):
    n = gt.max()
    ious = np.empty(n)
    for i in range(1, n+1):
        mask = (gt == i) | (pred == i)
        inter = np.count_nonzero((gt == i) & (pred == i))
        ious[i-1] = inter / np.count_nonzero(mask)
    return ious



N_POINT_PROMPTS = 3

SCRIPT_NAME = ""

DEEPBACS_DIR = ""
REAL_FOLDER = ""
MASK_FOLDER = ""
RESULTS_PATH = "/home/carlos/Desktop/Fiji.app/samj/deepbacs/tmp/"

f_names = []
model_types = ["tiny", "small", "large", "eff", "effvit"]
promtp_types = ["points", "bboxes"]
scores_mat = np.zeros((len(os.listdir(os.path.join(DEEPBACS_DIR, REAL_FOLDER))), len(model_types) * len(promtp_types)), dtype="float64")

for ii, ff in enumerate(os.listdir(os.path.join(DEEPBACS_DIR, REAL_FOLDER))):
    mask = np.load(os.path.join(DEEPBACS_DIR, MASK_FOLDER, ff))
    im = np.load(os.path.join(DEEPBACS_DIR, MASK_FOLDER, ff))
    bboxes = [[]]
    points = [[[]]]
    for i in range(1, mask.max() + 1):
        inds = np.where(mask == i)
        bottom, top = inds[0].min(), inds[0].max()
        left, right = inds[1].min(), inds[1].max()
        bboxes.append([[left, bottom, right, top]])

        point_inds = random.sample(range(inds[0].shape[0]), N_POINT_PROMPTS)
        xs = inds[0][point_inds]
        ys = inds[1][point_inds]
        pps = [[]]
        for j in range(N_POINT_PROMPTS):
            pps.append([[xs[j], ys[j]]])

        points.append([pps])

    command = [
        "/home/carlos/Desktop/Fiji.app/fiji-linux-x64",
        "--headless",
        "--console",
        f"/home/carlos/Desktop/Fiji.app/samj/scripts/{SCRIPT_NAME}"
    ]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    for j, model_type in enumerate(model_types):
        for k, prompt_type in enumerate(promtp_types):
            path_to_tmp = os.path.join(RESULTS_PATH, f"pred_{model_type}_{prompt_type}.npy")
            tmp_file = np.load(path_to_tmp)
            iou = iou_diagonal_fast(mask, tmp_file)
            scores_mat[ii, j * len(model_types) + k] = iou