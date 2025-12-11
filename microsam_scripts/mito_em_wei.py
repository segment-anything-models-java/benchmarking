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
import polars as pl

import napari

import numpy as np
from skimage import io, img_as_float

from micro_sam.sam_annotator import _widgets as widgets
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator import util as vutil

from napari.layers import Image as im_layer


def call(image: im_layer, model_type: str, skip_validate=False):

    # Update the image embeddings:
    state = AnnotatorState()

    state.skip_recomputing_embeddings = False
    # Reset the state.
    state.reset_state()

    # Get image dimensions.
    if image.rgb:
        ndim = image.data.ndim - 1
        state.image_shape = image.data.shape[:-1]
    else:
        ndim = image.data.ndim
        state.image_shape = image.data.shape

    # Set layer scale
    state.image_scale = tuple(image.scale)

    # Process tile_shape and halo, set other data.
    tile_shape, halo = widgets._process_tiling_inputs(0, 0, 0, 0)
    image_data = image.data

    # @thread_worker()
    def compute_image_embedding():

        # Whether to prefer decoder.
        # With 'amg', it is set to 'False', else it is 'True' for the default 'auto' and 'ais' mode.
        prefer_decoder = True

        state.initialize_predictor(
            image_data, model_type=model_type, save_path=None, ndim=ndim,
            device="cuda", checkpoint_path=None, tile_shape=tile_shape, halo=halo,
            prefer_decoder=prefer_decoder, pbar_init=None,
            pbar_update=None,
        )

    compute_image_embedding()
    return state


def load_image(image_input):
    """
    Load an image that can be:
      - a NumPy array (returned as float)
      - a PNG or TIFF file path (loaded with skimage.io.imread)
    """
    if isinstance(image_input, np.ndarray):
        # Already a NumPy array — just ensure correct dtype
        return img_as_float(image_input)
    elif isinstance(image_input, str):
        # Assume it's a path to a PNG/TIF file
        return img_as_float(io.imread(image_input))
    else:
        raise TypeError("Input must be a NumPy array or a file path (str).")
    
def segment(viewer: "napari.viewer.Viewer", batched: bool = False) -> None:
    """Segment object(s) for the current prompts.

    Args:
        viewer: The napari viewer.
        batched: Choose if you want to segment multiple objects with point prompts.
    """

    shape = viewer.layers["image"].data.shape

    # get the current box and point prompts
    boxes, masks = vutil.shape_layer_to_prompts(viewer.layers["prompts"], shape)
    points, labels = vutil.point_layer_to_prompts(viewer.layers["point_prompts"], with_stop_annotation=False)

    predictor = AnnotatorState().predictor
    image_embeddings = AnnotatorState().image_embeddings
    seg = vutil.prompt_segmentation(
        predictor, points, labels, boxes, masks, shape, image_embeddings=image_embeddings,
        multiple_box_prompts=True, batched=batched, previous_segmentation=viewer.layers[0].data,
    )

    return seg


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

def iou_diagonal_fast(gt, pred):
    n = gt.max()
    ious = np.empty(n)
    for i in range(1, n+1):
        mask = (gt == i) | (pred == i)
        inter = np.count_nonzero((gt == i) & (pred == i))
        ious[i-1] = inter / np.count_nonzero(mask)
    return ious



N_POINT_PROMPTS = 3



MITO_EM_MITOLAB_DIR = r'C:\Users\carlos\datasets\mito_em_wei'
REAL_FOLDER = "EM30-R-im\\im"
MASK_FOLDER = "EM30-R-mito-train-val-v2\\mito-val-v2"
QUPATH_PATH = os.path.join(MITO_EM_MITOLAB_DIR, "qupath")
RES_DIR = "C:\\Users\\carlos\\git\\benchmarking\\scripts\\res_microsam\\mito_em_wei.csv"
os.makedirs(os.path.dirname(RES_DIR), exist_ok=True)



f_names = []
model_types = [
    "vit_t_lm",
    "vit_b_lm",
    "vit_l_lm",

    "vit_t_em_organelles",
    "vit_b_em_organelles",
    "vit_l_em_organelles",

    "vit_h",
            ]
promtp_types = ["points", "bboxes"]

all_files = os.listdir(os.path.join(MITO_EM_MITOLAB_DIR, MASK_FOLDER))
all_files.sort()
scores_mat = np.zeros((len(all_files), len(model_types) * len(promtp_types)), dtype="float64")
all_files.sort()





cc = -1
viewer = napari.Viewer()
point_prompt_layer = viewer.add_points(
    name="point_prompts",
    property_choices={"label": ["positive", "negative"]},
    border_color="label",
    border_color_cycle=vutil.LABEL_COLOR_CYCLE,
    symbol="o",
    face_color="transparent",
    border_width=0.5,
    size=12,
    ndim=2,
)
point_prompt_layer.border_color_mode = "cycle"
rect_prompt_layer = viewer.add_shapes(
    face_color="transparent", shape_type='rectangle', edge_color="green", edge_width=4, name="prompts", ndim=2,
)
for cc, ff in enumerate(all_files):
    print(cc)
    f_names.append(ff)
    mask = tifffile.imread(os.path.join(MITO_EM_MITOLAB_DIR, MASK_FOLDER, ff))
    mask = relabel_consecutive(split_disconnected(mask, connectivity=2))
    im_number = ff[len("seg"):-len(".tif")]
    im_name = "im" + im_number + ".png"


    with open(os.path.join(QUPATH_PATH, f"point_prompts_{ff}.json"), "r") as f:
        point_prompts = json.load(f)
    with open(os.path.join(QUPATH_PATH, f"bbox_prompts_{ff}.json"), "r") as f:
        bbox_prompts = json.load(f)




    im = load_image(os.path.join(MITO_EM_MITOLAB_DIR, REAL_FOLDER, im_name))
    image_layer = viewer.add_image(im, name="image")

    for j, model_type in enumerate(model_types):

        state = call(image=image_layer, model_type=model_type)


        ious = []
        for ip, pp in enumerate(point_prompts):
            point_prompt_layer.data = np.array([[b, a] for a, b in pp])
            seg = segment(viewer=viewer)
            iou = iou_diagonal_fast((mask == ip + 1) * 1, seg)
            ious.append(iou[0])
        point_prompt_layer.data = None

        ious = np.array(ious)
        scores_mat[cc, j * len(promtp_types)] = ious.mean()

        ious = []
        for ib, bb in enumerate(bbox_prompts):
            rect_prompt_layer.data = np.array([
                                                [bb[1], bb[0]],
                                                [bb[1], bb[0] + bb[2]],
                                                [bb[1] + bb[3], bb[0] + bb[2]],
                                                [bb[1] + bb[3], bb[0]]
                                            ])
            rect_prompt_layer.shape_type = "rectangle"
            seg = segment(viewer=viewer)
            if len(seg.shape) == 3:
                seg = seg[:, :, 0]
            iou = iou_diagonal_fast((mask == ib + 1) * 1, seg)
            ious.append(iou[0])
        rect_prompt_layer.data = np.array([])
        ious = np.array(ious)
        scores_mat[cc, j * len(promtp_types) + 1] = ious.mean()
    viewer.layers.remove("image")

cols = []
for model_type in (model_types):
    for prompt_type in (promtp_types):
        cols.append(f"{model_type}_{prompt_type}")

df = pl.DataFrame(scores_mat, schema=cols)
df = df.with_columns(pl.Series("file_names", f_names))
df.write_csv(RES_DIR)