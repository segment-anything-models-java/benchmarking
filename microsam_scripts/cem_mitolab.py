import json
import os
import numpy as np
import tifffile
from skimage.measure import label as cc_label
from skimage import io, img_as_float
import polars as pl
import napari
import gc

import torch  # For GPU memory cleanup

from micro_sam.sam_annotator import _widgets as widgets
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator import util as vutil
from napari.layers import Image as im_layer


def call(image: im_layer, model_type: str):
    state = AnnotatorState()
    state.skip_recomputing_embeddings = False
    state.reset_state()

    # Image dimensions
    if image.rgb:
        ndim = image.data.ndim - 1
        state.image_shape = image.data.shape[:-1]
    else:
        ndim = image.data.ndim
        state.image_shape = image.data.shape

    state.image_scale = tuple(image.scale)
    tile_shape, halo = widgets._process_tiling_inputs(0, 0, 0, 0)
    image_data = image.data

    # Initialize predictor
    prefer_decoder = True
    state.initialize_predictor(
        image_data, model_type=model_type, save_path=None, ndim=ndim,
        device="cuda", checkpoint_path=None, tile_shape=tile_shape, halo=halo,
        prefer_decoder=prefer_decoder, pbar_init=None, pbar_update=None,
    )

    return state


def load_image(image_input):
    if isinstance(image_input, np.ndarray):
        return img_as_float(image_input)
    elif isinstance(image_input, str):
        return img_as_float(io.imread(image_input))
    else:
        raise TypeError("Input must be a NumPy array or a file path (str).")


def segment(viewer: "napari.viewer.Viewer", state: AnnotatorState, batched: bool = False):
    shape = viewer.layers["image"].data.shape
    boxes, masks = vutil.shape_layer_to_prompts(viewer.layers["prompts"], shape)
    points, labels = vutil.point_layer_to_prompts(viewer.layers["point_prompts"], with_stop_annotation=False)

    seg = vutil.prompt_segmentation(
        state.predictor, points, labels, boxes, masks, shape,
        image_embeddings=state.image_embeddings,
        multiple_box_prompts=True, batched=batched,
        previous_segmentation=viewer.layers[0].data,
    )
    return seg


def relabel_consecutive(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.size == 0:
        return mask.copy()
    fg = mask > 0
    if not np.any(fg):
        return mask.copy()
    uniq = np.unique(mask[fg])
    out = mask.copy()
    out[fg] = 1 + np.searchsorted(uniq, out[fg])
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


def iou_diagonal_fast(gt, pred):
    n = gt.max()
    ious = np.empty(n)
    for i in range(1, n+1):
        mask = (gt == i) | (pred == i)
        inter = np.count_nonzero((gt == i) & (pred == i))
        ious[i-1] = inter / np.count_nonzero(mask)
    return ious


# Paths and variables
CEM_MITOLAB_DIR = r'C:\Users\carlos\datasets\cem_mitolab\cem_mitolab\11037\data\cem_mitolab\cem_mitolab'
REAL_FOLDER = "images"
MASK_FOLDER = "masks"
QUPATH_PATH = os.path.join(r'C:\Users\carlos\datasets\cem_mitolab\qupath')
RES_DIR = "C:\\Users\\carlos\\git\\benchmarking\\scripts\\res_microsam\\cem_mitolab.csv"
os.makedirs(os.path.dirname(RES_DIR), exist_ok=True)

f_names = []
model_types = [
    "vit_t_lm", "vit_b_lm", "vit_l_lm",
    "vit_t_em_organelles", "vit_b_em_organelles", "vit_l_em_organelles",
    "vit_h",
]
promtp_types = ["points", "bboxes"]

all_files = sorted(os.listdir(CEM_MITOLAB_DIR))
n_ims = sum(len(os.listdir(os.path.join(CEM_MITOLAB_DIR, ff, REAL_FOLDER))) for ff in all_files)
scores_mat = np.zeros((n_ims, len(model_types) * len(promtp_types)), dtype="float64")

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
    face_color="transparent", shape_type='rectangle', edge_color="green",
    edge_width=4, name="prompts", ndim=2,
)

cc = -1
for ff in all_files:
    all_file_2 = sorted(os.listdir(os.path.join(CEM_MITOLAB_DIR, ff, REAL_FOLDER)))
    for ff2 in all_file_2:
        cc += 1
        print(cc)
        f_names.append(ff + "______" + ff2)

        im_path = os.path.join(CEM_MITOLAB_DIR, ff, REAL_FOLDER, ff2)
        mask_path = os.path.join(CEM_MITOLAB_DIR, ff, MASK_FOLDER, ff2)
        im = load_image(tifffile.imread(im_path))
        mask = relabel_consecutive(split_disconnected(tifffile.imread(mask_path), connectivity=2))

        with open(os.path.join(QUPATH_PATH, f"point_prompts_{ff + '_' + ff2}.json"), "r") as f:
            point_prompts = json.load(f)
        with open(os.path.join(QUPATH_PATH, f"bbox_prompts_{ff + '_' + ff2}.json"), "r") as f:
            bbox_prompts = json.load(f)

        image_layer = viewer.add_image(im, name="image")

        for j, model_type in enumerate(model_types):
            state = call(image=image_layer, model_type=model_type)

            # Points prompts
            ious = []
            for ip, pp in enumerate(point_prompts):
                point_prompt_layer.data = np.array([[b, a] for a, b in pp])
                seg = segment(viewer=viewer, state=state)
                iou = iou_diagonal_fast((mask == (ip + 1)) * 1, seg)
                ious.append(iou[0])
            point_prompt_layer.data = None
            ious = np.array(ious)
            if ious.size == 0:
                scores_mat[cc, j * len(promtp_types)] = 0.0
            else:
                scores_mat[cc, j * len(promtp_types)] = ious.mean()

            # BBox prompts
            ious = []
            for ib, bb in enumerate(bbox_prompts):
                rect_prompt_layer.data = np.array([
                    [bb[1], bb[0]],
                    [bb[1], bb[0] + bb[2]],
                    [bb[1] + bb[3], bb[0] + bb[2]],
                    [bb[1] + bb[3], bb[0]]
                ])
                seg = segment(viewer=viewer, state=state)
                if len(seg.shape) == 3:
                    seg = seg[:, :, 0]
                iou = iou_diagonal_fast((mask == (ib + 1)) * 1, seg)
                ious.append(iou[0])
            rect_prompt_layer.data = np.array([])
            ious = np.array(ious)
            if ious.size == 0:
                scores_mat[cc, j * len(promtp_types) + 1] = 0.0
            else:
                scores_mat[cc, j * len(promtp_types) + 1] = ious.mean()

            # Clean up GPU memory
            del state
            torch.cuda.empty_cache()
            gc.collect()

        print(scores_mat[cc])
        viewer.layers.remove("image")
        del im, mask
        gc.collect()

        if cc % 50 == 0:
            print("Saving intermediate results...")
            cols = [f"{m}_{p}" for m in model_types for p in promtp_types]
            df = pl.DataFrame(scores_mat[:len(f_names)], schema=cols)
            df = df.with_columns(pl.Series("file_names", f_names))
            df.write_csv(RES_DIR)

# Save final results
cols = [f"{m}_{p}" for m in model_types for p in promtp_types]
df = pl.DataFrame(scores_mat, schema=cols)
df = df.with_columns(pl.Series("file_names", f_names))
df.write_csv(RES_DIR)
