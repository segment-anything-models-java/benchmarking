from ai.models.samj import SAMJ_Annotator
from ai.models.samj.communication import SAM2Tiny, SAM2Small, SAM2Large, EfficientSAM, EfficientViTSAM
from ai.nets.samj.annotation import Mask

from io.bioimage.modelrunner.numpy import DecodeNumpy

from ij import IJ
from net.imglib2.img.display.imagej import ImageJFunctions

import os
import sys

FILE_PATH = "/home/carlos/Desktop/Fiji.app/samj/deepbacs/tmp/"

## PARSE ARGS
im_path = ""
point_prompts = ""
rect_prompts = ""

models = [SAM2Tiny(), SAM2Small(), SAM2Large(), EfficientSAM(), EfficientViTSAM()]
models_str = ["tiny", "small", "large", "eff", "effvit"]

for model, model_str in zip(models, models_str):

    wrapped = ImageJFunctions.wrap(IJ.open(im_path))
    model.setImage(wrapped, None)
    model.setReturnOnlyBiggest(True)

    mask = SAMJ_Annotator.samJReturnContours(model, wrapped, point_prompts, None)
    mask = Mask.getMask(wrapped.dimensionsAsLongArray()[0], wrapped.dimensionsAsLongArray()[1], mask)
    name = "pred_" + model_str + "_points.npy"
    DecodeNumpy.saveNpy(os.path.join(FILE_PATH, name), mask)

    mask = SAMJ_Annotator.samJReturnContours(model, wrapped, None, rect_prompts)
    mask = Mask.getMask(wrapped.dimensionsAsLongArray()[0], wrapped.dimensionsAsLongArray()[1], mask)
    name = "pred_" + model_str + "_bboxes.npy"
    DecodeNumpy.saveNpy(os.path.join(FILE_PATH, name), mask)