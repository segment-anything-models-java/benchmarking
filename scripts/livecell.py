#@String im_path
#@String bboxes
#@String points
#@String tmp_path

from io.bioimage.modelrunner.numpy import DecodeNumpy
from ai.nets.samj.ij import SAMJ_Annotator
from ai.nets.samj.communication.model import SAM2Tiny, SAM2Small, SAM2Large, EfficientSAM, EfficientViTSAML2
from ai.nets.samj.annotation import Mask
from net.imglib2 import Point

from java.util import ArrayList
from jarray import array as jarray
from java.awt import Rectangle

from ij import IJ
from net.imglib2.img import ImagePlusAdapter
from net.imglib2 import FinalInterval

import os
import json



def to_point_prompts_java(points_py):
    # If your data is nested like [[[[[x,y]], [[x,y]]]]], flatten it:
    lst = ArrayList()
    for instance in points_py:
        lst2 = ArrayList()
        for pp in instance:
            lst2.add(Point(pp[0], pp[1]))
        lst.add(lst2)
    return lst

def to_rect_prompts_java(bboxes_py):
    lst = ArrayList()
    for bb in bboxes_py:
        x, y, w, h = map(int, bb)
        lst.add(FinalInterval(jarray([x, y], 'l'), jarray([w + x - 1, h + y - 1], 'l')))
    return lst


FILE_PATH = tmp_path

## PARSE ARGS
point_prompts = to_point_prompts_java(json.loads(points))
rect_prompts = to_rect_prompts_java(json.loads(bboxes))


models = [SAM2Tiny(), SAM2Small(), SAM2Large(), EfficientSAM(), EfficientViTSAML2()]
models_str = ["tiny", "small", "large", "eff", "effvit"]

wrapped = ImagePlusAdapter.wrap(IJ.openImage(im_path))
for model, model_str in zip(models, models_str):

    model.setImage(wrapped, None)
    model.setReturnOnlyBiggest(True)

    for ii, bbox in enumerate(rect_prompts):
        segs = model.fetch2dSegmentation(bbox)
        mask = Mask.getMask(wrapped.dimensionsAsLongArray()[0], wrapped.dimensionsAsLongArray()[1], segs)
        name = "pred_" + model_str + "_bboxes_" + str(ii) + ".npy"
        DecodeNumpy.saveNpy(os.path.join(FILE_PATH, name), mask)
        
    for ii, point_list in enumerate(point_prompts):
        segs = model.fetch2dSegmentation(point_list, ArrayList())
        mask = Mask.getMask(wrapped.dimensionsAsLongArray()[0], wrapped.dimensionsAsLongArray()[1], segs)
        name = "pred_" + model_str + "_points_" + str(ii) + ".npy"
        DecodeNumpy.saveNpy(os.path.join(FILE_PATH, name), mask)

    model.closeProcess()
    del model
