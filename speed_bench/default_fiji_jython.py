from io.bioimage.modelrunner.numpy import DecodeNumpy
from ai.nets.samj.ij import SAMJ_Annotator
from ai.nets.samj.communication.model import SAM2Tiny, SAM2Small, SAM2Large, EfficientSAM, EfficientViTSAML2
from ai.nets.samj.annotation import Mask
from net.imglib2 import Point

from java.util import ArrayList
from jarray import array as jarray
from java.awt import Rectangle

from ij import IJ
from ij import ImagePlus
from ij.plugin import CompositeConverter
from net.imglib2.img import ImagePlusAdapter
from net.imglib2 import FinalInterval

import os
import json
from time import time



def to_point_prompts_java(points_py):
    # If your data is nested like [[[[[x,y]], [[x,y]]]]], flatten it:
    lst = ArrayList()
    for instance in points_py:
        lst2 = ArrayList()
        for pp in instance:
            if pp[0] == -1 and pp[1] == -1:
                continue
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


models = [SAM2Tiny(), SAM2Small(), SAM2Large(), EfficientSAM(), EfficientViTSAML2(), ]
models_str = ["tiny", "small", "large", "eff", "effvit", ]

imp = IJ.openImage(im_path)
isColorRGB = imp.getType() == ImagePlus.COLOR_RGB
if isColorRGB:
    imp = CompositeConverter.makeComposite(imp)
wrapped = ImagePlusAdapter.wrap(imp)

## WARM UP
for model, model_str in zip(models[:1], models_str[:1]):

    start_time = time()
    model.setImage(wrapped, None)
    end_time = time()
    print("WARMUP --- Loaging time for " + model_str + " ---- " + str(end_time - start_time))
    model.setReturnOnlyBiggest(True)
    start_time = time()
    for ii, bbox in enumerate(rect_prompts):
        segs = model.fetch2dSegmentation(bbox)
    end_time = time()

    time_per_prompt = (end_time - start_time) / len(rect_prompts)

    print("WARMUP --- time per prompt for " + model_str + " ---- " + str(time_per_prompt))
    print("")



for model, model_str in zip(models, models_str):

    start_time = time()
    model.setImage(wrapped, None)
    end_time = time()
    print("Loaging time for " + model_str + " ---- " + str(end_time - start_time))
    model.setReturnOnlyBiggest(True)
    start_time = time()
    for ii, bbox in enumerate(rect_prompts):
        segs = model.fetch2dSegmentation(bbox)
        
    for ii, point_list in enumerate(point_prompts):
        segs = model.fetch2dSegmentation(point_list, ArrayList())
    end_time = time()

    time_per_prompt = (end_time - start_time) / (len(rect_prompts) + len(point_prompts))

    print("time per prompt for " + model_str + " ---- " + str(time_per_prompt))
    print("")

    model.closeProcess()
    del model
