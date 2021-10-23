# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import pyximport; pyximport.install()
from keras_csp.nms.gpu_nms import gpu_nms
from keras_csp.nms.cpu_nms import cpu_nms
import numpy as np

def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):

    keep = cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
                        np.float32(sigma), np.float32(Nt),
                        np.float32(threshold),
                        np.uint8(method))
    return keep

def nms(dets, thresh, usegpu,gpu_id):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if usegpu:
        return gpu_nms(dets, thresh, device_id=gpu_id)
    else:
        return cpu_nms(dets, thresh)
