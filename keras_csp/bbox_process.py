from __future__ import division
import numpy as np
from nms_wrapper import nms

def parse_det(Y, C, score=0.1, down=4,scale='h'):
    seman = Y[0][0, :, :, 0]
    if scale=='h':
        height = np.exp(Y[1][0, :, :, 0])*down
        width = 0.41*height
    elif scale=='w':
        width = np.exp(Y[1][0, :, :, 0])*down
        height = width/0.41
    elif scale=='hw':
        height = np.exp(Y[1][0, :, :, 0])*down
        width = np.exp(Y[1][0, :, :, 1])*down
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = height[y_c[i], x_c[i]]
            w = width[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i]+0.5) * down - w / 2), max(0, (y_c[i]+0.5) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs

def parse_det_top(Y, C, score=0.1):
    seman = Y[0][0, :, :, 0]
    height = Y[1][0, :, :, 0]
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * 4
            w = 0.41 * h
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, x_c[i] * 4 + 2 - w / 2), max(0, y_c[i] * 4 + 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs

def parse_det_bottom(Y, C, score=0.1):
    seman = Y[0][0, :, :, 0]
    height = Y[1][0, :, :, 0]
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * 4
            w = 0.41 * h
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, x_c[i] * 4 + 2 - w / 2), max(0, y_c[i] * 4 + 2-h)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs

def parse_det_offset(Y, C, score=0.1,down=4):
    seman = Y[0][0, :, :, 0]
    height = Y[1][0, :, :, 0]
    offset_y = Y[2][0, :, :, 0]
    offset_x = Y[2][0, :, :, 1]
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs

def parse_wider_offset(Y, C, score=0.1,down=4,nmsthre=0.5):
    seman = Y[0][0, :, :, 0]
    height = Y[1][0, :, :, 0]
    width = Y[1][0, :, :, 1]
    offset_y = Y[2][0, :, :, 0]
    offset_x = Y[2][0, :, :, 1]
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            w = np.exp(width[y_c[i], x_c[i]]) * down
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            x1, y1 = min(x1, C.size_test[1]), min(y1, C.size_test[0])
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        #keep = nms(boxs, nmsthre, usegpu=False, gpu_id=0)
        #boxs = boxs[keep, :]
	boxs = soft_bbox_vote(boxs,thre=nmsthre)
    return boxs

def soft_bbox_vote(det,thre=0.35,score=0.05):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= thre)[0]
        det_accu = det[merge_index, :]
        det_accu_iou = o[merge_index]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            soft_det_accu = det_accu.copy()
            soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
            soft_index = np.where(soft_det_accu[:, 4] >= score)[0]
            soft_det_accu = soft_det_accu[soft_index, :]

            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score

            if soft_det_accu.shape[0] > 0:
                det_accu_sum = np.row_stack((soft_det_accu, det_accu_sum))

            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    order = dets[:, 4].ravel().argsort()[::-1]
    dets = dets[order, :]
    return dets

def bbox_vote(det,thre):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    # det = det[np.where(det[:, 4] > 0.2)[0], :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= thre)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    return dets
