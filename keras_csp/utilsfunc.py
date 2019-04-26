from __future__ import division
import cv2
import numpy as np

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape

	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio
def format_img_channels(img, C):
	""" formats the image channels based on config """
	# img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	# img /= C.img_scaling_factor
	# img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img_batch(img, C):
	""" formats the image channels based on config """
	# img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, :, 0] -= C.img_channel_mean[0]
	img[:, :, :, 1] -= C.img_channel_mean[1]
	img[:, :, :, 2] -= C.img_channel_mean[2]
	# img /= C.img_scaling_factor
	# img = np.transpose(img, (2, 0, 1))
	# img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	# img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img #return img, ratio

def format_img_inria(img, C):
	img_h, img_w = img.shape[:2]
	# img_h_new, img_w_new = int(round(img_h/16)*16), int(round(img_w/16)*16)
	# img = cv2.resize(img, (img_w_new, img_h_new))
	img_h_new, img_w_new = int(np.ceil(img_h/16)*16), int(np.ceil(img_w/16)*16)
	paved_image = np.zeros((img_h_new, img_w_new, 3), dtype=img.dtype)
	paved_image[0:img_h,0:img_w] = img
	img = format_img_channels(paved_image, C)
	return img

def format_img_ratio(img, C, ratio):
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img = cv2.resize(img, None, None, fx=ratio, fy=ratio)
	# img = cv2.resize(img, None, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
	img = np.expand_dims(img, axis=0)
	return img #return img, ratio

def preprocess_input_test(x):
    x = x.astype(np.float32)
    x /= 255.
    x -= 0.5
    x *= 2.
    x = np.expand_dims(x, axis=0)
    return x

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

def intersection(ai, bi, area):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h/area

def box_grid_overlap(bboxes, get_img_output_length):
    width, height = 960, 540
    (resized_width, resized_height) = 960, 540
    # for VGG16, calculate the output map size
    (output_width, output_height) = get_img_output_length(resized_width, resized_height)

    downscale = float(16)

    num_bboxes = len(bboxes)
    # initialise output objectiveness
    y_grid_overlap = np.zeros((output_height, output_width))

    if num_bboxes > 0:
        # get the GT box coordinates, and resize to account for image resizing
        gta = np.zeros((num_bboxes, 4))
        gta[:,0],gta[:,1],gta[:,2],gta[:,3] = bboxes[:,0],bboxes[:,1],bboxes[:,2]+bboxes[:,0],bboxes[:,3]+bboxes[:,1]
        # for bbox_num in range(num_bboxes):
        #     # get the GT box coordinates, and resize to account for image resizing
        #     gta[bbox_num, 0] = bboxes[bbox_num][0]
        #     gta[bbox_num, 1] = bboxes[bbox_num][1]
        #     gta[bbox_num, 2] = bboxes[bbox_num][2]
        #     gta[bbox_num, 3] = bboxes[bbox_num][3]
        for ix in range(output_width):
            x1_anc = downscale * ix
            x2_anc = downscale * (ix + 1)
            for jy in range(output_height):
                y1_anc = downscale * jy
                y2_anc = downscale * (jy + 1)
                best_op = 0
                for b in range(num_bboxes):
                    grid = [x1_anc,y1_anc,x2_anc,y2_anc]
                    op = intersection(grid,gta[b,:],downscale**2)
                    best_op = op if op>best_op else best_op
                y_grid_overlap[jy,ix] = best_op

    y_grid_overlap = np.expand_dims(y_grid_overlap.reshape((1,-1)), axis=0)
    return  y_grid_overlap

def integrate_motion_score(bboxes, probs, pred, stride=16):
	if len(bboxes) == 0:
		return []
	probs = probs.reshape((-1,1))
	pred_anchor_score = np.zeros((probs.shape[0], 1))
	for i in range(len(bboxes)):
		x1, y1, x2, y2 = int(bboxes[i][0]/stride), int(bboxes[i][1]/stride), int(bboxes[i][2]/stride), int(bboxes[i][3]/stride)
		pred_anchor_score[i,0] = np.sum(pred[y1:y2, x1:x2])/((x2-x1)*(y2-y1))
	# alpha, belta = 2/0.7, 0.1
	# pred_anchor_score = np.where(pred_anchor_score>0.7, pred_anchor_score*alpha, pred_anchor_score)
	# pred_anchor_score = np.maximum(pred_anchor_score*alpha, np.ones_like(pred_anchor_score)*belta)
	# all_probs = pred_anchor_score
	all_probs = probs*pred_anchor_score
	return all_probs

def box_encoder_pp(anchors, boxes, Y1):
	A = np.copy(anchors[:, :, :, :4])
	A = A.reshape((-1, 4))

	# 1 calculate the iou scores
	max_overlaps = np.zeros((anchors.shape[0] * anchors.shape[1] * anchors.shape[2],), dtype=np.float32)
	if len(boxes) > 0:
		boxes[:, 2] += boxes[:, 0]
		boxes[:, 3] += boxes[:, 1]
		overlaps = bbox_overlaps(np.ascontiguousarray(A, dtype=np.float64),
								 np.ascontiguousarray(boxes, dtype=np.float64))
		max_overlaps = overlaps.max(axis=1)
	# normalize the iou scores
	if np.max(max_overlaps) > 0:
		max_overlaps = (max_overlaps - np.min(max_overlaps)) / np.max(max_overlaps)
	# 2 calculate the rpn scores
	rpn_score = Y1.reshape((-1)).astype(np.float32)
	inds = np.where(max_overlaps == 0)
	rpn_score[inds] = np.min(rpn_score)
	scores = (rpn_score + max_overlaps) / 2
	scores = np.expand_dims(scores.reshape((1,-1)).astype(np.float32), axis=0)
	return scores

def box_encoder_iou(anchors, boxes):
	A = np.copy(anchors[:, :, :, :4])
	A = A.reshape((-1, 4))

	max_overlaps = np.zeros((anchors.shape[0] * anchors.shape[1] * anchors.shape[2],), dtype=np.float32)
	if len(boxes) > 0:
		boxes[:, 2] += boxes[:, 0]
		boxes[:, 3] += boxes[:, 1]
		overlaps = bbox_overlaps(np.ascontiguousarray(A, dtype=np.float64),
								 np.ascontiguousarray(boxes, dtype=np.float64))
		max_overlaps = overlaps.max(axis=1)
	scores = np.expand_dims(max_overlaps.reshape((1,-1)).astype(np.float32), axis=0)
	return scores
